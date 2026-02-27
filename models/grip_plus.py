"""
GRIP++: Graph-based Interaction-aware Trajectory Prediction.

This model uses a graph-based approach to model interactions between
traffic agents. It constructs a dynamic interaction graph where nodes
represent agents and edges represent their spatial relationships.
Graph convolutions are used to propagate interaction information.

Unlike V2X-Graph, GRIP++ is a NON-COOPERATIVE model: it only uses
vehicle-side observations without infrastructure data.

Key components:
  1. Trajectory encoding with 1D CNN
  2. Dynamic graph construction based on spatial proximity
  3. Graph convolutional interaction modeling
  4. Multi-modal trajectory decoding

Reference:
  - Li et al., "GRIP++: Enhanced Graph-based Interaction-aware
    Trajectory Prediction for Autonomous Driving," arXiv:1907.07792.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryEncoder(nn.Module):
    """
    Encode trajectory using 1D CNN followed by GRU.

    The CNN extracts local motion patterns while the GRU captures
    temporal dependencies across the full sequence.
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            hidden_dim, hidden_dim, num_layers=1,
            batch_first=True, bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, T, input_dim)
        Returns:
            summary: (B, hidden_dim)
            sequence: (B, T, hidden_dim)
        """
        conv_out = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        conv_out = self.dropout(conv_out)
        gru_out, hidden = self.gru(conv_out)
        return hidden.squeeze(0), gru_out


class GraphConvLayer(nn.Module):
    """
    Graph convolutional layer for interaction modeling.

    Uses edge features (relative positions/velocities) to weight
    the message passing between nodes.
    """

    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        # Node feature transformation
        self.node_fc = nn.Linear(in_dim, out_dim)

        # Edge feature computation and weighting
        # +3 for relative position (2D) & distance (1D)
        self.edge_fc = nn.Sequential(
            nn.Linear(in_dim * 2 + 3, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1),
        )

        # Aggregation
        self.aggregate_fc = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, node_feats, adj_mask, rel_positions=None):
        """
        Args:
            node_feats:    (B, N, in_dim)
            adj_mask:      (B, N, N) adjacency mask
            rel_positions: (B, N, N, 2) relative positions between nodes
        Returns:
            updated_feats: (B, N, out_dim)
        """
        B, N, D = node_feats.shape

        # Transform node features
        h = self.node_fc(node_feats)  # (B, N, out_dim)

        # Compute edge weights
        h_i = node_feats.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, D)
        h_j = node_feats.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, D)

        if rel_positions is not None:
            # Compute relative distances
            rel_dist = torch.norm(rel_positions, dim=-1, keepdim=True)  # (B, N, N, 1)
            edge_input = torch.cat([h_i, h_j, rel_positions, rel_dist], dim=-1)
        else:
            # Use zero placeholders
            zeros = torch.zeros(B, N, N, 3, device=node_feats.device)
            edge_input = torch.cat([h_i, h_j, zeros], dim=-1)

        edge_weights = self.edge_fc(edge_input).squeeze(-1)  # (B, N, N)

        # Apply adjacency mask
        edge_weights = edge_weights.masked_fill(adj_mask == 0, float('-inf'))
        edge_weights = F.softmax(edge_weights, dim=-1)
        edge_weights = edge_weights.masked_fill(torch.isnan(edge_weights), 0.0)

        # Message passing
        messages = torch.bmm(edge_weights, h)  # (B, N, out_dim)
        aggregated = self.aggregate_fc(messages)

        # Residual connection + normalization
        if D == aggregated.shape[-1]:
            updated = self.norm(node_feats + aggregated)
        else:
            updated = self.norm(aggregated)

        return updated


class GRIPPlusPlus(nn.Module):
    """
    GRIP++: Graph-based Interaction-aware Trajectory Prediction.

    Architecture:
      1. Encode ego and neighbor trajectories with CNN+GRU
      2. Construct interaction graph
      3. Apply graph convolutions for interaction modeling
      4. Decode multi-modal future trajectories

    This is a NON-COOPERATIVE model.
    """

    def __init__(self, config):
        super().__init__()
        self.model_name = 'GRIP++'
        self.cooperative = False  # Non-cooperative

        input_dim = config['input_dim']
        hidden_dim = config['hidden_dim']
        output_dim = config['output_dim']
        dropout = config['dropout']
        num_gcn_layers = config.get('num_gcn_layers', 3)
        self.future_steps = config['future_steps']
        self.max_agents = config['max_agents']
        self.num_modes = config.get('num_modes', 1)

        # Trajectory encoders
        self.ego_encoder = TrajectoryEncoder(input_dim, hidden_dim, dropout)
        self.neighbor_encoder = TrajectoryEncoder(input_dim, hidden_dim, dropout)

        # Node type embedding
        self.node_type_embed = nn.Embedding(2, hidden_dim)  # 0: ego, 1: neighbor

        # Graph convolution layers
        self.gcn_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_gcn_layers)
        ])

        # Context aggregation (from graph to ego)
        self.context_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Multi-modal decoder
        self.mode_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.future_steps * output_dim),
            )
            for _ in range(self.num_modes)
        ])

        # Mode probability head
        self.mode_prob_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_modes),
        )

    def _build_interaction_graph(self, ego_feat, nbr_feats, neighbor_mask,
                                  ego_last_pos, nbr_last_pos):
        """
        Build interaction graph between ego and neighbors.

        Args:
            ego_feat:     (B, hidden_dim)
            nbr_feats:    (B, N, hidden_dim)
            neighbor_mask: (B, N)
            ego_last_pos: (B, 2) ego's last observed position
            nbr_last_pos: (B, N, 2) neighbors' last observed positions
        Returns:
            node_feats: (B, 1+N, hidden_dim)
            adj_mask:   (B, 1+N, 1+N)
            rel_pos:    (B, 1+N, 1+N, 2)
        """
        B = ego_feat.shape[0]
        N = nbr_feats.shape[1]
        device = ego_feat.device
        num_nodes = 1 + N

        # Construct node features
        node_feats = torch.zeros(B, num_nodes, ego_feat.shape[-1], device=device)
        node_feats[:, 0] = ego_feat + self.node_type_embed(
            torch.zeros(B, dtype=torch.long, device=device)
        )
        for i in range(N):
            node_feats[:, 1 + i] = nbr_feats[:, i] + self.node_type_embed(
                torch.ones(B, dtype=torch.long, device=device)
            )

        # Build adjacency mask
        adj_mask = torch.zeros(B, num_nodes, num_nodes, device=device)
        # Ego connects to all valid neighbors
        adj_mask[:, 0, 0] = 1.0
        for i in range(N):
            m = neighbor_mask[:, i]
            adj_mask[:, 0, 1 + i] = m
            adj_mask[:, 1 + i, 0] = m
            adj_mask[:, 1 + i, 1 + i] = m  # self-loop
            # Neighbor-neighbor connections
            for j in range(N):
                if i != j:
                    adj_mask[:, 1 + i, 1 + j] = m * neighbor_mask[:, j]

        # Compute relative positions
        all_pos = torch.zeros(B, num_nodes, 2, device=device)
        all_pos[:, 0] = ego_last_pos
        all_pos[:, 1:] = nbr_last_pos

        rel_pos = all_pos.unsqueeze(2) - all_pos.unsqueeze(1)  # (B, N+1, N+1, 2)

        return node_feats, adj_mask, rel_pos

    def forward(self, batch):
        """
        Args:
            batch: dict with keys:
                'history':       (B, T_h, 2)
                'neighbors':     (B, N, T_h, 2)
                'neighbor_mask': (B, N)

        Returns:
            dict with:
                'predictions': (B, K, T_f, 2)
                'mode_probs':  (B, K)
        """
        history = batch['history']
        neighbors = batch['neighbors']
        neighbor_mask = batch['neighbor_mask']

        B = history.shape[0]
        N = neighbors.shape[1]

        # 1. Encode trajectories
        ego_feat, _ = self.ego_encoder(history)  # (B, hidden_dim)

        nbr_flat = neighbors.reshape(B * N, neighbors.shape[2], -1)
        nbr_feat_flat, _ = self.neighbor_encoder(nbr_flat)
        nbr_feats = nbr_feat_flat.reshape(B, N, -1)
        nbr_feats = nbr_feats * neighbor_mask.unsqueeze(-1)

        # Get last observed positions for graph construction
        ego_last_pos = history[:, -1, :2]  # (B, 2)
        nbr_last_pos = neighbors[:, :, -1, :2]  # (B, N, 2)

        # 2. Build interaction graph
        node_feats, adj_mask, rel_pos = self._build_interaction_graph(
            ego_feat, nbr_feats, neighbor_mask, ego_last_pos, nbr_last_pos
        )

        # 3. Graph convolutions
        for gcn_layer in self.gcn_layers:
            node_feats = gcn_layer(node_feats, adj_mask, rel_pos)

        # 4. Extract enriched ego feature
        ego_enriched = node_feats[:, 0]  # (B, hidden_dim)

        # Also aggregate neighbor context via attention
        if N > 0:
            nbr_enriched = node_feats[:, 1:]  # (B, N, hidden_dim)
            attn_scores = self.context_attn(nbr_enriched).squeeze(-1)  # (B, N)
            attn_scores = attn_scores.masked_fill(neighbor_mask == 0, float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = attn_weights.masked_fill(torch.isnan(attn_weights), 0.0)
            nbr_context = torch.bmm(
                attn_weights.unsqueeze(1), nbr_enriched
            ).squeeze(1)  # (B, hidden_dim)
            context = ego_enriched + nbr_context
        else:
            context = ego_enriched

        # 5. Multi-modal decoding
        all_preds = []
        for head in self.mode_heads:
            pred = head(context)
            pred = pred.reshape(B, self.future_steps, -1)
            all_preds.append(pred)

        predictions = torch.stack(all_preds, dim=1)  # (B, K, T_f, 2)
        mode_logits = self.mode_prob_head(context)
        mode_probs = F.softmax(mode_logits, dim=-1)

        return {
            'predictions': predictions,
            'mode_probs': mode_probs,
        }
