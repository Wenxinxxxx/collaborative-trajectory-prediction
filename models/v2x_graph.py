"""
V2X-Graph: Graph Neural Network for Cooperative Trajectory Prediction.

This model builds a heterogeneous graph to model interactions between:
  - Vehicle agents (agent-agent edges)
  - Infrastructure observations (agent-infra edges)
  - Road topology (agent-lane edges)

It uses Graph Attention Networks (GAT) for message passing and fuses
multi-source information for cooperative prediction.

This is a COOPERATIVE model that leverages V2X (Vehicle-to-Everything)
communication data.

Reference:
  - Yu et al., "V2X-Seq: A Large-Scale Sequential Dataset for
    Vehicle-Infrastructure Cooperative Perception and Forecasting,"
    CVPR 2023.
  - Veličković et al., "Graph Attention Networks," ICLR 2018.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    """Encode temporal trajectory using 1D convolution + GRU."""

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            features: (batch, hidden_dim) temporal summary
            sequence:  (batch, seq_len, hidden_dim) full sequence features
        """
        # Conv1D expects (batch, channels, seq_len)
        conv_out = self.conv1d(x.permute(0, 2, 1)).permute(0, 2, 1)
        conv_out = self.dropout(conv_out)
        gru_out, hidden = self.gru(conv_out)
        return hidden.squeeze(0), gru_out


class GraphAttentionLayer(nn.Module):
    """Single-head Graph Attention layer."""

    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj_mask):
        """
        Args:
            h: (batch, num_nodes, in_dim) node features
            adj_mask: (batch, num_nodes, num_nodes) adjacency mask (1=connected)
        Returns:
            h_out: (batch, num_nodes, out_dim)
        """
        B, N, _ = h.shape
        Wh = self.W(h)  # (B, N, out_dim)

        # Compute attention coefficients
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, out_dim)
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, out_dim)
        e = self.a(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1)  # (B, N, N)
        e = self.leaky_relu(e)

        # Mask invalid connections
        e = e.masked_fill(adj_mask == 0, float('-inf'))
        alpha = F.softmax(e, dim=-1)
        alpha = alpha.masked_fill(torch.isnan(alpha), 0.0)
        alpha = self.dropout(alpha)

        h_out = torch.bmm(alpha, Wh)  # (B, N, out_dim)
        return h_out


class MultiHeadGAT(nn.Module):
    """Multi-head Graph Attention Network layer."""

    def __init__(self, in_dim, out_dim, num_heads, dropout=0.1):
        super().__init__()
        assert out_dim % num_heads == 0
        head_dim = out_dim // num_heads

        self.heads = nn.ModuleList([
            GraphAttentionLayer(in_dim, head_dim, dropout)
            for _ in range(num_heads)
        ])
        self.norm = nn.LayerNorm(out_dim)
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
        )
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, h, adj_mask):
        """
        Args:
            h: (batch, num_nodes, in_dim)
            adj_mask: (batch, num_nodes, num_nodes)
        Returns:
            h_out: (batch, num_nodes, out_dim)
        """
        head_outputs = [head(h, adj_mask) for head in self.heads]
        h_cat = torch.cat(head_outputs, dim=-1)  # (B, N, out_dim)
        h_cat = self.norm(h_cat + (h if h.shape[-1] == h_cat.shape[-1]
                                   else torch.zeros_like(h_cat)))
        h_out = self.norm2(h_cat + self.ffn(h_cat))
        return h_out


class LaneEncoder(nn.Module):
    """Encode lane/map features using PointNet-like architecture."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, lanes, lane_mask):
        """
        Args:
            lanes: (B, max_lanes, lane_points, 2)
            lane_mask: (B, max_lanes)
        Returns:
            lane_features: (B, max_lanes, hidden_dim)
        """
        B, L, P, D = lanes.shape
        lanes_flat = lanes.reshape(B * L, P, D)
        encoded = self.mlp(lanes_flat)  # (B*L, P, hidden_dim)
        # Max-pool over points
        pooled = encoded.max(dim=1)[0]  # (B*L, hidden_dim)
        lane_features = pooled.reshape(B, L, -1)
        # Zero out invalid lanes
        lane_features = lane_features * lane_mask.unsqueeze(-1)
        return lane_features


class InfrastructureFusion(nn.Module):
    """Fuse vehicle-side and infrastructure-side observations."""

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, vehicle_feat, infra_feat, infra_mask):
        """
        Gated fusion of vehicle and infrastructure features.

        Args:
            vehicle_feat: (B, hidden_dim)
            infra_feat:   (B, hidden_dim)
            infra_mask:   (B,) whether infra view is available
        Returns:
            fused: (B, hidden_dim)
        """
        combined = torch.cat([vehicle_feat, infra_feat], dim=-1)
        gate = self.gate(combined)
        fused = self.fusion(combined)
        # Apply gate and mask
        mask = infra_mask.unsqueeze(-1)  # (B, 1)
        output = vehicle_feat + gate * fused * mask
        return self.norm(output)


class TrajectoryDecoder(nn.Module):
    """Decode future trajectories from context features."""

    def __init__(self, hidden_dim, output_dim, future_steps, num_modes, dropout=0.1):
        super().__init__()
        self.future_steps = future_steps
        self.num_modes = num_modes

        self.mode_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, future_steps * output_dim),
            )
            for _ in range(num_modes)
        ])

        self.mode_prob = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_modes),
        )

    def forward(self, context):
        """
        Args:
            context: (B, hidden_dim)
        Returns:
            predictions: (B, num_modes, future_steps, 2)
            mode_probs:  (B, num_modes)
        """
        B = context.shape[0]
        preds = []
        for head in self.mode_heads:
            pred = head(context)  # (B, future_steps * 2)
            pred = pred.reshape(B, self.future_steps, -1)
            preds.append(pred)

        predictions = torch.stack(preds, dim=1)  # (B, K, T_f, 2)
        mode_probs = F.softmax(self.mode_prob(context), dim=-1)

        return predictions, mode_probs


class V2XGraphPredictor(nn.Module):
    """
    V2X-Graph: Cooperative trajectory prediction using GNN.

    The model constructs a heterogeneous scene graph with:
      - Ego node: the target vehicle
      - Neighbor nodes: surrounding vehicles
      - Infrastructure node: roadside observation
      - Lane nodes: road topology

    Message passing via GAT aggregates information across all nodes,
    and the enriched ego node feature is decoded into future trajectories.
    """

    def __init__(self, config):
        super().__init__()
        self.model_name = 'V2X-Graph'
        self.cooperative = True

        hidden_dim = config['hidden_dim']
        input_dim = config['input_dim']
        output_dim = config['output_dim']
        num_gnn_layers = config['num_gnn_layers']
        num_heads = config['num_heads']
        dropout = config['dropout']
        self.future_steps = config['future_steps']
        self.max_agents = config['max_agents']
        self.num_modes = config.get('num_modes', 1)

        # Temporal encoders
        self.ego_encoder = TemporalEncoder(input_dim, hidden_dim, dropout)
        self.neighbor_encoder = TemporalEncoder(input_dim, hidden_dim, dropout)
        self.infra_encoder = TemporalEncoder(input_dim, hidden_dim, dropout)

        # Lane encoder
        self.lane_encoder = LaneEncoder(config['map_dim'], hidden_dim)

        # Node type embeddings
        self.node_type_embed = nn.Embedding(4, hidden_dim)
        # 0: ego, 1: neighbor, 2: infrastructure, 3: lane

        # Input projection (to unify dimensions for first GAT layer)
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            MultiHeadGAT(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_gnn_layers)
        ])

        # Infrastructure fusion
        self.infra_fusion = InfrastructureFusion(hidden_dim, dropout)

        # Trajectory decoder
        self.decoder = TrajectoryDecoder(
            hidden_dim, output_dim, self.future_steps, self.num_modes, dropout
        )

    def _build_graph(self, ego_feat, nbr_feats, nbr_mask, infra_feat,
                     infra_mask, lane_feats, lane_mask):
        """
        Build heterogeneous scene graph.

        Returns:
            node_features: (B, num_nodes, hidden_dim)
            adj_mask:      (B, num_nodes, num_nodes)
        """
        B = ego_feat.shape[0]
        device = ego_feat.device
        N_nbr = nbr_feats.shape[1]
        N_lane = lane_feats.shape[1]

        # Node features: [ego, neighbors, infra, lanes]
        # Total nodes = 1 + N_nbr + 1 + N_lane
        num_nodes = 1 + N_nbr + 1 + N_lane

        node_features = torch.zeros(B, num_nodes, ego_feat.shape[-1],
                                    device=device)

        # Ego node (index 0)
        node_features[:, 0] = ego_feat + self.node_type_embed(
            torch.zeros(B, dtype=torch.long, device=device))

        # Neighbor nodes (indices 1 to N_nbr)
        for i in range(N_nbr):
            node_features[:, 1 + i] = nbr_feats[:, i] + self.node_type_embed(
                torch.ones(B, dtype=torch.long, device=device))

        # Infrastructure node (index 1 + N_nbr)
        infra_idx = 1 + N_nbr
        node_features[:, infra_idx] = infra_feat + self.node_type_embed(
            2 * torch.ones(B, dtype=torch.long, device=device))

        # Lane nodes (indices 1 + N_nbr + 1 to end)
        lane_start = infra_idx + 1
        for i in range(N_lane):
            node_features[:, lane_start + i] = lane_feats[:, i] + \
                self.node_type_embed(
                    3 * torch.ones(B, dtype=torch.long, device=device))

        # Build adjacency mask
        adj_mask = torch.zeros(B, num_nodes, num_nodes, device=device)

        # Ego connects to all valid nodes
        adj_mask[:, 0, :] = 1.0
        adj_mask[:, :, 0] = 1.0

        # Neighbor connections (ego <-> neighbor, neighbor <-> neighbor)
        for i in range(N_nbr):
            mask_i = nbr_mask[:, i].unsqueeze(-1)  # (B, 1)
            adj_mask[:, 1 + i, :1 + N_nbr] = mask_i.squeeze(-1).unsqueeze(-1).expand(-1, 1 + N_nbr)
            adj_mask[:, :1 + N_nbr, 1 + i] = mask_i.squeeze(-1).unsqueeze(-1).expand(-1, 1 + N_nbr)

        # Infrastructure connection
        infra_m = infra_mask.unsqueeze(-1)  # (B, 1)
        adj_mask[:, infra_idx, 0] = infra_mask
        adj_mask[:, 0, infra_idx] = infra_mask

        # Lane connections (ego <-> lane)
        for i in range(N_lane):
            mask_i = lane_mask[:, i]
            adj_mask[:, lane_start + i, 0] = mask_i
            adj_mask[:, 0, lane_start + i] = mask_i

        # Self-connections
        adj_mask[:, range(num_nodes), range(num_nodes)] = 1.0

        return node_features, adj_mask

    def forward(self, batch):
        """
        Args:
            batch: dict with keys:
                'history':       (B, T_h, 2)
                'neighbors':     (B, N, T_h, 2)
                'neighbor_mask': (B, N)
                'infra_history': (B, T_h, 2)
                'infra_mask':    (B,)
                'lanes':         (B, L, P, 2)
                'lane_mask':     (B, L)

        Returns:
            dict with:
                'predictions': (B, K, T_f, 2)
                'mode_probs':  (B, K)
        """
        history = batch['history']
        neighbors = batch['neighbors']
        neighbor_mask = batch['neighbor_mask']
        infra_history = batch['infra_history']
        infra_mask = batch['infra_mask']
        lanes = batch['lanes']
        lane_mask = batch['lane_mask']

        B = history.shape[0]
        N = neighbors.shape[1]

        # 1. Encode ego trajectory
        ego_feat, _ = self.ego_encoder(history)  # (B, hidden_dim)

        # 2. Encode neighbor trajectories
        nbr_flat = neighbors.reshape(B * N, neighbors.shape[2], -1)
        nbr_feat_flat, _ = self.neighbor_encoder(nbr_flat)
        nbr_feats = nbr_feat_flat.reshape(B, N, -1)  # (B, N, hidden_dim)
        nbr_feats = nbr_feats * neighbor_mask.unsqueeze(-1)

        # 3. Encode infrastructure view
        infra_feat, _ = self.infra_encoder(infra_history)  # (B, hidden_dim)
        infra_feat = infra_feat * infra_mask.unsqueeze(-1)

        # 4. Encode lanes
        lane_feats = self.lane_encoder(lanes, lane_mask)  # (B, L, hidden_dim)

        # 5. Build graph and run GNN
        node_features, adj_mask = self._build_graph(
            ego_feat, nbr_feats, neighbor_mask,
            infra_feat, infra_mask, lane_feats, lane_mask
        )

        node_features = self.input_proj(node_features)

        for gnn_layer in self.gnn_layers:
            node_features = gnn_layer(node_features, adj_mask)

        # 6. Extract ego node feature (enriched by GNN)
        ego_enriched = node_features[:, 0]  # (B, hidden_dim)

        # 7. Infrastructure fusion
        infra_enriched = node_features[:, 1 + N]  # (B, hidden_dim)
        context = self.infra_fusion(ego_enriched, infra_enriched, infra_mask)

        # 8. Decode future trajectories
        predictions, mode_probs = self.decoder(context)

        return {
            'predictions': predictions,
            'mode_probs': mode_probs,
        }
