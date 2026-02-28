"""
V2X-Graph+: Enhanced Graph Neural Network for Cooperative Trajectory Prediction.

Improvements over V2X-Graph:
  1. Multi-scale temporal encoding (kernel sizes 3, 5, 7)
  2. Bidirectional GRU for richer temporal features
  3. Cross-attention infrastructure fusion (applied before GNN)
  4. Shared decoder backbone with lightweight mode-specific heads
  5. Learnable positional encoding for temporal features
  6. Trajectory smoothness auxiliary loss support

Parameter budget: ~1.1-1.2M (controlled increase from 998K)

Reference:
  - Based on V2X-Graph (Yu et al., CVPR 2023)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for temporal sequences."""

    def __init__(self, hidden_dim, max_len=200):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.02)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            x + positional encoding
        """
        return x + self.pos_embed[:, :x.shape[1], :]


class MultiScaleTemporalEncoder(nn.Module):
    """
    Multi-scale temporal encoding using parallel Conv1D with different kernel sizes,
    followed by a bidirectional GRU.
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        # Multi-scale convolutions
        self.conv_k3 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 3, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.conv_k5 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 3, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.conv_k7 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim - 2 * (hidden_dim // 3),
                      kernel_size=7, padding=3),
            nn.GELU(),
        )

        # Projection after concatenation
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Positional encoding
        self.pos_enc = PositionalEncoding(hidden_dim)

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            features: (batch, hidden_dim) temporal summary
            sequence:  (batch, seq_len, hidden_dim) full sequence features
        """
        # Multi-scale conv: (batch, input_dim, seq_len)
        x_t = x.permute(0, 2, 1)
        c3 = self.conv_k3(x_t)  # (B, hidden_dim//3, T)
        c5 = self.conv_k5(x_t)  # (B, hidden_dim//3, T)
        c7 = self.conv_k7(x_t)  # (B, remaining, T)

        # Concatenate and project
        conv_out = torch.cat([c3, c5, c7], dim=1).permute(0, 2, 1)  # (B, T, hidden_dim)
        conv_out = self.proj(conv_out)

        # Add positional encoding
        conv_out = self.pos_enc(conv_out)
        conv_out = self.dropout(conv_out)

        # Bidirectional GRU
        gru_out, hidden = self.gru(conv_out)  # gru_out: (B, T, hidden_dim)

        # Combine forward and backward final hidden states
        # hidden: (2, B, hidden_dim//2)
        features = torch.cat([hidden[0], hidden[1]], dim=-1)  # (B, hidden_dim)
        features = self.norm(features)

        return features, gru_out


class GraphAttentionLayer(nn.Module):
    """Single-head Graph Attention layer (same as V2X-Graph)."""

    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj_mask):
        B, N, _ = h.shape
        Wh = self.W(h)

        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)
        e = self.a(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1)
        e = self.leaky_relu(e)

        e = e.masked_fill(adj_mask == 0, float('-inf'))
        alpha = F.softmax(e, dim=-1)
        alpha = alpha.masked_fill(torch.isnan(alpha), 0.0)
        alpha = self.dropout(alpha)

        h_out = torch.bmm(alpha, Wh)
        return h_out


class MultiHeadGAT(nn.Module):
    """Multi-head Graph Attention Network layer (same as V2X-Graph)."""

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
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
        )
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, h, adj_mask):
        head_outputs = [head(h, adj_mask) for head in self.heads]
        h_cat = torch.cat(head_outputs, dim=-1)
        h_cat = self.norm(h_cat + (h if h.shape[-1] == h_cat.shape[-1]
                                   else torch.zeros_like(h_cat)))
        h_out = self.norm2(h_cat + self.ffn(h_cat))
        return h_out


class LaneEncoder(nn.Module):
    """Encode lane/map features using PointNet-like architecture (same as V2X-Graph)."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, lanes, lane_mask):
        B, L, P, D = lanes.shape
        lanes_flat = lanes.reshape(B * L, P, D)
        encoded = self.mlp(lanes_flat)
        pooled = encoded.max(dim=1)[0]
        lane_features = pooled.reshape(B, L, -1)
        lane_features = lane_features * lane_mask.unsqueeze(-1)
        return lane_features


class CrossAttentionFusion(nn.Module):
    """
    Lightweight cross-attention fusion between vehicle and infrastructure features.
    Applied BEFORE GNN to allow infrastructure info to propagate through the graph.
    """

    def __init__(self, hidden_dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        # Vehicle attends to infrastructure
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, vehicle_feat, infra_seq, infra_mask):
        """
        Args:
            vehicle_feat: (B, hidden_dim) vehicle ego feature
            infra_seq:    (B, T, hidden_dim) infrastructure temporal sequence
            infra_mask:   (B,) whether infra view is available
        Returns:
            fused: (B, hidden_dim)
        """
        B = vehicle_feat.shape[0]

        # Query from vehicle, Key/Value from infrastructure sequence
        Q = self.q_proj(vehicle_feat).unsqueeze(1)  # (B, 1, hidden_dim)
        K = self.k_proj(infra_seq)  # (B, T, hidden_dim)
        V = self.v_proj(infra_seq)  # (B, T, hidden_dim)

        # Reshape for multi-head attention
        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, 1, T)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)  # (B, H, 1, head_dim)
        context = context.transpose(1, 2).reshape(B, -1)  # (B, hidden_dim)
        context = self.out_proj(context)  # (B, hidden_dim)

        # Gated fusion
        combined = torch.cat([vehicle_feat, context], dim=-1)
        gate = self.gate(combined)

        # Apply infrastructure mask
        mask = infra_mask.unsqueeze(-1)  # (B, 1)
        fused = vehicle_feat + gate * context * mask

        return self.norm(fused)


class SharedTrajectoryDecoder(nn.Module):
    """
    Trajectory decoder with shared backbone and lightweight mode-specific heads.
    More parameter-efficient than independent heads per mode.
    """

    def __init__(self, hidden_dim, output_dim, future_steps, num_modes, dropout=0.1):
        super().__init__()
        self.future_steps = future_steps
        self.num_modes = num_modes

        # Shared backbone
        self.shared_backbone = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Lightweight mode-specific output layers
        self.mode_outputs = nn.ModuleList([
            nn.Linear(hidden_dim, future_steps * output_dim)
            for _ in range(num_modes)
        ])

        # Mode probability head
        self.mode_prob = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
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

        # Shared feature extraction
        shared_feat = self.shared_backbone(context)  # (B, hidden_dim)

        # Mode-specific predictions
        preds = []
        for head in self.mode_outputs:
            pred = head(shared_feat)
            pred = pred.reshape(B, self.future_steps, -1)
            preds.append(pred)

        predictions = torch.stack(preds, dim=1)  # (B, K, T_f, 2)
        mode_probs = F.softmax(self.mode_prob(shared_feat), dim=-1)

        return predictions, mode_probs


class V2XGraphPlusPredictor(nn.Module):
    """
    V2X-Graph+: Enhanced cooperative trajectory prediction using GNN.

    Improvements over V2X-Graph:
      1. Multi-scale temporal encoding with parallel Conv1D (kernel 3,5,7)
      2. Bidirectional GRU for richer temporal representation
      3. Cross-attention infrastructure fusion before GNN
      4. Shared decoder backbone for better parameter efficiency
      5. Learnable positional encoding
      6. GELU activation throughout
    """

    def __init__(self, config):
        super().__init__()
        self.model_name = 'V2X-Graph+'
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

        # [NEW] Multi-scale temporal encoders (replaces single-scale)
        self.ego_encoder = MultiScaleTemporalEncoder(input_dim, hidden_dim, dropout)
        self.neighbor_encoder = MultiScaleTemporalEncoder(input_dim, hidden_dim, dropout)
        self.infra_encoder = MultiScaleTemporalEncoder(input_dim, hidden_dim, dropout)

        # Lane encoder (unchanged)
        self.lane_encoder = LaneEncoder(config['map_dim'], hidden_dim)

        # Node type embeddings (unchanged)
        self.node_type_embed = nn.Embedding(4, hidden_dim)

        # Input projection
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)

        # [NEW] Cross-attention infrastructure fusion (before GNN)
        self.cross_attn_fusion = CrossAttentionFusion(
            hidden_dim, num_heads=2, dropout=dropout
        )

        # GNN layers (unchanged structure)
        self.gnn_layers = nn.ModuleList([
            MultiHeadGAT(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_gnn_layers)
        ])

        # [NEW] Post-GNN fusion (lightweight, replaces old InfrastructureFusion)
        self.post_fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.post_fusion_norm = nn.LayerNorm(hidden_dim)

        # [NEW] Shared trajectory decoder
        self.decoder = SharedTrajectoryDecoder(
            hidden_dim, output_dim, self.future_steps, self.num_modes, dropout
        )

    def _build_graph(self, ego_feat, nbr_feats, nbr_mask, infra_feat,
                     infra_mask, lane_feats, lane_mask):
        """Build heterogeneous scene graph (same logic as V2X-Graph)."""
        B = ego_feat.shape[0]
        device = ego_feat.device
        N_nbr = nbr_feats.shape[1]
        N_lane = lane_feats.shape[1]

        num_nodes = 1 + N_nbr + 1 + N_lane

        node_features = torch.zeros(B, num_nodes, ego_feat.shape[-1],
                                    device=device)

        # Ego node (index 0)
        node_features[:, 0] = ego_feat + self.node_type_embed(
            torch.zeros(B, dtype=torch.long, device=device))

        # Neighbor nodes
        for i in range(N_nbr):
            node_features[:, 1 + i] = nbr_feats[:, i] + self.node_type_embed(
                torch.ones(B, dtype=torch.long, device=device))

        # Infrastructure node
        infra_idx = 1 + N_nbr
        node_features[:, infra_idx] = infra_feat + self.node_type_embed(
            2 * torch.ones(B, dtype=torch.long, device=device))

        # Lane nodes
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

        # Neighbor connections
        for i in range(N_nbr):
            mask_i = nbr_mask[:, i].unsqueeze(-1)
            adj_mask[:, 1 + i, :1 + N_nbr] = mask_i.squeeze(-1).unsqueeze(-1).expand(-1, 1 + N_nbr)
            adj_mask[:, :1 + N_nbr, 1 + i] = mask_i.squeeze(-1).unsqueeze(-1).expand(-1, 1 + N_nbr)

        # Infrastructure connection
        adj_mask[:, infra_idx, 0] = infra_mask
        adj_mask[:, 0, infra_idx] = infra_mask

        # Lane connections
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

        # 1. [NEW] Multi-scale temporal encoding
        ego_feat, ego_seq = self.ego_encoder(history)  # (B, hidden_dim), (B, T, hidden_dim)

        nbr_flat = neighbors.reshape(B * N, neighbors.shape[2], -1)
        nbr_feat_flat, _ = self.neighbor_encoder(nbr_flat)
        nbr_feats = nbr_feat_flat.reshape(B, N, -1)
        nbr_feats = nbr_feats * neighbor_mask.unsqueeze(-1)

        infra_feat, infra_seq = self.infra_encoder(infra_history)  # (B, hidden_dim), (B, T, hidden_dim)
        infra_feat = infra_feat * infra_mask.unsqueeze(-1)

        # 2. Encode lanes
        lane_feats = self.lane_encoder(lanes, lane_mask)

        # 3. [NEW] Cross-attention fusion BEFORE GNN
        # Vehicle ego attends to infrastructure temporal sequence
        ego_feat = self.cross_attn_fusion(ego_feat, infra_seq, infra_mask)

        # 4. Build graph and run GNN
        node_features, adj_mask = self._build_graph(
            ego_feat, nbr_feats, neighbor_mask,
            infra_feat, infra_mask, lane_feats, lane_mask
        )

        node_features = self.input_proj(node_features)

        for gnn_layer in self.gnn_layers:
            node_features = gnn_layer(node_features, adj_mask)

        # 5. Extract ego node feature (enriched by GNN)
        ego_enriched = node_features[:, 0]  # (B, hidden_dim)

        # 6. [NEW] Post-GNN lightweight fusion
        infra_enriched = node_features[:, 1 + N]  # (B, hidden_dim)
        combined = torch.cat([ego_enriched, infra_enriched], dim=-1)
        gate = self.post_fusion_gate(combined)
        context = ego_enriched + gate * infra_enriched * infra_mask.unsqueeze(-1)
        context = self.post_fusion_norm(context)

        # 7. [NEW] Shared decoder
        predictions, mode_probs = self.decoder(context)

        return {
            'predictions': predictions,
            'mode_probs': mode_probs,
        }
