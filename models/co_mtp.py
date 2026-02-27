"""
Co-MTP: Cooperative Multi-Temporal Prediction Model.

This is the most advanced model in the project, combining:
  1. Multi-temporal feature extraction at different time scales
  2. Graph-based agent interaction modeling
  3. Transformer-based temporal reasoning
  4. V2X cooperative fusion (vehicle + infrastructure)
  5. Multi-modal trajectory prediction with confidence scores

The key innovation is the multi-temporal fusion module that captures
trajectory patterns at different granularities (short-term maneuvers,
medium-term intentions, long-term goals).

This is a COOPERATIVE model.

Reference:
  - Based on concepts from:
    - V2X-Seq (Yu et al., CVPR 2023)
    - HiVT (Zhou et al., CVPR 2022)
    - LaneGCN (Liang et al., ECCV 2020)
    - TNT (Zhao et al., CVPR 2021)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleTemporalEncoder(nn.Module):
    """
    Extract features at multiple temporal scales.

    Uses different kernel sizes to capture:
      - Short-term: fine-grained motion (small kernel)
      - Medium-term: maneuver-level patterns (medium kernel)
      - Long-term: trajectory-level trends (large kernel)
    """

    def __init__(self, input_dim, hidden_dim, num_scales=3, dropout=0.1):
        super().__init__()
        self.num_scales = num_scales

        # Different temporal scales
        kernel_sizes = [3, 5, 7][:num_scales]
        self.scale_convs = nn.ModuleList()
        for ks in kernel_sizes:
            self.scale_convs.append(nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=ks,
                          padding=ks // 2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=ks,
                          padding=ks // 2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ))

        # Scale attention: learn to weight different scales
        self.scale_attention = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_scales),
            nn.Softmax(dim=-1),
        )

        # Final GRU for sequential encoding
        self.gru = nn.GRU(
            hidden_dim, hidden_dim, num_layers=2,
            batch_first=True, dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, T, input_dim)
        Returns:
            summary: (B, hidden_dim) fused multi-scale feature
            sequence: (B, T, hidden_dim) full sequence features
        """
        B, T, D = x.shape
        x_conv = x.permute(0, 2, 1)  # (B, D, T)

        # Extract features at each scale
        scale_features = []
        for conv in self.scale_convs:
            feat = conv(x_conv).permute(0, 2, 1)  # (B, T, hidden_dim)
            scale_features.append(feat)

        # Compute scale attention weights
        # Pool each scale over time for attention computation
        scale_summaries = [f.mean(dim=1) for f in scale_features]
        concat_summary = torch.cat(scale_summaries, dim=-1)  # (B, hidden_dim * num_scales)
        scale_weights = self.scale_attention(concat_summary)  # (B, num_scales)

        # Weighted fusion of scales
        fused = torch.zeros_like(scale_features[0])
        for i, feat in enumerate(scale_features):
            fused = fused + scale_weights[:, i:i+1].unsqueeze(-1) * feat

        fused = self.dropout(fused)

        # Sequential encoding
        gru_out, hidden = self.gru(fused)
        summary = hidden[-1]  # (B, hidden_dim)

        return summary, gru_out


class SpatialInteractionModule(nn.Module):
    """
    Model spatial interactions between agents using attention.
    Implements a simplified version of the social attention mechanism.
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, ego_feat, neighbor_feats, neighbor_mask):
        """
        Args:
            ego_feat:       (B, hidden_dim)
            neighbor_feats: (B, N, hidden_dim)
            neighbor_mask:  (B, N)
        Returns:
            enriched: (B, hidden_dim)
        """
        key_padding_mask = (neighbor_mask == 0)
        all_masked = key_padding_mask.all(dim=1)  # (B,)

        # Handle fully-masked samples to avoid NaN in attention softmax
        if all_masked.any():
            safe_mask = key_padding_mask.clone()
            safe_mask[all_masked, 0] = False
            safe_feats = neighbor_feats.clone()
            safe_feats[all_masked, 0] = 0.0
        else:
            safe_mask = key_padding_mask
            safe_feats = neighbor_feats

        # Self-attention among neighbors
        nbr_attended, _ = self.self_attn(
            safe_feats, safe_feats, safe_feats,
            key_padding_mask=safe_mask,
        )
        safe_feats = self.norm1(safe_feats + nbr_attended)

        # Cross-attention: ego attends to neighbors
        ego_query = ego_feat.unsqueeze(1)  # (B, 1, hidden_dim)
        cross_out, _ = self.cross_attn(
            ego_query, safe_feats, safe_feats,
            key_padding_mask=safe_mask,
        )
        # Zero out cross-attention for fully-masked samples
        if all_masked.any():
            cross_out = cross_out * (~all_masked).float().unsqueeze(-1).unsqueeze(-1)

        ego_enriched = self.norm2(ego_query + cross_out).squeeze(1)

        # FFN
        ego_enriched = self.norm3(ego_enriched + self.ffn(ego_enriched))

        return ego_enriched


class CooperativeFusionModule(nn.Module):
    """
    Advanced fusion of vehicle-side and infrastructure-side features.

    Uses cross-attention and gated fusion to combine information
    from both viewpoints.
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        # Cross-attention: vehicle attends to infrastructure
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Gated fusion
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Sigmoid(),
        )
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, vehicle_seq, infra_seq, vehicle_summary,
                infra_summary, infra_mask):
        """
        Args:
            vehicle_seq:     (B, T, hidden_dim) vehicle temporal features
            infra_seq:       (B, T, hidden_dim) infra temporal features
            vehicle_summary: (B, hidden_dim)
            infra_summary:   (B, hidden_dim)
            infra_mask:      (B,)
        Returns:
            fused: (B, hidden_dim)
        """
        # Cross-attention over temporal sequences
        cross_out, _ = self.cross_attn(
            vehicle_seq, infra_seq, infra_seq,
        )
        cross_out = self.norm1(vehicle_seq + cross_out)
        cross_summary = cross_out.mean(dim=1)  # (B, hidden_dim)

        # Gated fusion of summaries
        combined = torch.cat([vehicle_summary, infra_summary, cross_summary],
                             dim=-1)
        gate = self.gate_net(combined)
        transformed = self.transform(combined)

        # Apply mask: if no infra data, fall back to vehicle only
        mask = infra_mask.unsqueeze(-1)
        fused = vehicle_summary + gate * transformed * mask
        return self.norm2(fused)


class MapEncoder(nn.Module):
    """Encode lane/map features with attention-based aggregation."""

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.lane_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, lanes, lane_mask, query):
        """
        Args:
            lanes:     (B, L, P, 2)
            lane_mask: (B, L)
            query:     (B, hidden_dim) ego feature as query
        Returns:
            map_context: (B, hidden_dim)
        """
        B, L, P, D = lanes.shape

        # Encode lane points
        lanes_flat = lanes.reshape(B * L, P, D)
        point_feats = self.point_encoder(lanes_flat)  # (B*L, P, hidden_dim)
        lane_feats = point_feats.max(dim=1)[0]  # (B*L, hidden_dim)
        lane_feats = lane_feats.reshape(B, L, -1)  # (B, L, hidden_dim)

        # Zero out invalid lanes
        lane_feats = lane_feats * lane_mask.unsqueeze(-1)

        # Attention: ego queries lane features
        key_padding_mask = (lane_mask == 0)
        all_masked = key_padding_mask.all(dim=1)

        if all_masked.any():
            safe_mask = key_padding_mask.clone()
            safe_mask[all_masked, 0] = False
            safe_feats = lane_feats.clone()
            safe_feats[all_masked, 0] = 0.0
        else:
            safe_mask = key_padding_mask
            safe_feats = lane_feats

        query_expanded = query.unsqueeze(1)  # (B, 1, hidden_dim)
        map_context, _ = self.lane_attn(
            query_expanded, safe_feats, safe_feats,
            key_padding_mask=safe_mask,
        )

        if all_masked.any():
            map_context = map_context * (~all_masked).float().unsqueeze(-1).unsqueeze(-1)

        map_context = self.norm(query_expanded + map_context).squeeze(1)

        return map_context


class MultiModalDecoder(nn.Module):
    """
    Decode multiple trajectory modes with confidence scores.

    Uses learnable mode queries and a Transformer decoder.
    """

    def __init__(self, hidden_dim, output_dim, future_steps, num_modes,
                 nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.future_steps = future_steps
        self.num_modes = num_modes

        # Learnable mode queries
        self.mode_queries = nn.Parameter(
            torch.randn(num_modes, hidden_dim) * 0.02
        )

        # Context integration
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Trajectory regression heads
        self.traj_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, future_steps * output_dim),
            )
            for _ in range(num_modes)
        ])

        # Confidence head
        self.conf_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, interaction_ctx, cooperative_ctx, map_ctx):
        """
        Args:
            interaction_ctx: (B, hidden_dim) agent interaction features
            cooperative_ctx: (B, hidden_dim) V2X cooperative features
            map_ctx:         (B, hidden_dim) map context features
        Returns:
            predictions: (B, K, T_f, 2)
            mode_probs:  (B, K)
        """
        B = interaction_ctx.shape[0]

        # Combine all context
        combined = torch.cat([interaction_ctx, cooperative_ctx, map_ctx],
                             dim=-1)
        context = self.context_proj(combined)  # (B, hidden_dim)

        # Generate mode-specific features
        mode_queries = self.mode_queries.unsqueeze(0).expand(B, -1, -1)
        mode_features = mode_queries + context.unsqueeze(1)  # (B, K, hidden_dim)

        # Decode each mode
        all_preds = []
        all_confs = []
        for k in range(self.num_modes):
            feat = mode_features[:, k]  # (B, hidden_dim)
            traj = self.traj_heads[k](feat)  # (B, T_f * 2)
            traj = traj.reshape(B, self.future_steps, -1)
            all_preds.append(traj)
            conf = self.conf_head(feat)  # (B, 1)
            all_confs.append(conf)

        predictions = torch.stack(all_preds, dim=1)  # (B, K, T_f, 2)
        conf_logits = torch.cat(all_confs, dim=-1)  # (B, K)
        mode_probs = F.softmax(conf_logits, dim=-1)

        return predictions, mode_probs


class CoMTP(nn.Module):
    """
    Co-MTP: Cooperative Multi-Temporal Prediction.

    Full architecture:
      1. Multi-Scale Temporal Encoding (vehicle + infra)
      2. Spatial Interaction (agent-agent)
      3. Cooperative Fusion (vehicle-infra)
      4. Map Context Encoding
      5. Multi-Modal Trajectory Decoding

    This model represents the state-of-the-art approach that leverages
    all available information sources for trajectory prediction.
    """

    def __init__(self, config):
        super().__init__()
        self.model_name = 'Co-MTP'
        self.cooperative = True

        hidden_dim = config['hidden_dim']
        input_dim = config['input_dim']
        output_dim = config['output_dim']
        dropout = config['dropout']
        nhead = config['nhead']
        self.future_steps = config['future_steps']
        self.num_modes = config.get('num_modes', 1)
        num_scales = config.get('num_temporal_scales', 3)

        # 1. Multi-scale temporal encoders
        self.ego_temporal = MultiScaleTemporalEncoder(
            input_dim, hidden_dim, num_scales, dropout
        )
        self.neighbor_temporal = MultiScaleTemporalEncoder(
            input_dim, hidden_dim, num_scales, dropout
        )
        self.infra_temporal = MultiScaleTemporalEncoder(
            input_dim, hidden_dim, num_scales, dropout
        )

        # 2. Spatial interaction
        self.spatial_interaction = SpatialInteractionModule(
            hidden_dim, nhead, dropout
        )

        # 3. Cooperative fusion
        self.cooperative_fusion = CooperativeFusionModule(
            hidden_dim, nhead, dropout
        )

        # 4. Map encoder
        self.map_encoder = MapEncoder(
            config['map_dim'], hidden_dim, dropout
        )

        # 5. Multi-modal decoder
        self.decoder = MultiModalDecoder(
            hidden_dim, output_dim, self.future_steps,
            self.num_modes, nhead, dropout=dropout,
        )

        # Initialize weights for numerical stability
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for numerical stability."""
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, batch):
        """
        Args:
            batch: dict with all keys from TrajectoryDataset

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

        # 1. Multi-scale temporal encoding
        ego_summary, ego_seq = self.ego_temporal(history)

        # Encode neighbors
        nbr_flat = neighbors.reshape(B * N, neighbors.shape[2], -1)
        nbr_summary_flat, _ = self.neighbor_temporal(nbr_flat)
        nbr_summaries = nbr_summary_flat.reshape(B, N, -1)
        nbr_summaries = nbr_summaries * neighbor_mask.unsqueeze(-1)

        # Encode infrastructure
        infra_summary, infra_seq = self.infra_temporal(infra_history)
        infra_summary = infra_summary * infra_mask.unsqueeze(-1)
        infra_seq = infra_seq * infra_mask.unsqueeze(-1).unsqueeze(-1)

        # 2. Spatial interaction
        interaction_ctx = self.spatial_interaction(
            ego_summary, nbr_summaries, neighbor_mask
        )

        # 3. Cooperative fusion
        cooperative_ctx = self.cooperative_fusion(
            ego_seq, infra_seq, ego_summary, infra_summary, infra_mask
        )

        # 4. Map context
        map_ctx = self.map_encoder(lanes, lane_mask, ego_summary)

        # 5. Multi-modal decoding
        predictions, mode_probs = self.decoder(
            interaction_ctx, cooperative_ctx, map_ctx
        )

        return {
            'predictions': predictions,
            'mode_probs': mode_probs,
        }
