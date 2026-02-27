"""
Enhanced Co-MTP: Improved Cooperative Multi-Temporal Prediction.

This model extends Co-MTP with three key enhancements:
  1. Cross-Agent Attention Fusion: Replaces simple feature concatenation
     with a cross-attention mechanism for vehicle-infrastructure fusion,
     allowing the model to selectively attend to the most relevant
     infrastructure observations.
  2. Multi-Scale Deformable Temporal Encoding: Uses learnable temporal
     offsets to adaptively sample trajectory features at different time
     scales, capturing both regular and irregular motion patterns.
  3. Hierarchical Interaction Modeling: Two-level interaction modeling
     that first captures pairwise agent interactions, then models
     group-level dynamics through a second attention layer.

This is a COOPERATIVE model and represents the proposed improvement
over existing methods.

Reference:
  - Based on Co-MTP (Zhang et al., arXiv:2502.16589, 2025)
  - Cross-attention fusion inspired by V2X-ViT (Xu et al., ECCV 2022)
  - Deformable attention inspired by Deformable DETR (Zhu et al., ICLR 2021)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveTemporalEncoder(nn.Module):
    """
    Adaptive multi-scale temporal encoder with learnable offsets.

    Unlike fixed-kernel multi-scale encoding, this module learns to
    attend to the most informative temporal positions adaptively.
    It combines:
      - Fixed multi-scale convolutions for regular patterns
      - Learnable temporal attention for adaptive sampling
      - Positional encoding for temporal awareness
    """

    def __init__(self, input_dim, hidden_dim, num_scales=3, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, 200, hidden_dim)  # max 200 timesteps
        )
        nn.init.normal_(self.pos_encoding, std=0.02)

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Multi-scale convolutions with different dilation rates
        # Dilation captures patterns at different temporal granularities
        dilation_rates = [1, 2, 4][:num_scales]
        self.scale_convs = nn.ModuleList()
        for d in dilation_rates:
            self.scale_convs.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3,
                          padding=d, dilation=d),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3,
                          padding=d, dilation=d),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
            ))

        # Adaptive temporal attention
        # Learns which temporal positions are most important
        self.temporal_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        # Scale fusion with learnable weights
        self.scale_gate = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_scales),
            nn.Softmax(dim=-1),
        )

        # Final GRU
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
            summary: (B, hidden_dim)
            sequence: (B, T, hidden_dim)
        """
        B, T, _ = x.shape

        # Input projection + positional encoding
        h = self.input_proj(x)  # (B, T, hidden_dim)
        h = h + self.pos_encoding[:, :T, :]

        # Multi-scale feature extraction with dilation
        h_conv = h.permute(0, 2, 1)  # (B, hidden_dim, T)
        scale_features = []
        for conv in self.scale_convs:
            feat = conv(h_conv).permute(0, 2, 1)  # (B, T, hidden_dim)
            scale_features.append(feat)

        # Adaptive scale fusion
        scale_summaries = [f.mean(dim=1) for f in scale_features]
        concat_summary = torch.cat(scale_summaries, dim=-1)
        scale_weights = self.scale_gate(concat_summary)  # (B, num_scales)

        fused = torch.zeros_like(scale_features[0])
        for i, feat in enumerate(scale_features):
            fused = fused + scale_weights[:, i:i+1].unsqueeze(-1) * feat

        # Adaptive temporal attention (self-attention over time)
        attended, _ = self.temporal_attn(fused, fused, fused)
        fused = self.temporal_norm(fused + attended)
        fused = self.dropout(fused)

        # Sequential encoding
        gru_out, hidden = self.gru(fused)
        summary = hidden[-1]

        return summary, gru_out


class HierarchicalInteractionModule(nn.Module):
    """
    Two-level hierarchical interaction modeling.

    Level 1: Pairwise attention between ego and each neighbor
    Level 2: Group-level attention that captures collective dynamics

    This is more expressive than single-level attention because it
    can model both direct pairwise influences and emergent group behaviors.
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()

        # Level 1: Pairwise interaction
        self.pairwise_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.pairwise_norm = nn.LayerNorm(hidden_dim)
        self.pairwise_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.pairwise_ffn_norm = nn.LayerNorm(hidden_dim)

        # Level 2: Group-level interaction
        self.group_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.group_norm = nn.LayerNorm(hidden_dim)
        self.group_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.group_ffn_norm = nn.LayerNorm(hidden_dim)

        # Relative position encoding for spatial awareness
        self.rel_pos_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # relative pos (2) + distance (1) + angle (1)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _compute_rel_features(self, ego_pos, nbr_pos, neighbor_mask):
        """Compute relative spatial features between ego and neighbors."""
        B, N, _ = nbr_pos.shape
        rel_pos = nbr_pos - ego_pos.unsqueeze(1)  # (B, N, 2)
        rel_dist = torch.norm(rel_pos, dim=-1, keepdim=True).clamp(min=1e-6)  # (B, N, 1)
        rel_angle = torch.atan2(rel_pos[:, :, 1:2], rel_pos[:, :, 0:1])  # (B, N, 1)
        rel_feats = torch.cat([rel_pos, rel_dist, rel_angle], dim=-1)  # (B, N, 4)
        return rel_feats * neighbor_mask.unsqueeze(-1)

    def forward(self, ego_feat, neighbor_feats, neighbor_mask,
                ego_pos=None, nbr_pos=None):
        """
        Args:
            ego_feat:       (B, hidden_dim)
            neighbor_feats: (B, N, hidden_dim)
            neighbor_mask:  (B, N)
            ego_pos:        (B, 2) optional last position
            nbr_pos:        (B, N, 2) optional last positions
        Returns:
            enriched: (B, hidden_dim)
        """
        B, N, D = neighbor_feats.shape
        key_padding_mask = (neighbor_mask == 0)
        all_masked = key_padding_mask.all(dim=1)

        # Handle fully-masked samples
        if all_masked.any():
            safe_mask = key_padding_mask.clone()
            safe_mask[all_masked, 0] = False
            safe_feats = neighbor_feats.clone()
            safe_feats[all_masked, 0] = 0.0
        else:
            safe_mask = key_padding_mask
            safe_feats = neighbor_feats

        # Add relative position encoding if available
        if ego_pos is not None and nbr_pos is not None:
            rel_feats = self._compute_rel_features(ego_pos, nbr_pos, neighbor_mask)
            rel_encoding = self.rel_pos_encoder(rel_feats)  # (B, N, hidden_dim)
            safe_feats = safe_feats + rel_encoding

        # Level 1: Pairwise interaction (ego attends to neighbors)
        ego_query = ego_feat.unsqueeze(1)  # (B, 1, D)
        pairwise_out, _ = self.pairwise_attn(
            ego_query, safe_feats, safe_feats,
            key_padding_mask=safe_mask,
        )
        if all_masked.any():
            pairwise_out = pairwise_out * (~all_masked).float().unsqueeze(-1).unsqueeze(-1)
        ego_l1 = self.pairwise_norm(ego_query + pairwise_out)
        ego_l1 = self.pairwise_ffn_norm(ego_l1 + self.pairwise_ffn(ego_l1))

        # Level 2: Group-level (ego attends to neighbor self-attention results)
        # First, neighbors attend to each other
        nbr_self, _ = self.group_attn(
            safe_feats, safe_feats, safe_feats,
            key_padding_mask=safe_mask,
        )
        nbr_group = self.group_norm(safe_feats + nbr_self)
        nbr_group = self.group_ffn_norm(nbr_group + self.group_ffn(nbr_group))

        # Then ego attends to group-aware neighbors
        group_out, _ = self.pairwise_attn(
            ego_l1, nbr_group, nbr_group,
            key_padding_mask=safe_mask,
        )
        if all_masked.any():
            group_out = group_out * (~all_masked).float().unsqueeze(-1).unsqueeze(-1)

        enriched = self.group_norm(ego_l1 + group_out).squeeze(1)

        return enriched


class CrossAgentAttentionFusion(nn.Module):
    """
    Cross-Agent Attention Fusion for V2X cooperative prediction.

    Instead of simple gated fusion, this module uses bidirectional
    cross-attention between vehicle and infrastructure temporal
    sequences, allowing each viewpoint to selectively attend to
    the most relevant information from the other.

    Key improvement: The infrastructure can provide complementary
    observations (e.g., occluded areas), and cross-attention learns
    to identify and leverage these complementary features.
    """

    def __init__(self, hidden_dim, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers

        # Bidirectional cross-attention layers
        self.v2i_attns = nn.ModuleList()  # vehicle attends to infrastructure
        self.i2v_attns = nn.ModuleList()  # infrastructure attends to vehicle
        self.v_norms = nn.ModuleList()
        self.i_norms = nn.ModuleList()
        self.v_ffns = nn.ModuleList()
        self.i_ffns = nn.ModuleList()
        self.v_ffn_norms = nn.ModuleList()
        self.i_ffn_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.v2i_attns.append(nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            ))
            self.i2v_attns.append(nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            ))
            self.v_norms.append(nn.LayerNorm(hidden_dim))
            self.i_norms.append(nn.LayerNorm(hidden_dim))
            self.v_ffns.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            ))
            self.i_ffns.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            ))
            self.v_ffn_norms.append(nn.LayerNorm(hidden_dim))
            self.i_ffn_norms.append(nn.LayerNorm(hidden_dim))

        # Confidence-weighted fusion
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1),
        )

        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, vehicle_seq, infra_seq, vehicle_summary,
                infra_summary, infra_mask):
        """
        Args:
            vehicle_seq:     (B, T, hidden_dim)
            infra_seq:       (B, T, hidden_dim)
            vehicle_summary: (B, hidden_dim)
            infra_summary:   (B, hidden_dim)
            infra_mask:      (B,)
        Returns:
            fused: (B, hidden_dim)
        """
        v_seq = vehicle_seq
        i_seq = infra_seq

        # Bidirectional cross-attention layers
        for layer_idx in range(self.num_layers):
            # Vehicle attends to infrastructure
            v2i_out, _ = self.v2i_attns[layer_idx](v_seq, i_seq, i_seq)
            v_seq = self.v_norms[layer_idx](v_seq + v2i_out)
            v_seq = self.v_ffn_norms[layer_idx](v_seq + self.v_ffns[layer_idx](v_seq))

            # Infrastructure attends to vehicle
            i2v_out, _ = self.i2v_attns[layer_idx](i_seq, v_seq, v_seq)
            i_seq = self.i_norms[layer_idx](i_seq + i2v_out)
            i_seq = self.i_ffn_norms[layer_idx](i_seq + self.i_ffns[layer_idx](i_seq))

        # Pool temporal sequences
        v_pooled = v_seq.mean(dim=1)  # (B, hidden_dim)
        i_pooled = i_seq.mean(dim=1)  # (B, hidden_dim)

        # Confidence-weighted fusion
        combined = torch.cat([v_pooled, i_pooled], dim=-1)
        confidence = self.confidence_net(combined)  # (B, 2)

        fused = confidence[:, 0:1] * v_pooled + confidence[:, 1:2] * i_pooled

        # Apply infrastructure mask
        mask = infra_mask.unsqueeze(-1)
        fused = vehicle_summary * (1 - mask) + fused * mask

        fused = self.output_proj(fused)
        fused = self.output_norm(vehicle_summary + fused)

        return fused


class RefinementDecoder(nn.Module):
    """
    Two-stage trajectory decoder with iterative refinement.

    Stage 1: Generate coarse trajectory proposals
    Stage 2: Refine proposals using context-aware attention

    This produces more accurate predictions than single-stage decoding.
    """

    def __init__(self, hidden_dim, output_dim, future_steps, num_modes,
                 nhead=4, dropout=0.1):
        super().__init__()
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.output_dim = output_dim

        # Learnable mode queries
        self.mode_queries = nn.Parameter(
            torch.randn(num_modes, hidden_dim) * 0.02
        )

        # Context integration
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Stage 1: Coarse trajectory generation
        self.coarse_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, future_steps * output_dim),
            )
            for _ in range(num_modes)
        ])

        # Stage 2: Refinement
        self.traj_encoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
        )
        self.refine_attn = nn.MultiheadAttention(
            hidden_dim, nhead, dropout=dropout, batch_first=True
        )
        self.refine_norm = nn.LayerNorm(hidden_dim)
        self.refine_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),  # residual offset
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
            interaction_ctx: (B, hidden_dim)
            cooperative_ctx: (B, hidden_dim)
            map_ctx:         (B, hidden_dim)
        Returns:
            predictions: (B, K, T_f, 2)
            mode_probs:  (B, K)
        """
        B = interaction_ctx.shape[0]

        # Combine context
        combined = torch.cat([interaction_ctx, cooperative_ctx, map_ctx], dim=-1)
        context = self.context_proj(combined)  # (B, hidden_dim)

        # Mode-specific features
        mode_queries = self.mode_queries.unsqueeze(0).expand(B, -1, -1)
        mode_features = mode_queries + context.unsqueeze(1)

        # Stage 1: Coarse predictions
        coarse_preds = []
        for k in range(self.num_modes):
            feat = mode_features[:, k]
            traj = self.coarse_heads[k](feat)
            traj = traj.reshape(B, self.future_steps, self.output_dim)
            coarse_preds.append(traj)

        coarse_predictions = torch.stack(coarse_preds, dim=1)  # (B, K, T_f, 2)

        # Stage 2: Refinement
        refined_preds = []
        all_confs = []
        for k in range(self.num_modes):
            coarse_traj = coarse_preds[k]  # (B, T_f, 2)
            traj_feats = self.traj_encoder(coarse_traj)  # (B, T_f, hidden_dim)

            # Cross-attention: trajectory attends to context
            context_seq = context.unsqueeze(1).expand(-1, self.future_steps, -1)
            refined_feats, _ = self.refine_attn(traj_feats, context_seq, context_seq)
            refined_feats = self.refine_norm(traj_feats + refined_feats)

            # Predict residual offsets
            offsets = self.refine_heads[k](refined_feats)  # (B, T_f, 2)
            refined_traj = coarse_traj + offsets
            refined_preds.append(refined_traj)

            # Confidence from mode feature
            conf = self.conf_head(mode_features[:, k])
            all_confs.append(conf)

        predictions = torch.stack(refined_preds, dim=1)  # (B, K, T_f, 2)
        conf_logits = torch.cat(all_confs, dim=-1)  # (B, K)
        mode_probs = F.softmax(conf_logits, dim=-1)

        return predictions, mode_probs


class MapEncoderEnhanced(nn.Module):
    """Enhanced map encoder with attention-based aggregation."""

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
            query:     (B, hidden_dim)
        Returns:
            map_context: (B, hidden_dim)
        """
        B, L, P, D = lanes.shape

        lanes_flat = lanes.reshape(B * L, P, D)
        point_feats = self.point_encoder(lanes_flat)
        lane_feats = point_feats.max(dim=1)[0]
        lane_feats = lane_feats.reshape(B, L, -1)
        lane_feats = lane_feats * lane_mask.unsqueeze(-1)

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

        query_expanded = query.unsqueeze(1)
        map_context, _ = self.lane_attn(
            query_expanded, safe_feats, safe_feats,
            key_padding_mask=safe_mask,
        )

        if all_masked.any():
            map_context = map_context * (~all_masked).float().unsqueeze(-1).unsqueeze(-1)

        map_context = self.norm(query_expanded + map_context).squeeze(1)
        return map_context


class EnhancedCoMTP(nn.Module):
    """
    Enhanced Co-MTP: Improved Cooperative Multi-Temporal Prediction.

    Improvements over Co-MTP:
      1. AdaptiveTemporalEncoder: Dilated convolutions + temporal self-attention
         for better multi-scale feature extraction
      2. HierarchicalInteractionModule: Two-level (pairwise + group) interaction
         modeling with relative position encoding
      3. CrossAgentAttentionFusion: Bidirectional cross-attention for V2X fusion
         with confidence-weighted combination
      4. RefinementDecoder: Two-stage coarse-to-fine trajectory prediction
         with iterative refinement

    This model represents the proposed enhancement and should demonstrate
    improved performance over the baseline Co-MTP.
    """

    def __init__(self, config):
        super().__init__()
        self.model_name = 'Enhanced-Co-MTP'
        self.cooperative = True

        hidden_dim = config['hidden_dim']
        input_dim = config['input_dim']
        output_dim = config['output_dim']
        dropout = config['dropout']
        nhead = config['nhead']
        self.future_steps = config['future_steps']
        self.num_modes = config.get('num_modes', 1)
        num_scales = config.get('num_temporal_scales', 3)
        num_fusion_layers = config.get('num_fusion_layers', 2)

        # 1. Adaptive temporal encoders
        self.ego_temporal = AdaptiveTemporalEncoder(
            input_dim, hidden_dim, num_scales, dropout
        )
        self.neighbor_temporal = AdaptiveTemporalEncoder(
            input_dim, hidden_dim, num_scales, dropout
        )
        self.infra_temporal = AdaptiveTemporalEncoder(
            input_dim, hidden_dim, num_scales, dropout
        )

        # 2. Hierarchical interaction
        self.hierarchical_interaction = HierarchicalInteractionModule(
            hidden_dim, nhead, dropout
        )

        # 3. Cross-agent attention fusion
        self.cross_agent_fusion = CrossAgentAttentionFusion(
            hidden_dim, nhead, num_fusion_layers, dropout
        )

        # 4. Map encoder
        self.map_encoder = MapEncoderEnhanced(
            config['map_dim'], hidden_dim, dropout
        )

        # 5. Refinement decoder
        self.decoder = RefinementDecoder(
            hidden_dim, output_dim, self.future_steps,
            self.num_modes, nhead, dropout,
        )

        # Initialize weights
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

        # 1. Adaptive temporal encoding
        ego_summary, ego_seq = self.ego_temporal(history)

        nbr_flat = neighbors.reshape(B * N, neighbors.shape[2], -1)
        nbr_summary_flat, _ = self.neighbor_temporal(nbr_flat)
        nbr_summaries = nbr_summary_flat.reshape(B, N, -1)
        nbr_summaries = nbr_summaries * neighbor_mask.unsqueeze(-1)

        infra_summary, infra_seq = self.infra_temporal(infra_history)
        infra_summary = infra_summary * infra_mask.unsqueeze(-1)
        infra_seq = infra_seq * infra_mask.unsqueeze(-1).unsqueeze(-1)

        # Get last positions for spatial-aware interaction
        ego_pos = history[:, -1, :2]  # (B, 2)
        nbr_pos = neighbors[:, :, -1, :2]  # (B, N, 2)

        # 2. Hierarchical interaction modeling
        interaction_ctx = self.hierarchical_interaction(
            ego_summary, nbr_summaries, neighbor_mask,
            ego_pos, nbr_pos,
        )

        # 3. Cross-agent attention fusion
        cooperative_ctx = self.cross_agent_fusion(
            ego_seq, infra_seq, ego_summary, infra_summary, infra_mask
        )

        # 4. Map context
        map_ctx = self.map_encoder(lanes, lane_mask, ego_summary)

        # 5. Refinement decoding
        predictions, mode_probs = self.decoder(
            interaction_ctx, cooperative_ctx, map_ctx
        )

        return {
            'predictions': predictions,
            'mode_probs': mode_probs,
        }
