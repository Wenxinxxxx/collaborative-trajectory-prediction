"""
Transformer-based Trajectory Predictor.

This model uses self-attention mechanisms to capture long-range temporal
dependencies in trajectory sequences. Compared to LSTM, Transformers
can process the entire sequence in parallel and better model complex
temporal patterns.

This model operates in a NON-COOPERATIVE setting but incorporates
neighbor interaction through cross-attention.

Reference:
  - Giuliari et al., "Transformer Networks for Trajectory Forecasting,"
    ICPR 2020.
  - Zhou et al., "HiVT: Hierarchical Vector Transformer for Multi-Agent
    Motion Prediction," CVPR 2022.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TrajectoryEmbedding(nn.Module):
    """Embed raw (x, y) coordinates into d_model dimensional space."""

    def __init__(self, input_dim, d_model):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        return self.fc(x)


class NeighborEncoder(nn.Module):
    """Encode neighbor trajectories and aggregate via attention."""

    def __init__(self, input_dim, d_model, nhead, num_layers, dropout):
        super().__init__()
        self.embed = TrajectoryEmbedding(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # Temporal encoding for each neighbor
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 2, dropout=dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=max(1, num_layers // 2)
        )

        # Cross-attention: ego attends to neighbors
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, ego_features, neighbors, neighbor_mask):
        """
        Args:
            ego_features: (B, T_h, d_model) encoded ego trajectory
            neighbors:    (B, N, T_h, 2) neighbor trajectories
            neighbor_mask: (B, N) which neighbors are valid

        Returns:
            interaction_features: (B, T_h, d_model)
        """
        B, N, T, _ = neighbors.shape

        # Encode each neighbor's trajectory
        neighbors_flat = neighbors.reshape(B * N, T, -1)
        nbr_embed = self.embed(neighbors_flat)
        nbr_embed = self.pos_enc(nbr_embed)
        nbr_encoded = self.temporal_encoder(nbr_embed)  # (B*N, T, d_model)

        # Pool each neighbor over time -> (B*N, d_model)
        nbr_pooled = nbr_encoded.mean(dim=1)
        nbr_pooled = nbr_pooled.reshape(B, N, -1)  # (B, N, d_model)

        # Create attention mask for invalid neighbors
        key_padding_mask = (neighbor_mask == 0)  # True = ignore

        # Handle case where ALL neighbors are masked for some samples
        # Set at least one key to unmasked (with zero value) to avoid NaN in softmax
        all_masked = key_padding_mask.all(dim=1)  # (B,)
        if all_masked.any():
            # For fully-masked samples, unmask the first key but zero out its value
            safe_mask = key_padding_mask.clone()
            safe_mask[all_masked, 0] = False
            safe_pooled = nbr_pooled.clone()
            safe_pooled[all_masked, 0] = 0.0
        else:
            safe_mask = key_padding_mask
            safe_pooled = nbr_pooled

        # Cross-attention: ego attends to neighbor summaries
        interaction, _ = self.cross_attn(
            query=ego_features,
            key=safe_pooled,
            value=safe_pooled,
            key_padding_mask=safe_mask,
        )

        # Zero out interaction for samples with no valid neighbors
        if all_masked.any():
            interaction = interaction * (~all_masked).float().unsqueeze(-1).unsqueeze(-1)

        return self.norm(ego_features + interaction)


class TransformerPredictor(nn.Module):
    """
    Transformer-based trajectory prediction model.

    Architecture:
      1. Trajectory Embedding: project (x,y) to d_model
      2. Positional Encoding: add temporal position info
      3. Transformer Encoder: self-attention over history
      4. Neighbor Interaction: cross-attention with neighbors
      5. Transformer Decoder: auto-regressive future generation
      6. Multi-modal Output: K trajectory modes with probabilities
    """

    def __init__(self, config):
        super().__init__()
        self.model_name = 'Transformer'
        self.cooperative = False

        input_dim = config['input_dim']
        d_model = config['d_model']
        nhead = config['nhead']
        num_enc_layers = config['num_encoder_layers']
        num_dec_layers = config['num_decoder_layers']
        dim_ff = config['dim_feedforward']
        dropout = config['dropout']
        output_dim = config['output_dim']
        self.future_steps = config['future_steps']
        self.num_modes = config.get('num_modes', 1)
        self.d_model = d_model

        # Embedding layers
        self.history_embed = TrajectoryEmbedding(input_dim, d_model)
        self.future_embed = TrajectoryEmbedding(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # Transformer Encoder (for history)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_enc_layers
        )

        # Neighbor interaction module
        self.neighbor_encoder = NeighborEncoder(
            input_dim, d_model, nhead, num_enc_layers, dropout
        )

        # Transformer Decoder (for future prediction)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_dec_layers
        )

        # Learnable future query tokens (one set per mode)
        self.future_queries = nn.Parameter(
            torch.randn(self.num_modes, self.future_steps, d_model) * 0.02
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, output_dim)

        # Mode probability head
        self.mode_prob_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Causal mask for decoder
        self._init_causal_mask()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for numerical stability."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _init_causal_mask(self):
        """Create causal mask to prevent attending to future positions."""
        mask = torch.triu(
            torch.ones(self.future_steps, self.future_steps), diagonal=1
        ).bool()
        self.register_buffer('causal_mask', mask)

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
        device = history.device

        # 1. Encode history trajectory
        hist_embed = self.history_embed(history)
        hist_embed = self.pos_enc(hist_embed)
        memory = self.transformer_encoder(hist_embed)  # (B, T_h, d_model)

        # 2. Neighbor interaction
        memory = self.neighbor_encoder(memory, neighbors, neighbor_mask)

        # 3. Decode for each mode
        all_predictions = []
        all_mode_features = []

        for k in range(self.num_modes):
            # Learnable query for this mode
            query = self.future_queries[k].unsqueeze(0).expand(B, -1, -1)
            query = self.pos_enc(query)

            # Transformer decoder
            decoded = self.transformer_decoder(
                tgt=query,
                memory=memory,
                tgt_mask=self.causal_mask.to(device),
            )  # (B, T_f, d_model)

            # Project to output space
            pred = self.output_proj(decoded)  # (B, T_f, 2)
            all_predictions.append(pred)

            # Pool decoded features for mode probability
            mode_feat = decoded.mean(dim=1)  # (B, d_model)
            all_mode_features.append(mode_feat)

        predictions = torch.stack(all_predictions, dim=1)  # (B, K, T_f, 2)

        # Mode probabilities
        mode_features = torch.stack(all_mode_features, dim=1)  # (B, K, d_model)
        mode_logits = self.mode_prob_head(mode_features).squeeze(-1)  # (B, K)
        mode_probs = torch.softmax(mode_logits, dim=-1)

        return {
            'predictions': predictions,
            'mode_probs': mode_probs,
        }
