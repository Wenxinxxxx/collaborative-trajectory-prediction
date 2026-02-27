"""
Social LSTM: Human/Vehicle Trajectory Prediction with Social Pooling.

This model extends the basic LSTM Encoder-Decoder by incorporating a
social pooling mechanism that captures interactions between the ego
agent and its neighbors. The social pooling layer aggregates hidden
states of nearby agents to model their influence on the ego agent's
future trajectory.

This is a NON-COOPERATIVE model: it uses neighbor trajectories from
the vehicle's own perception (no infrastructure data).

Reference:
  - Alahi et al., "Social LSTM: Human Trajectory Prediction in Crowded
    Spaces," CVPR 2016.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SocialPooling(nn.Module):
    """
    Social pooling layer that aggregates neighbor information.

    Instead of the original grid-based pooling from the paper, we use
    an attention-based pooling mechanism which is more flexible and
    performs better in practice.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        # Attention-based social pooling
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, ego_hidden, neighbor_hiddens, neighbor_mask):
        """
        Args:
            ego_hidden:       (B, hidden_dim)
            neighbor_hiddens: (B, N, hidden_dim)
            neighbor_mask:    (B, N) binary mask
        Returns:
            social_context: (B, hidden_dim) pooled neighbor information
        """
        B, N, D = neighbor_hiddens.shape

        query = self.query_proj(ego_hidden).unsqueeze(1)  # (B, 1, D)
        keys = self.key_proj(neighbor_hiddens)  # (B, N, D)
        values = self.value_proj(neighbor_hiddens)  # (B, N, D)

        # Compute attention scores
        attn_scores = torch.bmm(query, keys.transpose(1, 2)) / self.scale  # (B, 1, N)

        # Mask invalid neighbors
        mask_expanded = neighbor_mask.unsqueeze(1)  # (B, 1, N)
        attn_scores = attn_scores.masked_fill(mask_expanded == 0, float('-inf'))

        # Check if all neighbors are masked
        all_masked = (neighbor_mask.sum(dim=1) == 0)  # (B,)

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, 1, N)
        attn_weights = attn_weights.masked_fill(torch.isnan(attn_weights), 0.0)

        # Weighted sum of neighbor values
        social_feat = torch.bmm(attn_weights, values).squeeze(1)  # (B, D)

        # Zero out for samples with no valid neighbors
        if all_masked.any():
            social_feat[all_masked] = 0.0

        social_context = self.output_proj(social_feat)
        social_context = self.norm(ego_hidden + social_context)

        return social_context


class SocialLSTMEncoder(nn.Module):
    """LSTM encoder with social pooling at each timestep."""

    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Ego trajectory embedding
        self.input_embed = nn.Linear(input_dim, hidden_dim)

        # Neighbor trajectory embedding
        self.neighbor_embed = nn.Linear(input_dim, hidden_dim)

        # LSTM cell (input = embedded position + social context)
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 2,  # ego embedding + social context
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Social pooling
        self.social_pooling = SocialPooling(hidden_dim, dropout)

        # Neighbor LSTM (shared across all neighbors)
        self.neighbor_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, history, neighbors, neighbor_mask):
        """
        Args:
            history:       (B, T, 2) ego trajectory
            neighbors:     (B, N, T, 2) neighbor trajectories
            neighbor_mask: (B, N) binary mask
        Returns:
            hidden: (num_layers, B, hidden_dim)
            cell:   (num_layers, B, hidden_dim)
            social_context: (B, hidden_dim) final social context
        """
        B, T, _ = history.shape
        N = neighbors.shape[1]
        device = history.device

        # Embed ego trajectory
        ego_embedded = self.input_embed(history)  # (B, T, hidden_dim)

        # Encode all neighbor trajectories
        nbr_embedded = self.neighbor_embed(
            neighbors.reshape(B * N, T, -1)
        )  # (B*N, T, hidden_dim)
        nbr_out, _ = self.neighbor_lstm(nbr_embedded)
        nbr_out = nbr_out.reshape(B, N, T, self.hidden_dim)
        # Apply mask
        nbr_out = nbr_out * neighbor_mask.unsqueeze(-1).unsqueeze(-1)

        # Process each timestep with social pooling
        social_inputs = []
        # Use the neighbor hidden states at each timestep
        for t in range(T):
            nbr_t = nbr_out[:, :, t, :]  # (B, N, hidden_dim)
            ego_t = ego_embedded[:, t, :]  # (B, hidden_dim)
            social_ctx = self.social_pooling(ego_t, nbr_t, neighbor_mask)
            social_inputs.append(social_ctx)

        social_sequence = torch.stack(social_inputs, dim=1)  # (B, T, hidden_dim)

        # Concatenate ego embedding and social context
        lstm_input = torch.cat([ego_embedded, social_sequence], dim=-1)

        # Run LSTM
        _, (hidden, cell) = self.lstm(lstm_input)

        return hidden, cell, social_inputs[-1]


class SocialLSTM(nn.Module):
    """
    Social LSTM for trajectory prediction.

    Extends LSTM Seq2Seq with social pooling to model agent interactions.
    This is a non-cooperative model that uses only vehicle-side perception
    of neighboring agents.
    """

    def __init__(self, config):
        super().__init__()
        self.model_name = 'Social-LSTM'
        self.cooperative = False  # Non-cooperative

        input_dim = config['input_dim']
        hidden_dim = config['hidden_dim']
        num_layers = config['num_layers']
        output_dim = config['output_dim']
        dropout = config['dropout']
        self.future_steps = config['future_steps']
        self.num_modes = config.get('num_modes', 1)

        # Encoder with social pooling
        self.encoder = SocialLSTMEncoder(
            input_dim, hidden_dim, num_layers, dropout
        )

        # Multiple decoder heads for multi-modal prediction
        self.decoders = nn.ModuleList([
            nn.LSTM(
                input_size=input_dim + hidden_dim,  # prev position + social context
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            for _ in range(self.num_modes)
        ])

        # Output projection for each mode
        self.output_projs = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim)
            for _ in range(self.num_modes)
        ])

        # Social context projection for decoder input
        self.social_proj = nn.Linear(hidden_dim, hidden_dim)

        # Mode probability head
        self.mode_prob_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),  # hidden + social context
            nn.ReLU(),
            nn.Linear(64, self.num_modes),
        )

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

        # Encode with social pooling
        hidden, cell, social_ctx = self.encoder(
            history, neighbors, neighbor_mask
        )

        # Compute mode probabilities
        mode_input = torch.cat([hidden[-1], social_ctx], dim=-1)
        mode_logits = self.mode_prob_head(mode_input)
        mode_probs = F.softmax(mode_logits, dim=-1)

        # Project social context for decoder
        social_feat = self.social_proj(social_ctx)  # (B, hidden_dim)

        # Decode for each mode
        all_predictions = []
        for mode_idx in range(self.num_modes):
            decoder = self.decoders[mode_idx]
            output_proj = self.output_projs[mode_idx]

            h, c = hidden.clone(), cell.clone()
            decoder_input_pos = history[:, -1:, :]  # (B, 1, 2)

            mode_predictions = []
            for t in range(self.future_steps):
                # Concatenate position and social context
                social_expanded = social_feat.unsqueeze(1)  # (B, 1, hidden_dim)
                decoder_input = torch.cat(
                    [decoder_input_pos, social_expanded], dim=-1
                )

                out, (h, c) = decoder(decoder_input, (h, c))
                pred_pos = output_proj(out.squeeze(1))  # (B, 2)
                mode_predictions.append(pred_pos)
                decoder_input_pos = pred_pos.unsqueeze(1)

            mode_pred = torch.stack(mode_predictions, dim=1)  # (B, T_f, 2)
            all_predictions.append(mode_pred)

        predictions = torch.stack(all_predictions, dim=1)  # (B, K, T_f, 2)

        return {
            'predictions': predictions,
            'mode_probs': mode_probs,
        }
