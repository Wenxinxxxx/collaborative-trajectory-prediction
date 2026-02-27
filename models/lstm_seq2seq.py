"""
LSTM Encoder-Decoder (Seq2Seq) Baseline Model.

This is the simplest baseline for trajectory prediction. It uses an LSTM
encoder to compress the history trajectory into a context vector, and an
LSTM decoder to generate future trajectory predictions step by step.

This model operates in a NON-COOPERATIVE setting: it only uses the ego
agent's own historical trajectory, without any neighbor, infrastructure,
or map information.

Reference:
  - Alahi et al., "Social LSTM: Human Trajectory Prediction in Crowded
    Spaces," CVPR 2016.
"""

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """LSTM encoder that compresses history trajectory into a context vector."""

    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x):
        """
        Args:
            x: (batch, history_steps, input_dim)
        Returns:
            hidden: (num_layers, batch, hidden_dim)
            cell:   (num_layers, batch, hidden_dim)
        """
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell


class LSTMDecoder(nn.Module):
    """LSTM decoder that generates future trajectory step by step."""

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        """
        Args:
            x:      (batch, 1, input_dim) single step input
            hidden: (num_layers, batch, hidden_dim)
            cell:   (num_layers, batch, hidden_dim)
        Returns:
            output: (batch, output_dim) predicted point
            hidden, cell: updated states
        """
        out, (hidden, cell) = self.lstm(x, (hidden, cell))
        output = self.fc(out.squeeze(1))
        return output, hidden, cell


class LSTMSeq2Seq(nn.Module):
    """
    LSTM Seq2Seq model for trajectory prediction.

    Supports multi-modal prediction by using multiple decoder heads.
    """

    def __init__(self, config):
        super().__init__()
        self.model_name = 'LSTM-Seq2Seq'
        self.cooperative = False  # Non-cooperative baseline

        input_dim = config['input_dim']
        hidden_dim = config['hidden_dim']
        num_layers = config['num_layers']
        output_dim = config['output_dim']
        dropout = config['dropout']
        self.future_steps = config['future_steps']
        self.num_modes = config.get('num_modes', 1)

        # Encoder
        self.encoder = LSTMEncoder(input_dim, hidden_dim, num_layers, dropout)

        # Multiple decoder heads for multi-modal prediction
        self.decoders = nn.ModuleList([
            LSTMDecoder(input_dim, hidden_dim, num_layers, output_dim, dropout)
            for _ in range(self.num_modes)
        ])

        # Mode probability head
        self.mode_prob_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_modes),
            nn.Softmax(dim=-1),
        )

    def forward(self, batch):
        """
        Args:
            batch: dict with at least 'history' key
                   history: (batch_size, history_steps, 2)

        Returns:
            dict with:
                'predictions': (batch_size, num_modes, future_steps, 2)
                'mode_probs':  (batch_size, num_modes)
        """
        history = batch['history']  # (B, T_h, 2)
        batch_size = history.shape[0]
        device = history.device

        # Encode history
        hidden, cell = self.encoder(history)

        # Compute mode probabilities from the final hidden state
        mode_probs = self.mode_prob_head(hidden[-1])  # (B, num_modes)

        # Decode for each mode
        all_predictions = []
        for mode_idx in range(self.num_modes):
            decoder = self.decoders[mode_idx]
            # Start with the last history point
            decoder_input = history[:, -1:, :]  # (B, 1, 2)
            h, c = hidden.clone(), cell.clone()

            mode_predictions = []
            for t in range(self.future_steps):
                output, h, c = decoder(decoder_input, h, c)
                mode_predictions.append(output)
                decoder_input = output.unsqueeze(1)  # (B, 1, 2)

            mode_pred = torch.stack(mode_predictions, dim=1)  # (B, T_f, 2)
            all_predictions.append(mode_pred)

        predictions = torch.stack(all_predictions, dim=1)  # (B, K, T_f, 2)

        return {
            'predictions': predictions,
            'mode_probs': mode_probs,
        }
