"""
CNN-BiLSTM hybrid model for stock direction prediction.
1D Convolutions extract local patterns, BiLSTM captures temporal dependencies.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CNNBiLSTMModel(nn.Module):
    """
    1D CNN + Bidirectional LSTM hybrid for binary stock direction classification.
    Input: (batch, seq_len, num_features)
    Output: (batch, 1) — probability of UP
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        cnn_filters: int = 64,
        kernel_sizes: tuple = (3, 5, 7),
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Multi-scale 1D CNN branches
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_size, cnn_filters, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(cnn_filters),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
            )
            for k in kernel_sizes
        ])

        cnn_out_size = cnn_filters * len(kernel_sizes)

        # Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=cnn_out_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention over BiLSTM outputs
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        # CNN expects (batch, channels, seq_len)
        x_cnn = x.permute(0, 2, 1)

        # Multi-scale CNN
        conv_outs = [conv(x_cnn) for conv in self.convs]
        # Each: (batch, cnn_filters, seq_len)
        cnn_cat = torch.cat(conv_outs, dim=1)  # (batch, cnn_filters*3, seq_len)
        cnn_cat = cnn_cat.permute(0, 2, 1)  # (batch, seq_len, cnn_filters*3)

        # BiLSTM
        bilstm_out, _ = self.bilstm(cnn_cat)
        # bilstm_out: (batch, seq_len, hidden_size*2)

        # Attention
        attn_weights = self.attention(bilstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = (bilstm_out * attn_weights).sum(dim=1)

        out = self.classifier(context)
        return out.squeeze(-1)
