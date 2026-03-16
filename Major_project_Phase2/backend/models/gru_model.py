"""
GRU model for stock direction prediction.
Multi-layer GRU with temporal attention mechanism.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """
    Stacked GRU for binary classification of stock direction.
    Input: (batch, seq_len, num_features)
    Output: (batch, 1) — probability of UP
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        # gru_out: (batch, seq_len, hidden_size)

        attn_weights = self.attention(gru_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = (gru_out * attn_weights).sum(dim=1)

        out = self.classifier(context)
        return out.squeeze(-1)
