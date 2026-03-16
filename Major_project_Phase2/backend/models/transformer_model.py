"""
Transformer-based models for stock direction prediction.
Includes BERT, RoBERTa, and DistilBERT variants.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional

class TransformerEncoder(nn.Module):
    """Custom Transformer Encoder for time series data."""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.fc1 = nn.Linear(d_model, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification head
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x.squeeze(-1)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class BERTStyleModel(nn.Module):
    """
    BERT-style architecture adapted for financial time series.
    Uses multi-head self-attention with pre-norm architecture.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.layer_norm_emb = nn.LayerNorm(hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout)
        
        # BERT-style encoder layers (pre-norm)
        self.layers = nn.ModuleList([
            BERTEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(hidden_dim)
        
        # Pooler (like BERT's [CLS] token pooler)
        self.pooler = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embed input
        x = self.embedding(x)
        x = self.layer_norm_emb(x)
        x = self.pos_encoding(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_layer_norm(x)
        
        # Pool (use mean of all positions)
        pooled = x.mean(dim=1)
        pooled = self.pooler(pooled)
        
        # Classify
        output = self.classifier(pooled)
        return output.squeeze(-1)


class BERTEncoderLayer(nn.Module):
    """Single BERT encoder layer with pre-norm."""
    
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self attention
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed)
        x = x + self.dropout(attn_out)
        
        # Pre-norm feedforward
        normed = self.norm2(x)
        ff_out = self.ff(normed)
        x = x + ff_out
        
        return x


class RoBERTaStyleModel(nn.Module):
    """
    RoBERTa-style architecture - similar to BERT but with:
    - Dynamic masking
    - Larger batch training simulation
    - No NSP task (classification only)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection with larger capacity
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Learnable positional embeddings (like RoBERTa)
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, hidden_dim) * 0.02)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            RoBERTaEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Classification head with more capacity
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_proj(x)
        
        # Add positional embeddings
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Pass through encoder
        for layer in self.encoder_layers:
            x = layer(x)
        
        x = self.final_norm(x)
        
        # Global pooling
        x = x.mean(dim=1)
        
        # Classify
        return self.classifier(x).squeeze(-1)


class RoBERTaEncoderLayer(nn.Module):
    """RoBERTa encoder layer with post-norm."""
    
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feedforward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class DistilBERTStyleModel(nn.Module):
    """
    DistilBERT-style architecture - lighter and faster:
    - Fewer layers (knowledge distillation concept)
    - Efficient attention
    - Faster inference
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        num_heads: int = 6,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Efficient embedding
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Sinusoidal positional encoding (more efficient)
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout)
        
        # Fewer but efficient layers
        self.layers = nn.ModuleList([
            DistilBERTLayer(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # Efficient classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embed
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        # Encode
        for layer in self.layers:
            x = layer(x)
        
        # Pool and classify
        x = x.mean(dim=1)
        return self.classifier(x).squeeze(-1)


class DistilBERTLayer(nn.Module):
    """Efficient DistilBERT layer."""
    
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.sa_layer_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim)
        )
        self.output_layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self attention
        attn_out, _ = self.attention(x, x, x)
        x = self.sa_layer_norm(x + self.dropout(attn_out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.output_layer_norm(x + self.dropout(ffn_out))
        
        return x


class HybridTransformerLSTM(nn.Module):
    """
    Hybrid model combining Transformer attention with LSTM memory.
    Best of both worlds for time series prediction.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        lstm_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer for global attention
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # LSTM for sequential memory
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # transformer + bidirectional lstm
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project input
        x = self.input_proj(x)
        
        # Transformer path
        x_trans = self.pos_encoding(x)
        x_trans = self.transformer(x_trans)
        x_trans = x_trans.mean(dim=1)  # Global pooling
        
        # LSTM path
        x_lstm, _ = self.lstm(x)
        x_lstm = x_lstm[:, -1, :]  # Last timestep (bidirectional)
        
        # Fuse both representations
        x_fused = torch.cat([x_trans, x_lstm], dim=-1)
        x_fused = self.fusion(x_fused)
        
        # Classify
        return self.classifier(x_fused).squeeze(-1)
