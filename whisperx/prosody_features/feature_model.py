import torch.nn as nn
import torch
import math
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encodings
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape ``[batch_size, seq_len, embedding_dim]``.

        Returns:
            Tensor: Output tensor of shape ``[batch_size, seq_len, embedding_dim]``.
        """
        batch_size, seq_len, _ = x.size()
        
        # Add positional encodings: Slice `self.pe` up to `seq_len` and broadcast over batch
        x = x + self.pe[:seq_len].unsqueeze(0)  # Shape: [1, seq_len, d_model]
        return self.dropout(x)

class ProsodyFeatureModel(nn.Module):
    
    def __init__(self, num_tokens: int, embedding_dim: int = 128, num_layers: int = 2, dropout: float = 0.0):
        
        super().__init__()
        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(num_embeddings=num_tokens, embedding_dim=embedding_dim)
        self.pos_encoding = PositionalEncoding(d_model=embedding_dim, dropout=dropout)
        self.encoder = nn.TransformerEncoder(
            encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, dropout=dropout),
            num_layers=num_layers
        )
        
    def forward(self, x: Tensor) -> Tensor:
        
        embeds = self.embedding(x)
        embeds_pe = self.pos_encoding(embeds)
        z = self.encoder(embeds_pe)
        z_mean = z.mean(1)
        return z_mean