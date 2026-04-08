"""
DualVLN System 1 sub-modules ported from InternNav.

Contains:
- SinusoidalPositionalEncoding: positional encoding for trajectory timesteps
- MemoryEncoder: self-attention on ViT patch tokens for temporal visual memory
- QFormer: cross-attention query module to compress visual memory into fixed tokens
"""

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal encoding of shape (B, T, embedding_dim)
    given timesteps of shape (B, T).
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        timesteps = timesteps.float()
        B, T = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        enc = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)
        return enc


class MemoryEncoder(nn.Module):
    """
    Self-attention encoder for visual memory features.
    Takes ViT patch tokens from two images (pixel_goal + current),
    adds positional embeddings, and encodes with TransformerEncoder.
    """

    def __init__(self, hidden_size=384, num_heads=6, num_layers=3, max_len=512, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, batch_first=True, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.memory_pos = nn.Parameter(torch.randn(max_len, hidden_size))

    def forward(self, memory, memory_mask=None):
        """
        Args:
            memory: (B, N, C) — flattened ViT patch tokens from 2 images
            memory_mask: (B, N) — optional padding mask
        Returns:
            encoded_memory: (B, N, C)
        """
        B, N, C = memory.shape
        pos = self.memory_pos[:N, :].unsqueeze(0).expand(B, -1, -1)
        memory = memory + pos
        encoded_memory = self.encoder(memory, src_key_padding_mask=memory_mask)
        return encoded_memory


class QFormer(nn.Module):
    """
    Query-Former: cross-attention module with learnable queries.
    Compresses variable-length visual memory into fixed-size tokens.

    In DualVLN:
      Q = 32 learnable query_tokens (768-dim)
      K, V = cat(original_vit_feat, memory_encoder_output) (768-dim)
      Output = (B, 32, 768) memory tokens
    """

    def __init__(self, num_query=32, hidden_size=768, num_layers=3, num_heads=12):
        super().__init__()
        self.num_query = num_query
        self.hidden_size = hidden_size

        self.query_tokens = nn.Parameter(torch.randn(num_query, hidden_size))
        self.query_pos = nn.Parameter(torch.randn(num_query, hidden_size))

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.visual_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, visual_feats, visual_attn_mask=None):
        """
        Args:
            visual_feats: (B, N, hidden_size) — concatenated original + encoded features
            visual_attn_mask: (B, N) — optional key padding mask
        Returns:
            (B, num_query, hidden_size)
        """
        B = visual_feats.size(0)
        query_tokens = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        query_tokens = query_tokens + self.query_pos.unsqueeze(0)
        out = self.decoder(query_tokens, visual_feats, memory_key_padding_mask=visual_attn_mask)
        return out
