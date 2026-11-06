"""Transformer encodeur écrit from scratch + branche de features manuelles.

    tokens (256) -> Embedding token + position (128)
                 -> 4 x [LayerNorm -> Self-Attention (4 têtes) -> résiduel
                         LayerNorm -> Feed-forward 128->512->128 -> résiduel]
                 -> LayerNorm -> mean pooling (on ignore le padding) -> 128
    features (12) ---------------------------------------------------> concat 140
                 -> Dense 140 -> 64 (ReLU, dropout) -> Dense 64 -> 1 (logit)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Attention bidirectionnelle : chaque token regarde tous les autres (sauf le padding)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)   # Q, K, V en une seule projection
        self.out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, têtes, T, head_dim)
        q, k, v = (t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in (q, k, v))

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # (B, têtes, T, T)
        # pad_mask : 1 = vrai token, 0 = padding -> on interdit de regarder le padding
        scores = scores.masked_fill(pad_mask[:, None, None, :] == 0, float("-inf"))
        att = self.attn_drop(F.softmax(scores, dim=-1))

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)       # on recolle les têtes
        return self.resid_drop(self.out(y))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, pad_mask):
        x = x + self.attn(self.ln1(x), pad_mask)   # connexion résiduelle
        x = x + self.ff(self.ln2(x))
        return x


class PhishingTransformer(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, n_heads, n_layers, d_ff,
                 n_features, head_hidden, dropout, head_dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model + n_features, head_hidden), nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),               # 1 neurone : logit de P(phishing)
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids, pad_mask, feats):
        T = ids.size(1)
        pos = torch.arange(T, device=ids.device)
        x = self.drop(self.tok_emb(ids) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x, pad_mask)
        x = self.ln_f(x)

        # Mean pooling : moyenne des vecteurs des vrais tokens seulement
        m = pad_mask.unsqueeze(-1).float()
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1.0)

        return self.head(torch.cat([pooled, feats], dim=1)).squeeze(-1)  # logits (B,)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
