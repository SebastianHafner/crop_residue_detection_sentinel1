"""
================================================================================
models/psenet.py
================================================================================

WHAT IT DOES
------------
Full Pixel-Set Encoder with Temporal Attention (PSENet) implementation,
tailored for the crop residue detection pipeline.

Architecture:
  Input (B, T, C, P)
      ↓
  PixelSetEncoder      — shared MLP per pixel → masked pooling → MLP
      ↓
  (B, T, E)            — one field embedding per timestep
      ↓
  TemporalAttentionEncoder — DOY positional encoding + transformer + master query
      ↓
  (B, E)               — single season summary embedding
      ↓
  ClassifierHead       — MLP → logits
      ↓
  (B, n_classes)

Reference: Garnot et al. (2020) "Satellite Image Time Series Classification
with Pixel-Set Encoders and Temporal Self-Attention", CVPR 2020.

USAGE
-----
  from models.psenet import PSENet
  model = PSENet(cfg)
  logits = model(x, doy, mask)         # standard forward
  logits, attn = model(x, doy, mask, return_attention=True)  # + attention weights
================================================================================
"""

import math
import copy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class PSENet(nn.Module):
    """
    Pixel-Set Encoder with Temporal Attention for crop residue classification.

    Args:
        cfg: Hydra DictConfig — reads model.* and dataloader.* keys
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        C        = cfg.model.in_channels     # 12
        mlp1     = [C] + list(cfg.model.mlp1)           # [12, 32, 64]
        pooling  = cfg.model.pooling                     # 'mean_std'
        n_pool   = len(pooling.split('_'))               # 2
        inter    = mlp1[-1] * n_pool                     # 128
        mlp2     = list(cfg.model.mlp2)                  # [128, 128]
        assert mlp2[0] == inter, (
            f'mlp2[0] ({mlp2[0]}) must equal mlp1[-1]*n_pool ({inter}). '
            f'Check config: mlp1={cfg.model.mlp1}, pooling={pooling}')

        E        = cfg.model.embed_dim       # 128  (must equal mlp2[-1])
        assert E == mlp2[-1], f'embed_dim ({E}) must equal mlp2[-1] ({mlp2[-1]})'

        n_classes = cfg.model.out_channels   # 2
        mlp4      = list(cfg.model.mlp4) + [n_classes]  # [128,64,32,2]
        assert mlp4[0] == E, f'mlp4[0] ({mlp4[0]}) must equal embed_dim ({E})'

        self.pixel_encoder   = PixelSetEncoder(mlp1, pooling, mlp2)
        self.temporal_encoder = TemporalAttentionEncoder(
            embed_dim = E,
            n_heads   = cfg.model.n_heads,
            n_layers  = cfg.model.n_layers,
            dropout   = cfg.model.dropout,
        )
        self.classifier = get_mlp(mlp4, dropout=cfg.model.dropout)
        self.name = 'psenet'

    def forward(
        self,
        x:    Tensor,                    # (B, T, C, P)
        doy:  Tensor,                    # (B, T)
        mask: Tensor,                    # (B, P)
        return_attention: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """
        Args:
            x:    pixel set tensor (B, T, C, P)
            doy:  day-of-year per timestep (B, T)  float
            mask: valid pixel mask (B, P)  1=valid 0=padded
            return_attention: if True also return (B, T) attention weights

        Returns:
            logits (B, n_classes)
            [optional] attn_weights (B, T)
        """
        # (B, T, C, P) → (B, T, E)
        field_emb = self.pixel_encoder(x, mask)

        # (B, T, E) + doy → (B, E) [+ optional (B, T) attn]
        if return_attention:
            summary, attn = self.temporal_encoder(field_emb, doy, return_attention=True)
        else:
            summary = self.temporal_encoder(field_emb, doy)

        logits = self.classifier(summary)   # (B, n_classes)

        if return_attention:
            return logits, attn
        return logits


# ---------------------------------------------------------------------------
# Pixel Set Encoder
# ---------------------------------------------------------------------------

class PixelSetEncoder(nn.Module):
    """
    Encodes a set of pixels per field per timestep into a fixed-size embedding.

    Input:  (B, T, C, P)
    Output: (B, T, E)

    Steps:
      1. Shared MLP (mlp1) applied to each pixel independently
      2. Masked pooling (mean, std, max, min — configurable)
      3. Second MLP (mlp2) on the concatenated pooled features
    """

    def __init__(self, mlp1: List[int], pooling: str, mlp2: List[int]):
        super().__init__()
        self.pooling_ops = pooling.split('_')

        # MLP1: per-pixel feature extraction
        layers = []
        for i in range(len(mlp1) - 1):
            layers.append(LinearBNReLU(mlp1[i], mlp1[i + 1]))
        self.mlp1 = nn.Sequential(*layers)

        # MLP2: after pooling
        layers = []
        for i in range(len(mlp2) - 1):
            layers.append(nn.Linear(mlp2[i], mlp2[i + 1]))
            if i < len(mlp2) - 2:
                layers += [nn.BatchNorm1d(mlp2[i + 1]), nn.ReLU()]
            else:
                layers.append(nn.BatchNorm1d(mlp2[i + 1]))
        self.mlp2 = nn.Sequential(*layers)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x:    (B, T, C, P)
            mask: (B, P)  1=valid 0=padded
        Returns:
            (B, T, E)
        """
        B, T, C, P = x.shape

        # Flatten batch and time for efficient MLP application
        x_flat = x.view(B * T, C, P)             # (B*T, C, P)

        # MLP1 — applied per pixel (channel-first → last → MLP → first)
        out = x_flat.permute(0, 2, 1)            # (B*T, P, C)
        out = self.mlp1(out)                      # (B*T, P, D)
        D   = out.shape[-1]

        # Expand mask to (B*T, P)
        mask_bt = mask.unsqueeze(1).expand(B, T, P).contiguous().view(B * T, P)

        # Pooling
        pooled_parts = []
        for op in self.pooling_ops:
            if op == 'mean':
                pooled_parts.append(masked_mean(out, mask_bt))
            elif op == 'std':
                pooled_parts.append(masked_std(out, mask_bt))
            elif op == 'max':
                pooled_parts.append(masked_max(out, mask_bt))
            elif op == 'min':
                pooled_parts.append(masked_min(out, mask_bt))
        pooled = torch.cat(pooled_parts, dim=-1)  # (B*T, D*n_ops)

        # MLP2
        out = self.mlp2(pooled)                   # (B*T, E)
        out = out.view(B, T, -1)                  # (B, T, E)
        return out


class LinearBNReLU(nn.Module):
    """Linear → BatchNorm → ReLU block that handles (N, P, C) inputs."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn     = nn.BatchNorm1d(out_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, P, C)
        N, P, C = x.shape
        out = self.linear(x.view(N * P, C))   # (N*P, out_dim)
        out = self.bn(out)
        out = F.relu(out)
        return out.view(N, P, -1)


# ---------------------------------------------------------------------------
# Pooling operations — all handle a boolean mask (1=valid, 0=padded)
# ---------------------------------------------------------------------------

def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    """x: (N, P, D), mask: (N, P) → (N, D)"""
    mask_f  = mask.unsqueeze(-1).float()          # (N, P, 1)
    n_valid = mask_f.sum(dim=1).clamp(min=1.0)   # (N, 1)
    return (x * mask_f).sum(dim=1) / n_valid      # (N, D)


def masked_std(x: Tensor, mask: Tensor) -> Tensor:
    """x: (N, P, D), mask: (N, P) → (N, D)"""
    mask_f  = mask.unsqueeze(-1).float()
    n_valid = mask_f.sum(dim=1).clamp(min=2.0)   # need ≥2 for std
    mean    = (x * mask_f).sum(dim=1, keepdim=True) / n_valid.unsqueeze(1)
    sq_diff = ((x - mean) ** 2) * mask_f
    return torch.sqrt(sq_diff.sum(dim=1) / (n_valid - 1) + 1e-8)


def masked_max(x: Tensor, mask: Tensor) -> Tensor:
    """x: (N, P, D), mask: (N, P) → (N, D)"""
    mask_f = mask.unsqueeze(-1).float()
    x_masked = x * mask_f + (1 - mask_f) * (-1e9)
    return x_masked.max(dim=1)[0]


def masked_min(x: Tensor, mask: Tensor) -> Tensor:
    """x: (N, P, D), mask: (N, P) → (N, D)"""
    mask_f = mask.unsqueeze(-1).float()
    x_masked = x * mask_f + (1 - mask_f) * 1e9
    return x_masked.min(dim=1)[0]


# ---------------------------------------------------------------------------
# Temporal Attention Encoder
# ---------------------------------------------------------------------------

class TemporalAttentionEncoder(nn.Module):
    """
    Processes the sequence of field embeddings (B, T, E) into a single
    summary embedding (B, E) using:
      1. Sinusoidal DOY positional encoding (handles irregular time intervals)
      2. Transformer encoder (multi-head self-attention over time)
      3. Learned master query that attends over all timesteps

    The master query output is the field's season-level representation.
    Attention weights from the master query reveal which dates the model
    found most informative — used for interpretability analysis in the thesis.
    """

    def __init__(self, embed_dim: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_enc   = DOYPositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = embed_dim,
            nhead           = n_heads,
            dim_feedforward = embed_dim * 2,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,   # pre-norm: more stable for small datasets
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Learnable master query — one vector that summarises the whole season
        self.master_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

    def forward(
        self,
        x:   Tensor,                        # (B, T, E)
        doy: Tensor,                        # (B, T)
        return_attention: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """
        Args:
            x:   field embeddings (B, T, E)
            doy: day-of-year per timestep (B, T)
            return_attention: if True also return master query attention weights

        Returns:
            summary: (B, E)
            [optional] attn_weights: (B, T)  — master query → each timestep
        """
        B, T, E = x.shape

        # Add DOY positional encoding
        x = x + self.pos_enc(doy)           # (B, T, E)

        # Prepend master query
        mq  = self.master_query.expand(B, 1, E)    # (B, 1, E)
        seq = torch.cat([mq, x], dim=1)            # (B, T+1, E)

        # Transformer processes full sequence (master query + all timesteps)
        out = self.transformer(seq)                 # (B, T+1, E)

        # Master query output is the season summary
        summary = out[:, 0, :]                      # (B, E)

        if not return_attention:
            return summary

        # Extract attention weights from the last transformer layer
        # by re-running the final layer's attention manually
        attn_weights = self._extract_master_attn(seq)   # (B, T)
        return summary, attn_weights

    def _extract_master_attn(self, seq: Tensor) -> Tensor:
        """
        Re-run the last transformer layer's self-attention to get the
        master query's attention distribution over timesteps.

        Returns: (B, T)  — attention weight for each real timestep
                           (excludes the master query position itself)
        """
        last_layer = self.transformer.layers[-1]
        x_norm     = last_layer.norm1(seq)

        with torch.no_grad():
            _, attn = last_layer.self_attn(
                x_norm, x_norm, x_norm,
                need_weights=True,
                average_attn_weights=True,
            )
        # attn: (B, T+1, T+1) — row 0 = master query's attention
        # Skip column 0 (master query attending to itself)
        return attn[:, 0, 1:]   # (B, T)


class DOYPositionalEncoding(nn.Module):
    """
    Converts day-of-year (DOY) to a learned embedding of dimension E.

    Uses sin/cos encoding with annual period (365 days) so the model
    knows DOY 1 and DOY 365 are temporally adjacent.  A small learned
    linear layer projects the 2D sinusoidal vector to embed_dim.

    Handles irregular acquisition intervals naturally — the encoding is
    a continuous function of DOY, not a discrete positional index.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(2, embed_dim)

    def forward(self, doy: Tensor) -> Tensor:
        """
        Args:
            doy: (B, T)  float tensor of day-of-year values
        Returns:
            (B, T, embed_dim)
        """
        angle = 2.0 * math.pi * doy / 365.0                   # (B, T)
        enc   = torch.stack([angle.sin(), angle.cos()], dim=-1)  # (B, T, 2)
        return self.proj(enc)                                   # (B, T, E)


# ---------------------------------------------------------------------------
# Classifier head
# ---------------------------------------------------------------------------

def get_mlp(dims: List[int], dropout: float = 0.0) -> nn.Sequential:
    """
    Build a fully-connected MLP with BatchNorm, ReLU, and Dropout.
    The last layer has no activation (raw logits for CrossEntropyLoss).
    """
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
    return nn.Sequential(*layers)
