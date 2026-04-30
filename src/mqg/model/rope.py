"""Rotary Position Embedding (RoPE).

Reference: Su et al. (2021), "RoFormer: Enhanced Transformer with Rotary Position Embedding".
We pre-compute cos/sin caches for the maximum sequence length and apply via
the standard "rotate half" trick.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


def _build_cache(max_seq_len: int, head_dim: int, base: float, device, dtype):
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    half = head_dim // 2
    # frequencies for each pair (i = 0..half-1)
    freqs = base ** (-torch.arange(0, half, device=device, dtype=torch.float32) / half)
    pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(pos, freqs)  # (max_seq_len, half)
    cos = angles.cos().to(dtype)
    sin = angles.sin().to(dtype)
    return cos, sin


def _rotate_half(x: Tensor) -> Tensor:
    # split last dim in half, rotate: (x1, x2) -> (-x2, x1)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE.

    Args:
        x: (B, S, H, D) where D = head_dim (even).
        cos, sin: (S, D/2) for the relevant positions.
    """
    # broadcast cos/sin to (1, S, 1, D)
    cos_full = torch.cat((cos, cos), dim=-1).unsqueeze(0).unsqueeze(2)
    sin_full = torch.cat((sin, sin), dim=-1).unsqueeze(0).unsqueeze(2)
    return (x * cos_full) + (_rotate_half(x) * sin_full)


class RoPECache(nn.Module):
    """Holds RoPE cos/sin caches as buffers."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        cos, sin = _build_cache(max_seq_len, head_dim, base, device="cpu", dtype=torch.float32)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def get(self, seq_len: int) -> tuple[Tensor, Tensor]:
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds RoPE cache max={self.max_seq_len}")
        return self.cos[:seq_len], self.sin[:seq_len]
