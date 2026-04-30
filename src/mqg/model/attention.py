"""Grouped Query Attention (GQA) with RoPE.

n_kv_heads divides n_heads. Each KV head is shared across n_heads / n_kv_heads
query heads. When n_kv_heads = 1, this is Multi-Query Attention.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from .rope import RoPECache, apply_rope


class GQAAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        n_kv_heads: int,
        rope: RoPECache,
    ) -> None:
        super().__init__()
        assert n_heads * head_dim == d_model, (
            f"n_heads ({n_heads}) * head_dim ({head_dim}) must equal d_model ({d_model})"
        )
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.repeat = n_heads // n_kv_heads
        self.rope = rope

        # Separate projections (no bias, like LLaMA / Qwen).
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(self, x: Tensor) -> Tensor:
        B, S, _ = x.shape

        q = self.q_proj(x)  # (B, S, n_heads * head_dim)
        k = self.k_proj(x)  # (B, S, n_kv_heads * head_dim)
        v = self.v_proj(x)

        q = rearrange(q, "b s (h d) -> b s h d", h=self.n_heads)
        k = rearrange(k, "b s (h d) -> b s h d", h=self.n_kv_heads)
        v = rearrange(v, "b s (h d) -> b s h d", h=self.n_kv_heads)

        # RoPE on q and k
        cos, sin = self.rope.get(S)
        q = apply_rope(q, cos.to(q.device), sin.to(q.device))
        k = apply_rope(k, cos.to(k.device), sin.to(k.device))

        # Repeat KV to match Q heads
        if self.repeat > 1:
            k = k.repeat_interleave(self.repeat, dim=2)
            v = v.repeat_interleave(self.repeat, dim=2)

        # (B, H, S, D) for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention with causal mask
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal_mask = torch.triu(
            torch.full((S, S), float("-inf"), device=x.device, dtype=attn_scores.dtype),
            diagonal=1,
        )
        attn_scores = attn_scores + causal_mask
        attn = F.softmax(attn_scores, dim=-1)

        out = torch.matmul(attn, v)  # (B, H, S, D)
        out = rearrange(out, "b h s d -> b s (h d)")
        return self.o_proj(out)
