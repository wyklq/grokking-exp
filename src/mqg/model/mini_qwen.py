"""Mini-Qwen decoder block and full model.

Block layout (pre-norm, Qwen-style):

    x = x + Attention(RMSNorm(x))
    x = x + SwiGLU(RMSNorm(x))
"""
from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from .attention import GQAAttention
from .ffn import SwiGLU
from .rmsnorm import RMSNorm
from .rope import RoPECache


@dataclass
class MiniQwenConfig:
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    head_dim: int = 16
    n_kv_heads: int = 1
    ffn_hidden: int = 170
    vocab_size: int = 115
    max_seq_len: int = 8
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    tied_embedding: bool = True


class MiniQwenBlock(nn.Module):
    def __init__(self, cfg: MiniQwenConfig, rope: RoPECache) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = GQAAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            head_dim=cfg.head_dim,
            n_kv_heads=cfg.n_kv_heads,
            rope=rope,
        )
        self.norm2 = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MiniQwen(nn.Module):
    def __init__(self, cfg: MiniQwenConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # RoPE cache shared by all layers (depends only on head_dim & seq).
        self.rope = RoPECache(cfg.head_dim, cfg.max_seq_len, base=cfg.rope_base)

        self.blocks = nn.ModuleList(
            [MiniQwenBlock(cfg, self.rope) for _ in range(cfg.n_layers)]
        )
        self.norm_f = RMSNorm(cfg.d_model, eps=cfg.norm_eps)

        if cfg.tied_embedding:
            self.lm_head = None  # use embedding.weight transposed
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        # Conservative small-scale init (works well for grokking experiments).
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        """tokens: (B, S) int64 -> logits: (B, S, vocab)."""
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        if self.lm_head is None:
            logits = x @ self.embedding.weight.T
        else:
            logits = self.lm_head(x)
        return logits

    def num_params(self, exclude_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if exclude_embedding:
            n -= self.embedding.weight.numel()
            if self.lm_head is not None:
                n -= self.lm_head.weight.numel()
        return n
