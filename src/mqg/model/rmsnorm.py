"""RMSNorm — root-mean-square layer normalization (pre-norm style).

Reference: Zhang & Sennrich (2019), "Root Mean Square Layer Normalization".
Used by LLaMA / Qwen / DeepSeek. No bias, single learnable gain.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., d)
        # Compute in fp32 for stability even when activations are fp16/bf16.
        dtype = x.dtype
        x_f = x.float()
        rms = x_f.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x_f * rms).to(dtype) * self.weight
