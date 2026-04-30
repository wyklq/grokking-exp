"""SwiGLU FFN.

SwiGLU(x) = down(silu(gate(x)) * up(x))

Three linear projections (no bias). Hidden dim is typically 8/3 * d_model
(matched to a vanilla 4*d_model FFN's parameter count, modulo rounding).
"""
from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))
