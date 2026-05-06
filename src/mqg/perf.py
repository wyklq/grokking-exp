"""Small performance switches used by CLI entrypoints."""
from __future__ import annotations

from typing import Literal

import torch

MatmulPrecision = Literal["highest", "high", "medium"]


def configure_matmul_precision(precision: MatmulPrecision | None) -> None:
    """Configure float32 matmul precision when explicitly requested."""
    if precision is not None:
        torch.set_float32_matmul_precision(precision)
