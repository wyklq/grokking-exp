"""Phase-diagram grid enumeration and scan orchestration (Phase 4).

A scan is a Cartesian product of `alpha_values` × `lambda_values`.
Each cell is identified by (alpha_idx, lambda_idx) in lattice space.
Boundary detection (boundary.py) operates on these indices.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import math


@dataclass(frozen=True)
class GridCell:
    alpha_idx: int
    lambda_idx: int
    alpha: float
    lam: float


@dataclass
class GridSpec:
    alpha_values: tuple[float, ...]
    lambda_values: tuple[float, ...]

    def cells(self) -> list[GridCell]:
        out: list[GridCell] = []
        for i, a in enumerate(self.alpha_values):
            for j, l in enumerate(self.lambda_values):
                out.append(GridCell(i, j, float(a), float(l)))
        return out

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.alpha_values), len(self.lambda_values))


def default_alpha_grid(n: int = 9) -> tuple[float, ...]:
    """Linear in [0.1, 0.9] (excludes 0 and 1; both are degenerate)."""
    return tuple(round(0.1 + 0.8 * k / (n - 1), 4) for k in range(n))


def default_lambda_grid(n: int = 7, lo: float = 1e-2, hi: float = 1e1) -> tuple[float, ...]:
    """Log-uniform between lo and hi."""
    log_lo, log_hi = math.log10(lo), math.log10(hi)
    return tuple(round(10 ** (log_lo + (log_hi - log_lo) * k / (n - 1)), 6) for k in range(n))


def default_grid_spec() -> GridSpec:
    return GridSpec(alpha_values=default_alpha_grid(), lambda_values=default_lambda_grid())
