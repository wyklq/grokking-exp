"""Phase-diagram grid enumeration and scan orchestration (Phase 4).

A scan is a Cartesian product of `alpha_values` × `lambda_values`.
Each cell is identified by (alpha_idx, lambda_idx) in lattice space.
Boundary detection (boundary.py) operates on these indices.
"""
from __future__ import annotations

from dataclasses import dataclass

import math

DEFAULT_LAMBDA_HI = 10 ** 0.5


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

    def __post_init__(self) -> None:
        if not self.alpha_values:
            raise ValueError("alpha_values must contain at least one value")
        if not self.lambda_values:
            raise ValueError("lambda_values must contain at least one value")
        bad_alphas = [a for a in self.alpha_values if not math.isfinite(a) or not 0.0 < a < 1.0]
        if bad_alphas:
            raise ValueError(f"alpha_values must all be finite values in (0, 1), got {bad_alphas}")
        bad_lambdas = [lam for lam in self.lambda_values if not math.isfinite(lam) or lam < 0.0]
        if bad_lambdas:
            raise ValueError(
                f"lambda_values must all be finite non-negative values, got {bad_lambdas}"
            )

    def cells(self) -> list[GridCell]:
        out: list[GridCell] = []
        for i, a in enumerate(self.alpha_values):
            for j, lam in enumerate(self.lambda_values):
                out.append(GridCell(i, j, float(a), float(lam)))
        return out

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.alpha_values), len(self.lambda_values))


def default_alpha_grid(n: int = 9) -> tuple[float, ...]:
    """Linear in [0.1, 0.9] (excludes 0 and 1; both are degenerate)."""
    if n < 2:
        raise ValueError(f"alpha grid size must be >= 2, got {n}")
    return tuple(round(0.1 + 0.8 * k / (n - 1), 4) for k in range(n))


def default_lambda_grid(
    n: int = 6,
    lo: float = 1e-2,
    hi: float = DEFAULT_LAMBDA_HI,
) -> tuple[float, ...]:
    """Log-uniform between lo and hi."""
    if n < 2:
        raise ValueError(f"lambda grid size must be >= 2, got {n}")
    if lo <= 0.0 or hi <= 0.0 or hi <= lo:
        raise ValueError(f"lambda grid bounds must satisfy 0 < lo < hi, got lo={lo}, hi={hi}")
    log_lo, log_hi = math.log10(lo), math.log10(hi)
    return tuple(round(10 ** (log_lo + (log_hi - log_lo) * k / (n - 1)), 6) for k in range(n))


def default_grid_spec() -> GridSpec:
    return GridSpec(alpha_values=default_alpha_grid(), lambda_values=default_lambda_grid())
