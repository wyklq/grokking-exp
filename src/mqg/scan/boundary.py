"""Boundary detection for phase-diagram cells (Phase 4).

A cell is "on a phase boundary" if at least one of its 4-neighbors (in
(alpha_idx, lambda_idx) lattice space) has a different majority phase.

Per rubber-duck design (#2 + #3), we return BOTH sides of every
disagreeing edge so Phase 2 can re-run them and disambiguate noise.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def _idx(d: dict, i: int, j: int) -> str | None:
    return d.get((i, j))


def detect_boundary_cells(
    cell_phase: dict[tuple[int, int], str],
    shape: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    """Return sorted list of (alpha_idx, lambda_idx) cells on a phase boundary.

    A cell c is included iff there exists a 4-neighbor n with cell_phase[n]
    defined and cell_phase[n] != cell_phase[c]. Both sides of each
    disagreeing edge are included.

    Args:
        cell_phase: mapping (i, j) -> phase label
        shape: optional (n_alpha, n_lambda); ignored for boundary computation
               (kept for API symmetry).

    Returns:
        Sorted list of (i, j) tuples.
    """
    boundary: set[tuple[int, int]] = set()
    for (i, j), ph in cell_phase.items():
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            n = (i + di, j + dj)
            if n in cell_phase and cell_phase[n] != ph:
                boundary.add((i, j))
                boundary.add(n)
    return sorted(boundary)


def majority_phase(phases: Iterable[str]) -> str:
    """Return the most common phase label; ties broken alphabetically (deterministic)."""
    counts: dict[str, int] = defaultdict(int)
    for p in phases:
        counts[p] += 1
    if not counts:
        return "fail"
    max_n = max(counts.values())
    candidates = sorted(p for p, n in counts.items() if n == max_n)
    return candidates[0]
