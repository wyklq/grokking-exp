"""Phase-diagram scan orchestration (Phase 4).

Two-stage protocol per findings.md D7+D8:
  Phase 1: 1 seed per cell across the full grid → coarse phase map.
  Phase 2: extra seeds at cells on phase boundaries → refined map with
           N=5 total per boundary cell.

Results are stored in long-format: one row per (group, alpha_idx,
lambda_idx, seed). Aggregation (majority phase, etc.) is downstream.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch
from torch import Tensor

from ..data import TaskSpec, make_split
from ..model import MiniQwenConfig
from ..train.trainer import TrainConfig, TrainResult
from .boundary import detect_boundary_cells, majority_phase
from .grid import GridCell, GridSpec
from .multi_seed import train_multi_seed


@dataclass
class CellRunRecord:
    group: str
    split_strategy: str
    tied_embedding: bool
    alpha_idx: int
    lambda_idx: int
    alpha: float
    lam: float
    seed: int
    split_seed: int
    phase: str
    t_train: Optional[int]
    t_test: Optional[int]
    final_step: int
    train_loss_final: Optional[float]
    test_loss_final: Optional[float]
    train_acc_final: Optional[float]
    test_acc_final: Optional[float]


def _record(
    group: str,
    split_strategy: str,
    tied: bool,
    cell: GridCell,
    seed: int,
    split_seed: int,
    res: TrainResult,
) -> CellRunRecord:
    last = res.history[-1] if res.history else None
    return CellRunRecord(
        group=group,
        split_strategy=split_strategy,
        tied_embedding=tied,
        alpha_idx=cell.alpha_idx,
        lambda_idx=cell.lambda_idx,
        alpha=cell.alpha,
        lam=cell.lam,
        seed=seed,
        split_seed=split_seed,
        phase=res.phase,
        t_train=res.t_train,
        t_test=res.t_test,
        final_step=res.final_step,
        train_loss_final=last.train_loss if last else None,
        test_loss_final=last.test_loss if last else None,
        train_acc_final=last.train_acc if last else None,
        test_acc_final=last.test_acc if last else None,
    )


def run_cell(
    *,
    group: str,
    split_strategy: str,
    spec: TaskSpec,
    cell: GridCell,
    seeds: list[int],
    base_train_cfg: TrainConfig,
    base_model_cfg: MiniQwenConfig,
    split_seed: int = 0,
    device: str = "cpu",
    log_steps: tuple[int, ...] | None = None,
) -> list[CellRunRecord]:
    """Run all `seeds` for a single (alpha, lambda) cell.

    Same train/test split is used across seeds (controlled by `split_seed`).
    """
    train_cfg = TrainConfig(
        lr=base_train_cfg.lr,
        betas=base_train_cfg.betas,
        weight_decay=cell.lam,
        T_min=base_train_cfg.T_min,
        T_max=base_train_cfg.T_max,
        grok_extension_factor=base_train_cfg.grok_extension_factor,
        grok_ratio=base_train_cfg.grok_ratio,
        acc_threshold=base_train_cfg.acc_threshold,
        seed=seeds[0],  # base seed; multi_seed handles per-seed init
    )
    train_idx, test_idx = make_split(split_strategy, spec, cell.alpha, seed=split_seed)
    results = train_multi_seed(
        model_cfg=base_model_cfg,
        train_cfg=train_cfg,
        spec=spec,
        train_idx=train_idx,
        test_idx=test_idx,
        seeds=seeds,
        device=device,
        log_steps=log_steps,
    )
    return [
        _record(group, split_strategy, base_model_cfg.tied_embedding,
                cell, seed=s, split_seed=split_seed, res=r)
        for s, r in zip(seeds, results)
    ]


def run_phase1(
    *,
    group: str,
    split_strategy: str,
    spec: TaskSpec,
    grid: GridSpec,
    base_train_cfg: TrainConfig,
    base_model_cfg: MiniQwenConfig,
    split_seed: int = 0,
    device: str = "cpu",
    log_steps: tuple[int, ...] | None = None,
    on_cell_done=None,
) -> list[CellRunRecord]:
    """Run 1 seed per cell across the full grid."""
    records: list[CellRunRecord] = []
    for cell in grid.cells():
        recs = run_cell(
            group=group,
            split_strategy=split_strategy,
            spec=spec,
            cell=cell,
            seeds=[0],
            base_train_cfg=base_train_cfg,
            base_model_cfg=base_model_cfg,
            split_seed=split_seed,
            device=device,
            log_steps=log_steps,
        )
        records.extend(recs)
        if on_cell_done is not None:
            on_cell_done(cell, recs)
    return records


def run_phase2(
    *,
    group: str,
    split_strategy: str,
    spec: TaskSpec,
    grid: GridSpec,
    phase1_records: list[CellRunRecord],
    n_seeds: int,
    base_train_cfg: TrainConfig,
    base_model_cfg: MiniQwenConfig,
    split_seed: int = 0,
    device: str = "cpu",
    log_steps: tuple[int, ...] | None = None,
    on_cell_done=None,
) -> list[CellRunRecord]:
    """Run additional seeds at boundary cells (totalling n_seeds per boundary cell).

    Reuses Phase 1's seed=0 result; runs seeds 1..n_seeds-1 here.
    Returns ONLY the new records (Phase 2 additions).
    """
    cell_phase: dict[tuple[int, int], str] = {
        (r.alpha_idx, r.lambda_idx): r.phase for r in phase1_records if r.seed == 0
    }
    boundary = detect_boundary_cells(cell_phase, shape=grid.shape)
    cells_by_idx = {(c.alpha_idx, c.lambda_idx): c for c in grid.cells()}
    new_records: list[CellRunRecord] = []
    for idx in boundary:
        cell = cells_by_idx[idx]
        extra_seeds = list(range(1, n_seeds))
        if not extra_seeds:
            continue
        recs = run_cell(
            group=group,
            split_strategy=split_strategy,
            spec=spec,
            cell=cell,
            seeds=extra_seeds,
            base_train_cfg=base_train_cfg,
            base_model_cfg=base_model_cfg,
            split_seed=split_seed,
            device=device,
            log_steps=log_steps,
        )
        new_records.extend(recs)
        if on_cell_done is not None:
            on_cell_done(cell, recs)
    return new_records
