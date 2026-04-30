"""Instrument multi-seed training with progress measures (Phase 5b).

At every log step, unstack each seed's parameters into a temporary
MiniQwen instance and run `compute_all_measures` on it. Collect the
results into a long-format DataFrame keyed by (group, alpha_idx,
lambda_idx, seed, step).

Hessian computation is the expensive part (~20 backward passes per
checkpoint per seed). Use `measures_steps` to subset log_steps if
trajectory density isn't needed everywhere.
"""
from __future__ import annotations

from typing import Iterable, Optional

from torch import Tensor

from ..data import TaskSpec, build_full_dataset, make_split
from ..measures import compute_all_measures
from ..model import MiniQwen, MiniQwenConfig
from ..train.trainer import TrainConfig
from .grid import GridCell
from .multi_seed import train_multi_seed


def unstack_seed(
    params: dict[str, Tensor],
    buffers: dict[str, Tensor],
    cfg: MiniQwenConfig,
    seed_idx: int,
    device: str = "cpu",
) -> MiniQwen:
    """Materialize seed `seed_idx` into a fresh MiniQwen instance.

    Used by measure callbacks that need a real nn.Module (Hessian, hooks).
    """
    model = MiniQwen(cfg).to(device)
    state = {}
    for name, stacked in params.items():
        state[name] = stacked[seed_idx].detach().clone()
    for name, stacked in buffers.items():
        state[name] = stacked[seed_idx].detach().clone()
    # MiniQwen's state_dict includes both params and buffers (e.g. RoPE cache).
    # Use strict=False because RoPE buffers may not be in `buffers` dict if
    # they were registered as `persistent=False`.
    expected_keys = set(model.state_dict().keys())
    loadable_state = {name: value for name, value in state.items() if name in expected_keys}
    load_info = model.load_state_dict(loadable_state, strict=False)
    if load_info.missing_keys:
        raise RuntimeError(f"Failed to load seed {seed_idx}; missing keys: {load_info.missing_keys}")
    # Non-persistent buffers such as RoPE caches are reconstructed in __init__.
    return model


def run_cell_with_measures(
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
    measures_steps: Optional[Iterable[int]] = None,
    skip_hessian: bool = False,
    hessian_iters: int = 10,
) -> tuple[list, list[dict]]:
    """Run multi-seed training on a single cell, instrumented with measures.

    Returns:
        (train_results, trajectory_rows)
        train_results: list[TrainResult] (one per seed)
        trajectory_rows: list[dict], each row keyed by (group, alpha_idx,
            lambda_idx, seed, step) with all measure values flattened in.
    """
    if not seeds:
        raise ValueError("seeds must be non-empty")
    train_cfg = TrainConfig(
        lr=base_train_cfg.lr,
        betas=base_train_cfg.betas,
        weight_decay=cell.lam,
        T_min=base_train_cfg.T_min,
        T_max=base_train_cfg.T_max,
        grok_extension_factor=base_train_cfg.grok_extension_factor,
        grok_ratio=base_train_cfg.grok_ratio,
        acc_threshold=base_train_cfg.acc_threshold,
        seed=seeds[0],
    )
    train_idx, test_idx = make_split(split_strategy, spec, cell.alpha, seed=split_seed)
    tokens, _ = build_full_dataset(spec)
    tokens = tokens.to(device)
    train_tokens = tokens[train_idx.to(device)]

    measures_steps_set = set(measures_steps) if measures_steps is not None else None
    rows: list[dict] = []

    base_meta = {
        "group": group,
        "split_strategy": split_strategy,
        "tied_embedding": base_model_cfg.tied_embedding,
        "alpha_idx": cell.alpha_idx,
        "lambda_idx": cell.lambda_idx,
        "alpha": cell.alpha,
        "lam": cell.lam,
        "split_seed": split_seed,
    }

    def hook(step: int, params: dict, buffers: dict, _base: MiniQwen) -> None:
        if measures_steps_set is not None and step not in measures_steps_set:
            return
        for i, sd in enumerate(seeds):
            model = unstack_seed(params, buffers, base_model_cfg, i, device=device)
            measures = compute_all_measures(
                model,
                train_tokens=train_tokens,
                answer_pos=spec.answer_pos,
                p=spec.p,
                skip_hessian=skip_hessian,
                hessian_iters=hessian_iters,
                hessian_seed=sd,
            )
            row = {**base_meta, "seed": sd, "step": step, **measures}
            rows.append(row)

    train_results = train_multi_seed(
        model_cfg=base_model_cfg,
        train_cfg=train_cfg,
        spec=spec,
        train_idx=train_idx,
        test_idx=test_idx,
        seeds=seeds,
        device=device,
        log_steps=log_steps,
        at_log_step_hook=hook,
    )
    return train_results, rows


def to_dataframe(rows: list[dict]):
    """Convert collected trajectory rows to a pandas DataFrame.

    Pandas is imported lazily so test envs without pandas can still import this.
    """
    import pandas as pd
    return pd.DataFrame(rows)
