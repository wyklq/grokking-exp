"""vmap-based multi-seed trainer (Phase 4 core).

Trains N seeds of the same (alpha, lambda) cell in parallel via
`torch.func.stack_module_state` + `vmap` + a single `torch.optim.AdamW`
acting on the stacked parameter tensors.

Key invariants (per rubber-duck design review):
  - All seeds share the same train/test split (only init differs).
  - Loss is summed across seeds before backward (gradient equivalence
    to N independent runs).
  - Per-seed bookkeeping (t_train, t_test, history) is *frozen* once a
    seed hits its stop condition; the seed's params keep being updated
    (wasted compute) but its phase classification is stable.
  - The outer loop terminates when all seeds are stopped or step >= T_max.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor
from torch.func import functional_call, stack_module_state, vmap

from ..data import TaskSpec, build_full_dataset
from ..model import MiniQwen, MiniQwenConfig
from ..train.checkpoints import LogStepIterator
from ..train.trainer import (
    PhaseLabel,
    StepLog,
    TrainConfig,
    TrainResult,
    classify_phase,
)


def _build_stacked(
    model_cfg: MiniQwenConfig, seeds: list[int], device: str
) -> tuple[dict, dict, MiniQwen]:
    """Instantiate N MiniQwen with given seeds and stack their state."""
    models: list[MiniQwen] = []
    for s in seeds:
        torch.manual_seed(s)
        models.append(MiniQwen(model_cfg).to(device))
    params, buffers = stack_module_state(models)
    base = MiniQwen(model_cfg).to("meta")
    return params, buffers, base


def _per_seed_loss_acc(
    logits: Tensor, tokens: Tensor, answer_pos: int
) -> tuple[Tensor, Tensor]:
    """Compute per-seed CE loss + accuracy.

    Args:
        logits: (N, B, S, V)
        tokens: (B, S)  -- shared across seeds
        answer_pos: int

    Returns:
        loss: (N,) sum-reduced over batch (matches single-seed sum, then mean
              would cancel; but we use sum across batch as torch.optim's grad
              equivalent to mean*B doesn't matter here -- consistency with
              single-seed trainer requires mean reduction).
        acc:  (N,)
    """
    # logits at predicting position = answer_pos - 1
    pred_logits = logits[:, :, answer_pos - 1, :]   # (N, B, V)
    targets = tokens[:, answer_pos]                 # (B,)
    N, B, V = pred_logits.shape
    flat = pred_logits.reshape(N * B, V)
    tgt = targets.repeat(N)                         # (N*B,)
    loss_flat = torch.nn.functional.cross_entropy(flat, tgt, reduction="none")
    loss = loss_flat.reshape(N, B).mean(dim=1)      # per-seed mean over batch
    pred = pred_logits.argmax(dim=-1)               # (N, B)
    acc = (pred == targets.unsqueeze(0)).float().mean(dim=1)  # (N,)
    return loss, acc


def train_multi_seed(
    model_cfg: MiniQwenConfig,
    train_cfg: TrainConfig,
    spec: TaskSpec,
    train_idx: Tensor,
    test_idx: Tensor,
    seeds: list[int],
    device: str = "cpu",
    log_steps: tuple[int, ...] | None = None,
    at_log_step_hook=None,
) -> list[TrainResult]:
    """Train `len(seeds)` seeds in parallel via vmap.

    Args:
        at_log_step_hook: optional callable invoked at every log step:
            ``hook(step: int, params: dict[str, Tensor], buffers: dict, base: MiniQwen)``
            where params/buffers have leading dim N=len(seeds). Hook can
            unstack a seed and run measures on it (see scan/instrumented.py).
            Hook MUST NOT mutate `params` or `buffers`.

    Returns a list of TrainResult, one per seed (in same order).
    """
    N = len(seeds)
    assert N >= 1

    params, buffers, base = _build_stacked(model_cfg, seeds, device)

    optimizer = torch.optim.AdamW(
        params.values(),
        lr=train_cfg.lr,
        betas=train_cfg.betas,
        weight_decay=train_cfg.weight_decay,
    )

    tokens, _ = build_full_dataset(spec)
    tokens = tokens.to(device)
    train_tokens = tokens[train_idx.to(device)]
    test_tokens = tokens[test_idx.to(device)]

    def fmodel(p, b, x):
        return functional_call(base, (p, b), (x,))

    vfwd = vmap(fmodel, in_dims=(0, 0, None))

    # per-seed bookkeeping
    histories: list[list[StepLog]] = [[] for _ in range(N)]
    t_train: list[Optional[int]] = [None] * N
    t_test: list[Optional[int]] = [None] * N
    final_step: list[int] = [0] * N
    T_target: list[int] = [train_cfg.T_min] * N
    stopped: list[bool] = [False] * N

    log_iter = LogStepIterator(log_steps) if log_steps is not None else LogStepIterator()
    T_cap = train_cfg.T_max

    step = 0
    while step < T_cap and not all(stopped):
        # forward + backward (all seeds, even stopped ones — wasted but harmless)
        for p in params.values():
            if p.grad is not None:
                p.grad = None
        logits = vfwd(params, buffers, train_tokens)
        loss_per_seed, _ = _per_seed_loss_acc(logits, train_tokens, spec.answer_pos)
        # sum across seeds: gradient on each seed slice == standalone backward
        loss_per_seed.sum().backward()
        optimizer.step()
        step += 1

        if log_iter.reached(step):
            with torch.no_grad():
                tr_logits = vfwd(params, buffers, train_tokens)
                tr_loss, tr_acc = _per_seed_loss_acc(tr_logits, train_tokens, spec.answer_pos)
                te_logits = vfwd(params, buffers, test_tokens)
                te_loss, te_acc = _per_seed_loss_acc(te_logits, test_tokens, spec.answer_pos)

            for i in range(N):
                if stopped[i]:
                    continue
                histories[i].append(
                    StepLog(
                        step=step,
                        train_loss=float(tr_loss[i].item()),
                        train_acc=float(tr_acc[i].item()),
                        test_loss=float(te_loss[i].item()),
                        test_acc=float(te_acc[i].item()),
                    )
                )
                final_step[i] = step

                if t_train[i] is None and tr_acc[i].item() >= train_cfg.acc_threshold:
                    t_train[i] = step
                    T_target[i] = min(
                        T_cap,
                        max(T_target[i], train_cfg.grok_extension_factor * step),
                    )
                if t_test[i] is None and te_acc[i].item() >= train_cfg.acc_threshold:
                    t_test[i] = step

                # per-seed termination
                if t_train[i] is not None and t_test[i] is not None and step >= T_target[i]:
                    stopped[i] = True
                elif step >= T_target[i]:
                    if t_train[i] is None and T_target[i] < T_cap:
                        T_target[i] = min(T_cap, T_target[i] * 10)
                    else:
                        stopped[i] = True

            if at_log_step_hook is not None:
                at_log_step_hook(step, params, buffers, base)

    # build results
    results: list[TrainResult] = []
    for i in range(N):
        results.append(
            TrainResult(
                phase=classify_phase(t_train[i], t_test[i], train_cfg.grok_ratio),
                t_train=t_train[i],
                t_test=t_test[i],
                history=histories[i],
                final_step=final_step[i],
            )
        )
    return results
