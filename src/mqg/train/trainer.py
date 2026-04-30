"""Single-cell training loop with adaptive T (decision D7).

Implements the protocol from findings.md D7:

    T_min = 1e5
    T_max = 5e6
    Train until t_train (train_acc >= 0.99) is reached or step >= T_min.
    If t_train reached:
        Continue training up to max(50 * t_train, T_min), then stop.
    Else:
        Extend by one order of magnitude (T_min * 10).
        If still not reached -> phase = 'fail'.

Returns a TrainResult with phase classification and full history at log steps.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from ..data import TaskSpec, build_full_dataset
from ..model import MiniQwen, MiniQwenConfig
from .checkpoints import LogStepIterator
from .loss import label_masked_loss_and_acc

PhaseLabel = str  # 'fail' | 'memorize' | 'grok' | 'comprehend'


@dataclass
class TrainConfig:
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.98)
    weight_decay: float = 1.0  # this is lambda
    T_min: int = 100_000
    T_max: int = 5_000_000
    grok_extension_factor: int = 50
    grok_ratio: float = 10.0
    acc_threshold: float = 0.99
    seed: int = 0


@dataclass
class StepLog:
    step: int
    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float


@dataclass
class TrainResult:
    phase: PhaseLabel
    t_train: Optional[int]
    t_test: Optional[int]
    history: list[StepLog] = field(default_factory=list)
    final_step: int = 0


def classify_phase(
    t_train: Optional[int],
    t_test: Optional[int],
    grok_ratio: float = 10.0,
) -> PhaseLabel:
    if grok_ratio <= 0:
        raise ValueError(f"grok_ratio must be positive, got {grok_ratio}")
    if t_train is None:
        return "fail"
    if t_test is None:
        return "memorize"
    gap = t_test - t_train
    if gap > grok_ratio * t_train:
        return "grok"
    return "comprehend"


def evaluate(
    model: MiniQwen, tokens: Tensor, answer_pos: int
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(tokens)
        loss, acc = label_masked_loss_and_acc(logits, tokens, answer_pos)
    return float(loss.item()), float(acc.item())


def validate_train_inputs(
    model_cfg: MiniQwenConfig,
    train_cfg: TrainConfig,
    spec: TaskSpec,
) -> None:
    if model_cfg.vocab_size != spec.vocab_size:
        raise ValueError(
            f"vocab_size mismatch: model_cfg.vocab_size={model_cfg.vocab_size}, "
            f"spec.vocab_size={spec.vocab_size}"
        )
    if model_cfg.max_seq_len < spec.seq_len:
        raise ValueError(
            f"model_cfg.max_seq_len={model_cfg.max_seq_len} is smaller than "
            f"spec.seq_len={spec.seq_len}"
        )
    if train_cfg.T_min < 1:
        raise ValueError(f"T_min must be >= 1, got {train_cfg.T_min}")
    if train_cfg.T_max < train_cfg.T_min:
        raise ValueError(
            f"T_max must be >= T_min, got T_max={train_cfg.T_max}, T_min={train_cfg.T_min}"
        )
    if train_cfg.grok_extension_factor < 1:
        raise ValueError(
            "grok_extension_factor must be >= 1, "
            f"got {train_cfg.grok_extension_factor}"
        )
    if not 0.0 <= train_cfg.acc_threshold <= 1.0:
        raise ValueError(f"acc_threshold must be in [0, 1], got {train_cfg.acc_threshold}")
    if train_cfg.weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {train_cfg.weight_decay}")
    if train_cfg.lr <= 0:
        raise ValueError(f"lr must be positive, got {train_cfg.lr}")
    if train_cfg.grok_ratio <= 0:
        raise ValueError(f"grok_ratio must be positive, got {train_cfg.grok_ratio}")


def train_one_cell(
    model_cfg: MiniQwenConfig,
    train_cfg: TrainConfig,
    spec: TaskSpec,
    train_idx: Tensor,
    test_idx: Tensor,
    device: str = "cpu",
    log_steps: tuple[int, ...] | None = None,
    on_log: Callable[[StepLog], None] | None = None,
) -> TrainResult:
    """Train a single (alpha, lambda) cell with adaptive T."""
    validate_train_inputs(model_cfg, train_cfg, spec)
    torch.manual_seed(train_cfg.seed)

    tokens, _ = build_full_dataset(spec)
    tokens = tokens.to(device)
    train_tokens = tokens[train_idx.to(device)]
    test_tokens = tokens[test_idx.to(device)]

    model = MiniQwen(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        betas=train_cfg.betas,
        weight_decay=train_cfg.weight_decay,
    )

    log_iter = LogStepIterator(log_steps) if log_steps is not None else LogStepIterator()
    history: list[StepLog] = []
    t_train: Optional[int] = None
    t_test: Optional[int] = None

    # Adaptive T bookkeeping
    T_target = train_cfg.T_min
    T_cap = train_cfg.T_max

    step = 0
    while step < T_cap:
        # one full-batch step
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_tokens)
        loss, _ = label_masked_loss_and_acc(logits, train_tokens, spec.answer_pos)
        loss.backward()
        optimizer.step()
        step += 1

        if log_iter.reached(step):
            tr_loss, tr_acc = evaluate(model, train_tokens, spec.answer_pos)
            te_loss, te_acc = evaluate(model, test_tokens, spec.answer_pos)
            log = StepLog(step, tr_loss, tr_acc, te_loss, te_acc)
            history.append(log)
            if on_log is not None:
                on_log(log)

            if t_train is None and tr_acc >= train_cfg.acc_threshold:
                t_train = step
                # Adaptive extension of training budget
                T_target = min(
                    T_cap,
                    max(T_target, train_cfg.grok_extension_factor * t_train),
                )
            if t_test is None and te_acc >= train_cfg.acc_threshold:
                t_test = step

            # Termination conditions
            if t_train is not None and t_test is not None and step >= T_target:
                break
            if step >= T_target:
                if t_train is None and T_target < T_cap:
                    # Extend by one order of magnitude
                    T_target = min(T_cap, T_target * 10)
                else:
                    break

    phase = classify_phase(t_train, t_test, train_cfg.grok_ratio)
    return TrainResult(
        phase=phase,
        t_train=t_train,
        t_test=t_test,
        history=history,
        final_step=step,
    )
