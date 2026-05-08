"""Train one grid-search cell and save a Mini-Qwen checkpoint.

This is the checkpoint-producing counterpart of ``scripts/train_one.py``. It
uses the same adaptive-T protocol as the project trainer, but keeps the trained
model object so the final weights can be saved for causal mechanism probes.

Example:

    python scripts/train_checkpoint.py \
        --group B --p 113 --alpha 0.4 --lambda 0.1 \
        --T-min 100000 --T-max 100000 \
        --device cuda --matmul-precision high
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch

from mqg.data import TaskSpec, build_full_dataset, make_split
from mqg.model import MiniQwen, MiniQwenConfig
from mqg.perf import configure_matmul_precision
from mqg.train.checkpoints import DEFAULT_LOG_STEPS, LogStepIterator
from mqg.train.loss import label_masked_loss_and_acc
from mqg.train.trainer import StepLog, TrainConfig, TrainResult, classify_phase, validate_train_inputs


GROUP_PRESETS = {
    "A": ("S1", True),
    "B": ("S3", True),
    "C": ("S3", False),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one cell and save a checkpoint.")
    parser.add_argument("--group", choices=sorted(GROUP_PRESETS), default=None)
    parser.add_argument("--p", type=int, default=113)
    parser.add_argument("--op", choices=["add", "mul"], default="add")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--lambda", dest="lam", type=float, required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--split", choices=["S1", "S3"], default=None)
    parser.add_argument("--tied", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--T-min", type=int, default=100_000)
    parser.add_argument("--T-max", type=int, default=5_000_000)
    parser.add_argument("--grok-extension-factor", type=int, default=50)
    parser.add_argument("--grok-ratio", type=float, default=10.0)
    parser.add_argument("--acc-threshold", type=float, default=0.99)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--matmul-precision", choices=["highest", "high", "medium"], default=None)
    parser.add_argument("--log-steps", type=int, nargs="*", default=list(DEFAULT_LOG_STEPS))
    parser.add_argument("--progress-interval-steps", type=int, default=10_000)
    parser.add_argument("--out-dir", type=Path, default=Path("results/checkpoints"))
    parser.add_argument("--tag", default=None)
    parser.add_argument("--include-optimizer", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def evaluate(model: MiniQwen, tokens: torch.Tensor, answer_pos: int) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(tokens)
        loss, acc = label_masked_loss_and_acc(logits, tokens, answer_pos)
    return float(loss.item()), float(acc.item())


def train_one_checkpoint(
    model_cfg: MiniQwenConfig,
    train_cfg: TrainConfig,
    spec: TaskSpec,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    *,
    device: str,
    log_steps: tuple[int, ...],
    progress_interval_steps: int | None,
    on_log,
    on_progress,
) -> tuple[MiniQwen, torch.optim.Optimizer, TrainResult]:
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

    log_iter = LogStepIterator(log_steps)
    history: list[StepLog] = []
    t_train: Optional[int] = None
    t_test: Optional[int] = None
    T_target = train_cfg.T_min
    T_cap = train_cfg.T_max
    extended_without_train = False
    step = 0

    while step < T_cap:
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
                on_log(log, T_target)

            if t_train is None and tr_acc >= train_cfg.acc_threshold:
                t_train = step
                T_target = min(
                    T_cap,
                    max(T_target, train_cfg.grok_extension_factor * t_train),
                )
            if t_test is None and te_acc >= train_cfg.acc_threshold:
                t_test = step

            if t_train is not None and t_test is not None and step >= T_target:
                break
            if step >= T_target:
                if t_train is None and not extended_without_train and T_target < T_cap:
                    T_target = min(T_cap, T_target * 10)
                    extended_without_train = True
                else:
                    break
        elif (
            progress_interval_steps is not None
            and progress_interval_steps > 0
            and step % progress_interval_steps == 0
            and on_progress is not None
        ):
            on_progress(step, T_target)

    phase = classify_phase(t_train, t_test, train_cfg.grok_ratio)
    result = TrainResult(phase=phase, t_train=t_train, t_test=t_test, history=history, final_step=step)
    return model, optimizer, result


def checkpoint_name(args: argparse.Namespace, split: str, tied: bool) -> str:
    if args.tag:
        return args.tag
    group = args.group or f"split{split}"
    alpha = f"{args.alpha:g}".replace(".", "p")
    lam = f"{args.lam:g}".replace(".", "p")
    return f"{group}_p{args.p}_a{alpha}_lam{lam}_seed{args.seed}_tied{tied}"


def main() -> int:
    args = parse_args()
    configure_matmul_precision(args.matmul_precision)

    if args.group is not None:
        preset_split, preset_tied = GROUP_PRESETS[args.group]
        split = args.split or preset_split
        tied = preset_tied if args.tied is None else args.tied
    else:
        if args.split is None or args.tied is None:
            raise ValueError("Without --group, pass both --split and --tied/--no-tied.")
        split = args.split
        tied = args.tied
    if args.log_steps == []:
        raise ValueError("--log-steps requires at least one value when provided.")

    spec = TaskSpec(p=args.p, op=args.op)
    model_cfg = MiniQwenConfig(vocab_size=spec.vocab_size, tied_embedding=tied)
    train_cfg = TrainConfig(
        lr=args.lr,
        weight_decay=args.lam,
        T_min=args.T_min,
        T_max=args.T_max,
        grok_extension_factor=args.grok_extension_factor,
        grok_ratio=args.grok_ratio,
        acc_threshold=args.acc_threshold,
        seed=args.seed,
    )
    train_idx, test_idx = make_split(split, spec, args.alpha, args.split_seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = checkpoint_name(args, split, tied)
    ckpt_path = args.out_dir / f"{tag}.pt"
    json_path = args.out_dir / f"{tag}.json"

    start_time = time.time()

    def log(message: str) -> None:
        if not args.quiet:
            print(message, flush=True)

    log(
        f"[setup] group={args.group} split={split} tied={tied} p={args.p} "
        f"alpha={args.alpha:g} lambda={args.lam:g} seed={args.seed} "
        f"train={len(train_idx)} test={len(test_idx)} device={args.device} "
        f"matmul_precision={args.matmul_precision}"
    )

    def on_log(step_log: StepLog, t_target: int) -> None:
        log(
            f"[log] step={step_log.step:,} train_loss={step_log.train_loss:.4f} "
            f"train_acc={step_log.train_acc:.4f} test_loss={step_log.test_loss:.4f} "
            f"test_acc={step_log.test_acc:.4f} T_target={t_target:,}"
        )

    def on_progress(step: int, t_target: int) -> None:
        elapsed = time.time() - start_time
        rate = step / elapsed if elapsed > 0 else 0.0
        log(f"[progress] step={step:,}/{args.T_max:,} T_target={t_target:,} rate={rate:.1f} step/s")

    model, optimizer, result = train_one_checkpoint(
        model_cfg,
        train_cfg,
        spec,
        train_idx,
        test_idx,
        device=args.device,
        log_steps=tuple(sorted(set(args.log_steps))),
        progress_interval_steps=args.progress_interval_steps,
        on_log=on_log,
        on_progress=on_progress,
    )
    wall_seconds = time.time() - start_time

    payload = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_cfg": asdict(model_cfg),
        "train_cfg": asdict(train_cfg),
        "task": asdict(spec),
        "group": args.group,
        "split": split,
        "tied_embedding": tied,
        "alpha": args.alpha,
        "lam": args.lam,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "result": {
            "phase": result.phase,
            "t_train": result.t_train,
            "t_test": result.t_test,
            "final_step": result.final_step,
            "history": [asdict(log_row) for log_row in result.history],
        },
        "wall_seconds": wall_seconds,
    }
    if args.include_optimizer:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, ckpt_path)
    json_path.write_text(json.dumps({k: v for k, v in payload.items() if k != "model_state_dict"}, indent=2, default=str))

    log(
        f"[result] phase={result.phase} t_train={result.t_train} t_test={result.t_test} "
        f"final_step={result.final_step:,} wall={wall_seconds:.1f}s"
    )
    log(f"[saved] {ckpt_path}")
    log(f"[saved] {json_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)