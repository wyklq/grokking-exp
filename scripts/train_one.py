"""Train a single (alpha, lambda) cell.

Examples:
    python scripts/train_one.py --p 23 --alpha 0.5 --lambda 1.0 --T-min 5000

For the GPU sanity-check run that should reproduce grokking:
    python scripts/train_one.py --p 113 --alpha 0.3 --lambda 1.0 --T-min 100000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from mqg.data import TaskSpec, make_split
from mqg.model import MiniQwenConfig
from mqg.perf import configure_matmul_precision
from mqg.train import TrainConfig, train_one_cell


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one Mini-Qwen cell")
    p.add_argument("--p", type=int, default=113, help="modulus p (prime)")
    p.add_argument("--op", choices=["add", "mul"], default="add")
    p.add_argument("--alpha", type=float, default=0.3, help="train fraction")
    p.add_argument("--lambda", dest="lam", type=float, default=1.0, help="weight decay")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--split", choices=["S1", "S3"], default="S1")
    p.add_argument("--tied", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--T-min", type=int, default=100_000)
    p.add_argument("--T-max", type=int, default=5_000_000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--matmul-precision", choices=["highest", "high", "medium"], default=None,
                   help="Optional torch float32 matmul precision. Use 'high' on NVIDIA GPUs "
                        "to allow TF32 acceleration.")
    p.add_argument("--out", type=Path, default=Path("results/single_cell"))
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    configure_matmul_precision(args.matmul_precision)

    spec = TaskSpec(p=args.p, op=args.op)
    model_cfg = MiniQwenConfig(
        vocab_size=spec.vocab_size,
        tied_embedding=args.tied,
    )
    train_cfg = TrainConfig(
        lr=args.lr,
        weight_decay=args.lam,
        T_min=args.T_min,
        T_max=args.T_max,
        seed=args.seed,
    )

    train_idx, test_idx = make_split(args.split, spec, args.alpha, args.split_seed)
    print(
        f"[setup] p={args.p} alpha={args.alpha} lambda={args.lam} split={args.split} "
        f"tied={args.tied} | train={len(train_idx)} test={len(test_idx)} "
        f"vocab={spec.vocab_size} matmul_precision={args.matmul_precision}",
        file=sys.stderr,
    )

    def on_log(log):
        if args.quiet:
            return
        print(
            f"  step={log.step:>9d}  train_loss={log.train_loss:.4f} "
            f"train_acc={log.train_acc:.3f}  "
            f"test_loss={log.test_loss:.4f}  test_acc={log.test_acc:.3f}",
            file=sys.stderr,
        )

    t0 = time.time()
    result = train_one_cell(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        spec=spec,
        train_idx=train_idx,
        test_idx=test_idx,
        device=args.device,
        on_log=on_log,
    )
    dt = time.time() - t0

    print(
        f"[result] phase={result.phase} t_train={result.t_train} "
        f"t_test={result.t_test} final_step={result.final_step} "
        f"({dt:.1f}s)",
        file=sys.stderr,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"p{args.p}_a{args.alpha}_l{args.lam}_split{args.split}_tied{args.tied}_seed{args.seed}"
    out_path = args.out / f"{tag}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "args": vars(args) | {"out": str(args.out)},
                "phase": result.phase,
                "t_train": result.t_train,
                "t_test": result.t_test,
                "final_step": result.final_step,
                "wall_seconds": dt,
                "history": [asdict(h) for h in result.history],
            },
            f,
            indent=2,
            default=str,
        )
    print(f"[saved] {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
