"""Run a full instrumented scan: train + measures trajectory.

For each cell in the grid, train N seeds and at each log step compute
the dual-track progress measures. Output is a long-format Parquet:

    one row per (group, alpha_idx, lambda_idx, seed, step)
    columns include: phase metadata + all measure values

Usage example (CPU smoke):
    python3 scripts/run_scan_instrumented.py \\
        --group A --p 7 --T-min 200 --T-max 200 \\
        --alpha 0.5 0.8 --lambda 0.0 1.0 --n-seeds 2 \\
        --measures-steps 100 200 --skip-hessian
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from mqg.data import TaskSpec
from mqg.model import MiniQwenConfig
from mqg.scan import (
    GridSpec,
    default_grid_spec,
    run_cell_with_measures,
    to_dataframe,
)
from mqg.scan.grid import default_alpha_grid, default_lambda_grid
from mqg.train.trainer import TrainConfig

GROUP_PRESETS = {
    "A": ("S1", True),
    "B": ("S3", True),
    "C": ("S3", False),
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--group", choices=sorted(GROUP_PRESETS), required=True)
    p.add_argument("--p", type=int, default=113)
    p.add_argument("--alpha", type=float, nargs="*", default=None)
    p.add_argument("--lambda", dest="lams", type=float, nargs="*", default=None)
    p.add_argument("--T-min", type=int, default=100_000)
    p.add_argument("--T-max", type=int, default=5_000_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--measures-steps", type=int, nargs="*", default=None,
                   help="Subset of log_steps at which to compute measures. "
                        "Default: all log_steps.")
    p.add_argument("--skip-hessian", action="store_true",
                   help="Skip Hessian top eigenvalue (saves ~20 backward passes per checkpoint).")
    p.add_argument("--hessian-iters", type=int, default=10)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    if args.alpha == []:
        p.error("--alpha requires at least one value when provided")
    if args.lams == []:
        p.error("--lambda requires at least one value when provided")
    if args.measures_steps == []:
        p.error("--measures-steps requires at least one value when provided")
    if args.n_seeds < 1:
        p.error("--n-seeds must be >= 1")
    if args.hessian_iters < 1:
        p.error("--hessian-iters must be >= 1")

    split_strategy, tied = GROUP_PRESETS[args.group]
    spec = TaskSpec(p=args.p)

    if args.alpha is not None or args.lams is not None:
        alphas = tuple(args.alpha) if args.alpha is not None else default_alpha_grid()
        lams = tuple(args.lams) if args.lams is not None else default_lambda_grid()
        grid = GridSpec(alpha_values=alphas, lambda_values=lams)
    else:
        grid = default_grid_spec()

    model_cfg = MiniQwenConfig(vocab_size=spec.vocab_size, tied_embedding=tied)
    train_cfg = TrainConfig(lr=args.lr, T_min=args.T_min, T_max=args.T_max)
    seeds = list(range(args.n_seeds))

    if not args.quiet:
        print(f"[setup] group={args.group} split={split_strategy} tied={tied} "
              f"p={args.p} grid={grid.shape} n_seeds={args.n_seeds}")
        print(f"[setup] alphas={list(grid.alpha_values)}")
        print(f"[setup] lambdas={list(grid.lambda_values)}")
        if args.measures_steps:
            print(f"[setup] measures_steps={args.measures_steps}")
        if args.skip_hessian:
            print("[setup] skip_hessian=True")

    t0 = time.time()
    all_rows: list[dict] = []
    cell_summary: list[dict] = []
    cells = grid.cells()
    log_steps = tuple(args.measures_steps) if args.measures_steps is not None else None
    for k, cell in enumerate(cells, 1):
        t_cell = time.time()
        results, rows = run_cell_with_measures(
            group=args.group,
            split_strategy=split_strategy,
            spec=spec,
            cell=cell,
            seeds=seeds,
            base_train_cfg=train_cfg,
            base_model_cfg=model_cfg,
            split_seed=args.split_seed,
            device=args.device,
            log_steps=log_steps,
            measures_steps=args.measures_steps,
            skip_hessian=args.skip_hessian,
            hessian_iters=args.hessian_iters,
        )
        all_rows.extend(rows)
        for sd, r in zip(seeds, results):
            cell_summary.append({
                "group": args.group, "alpha_idx": cell.alpha_idx,
                "lambda_idx": cell.lambda_idx, "alpha": cell.alpha,
                "lam": cell.lam, "seed": sd, "phase": r.phase,
                "t_train": r.t_train, "t_test": r.t_test,
                "final_step": r.final_step,
            })
        if not args.quiet:
            phases = [r.phase for r in results]
            print(f"[cell {k:>3}/{len(cells)}] a={cell.alpha:.3f} lam={cell.lam:.4g} "
                  f"phases={phases} ({time.time()-t_cell:.1f}s; rows={len(rows)})")

    df = to_dataframe(all_rows)
    out = Path(args.out) if args.out else Path(f"results/scans/{args.group}_p{args.p}_traj.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)

    summary_path = out.with_suffix(".cells.parquet")
    import pandas as pd
    pd.DataFrame(cell_summary).to_parquet(summary_path)

    if not args.quiet:
        print(f"[saved] traj: {out}  ({len(df)} rows, {len(df.columns)} cols)")
        print(f"[saved] cells summary: {summary_path}  ({len(cell_summary)} rows)")
        print(f"[done]  total elapsed {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
