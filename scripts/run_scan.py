"""Run a full 2-stage phase-diagram scan for one experimental group.

Usage example (CPU smoke):
    python3 scripts/run_scan.py --group A --p 7 --T-min 200 --T-max 400 \\
        --alpha 0.3 0.6 --lambda 0.0 1.0

For the actual GPU runs, use the default 9x7 grid.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from mqg.data import TaskSpec
from mqg.model import MiniQwenConfig
from mqg.scan import (
    GridSpec,
    default_grid_spec,
    run_phase1,
    run_phase2,
)
from mqg.train.trainer import TrainConfig

GROUP_PRESETS = {
    # name -> (split_strategy, tied_embedding)
    "A": ("S1", True),
    "B": ("S3", True),
    "C": ("S3", False),
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--group", choices=sorted(GROUP_PRESETS), required=True)
    p.add_argument("--p", type=int, default=113)
    p.add_argument("--alpha", type=float, nargs="*", default=None,
                   help="Override alpha grid (default: 9-pt linear in [0.1,0.9])")
    p.add_argument("--lambda", dest="lams", type=float, nargs="*", default=None,
                   help="Override lambda grid (default: 7-pt log in [1e-2,1e1])")
    p.add_argument("--T-min", type=int, default=100_000)
    p.add_argument("--T-max", type=int, default=5_000_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-seeds", type=int, default=5,
                   help="Total seeds per boundary cell in Phase 2")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--skip-phase2", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    split_strategy, tied = GROUP_PRESETS[args.group]
    spec = TaskSpec(p=args.p)

    if args.alpha is not None or args.lams is not None:
        from mqg.scan.grid import default_alpha_grid, default_lambda_grid
        alphas = tuple(args.alpha) if args.alpha is not None else default_alpha_grid()
        lams = tuple(args.lams) if args.lams is not None else default_lambda_grid()
        grid = GridSpec(alpha_values=alphas, lambda_values=lams)
    else:
        grid = default_grid_spec()

    model_cfg = MiniQwenConfig(vocab_size=spec.vocab_size, tied_embedding=tied)
    train_cfg = TrainConfig(lr=args.lr, T_min=args.T_min, T_max=args.T_max)

    n_alpha, n_lambda = grid.shape
    if not args.quiet:
        print(f"[setup] group={args.group} split={split_strategy} tied={tied} "
              f"p={args.p} grid={n_alpha}x{n_lambda} "
              f"T_min={args.T_min} T_max={args.T_max}")
        print(f"[setup] alphas={list(grid.alpha_values)}")
        print(f"[setup] lambdas={list(grid.lambda_values)}")

    t0 = time.time()
    cell_counter = {"n": 0, "total": n_alpha * n_lambda}

    def _on_cell_done_p1(cell, recs):
        cell_counter["n"] += 1
        if not args.quiet:
            r = recs[0]
            print(f"[p1 {cell_counter['n']:>3}/{cell_counter['total']}] "
                  f"a={cell.alpha:.3f} lam={cell.lam:.4g} -> {r.phase} "
                  f"(t_train={r.t_train}, t_test={r.t_test})")

    p1_records = run_phase1(
        group=args.group,
        split_strategy=split_strategy,
        spec=spec,
        grid=grid,
        base_train_cfg=train_cfg,
        base_model_cfg=model_cfg,
        split_seed=args.split_seed,
        device=args.device,
        on_cell_done=_on_cell_done_p1,
    )
    p1_dt = time.time() - t0
    if not args.quiet:
        print(f"[phase1] done in {p1_dt:.1f}s, {len(p1_records)} records")

    p2_records: list = []
    if not args.skip_phase2 and args.n_seeds > 1:
        t1 = time.time()

        def _on_cell_done_p2(cell, recs):
            if not args.quiet:
                phases = [r.phase for r in recs]
                print(f"[p2 boundary] a={cell.alpha:.3f} lam={cell.lam:.4g} -> {phases}")

        p2_records = run_phase2(
            group=args.group,
            split_strategy=split_strategy,
            spec=spec,
            grid=grid,
            phase1_records=p1_records,
            n_seeds=args.n_seeds,
            base_train_cfg=train_cfg,
            base_model_cfg=model_cfg,
            split_seed=args.split_seed,
            device=args.device,
            on_cell_done=_on_cell_done_p2,
        )
        p2_dt = time.time() - t1
        if not args.quiet:
            print(f"[phase2] done in {p2_dt:.1f}s, {len(p2_records)} records "
                  f"(across {len({(r.alpha_idx, r.lambda_idx) for r in p2_records})} boundary cells)")

    all_records = p1_records + p2_records
    out = (Path(args.out) if args.out is not None
           else Path(f"results/scans/{args.group}_p{args.p}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(r) for r in all_records], indent=2))
    if not args.quiet:
        print(f"[saved] {out}  ({len(all_records)} total records, "
              f"elapsed {time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
