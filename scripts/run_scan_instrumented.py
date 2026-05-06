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
from mqg.perf import configure_matmul_precision
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
    p.add_argument("--matmul-precision", choices=["highest", "high", "medium"], default=None,
                   help="Optional torch float32 matmul precision. Use 'high' on NVIDIA GPUs "
                        "to allow TF32 acceleration.")
    p.add_argument("--progress-interval-steps", type=int, default=100_000,
                   help="Emit a lightweight per-cell heartbeat every N training steps. "
                        "Use 0 to disable.")
    p.add_argument("--no-save-partial", action="store_true",
                   help="Disable writing *.partial.parquet progress files after each cell.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from existing partial/final Parquet files by skipping completed cells.")
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
    if args.progress_interval_steps < 0:
        p.error("--progress-interval-steps must be >= 0")
    configure_matmul_precision(args.matmul_precision)

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
    progress_interval_steps = args.progress_interval_steps or None

    def log(message: str) -> None:
        if not args.quiet:
            print(message, flush=True)

    if not args.quiet:
        log(f"[setup] group={args.group} split={split_strategy} tied={tied} "
            f"p={args.p} grid={grid.shape} n_seeds={args.n_seeds} "
            f"matmul_precision={args.matmul_precision}")
        log(f"[setup] alphas={list(grid.alpha_values)}")
        log(f"[setup] lambdas={list(grid.lambda_values)}")
        if args.measures_steps:
            log(f"[setup] measures_steps={args.measures_steps}")
        if args.skip_hessian:
            log("[setup] skip_hessian=True")
        if progress_interval_steps is not None:
            log(f"[setup] progress_interval_steps={progress_interval_steps}")

    t0 = time.time()
    all_rows: list[dict] = []
    cell_summary: list[dict] = []
    cells = grid.cells()
    log_steps = tuple(args.measures_steps) if args.measures_steps is not None else None
    out = Path(args.out) if args.out else Path(f"results/scans/{args.group}_p{args.p}_traj.parquet")
    summary_path = out.with_suffix(".cells.parquet")
    partial_path = out.with_suffix(".partial.parquet")
    partial_summary_path = out.with_suffix(".partial.cells.parquet")

    completed_seeds_by_cell: dict[tuple[int, int], set[int]] = {}
    expected_seeds = set(seeds)
    requested_cells = {(cell.alpha_idx, cell.lambda_idx) for cell in cells}

    def dedupe_records(rows: list[dict], key_fields: tuple[str, ...]) -> list[dict]:
        deduped: dict[tuple, dict] = {}
        for row in rows:
            deduped[tuple(row[field] for field in key_fields)] = row
        return list(deduped.values())

    def in_requested_scope(row: dict) -> bool:
        cell_key = (int(row["alpha_idx"]), int(row["lambda_idx"]))
        return cell_key in requested_cells and int(row["seed"]) in expected_seeds

    if args.resume:
        import pandas as pd

        resume_traj_path = partial_path if partial_path.exists() else out
        resume_cells_path = partial_summary_path if partial_summary_path.exists() else summary_path
        if resume_cells_path.exists():
            cells_df = pd.read_parquet(resume_cells_path)
            cell_summary = dedupe_records(
                [row for row in cells_df.to_dict("records") if in_requested_scope(row)],
                ("alpha_idx", "lambda_idx", "seed"),
            )
            if resume_traj_path.exists():
                all_rows = dedupe_records(
                    [
                        row for row in pd.read_parquet(resume_traj_path).to_dict("records")
                        if in_requested_scope(row)
                    ],
                    ("alpha_idx", "lambda_idx", "seed", "step"),
                )

            summary_seeds_by_cell: dict[tuple[int, int], set[int]] = {}
            for row in cell_summary:
                key = (int(row["alpha_idx"]), int(row["lambda_idx"]))
                summary_seeds_by_cell.setdefault(key, set()).add(int(row["seed"]))

            traj_seeds_by_cell: dict[tuple[int, int], set[int]] = {}
            for row in all_rows:
                key = (int(row["alpha_idx"]), int(row["lambda_idx"]))
                traj_seeds_by_cell.setdefault(key, set()).add(int(row["seed"]))

            for key, summary_seeds in summary_seeds_by_cell.items():
                durable_seeds = summary_seeds & traj_seeds_by_cell.get(key, set())
                if durable_seeds:
                    completed_seeds_by_cell[key] = durable_seeds & expected_seeds
            completed_cells = {
                key for key, done_seeds in completed_seeds_by_cell.items()
                if done_seeds >= expected_seeds
            }
            log(f"[resume] loaded {len(all_rows)} trajectory rows and {len(cell_summary)} "
                f"cell rows; skipping {len(completed_cells)} completed cells")
        else:
            log("[resume] no partial/final cell summary found; starting from scratch")

    def write_outputs(traj_path: Path, cells_path: Path) -> None:
        nonlocal all_rows, cell_summary
        all_rows = dedupe_records(
            [row for row in all_rows if in_requested_scope(row)],
            ("alpha_idx", "lambda_idx", "seed", "step"),
        )
        cell_summary = dedupe_records(
            [row for row in cell_summary if in_requested_scope(row)],
            ("alpha_idx", "lambda_idx", "seed"),
        )
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        traj_tmp = traj_path.with_name(f"{traj_path.stem}.tmp{traj_path.suffix}")
        cells_tmp = cells_path.with_name(f"{cells_path.stem}.tmp{cells_path.suffix}")
        to_dataframe(all_rows).to_parquet(traj_tmp)
        import pandas as pd
        pd.DataFrame(cell_summary).to_parquet(cells_tmp)
        traj_tmp.replace(traj_path)
        cells_tmp.replace(cells_path)

    for k, cell in enumerate(cells, 1):
        cell_key = (cell.alpha_idx, cell.lambda_idx)
        completed_seeds = completed_seeds_by_cell.get(cell_key, set())
        cell_seeds = [seed for seed in seeds if seed not in completed_seeds]
        if not cell_seeds:
            log(f"[cell {k:>3}/{len(cells)} skip] a={cell.alpha:.3f} "
                f"lam={cell.lam:.4g} already complete")
            continue
        if completed_seeds:
            log(f"[cell {k:>3}/{len(cells)} resume] a={cell.alpha:.3f} "
                f"lam={cell.lam:.4g} running missing seeds={cell_seeds}")
        t_cell = time.time()

        def on_progress(step: int) -> None:
            elapsed = time.time() - t_cell
            rate = step / elapsed if elapsed > 0 else 0.0
            log(f"[cell {k:>3}/{len(cells)} progress] a={cell.alpha:.3f} "
                f"lam={cell.lam:.4g} step={step:,}/{args.T_max:,} "
                f"elapsed={elapsed:.1f}s rate={rate:.1f} step/s")

        def on_train_log(step, step_logs, t_train, t_test, t_target, stopped) -> None:
            parts = []
            for idx, (sd, step_log) in enumerate(zip(cell_seeds, step_logs)):
                if step_log is None:
                    continue
                parts.append(
                    f"s{sd}: train={step_log.train_acc:.3f} test={step_log.test_acc:.3f} "
                    f"t_train={t_train[idx]} t_test={t_test[idx]} "
                    f"T_target={t_target[idx]} stopped={stopped[idx]}"
                )
            if parts:
                log(f"[cell {k:>3}/{len(cells)} log] a={cell.alpha:.3f} "
                    f"lam={cell.lam:.4g} step={step:,} | " + " | ".join(parts))

        def on_measure_step(step: int, new_rows: int, total_rows: int) -> None:
            log(f"[cell {k:>3}/{len(cells)} measures] a={cell.alpha:.3f} "
                f"lam={cell.lam:.4g} step={step:,} +{new_rows} rows "
                f"(cell_rows={total_rows})")

        results, rows = run_cell_with_measures(
            group=args.group,
            split_strategy=split_strategy,
            spec=spec,
            cell=cell,
            seeds=cell_seeds,
            base_train_cfg=train_cfg,
            base_model_cfg=model_cfg,
            split_seed=args.split_seed,
            device=args.device,
            log_steps=log_steps,
            measures_steps=args.measures_steps,
            skip_hessian=args.skip_hessian,
            hessian_iters=args.hessian_iters,
            on_train_log=on_train_log if not args.quiet else None,
            progress_interval_steps=progress_interval_steps,
            on_progress=on_progress if not args.quiet else None,
            on_measure_step=on_measure_step if not args.quiet else None,
        )
        all_rows.extend(rows)
        for sd, r in zip(cell_seeds, results):
            cell_summary.append({
                "group": args.group, "alpha_idx": cell.alpha_idx,
                "lambda_idx": cell.lambda_idx, "alpha": cell.alpha,
                "lam": cell.lam, "seed": sd, "phase": r.phase,
                "t_train": r.t_train, "t_test": r.t_test,
                "final_step": r.final_step,
            })
        if not args.quiet:
            phases = [r.phase for r in results]
            log(f"[cell {k:>3}/{len(cells)}] a={cell.alpha:.3f} lam={cell.lam:.4g} "
                f"phases={phases} ({time.time()-t_cell:.1f}s; rows={len(rows)})")
        if not args.no_save_partial:
            write_outputs(partial_path, partial_summary_path)
            log(f"[partial] saved {partial_path} ({len(all_rows)} rows), "
                f"{partial_summary_path} ({len(cell_summary)} cells)")

    write_outputs(out, summary_path)

    if not args.quiet:
        df = to_dataframe(all_rows)
        log(f"[saved] traj: {out}  ({len(df)} rows, {len(df.columns)} cols)")
        log(f"[saved] cells summary: {summary_path}  ({len(cell_summary)} rows)")
        log(f"[done]  total elapsed {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
