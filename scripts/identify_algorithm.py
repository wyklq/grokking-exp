"""Identify Fourier-like algorithms from Mini-Qwen scan results.

This script analyzes the artifacts that already exist in ``results/scans``:
instrumented trajectory Parquet files plus ``*.cells.parquet`` phase summaries.

It produces trajectory-level evidence for whether comprehension-region models
look Fourier-like. It does *not* claim causal proof unless saved checkpoints are
available for ablation; see the generated report for the checkpoint protocol.

Typical use:

    python scripts/identify_algorithm.py \
        --results-dir results/scans \
        --out-dir results/algorithm_identification \
        --p 113
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


KEYS = ["group", "alpha_idx", "lambda_idx", "alpha", "lam", "seed"]
TRAJ_KEY = [*KEYS, "step"]
PHASE_ORDER = ["fail", "memorize", "grok", "comprehend"]
PHASE_COLORS = {
    "fail": "#8c8c8c",
    "memorize": "#d95f02",
    "grok": "#7570b3",
    "comprehend": "#1b9e77",
    "unknown": "#e6e6e6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Mini-Qwen scan outputs for Fourier-like algorithm evidence."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/scans"),
        help="Directory containing scan trajectory Parquet and *.cells.parquet files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/algorithm_identification"),
        help="Directory for analysis tables, report, and figures.",
    )
    parser.add_argument(
        "--p",
        type=int,
        default=113,
        help="Modulus p. Used for interpreting dominant Fourier frequencies.",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Optional subset of groups to analyze, e.g. --groups B C.",
    )
    parser.add_argument(
        "--include-partial-duplicates",
        action="store_true",
        help=(
            "By default, if both X.parquet and X.partial.parquet exist, the final file "
            "wins. Set this to include both before key-level deduplication."
        ),
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Only write tables and report; skip PNG generation.",
    )
    return parser.parse_args()


def is_partial(path: Path) -> bool:
    return ".partial" in path.name


def final_partner(path: Path) -> Path:
    return path.with_name(path.name.replace(".partial", ""))


def preferred_parquet_files(
    results_dir: Path,
    *,
    cells: bool,
    include_partial_duplicates: bool,
) -> list[Path]:
    pattern = "*.cells.parquet" if cells else "*.parquet"
    paths = sorted(results_dir.glob(pattern))
    if not cells:
        paths = [p for p in paths if not p.name.endswith(".cells.parquet")]
    if include_partial_duplicates:
        return paths

    preferred: list[Path] = []
    for path in paths:
        if is_partial(path) and final_partner(path).exists():
            continue
        preferred.append(path)
    return preferred


def load_parquet_set(paths: Iterable[Path], *, kind: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_parquet(path)
        if df.empty:
            continue
        df = df.copy()
        df["source_file"] = str(path)
        df["source_kind"] = kind
        df["source_is_partial"] = is_partial(path)
        df["source_priority"] = np.where(df["source_is_partial"], 0, 1)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No non-empty {kind} Parquet files found.")
    return pd.concat(frames, ignore_index=True, sort=False)


def dedupe_by_priority(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    missing = sorted(set(key_cols) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required key columns: {missing}")
    order_cols = [*key_cols, "source_priority", "source_file"]
    return (
        df.sort_values(order_cols)
        .drop_duplicates(key_cols, keep="last")
        .reset_index(drop=True)
    )


def filter_groups(df: pd.DataFrame, groups: list[str] | None) -> pd.DataFrame:
    if not groups:
        return df
    return df[df["group"].isin(groups)].reset_index(drop=True)


def canonical_frequency(freq: float | int | None, p: int) -> float:
    if freq is None or pd.isna(freq):
        return float("nan")
    k = int(freq) % p
    return float(min(k, (-k) % p))


def clipped01(x: pd.Series | float) -> pd.Series | float:
    return np.clip(x, 0.0, 1.0)


def dominant_frequency_string(row: pd.Series) -> str:
    cols = sorted(c for c in row.index if c.startswith("dom_freq_"))
    values: list[str] = []
    for col in cols:
        val = row[col]
        if pd.notna(val):
            values.append(str(int(val)))
    return ",".join(values)


def add_algorithm_features(traj: pd.DataFrame, cells: pd.DataFrame, p: int) -> pd.DataFrame:
    metric_cols = [
        "fourier_sparsity",
        "circularity_top_freq",
        "weight_norm_total",
        "embedding_stable_rank",
        "embedding_effective_rank",
        "lm_head_effective_rank",
        "hessian_top_eig",
    ]
    present_metrics = [c for c in metric_cols if c in traj.columns]
    freq_cols = sorted(c for c in traj.columns if c.startswith("dom_freq_"))

    first_idx = traj.groupby(KEYS, dropna=False)["step"].idxmin()
    final_idx = traj.groupby(KEYS, dropna=False)["step"].idxmax()

    first = traj.loc[first_idx, [*KEYS, "step", *present_metrics]].copy()
    first = first.rename(columns={c: f"first_{c}" for c in ["step", *present_metrics]})

    final = traj.loc[
        final_idx,
        [*KEYS, "step", *present_metrics, *freq_cols, "source_file"],
    ].copy()
    final = final.rename(columns={"step": "final_step_measured"})

    features = final.merge(first, on=KEYS, how="left")

    cell_cols = [*KEYS, "phase", "t_train", "t_test", "final_step"]
    available_cell_cols = [c for c in cell_cols if c in cells.columns]
    features = features.merge(cells[available_cell_cols], on=KEYS, how="left")
    if "phase" not in features.columns:
        features["phase"] = "unknown"
    features["phase"] = features["phase"].fillna("unknown")

    required = {"fourier_sparsity", "circularity_top_freq"}
    if not required.issubset(features.columns):
        raise ValueError("Trajectory files must contain fourier_sparsity and circularity_top_freq.")

    features["delta_fourier_sparsity"] = (
        features["fourier_sparsity"] - features["first_fourier_sparsity"]
    )
    features["delta_circularity_drop"] = (
        features["first_circularity_top_freq"] - features["circularity_top_freq"]
    )
    features["dominant_freqs"] = features.apply(dominant_frequency_string, axis=1)
    if "dom_freq_0" in features.columns:
        features["top_canonical_freq"] = features["dom_freq_0"].map(
            lambda x: canonical_frequency(x, p)
        )
    else:
        features["top_canonical_freq"] = np.nan

    # Absolute, interpretable Fourier evidence score. This is not a learned
    # classifier; it encodes the mechanistic hypothesis directly.
    sparsity_component = clipped01((features["fourier_sparsity"] - 0.30) / 0.55)
    circularity_component = clipped01((0.55 - features["circularity_top_freq"]) / 0.45)
    trend_component = clipped01((features["delta_fourier_sparsity"] + 0.05) / 0.65)
    features["fourier_score"] = 100.0 * (
        0.55 * sparsity_component + 0.35 * circularity_component + 0.10 * trend_component
    )
    features["fourier_score"] = features["fourier_score"].round(2)

    # Group-local percentiles help compare tied vs untied settings whose
    # absolute embedding spectra sit at different baselines.
    features["fourier_sparsity_pct_in_group"] = features.groupby("group")[
        "fourier_sparsity"
    ].rank(pct=True)
    features["circularity_good_pct_in_group"] = 1.0 - features.groupby("group")[
        "circularity_top_freq"
    ].rank(pct=True)

    features["fourier_evidence"] = pd.cut(
        features["fourier_score"],
        bins=[-0.1, 20.0, 35.0, 70.0, 100.1],
        labels=["none", "weak", "moderate", "strong"],
    ).astype(str)

    def mechanism_call(row: pd.Series) -> str:
        phase = row.get("phase", "unknown")
        score = float(row.get("fourier_score", float("nan")))
        if phase == "fail":
            return "no_generalization_or_fail"
        if phase == "memorize":
            return "memorization_or_lookup"
        if phase in {"grok", "comprehend"}:
            if score >= 70.0:
                return "fourier_like_strong"
            if score >= 35.0:
                return "fourier_like_moderate"
            return "generalizes_but_fourier_weak"
        return "unknown"

    features["mechanism_call"] = features.apply(mechanism_call, axis=1)
    return features.sort_values(KEYS).reset_index(drop=True)


def summarize_phase(cells: pd.DataFrame) -> pd.DataFrame:
    summary = (
        cells.groupby(["group", "phase"], dropna=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    summary["phase"] = pd.Categorical(summary["phase"], PHASE_ORDER, ordered=True)
    return summary.sort_values(["group", "phase"]).reset_index(drop=True)


def summarize_metrics(features: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "fourier_score",
        "fourier_sparsity",
        "circularity_top_freq",
        "delta_fourier_sparsity",
        "delta_circularity_drop",
        "embedding_stable_rank",
        "embedding_effective_rank",
        "lm_head_effective_rank",
        "weight_norm_total",
    ]
    metrics = [c for c in metrics if c in features.columns]
    rows: list[dict[str, object]] = []
    for (group, phase), sub in features.groupby(["group", "phase"], dropna=False):
        row: dict[str, object] = {"group": group, "phase": phase, "n": len(sub)}
        for metric in metrics:
            row[f"{metric}_median"] = float(sub[metric].median())
            row[f"{metric}_mean"] = float(sub[metric].mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["phase"] = pd.Categorical(out["phase"], PHASE_ORDER, ordered=True)
        out = out.sort_values(["group", "phase"]).reset_index(drop=True)
    return out


def summarize_top_frequencies(features: pd.DataFrame) -> pd.DataFrame:
    sub = features[features["phase"].isin(["comprehend", "grok"])].copy()
    if sub.empty:
        return pd.DataFrame(columns=["group", "phase", "top_canonical_freq", "n"])
    return (
        sub.groupby(["group", "phase", "top_canonical_freq"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["group", "phase", "n"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def save_tables(
    out_dir: Path,
    features: pd.DataFrame,
    phase_summary: pd.DataFrame,
    metric_summary: pd.DataFrame,
    top_freqs: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_dir / "cell_algorithm_scores.csv", index=False)
    features.to_parquet(out_dir / "cell_algorithm_scores.parquet", index=False)
    features[features["phase"] == "comprehend"].to_csv(
        out_dir / "comprehend_cells.csv", index=False
    )
    phase_summary.to_csv(out_dir / "phase_summary.csv", index=False)
    metric_summary.to_csv(out_dir / "metric_summary_by_phase.csv", index=False)
    top_freqs.to_csv(out_dir / "top_frequency_counts.csv", index=False)


def phase_code(phase: str) -> int:
    try:
        return PHASE_ORDER.index(phase)
    except ValueError:
        return len(PHASE_ORDER)


def plot_phase_grids(cells: pd.DataFrame, out_dir: Path) -> None:
    groups = sorted(cells["group"].dropna().unique())
    if not groups:
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(5.2 * len(groups), 4.4), squeeze=False)
    cmap = ListedColormap([PHASE_COLORS[p] for p in PHASE_ORDER] + [PHASE_COLORS["unknown"]])
    for ax, group in zip(axes.flat, groups):
        sub = cells[cells["group"] == group].copy()
        modal = (
            sub.groupby(["alpha", "lam"])["phase"]
            .agg(lambda s: s.value_counts().idxmax())
            .reset_index()
        )
        alphas = sorted(modal["alpha"].unique())
        lams = sorted(modal["lam"].unique())
        grid = np.full((len(lams), len(alphas)), np.nan)
        for _, row in modal.iterrows():
            grid[lams.index(row["lam"]), alphas.index(row["alpha"])] = phase_code(str(row["phase"]))
        ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=len(PHASE_ORDER))
        ax.set_title(f"Group {group}: phase")
        ax.set_xlabel("alpha")
        ax.set_ylabel("lambda")
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f"{a:g}" for a in alphas], rotation=45, ha="right")
        ax.set_yticks(range(len(lams)))
        ax.set_yticklabels([f"{lam:g}" for lam in lams])
        for i in range(len(lams)):
            for j in range(len(alphas)):
                val = grid[i, j]
                if not np.isnan(val):
                    ax.text(j, i, PHASE_ORDER[int(val)][0].upper(), ha="center", va="center", fontsize=8)
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=PHASE_COLORS[p], label=p)
        for p in PHASE_ORDER
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_dir / "phase_grids.png", dpi=180)
    plt.close(fig)


def plot_fourier_scatter(features: pd.DataFrame, out_dir: Path) -> None:
    groups = sorted(features["group"].dropna().unique())
    if not groups:
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(5.2 * len(groups), 4.4), squeeze=False)
    for ax, group in zip(axes.flat, groups):
        sub = features[features["group"] == group]
        for phase in PHASE_ORDER:
            part = sub[sub["phase"] == phase]
            if part.empty:
                continue
            ax.scatter(
                part["fourier_sparsity"],
                part["circularity_top_freq"],
                s=32 + 0.8 * part["fourier_score"],
                alpha=0.78,
                color=PHASE_COLORS[phase],
                label=phase,
                edgecolor="white",
                linewidth=0.5,
            )
        ax.axvline(0.40, color="#444444", linestyle="--", linewidth=0.8)
        ax.axvline(0.70, color="#111111", linestyle=":", linewidth=0.9)
        ax.axhline(0.35, color="#444444", linestyle="--", linewidth=0.8)
        ax.axhline(0.25, color="#111111", linestyle=":", linewidth=0.9)
        ax.set_title(f"Group {group}: final Fourier evidence")
        ax.set_xlabel("Fourier sparsity (higher = fewer modes)")
        ax.set_ylabel("Circularity CV (lower = cleaner circles)")
        ax.grid(alpha=0.18)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=PHASE_COLORS[p], label=p)
        for p in PHASE_ORDER
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out_dir / "fourier_evidence_scatter.png", dpi=180)
    plt.close(fig)


def plot_score_heatmaps(features: pd.DataFrame, out_dir: Path) -> None:
    groups = sorted(features["group"].dropna().unique())
    if not groups:
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(5.1 * len(groups), 4.4), squeeze=False)
    last_im = None
    for ax, group in zip(axes.flat, groups):
        sub = features[features["group"] == group]
        piv = sub.pivot_table(
            index="lam",
            columns="alpha",
            values="fourier_score",
            aggfunc="mean",
            observed=False,
        ).sort_index().sort_index(axis=1)
        last_im = ax.imshow(
            piv.values,
            origin="lower",
            aspect="auto",
            vmin=0,
            vmax=100,
            cmap="viridis",
        )
        ax.set_title(f"Group {group}: Fourier score")
        ax.set_xlabel("alpha")
        ax.set_ylabel("lambda")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{a:g}" for a in piv.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{lam:g}" for lam in piv.index])
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                val = piv.values[i, j]
                if not np.isnan(val):
                    color = "white" if val < 55 else "black"
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8, color=color)
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.78)
        cbar.set_label("Fourier evidence score (0-100)")
    fig.subplots_adjust(wspace=0.38, right=0.88)
    fig.savefig(out_dir / "fourier_score_heatmaps.png", dpi=180)
    plt.close(fig)


def plot_metric_boxplots(features: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        ("fourier_score", "Fourier score"),
        ("fourier_sparsity", "Fourier sparsity"),
        ("circularity_top_freq", "Circularity CV"),
        ("embedding_effective_rank", "Embedding effective rank"),
    ]
    metrics = [(c, t) for c, t in metrics if c in features.columns]
    if not metrics:
        return
    labels = []
    grouped = []
    for group in sorted(features["group"].dropna().unique()):
        for phase in PHASE_ORDER:
            sub = features[(features["group"] == group) & (features["phase"] == phase)]
            if not sub.empty:
                labels.append(f"{group}\n{phase[:4]}")
                grouped.append(sub)
    if not grouped:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.8 * len(metrics), 4.6), squeeze=False)
    for ax, (metric, title) in zip(axes.flat, metrics):
        data = [g[metric].dropna().values for g in grouped]
        ax.boxplot(data, tick_labels=labels, showfliers=True, patch_artist=True)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_by_phase.png", dpi=180)
    plt.close(fig)


def plot_phase_trajectories(traj: pd.DataFrame, cells: pd.DataFrame, out_dir: Path) -> None:
    merged = traj.merge(cells[[*KEYS, "phase"]], on=KEYS, how="left")
    merged["phase"] = merged["phase"].fillna("unknown")
    groups = sorted(merged["group"].dropna().unique())
    if not groups:
        return
    metrics = [
        ("fourier_sparsity", "Fourier sparsity"),
        ("circularity_top_freq", "Circularity CV"),
    ]
    fig, axes = plt.subplots(len(metrics), len(groups), figsize=(5.0 * len(groups), 6.8), squeeze=False)
    for col_idx, group in enumerate(groups):
        sub_g = merged[merged["group"] == group]
        for row_idx, (metric, title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for phase in PHASE_ORDER:
                sub = sub_g[sub_g["phase"] == phase]
                if sub.empty:
                    continue
                med = sub.groupby("step", as_index=False)[metric].median().sort_values("step")
                ax.semilogx(
                    med["step"],
                    med[metric],
                    marker="o",
                    linewidth=1.4,
                    markersize=3.2,
                    color=PHASE_COLORS.get(phase, "#333333"),
                    label=phase,
                )
            ax.set_title(f"Group {group}: {title}")
            ax.set_xlabel("step")
            ax.set_ylabel(title)
            ax.grid(alpha=0.18)
    handles = [
        plt.Line2D([0], [0], marker="o", color=PHASE_COLORS[p], label=p)
        for p in PHASE_ORDER
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "phase_metric_trajectories.png", dpi=180)
    plt.close(fig)


def write_figures(traj: pd.DataFrame, cells: pd.DataFrame, features: pd.DataFrame, out_dir: Path) -> None:
    plot_phase_grids(cells, out_dir)
    plot_fourier_scatter(features, out_dir)
    plot_score_heatmaps(features, out_dir)
    plot_metric_boxplots(features, out_dir)
    plot_phase_trajectories(traj, cells, out_dir)


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    show = df.head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    columns = [str(c) for c in show.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in show.columns) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def compact_group_summary(features: pd.DataFrame) -> pd.DataFrame:
    comp = features[features["phase"] == "comprehend"]
    if comp.empty:
        return pd.DataFrame(
            columns=[
                "group",
                "n_comprehend",
                "median_fourier_score",
                "median_fourier_sparsity",
                "median_circularity",
                "median_delta_sparsity",
                "strong_or_moderate",
            ]
        )
    rows = []
    for group, sub in comp.groupby("group"):
        rows.append(
            {
                "group": group,
                "n_comprehend": len(sub),
                "median_fourier_score": sub["fourier_score"].median(),
                "median_fourier_sparsity": sub["fourier_sparsity"].median(),
                "median_circularity": sub["circularity_top_freq"].median(),
                "median_delta_sparsity": sub["delta_fourier_sparsity"].median(),
                "strong_or_moderate": int(
                    sub["mechanism_call"]
                    .isin(["fourier_like_strong", "fourier_like_moderate"])
                    .sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def write_report(
    out_dir: Path,
    *,
    p: int,
    traj_files: list[Path],
    cell_files: list[Path],
    cells: pd.DataFrame,
    features: pd.DataFrame,
    phase_summary: pd.DataFrame,
    metric_summary: pd.DataFrame,
    top_freqs: pd.DataFrame,
) -> None:
    mechanism_counts = (
        features.groupby(["group", "phase", "mechanism_call"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["group", "phase", "mechanism_call"])
    )
    group_summary = compact_group_summary(features)
    comp_cols = [
        "group",
        "alpha",
        "lam",
        "seed",
        "phase",
        "final_step_measured",
        "fourier_score",
        "fourier_sparsity",
        "circularity_top_freq",
        "delta_fourier_sparsity",
        "embedding_effective_rank",
        "lm_head_effective_rank",
        "mechanism_call",
        "top_canonical_freq",
        "dominant_freqs",
    ]
    comp_cols = [c for c in comp_cols if c in features.columns]
    comp_cells = features[features["phase"] == "comprehend"][comp_cols].sort_values(
        ["group", "alpha", "lam", "seed"]
    )
    checkpoint_files = sorted(
        pth for pattern in ("*.pt", "*.pth", "*.ckpt") for pth in Path("results").rglob(pattern)
    )
    if checkpoint_files:
        checkpoint_note = "Checkpoint files were found; run causal Fourier ablations on them next."
    else:
        checkpoint_note = (
            "No model checkpoint files (`*.pt`, `*.pth`, `*.ckpt`) were found under `results/`, "
            "so this run cannot perform causal ablations. The current conclusion is trajectory-level "
            "evidence, not a definitive proof."
        )

    report = f"""# Algorithm identification report

Generated by `scripts/identify_algorithm.py`.

## Data loaded

- Modulus: `p={p}`
- Trajectory files:
{chr(10).join(f"  - `{path}`" for path in traj_files)}
- Cell summary files:
{chr(10).join(f"  - `{path}`" for path in cell_files)}
- Rows after key-level deduplication: `{len(features)}` final cell/seed rows, `{len(cells)}` cell-summary rows.

## Method

The current scan artifacts contain per-checkpoint progress measures, not saved
model weights. I therefore use a two-tier methodology:

1. **Trajectory-level identification, available now.** For each `(group, alpha,
   lambda, seed)` I take the final measured checkpoint, merge its phase label,
   and score Fourier evidence from final `fourier_sparsity`, final
   `circularity_top_freq`, and `delta_fourier_sparsity`.
2. **Causal validation, required to rule out other algorithms.** On saved
    checkpoints, run `scripts/probe_fourier_checkpoint.py`: Fourier sufficiency,
    Fourier necessity, triadic logit FFT on `L[a,b,c]`, and cyclic equivariance
    checks.

{checkpoint_note}

The score thresholds are deliberately interpretable rather than trained:
`>=70` strong, `35-70` moderate, `20-35` weak, `<20` none.

## Phase counts

{markdown_table(phase_summary)}

## Comprehension-region Fourier evidence

{markdown_table(group_summary)}

## Mechanism calls

{markdown_table(mechanism_counts)}

## Metric medians by phase

{markdown_table(metric_summary)}

## Comprehend cells

{markdown_table(comp_cells, max_rows=80)}

## Dominant canonical frequencies in generalizing cells

For real embeddings, conjugate FFT bins are mathematically paired, so the
important summary is the canonical top frequency distribution rather than mere
presence of a conjugate partner.

{markdown_table(top_freqs, max_rows=80)}

## Figures

- `phase_grids.png` — phase map over `(alpha, lambda)`.
- `fourier_evidence_scatter.png` — final Fourier sparsity vs circularity.
- `fourier_score_heatmaps.png` — Fourier evidence score over the grid.
- `metrics_by_phase.png` — metric distributions by group/phase.
- `phase_metric_trajectories.png` — median Fourier/circularity trajectories by phase.

## Conservative conclusion

- A cell with `phase=comprehend` or `phase=grok` and strong/moderate Fourier
  evidence is a Fourier-like algorithm candidate.
- A generalizing cell with weak evidence is an “other algorithm or incomplete
  evidence” candidate.
- A `memorize` cell is not evidence of an algorithmic solution even if its
  embedding has Fourier structure; it did not generalize.
- A `fail` cell has no validated learned algorithm.

Because the current stored artifacts are trajectories rather than weights, the
causal Fourier-vs-other decision should be finalized only after saving and
ablating representative checkpoints from the comprehension region.
"""
    (out_dir / "algorithm_identification_report.md").write_text(report, encoding="utf-8")
    metadata = {
        "p": p,
        "trajectory_files": [str(path) for path in traj_files],
        "cell_files": [str(path) for path in cell_files],
        "n_final_rows": int(len(features)),
        "n_cell_rows": int(len(cells)),
        "score_formula": {
            "sparsity_component": "clip((fourier_sparsity - 0.30) / 0.55, 0, 1)",
            "circularity_component": "clip((0.55 - circularity_top_freq) / 0.45, 0, 1)",
            "trend_component": "clip((delta_fourier_sparsity + 0.05) / 0.65, 0, 1)",
            "fourier_score": "100*(0.55*sparsity + 0.35*circularity + 0.10*trend)",
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    traj_files = preferred_parquet_files(
        args.results_dir,
        cells=False,
        include_partial_duplicates=args.include_partial_duplicates,
    )
    cell_files = preferred_parquet_files(
        args.results_dir,
        cells=True,
        include_partial_duplicates=args.include_partial_duplicates,
    )
    traj = load_parquet_set(traj_files, kind="trajectory")
    cells = load_parquet_set(cell_files, kind="cells")
    traj = filter_groups(traj, args.groups)
    cells = filter_groups(cells, args.groups)
    if traj.empty:
        raise ValueError("No trajectory rows remain after filtering.")
    if cells.empty:
        raise ValueError("No cell summary rows remain after filtering.")
    traj = dedupe_by_priority(traj, TRAJ_KEY)
    cells = dedupe_by_priority(cells, KEYS)

    features = add_algorithm_features(traj, cells, args.p)
    phase_summary = summarize_phase(cells)
    metric_summary = summarize_metrics(features)
    top_freqs = summarize_top_frequencies(features)

    save_tables(args.out_dir, features, phase_summary, metric_summary, top_freqs)
    if not args.no_figures:
        write_figures(traj, cells, features, args.out_dir)
    write_report(
        args.out_dir,
        p=args.p,
        traj_files=traj_files,
        cell_files=cell_files,
        cells=cells,
        features=features,
        phase_summary=phase_summary,
        metric_summary=metric_summary,
        top_freqs=top_freqs,
    )

    print(f"[saved] {args.out_dir / 'cell_algorithm_scores.csv'}")
    print(f"[saved] {args.out_dir / 'algorithm_identification_report.md'}")
    if not args.no_figures:
        print(f"[saved] figures under {args.out_dir}")
    comp = compact_group_summary(features)
    if not comp.empty:
        print("\n[comprehension summary]")
        print(comp.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())