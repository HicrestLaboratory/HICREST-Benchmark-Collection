"""
plot_concurrent.py
==================
Plots concurrent-run performance and, when baseline data is supplied,
computes and plots interference (slowdown) metrics.

Layout of each standard performance plot
-----------------------------------------
Three metrics (runtime, commtime, throughput) × one subplot each.
  X axis  : repetition index
  Lines   : one per job_name within the sbatchman job
  Band    : ±1 std across MPI ranks / iterations

Interference metrics (requires --baseline)
-------------------------------------------
For each sbatchman job in the concurrent data, four scalar summaries are
computed from per-job slowdown  σ_j = T0(strategy, gpus) / T(job | C):

  - Mean slowdown          σ̄  = mean(σ_j)
  - GPU-weighted slowdown  σ̃  = Σ (g_j / G) · σ_j
  - Worst-case slowdown    σ_max = max(σ_j)
  - Per-strategy histogram : one bar-plot bin per strategy, aggregated
                             with the chosen aggregate function

The metric and its aggregation function are fully configurable via
SLOWDOWN_METRIC and SLOWDOWN_AGG at the top of this file, or at runtime
via --metric and --agg.

Usage
-----
    # Performance plots only
    python plot_concurrent.py results/concurrent.parquet

    # With interference metrics
    python plot_concurrent.py results/concurrent.parquet \\
        --baseline results/baseline.parquet

    # Custom metric / aggregation
    python plot_concurrent.py results/concurrent.parquet \\
        --baseline results/baseline.parquet \\
        --metric throughput --agg mean
"""

from __future__ import annotations

import sys
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent / "common"))
import import_export


# ===========================================================================
# ❶  USER-CONFIGURABLE SECTION
#    Change these to switch how slowdown is extracted and aggregated.
# ===========================================================================

# Which column in dfs['measurements'] to use as the throughput proxy.
# Must be a higher-is-better metric (slowdown = T0 / T).
SLOWDOWN_METRIC: str = "throughput"

# How to reduce per-rank/iteration rows to a single scalar for one run.
# Options: "mean" | "median" | "max" | "min"
SLOWDOWN_AGG: str = "mean"


# ===========================================================================
# Config
# ===========================================================================

OUT_DIR = Path("plots") / "concurrent"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["throughput", "runtime", "commtime"]

METRIC_LABELS = {
    "throughput": "Throughput (img/s)",
    "runtime":    "Runtime (s)",
    "commtime":   "Comm. time (s)",
}

_PALETTE = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ===========================================================================
# Aggregation helpers
# ===========================================================================

def _agg_fn(name: str) -> Callable[[pd.Series], float]:
    """Return a scalar aggregation function by name."""
    fns: dict[str, Callable] = {
        "mean":   lambda s: float(s.mean()),
        "median": lambda s: float(s.median()),
        "max":    lambda s: float(s.max()),
        "min":    lambda s: float(s.min()),
    }
    if name not in fns:
        raise ValueError(f"Unknown aggregation '{name}'. Choose from {list(fns)}")
    return fns[name]


# ===========================================================================
# Debug / pretty-print helpers
# ===========================================================================

def _table(rows: list[list[str]], header: list[str], title: str = "", indent: int = 2) -> None:
    """Print a plain-text aligned table to stdout."""
    all_rows   = [header] + rows
    col_widths = [max(len(str(r[c])) for r in all_rows) for c in range(len(header))]
    pad        = " " * indent
    sep        = pad + "  ".join("-" * w for w in col_widths)
    if title:
        total_w = sum(col_widths) + 2 * (len(col_widths) - 1)
        print(f"\n{pad}{title:=^{total_w}}")
    print(pad + "  ".join(str(h).ljust(w) for h, w in zip(header, col_widths)))
    print(sep)
    for row in rows:
        print(pad + "  ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))
    print()


def _print_baseline_table(
    baseline: dict[tuple[str, int], float],
    metric: str,
    counts: dict[tuple[str, int], int],
) -> None:
    """Print a summary table of all T0 baseline values."""
    if not baseline:
        print("  (no baseline entries)\n")
        return
    rows = [
        [strategy, str(gpus), f"{t0:.4f}", str(counts.get((strategy, gpus), "?"))]
        for (strategy, gpus), t0 in sorted(baseline.items())
    ]
    _table(
        rows,
        header=["Strategy", "GPUs", f"T0 ({metric})", "Runs averaged"],
        title=" Baseline T0 summary ",
    )


def _print_concurrent_job_stats(
    sbm_job_id: str,
    tag: str,
    entries: list[tuple[dict, pd.DataFrame]],
    metric: str,
    agg: str,
    slowdowns: dict[str, dict[int, float]] | None,
    baseline: dict[tuple[str, int], float] | None,
) -> None:
    """
    Print two tables for one sbatchman job:

    Table 1 — per (job_name, repetition) row:
        strategy/gpus  n_rows  <metric> mean  <metric> std  <metric> <agg>  T0  σ_j

    Table 2 — per job_name summary (σ aggregated over repetitions):
        job_name  n_reps  σ mean  σ std  σ min  σ max
    """
    agg_fn = _agg_fn(agg)

    detail_rows = []
    for meta, df in sorted(entries, key=lambda t: (t[0]["job_name"], t[0]["repetition"])):
        jname    = meta["job_name"]
        rep      = meta["repetition"]
        strategy = meta.get("strategy") or jname.split('_')[0]
        gpus     = meta.get("gpus") or int(jname.split('_')[1][1:])

        if metric in df.columns:
            col    = pd.to_numeric(df[metric], errors="coerce").dropna()
            n      = len(col)
            mean_v = col.mean()
            std_v  = col.std(ddof=0)
            agg_v  = agg_fn(col)
        else:
            n = mean_v = std_v = agg_v = float("nan")

        t0    = baseline.get((strategy, int(gpus)), float("nan")) if baseline else float("nan")
        sigma = (
            slowdowns[jname][rep]
            if slowdowns and jname in slowdowns and rep in slowdowns[jname]
            else float("nan")
        )

        detail_rows.append([
            jname,
            str(rep),
            f"{strategy}/{gpus}g",
            str(n),
            f"{mean_v:.4f}" if not np.isnan(mean_v) else "—",
            f"{std_v:.4f}"  if not np.isnan(std_v)  else "—",
            f"{agg_v:.4f}"  if not np.isnan(agg_v)  else "—",
            f"{t0:.4f}"     if not np.isnan(t0)      else "—",
            f"{sigma:.4f}"  if not np.isnan(sigma)   else "—",
        ])

    _table(
        detail_rows,
        header=["job_name", "rep", "strategy/gpus", "n_rows",
                f"{metric} mean", f"{metric} std", f"{metric} ({agg})",
                "T0", "σ_j"],
        title=f" sbm_job={sbm_job_id} / tag={tag} — per-repetition detail ",
    )

    # Table 2: per-job σ summary (only when slowdowns available)
    if slowdowns:
        summary_rows = []
        for jname in sorted(slowdowns):
            reps = slowdowns[jname]
            if not reps:
                continue
            vals = list(reps.values())
            summary_rows.append([
                jname,
                str(len(vals)),
                f"{np.mean(vals):.4f}",
                f"{np.std(vals, ddof=0):.4f}",
                f"{np.min(vals):.4f}",
                f"{np.max(vals):.4f}",
            ])
        if summary_rows:
            _table(
                summary_rows,
                header=["job_name", "n_reps", "σ mean", "σ std", "σ min", "σ max"],
                title=f" sbm_job={sbm_job_id} — per-job σ summary (over repetitions) ",
            )


# ===========================================================================
# Data loading
# ===========================================================================

def _load_parquet(path: Path) -> list[tuple[dict, dict[str, pd.DataFrame]]]:
    mapping, _ = import_export.read_multiple_from_parquet(path)
    return mapping


def _group_by_sbm_job(
    mapping: list[tuple[dict, dict[str, pd.DataFrame]]],
) -> dict[str, list[tuple[dict, pd.DataFrame]]]:
    """Returns { sbm_job_id -> [(meta, df), ...] } sorted by (job_name, repetition)."""
    groups: dict[str, list[tuple[dict, pd.DataFrame]]] = defaultdict(list)
    for meta, dfs in mapping:
        df = dfs.get("measurements")
        if df is None or df.empty:
            continue
        groups[str(meta["sbm_job_id"])].append((meta, df))

    for jid in groups:
        groups[jid].sort(key=lambda t: (t[0]["job_name"], t[0]["repetition"]))

    return groups


def _build_baseline_table_from_mapping(
    mapping: list[tuple[dict, dict[str, pd.DataFrame]]],
    metric: str,
    agg: str,
    exclude_first_n: int = 2
) -> tuple[dict[tuple[str, int], float], dict[tuple[str, int], int]]:
    """
    Build { (strategy, gpus) -> T0 } from an already-loaded mapping.

    Multiple runs with the same key are averaged together.
    
    exclude_first_n Excludes warmup runs

    Returns (baseline, counts) where counts[key] is how many runs
    contributed to that T0 value.
    """
    agg_fn      = _agg_fn(agg)
    accumulator: dict[tuple[str, int], list[float]] = defaultdict(list)
    counts:   dict[tuple[str, int], int]   = defaultdict(int)

    for meta, dfs in mapping:
        df = dfs.get("measurements")
        if df is None or df.empty:
            continue

        strategy = meta.get("strategy")
        gpus     = meta.get("gpus")
        uid      = meta.get("uid")
        if not strategy and uid:
            strategy = uid.split('_')[0]
        if not gpus and uid:
            gpus = int(uid.split('_')[1][1:])
            
        if metric not in df.columns:
            print(f"  [baseline] metric '{metric}' not found in job {meta.get('sbm_job_id')} ({strategy=} / {gpus=}), skipping.")
            continue
            
        if strategy is None or gpus is None:
            print(f"  [baseline] missing 'strategy' or 'gpus' or 'uid' in meta {meta}, skipping.")
            continue

        col = pd.to_numeric(df[metric], errors="coerce").dropna()
        if col.empty:
            continue
        # Remove the first exclude_first_n values from col
        if len(col) > exclude_first_n:
            col = col.iloc[exclude_first_n:]
        else:
            print(f"  [baseline] cannot remove {exclude_first_n} warmup runs for {strategy=} / {gpus=}, not enough values.")
        
        # print(metric)
        # print(agg)
        # print(meta)
        # print(df)        
        # print(col)
        # print('='*80)
        # print()
        
        counts[(strategy, int(gpus))] += len(col) 
        accumulator[(strategy, int(gpus))].append(agg_fn(col))

    baseline: dict[tuple[str, int], float] = {}
    for key, vals in accumulator.items():
        baseline[key] = float(np.mean(vals))

    return baseline, counts


def _build_baseline_table(
    path: Path,
    metric: str,
    agg: str,
) -> tuple[dict[tuple[str, int], float], dict[tuple[str, int], int]]:
    """Convenience wrapper: load a single parquet file and build the T0 table."""
    mapping = _load_parquet(path)
    return _build_baseline_table_from_mapping(mapping, metric, agg)


# ===========================================================================
# Per-run aggregation for concurrent data
# ===========================================================================

def _aggregate_runs(
    entries: list[tuple[dict, pd.DataFrame]],
) -> dict[str, dict[int, dict[str, tuple[float, float]]]]:
    """
    Returns { job_name -> { repetition -> { metric -> (mean, std) } } }.
    mean/std are over MPI ranks / iterations within one run.
    """
    result: dict[str, dict[int, dict[str, tuple[float, float]]]] = defaultdict(dict)
    for meta, df in entries:
        jname = meta["job_name"]
        rep   = meta["repetition"]
        stats: dict[str, tuple[float, float]] = {}
        for m in METRICS:
            if m in df.columns:
                col = pd.to_numeric(df[m], errors="coerce").dropna()
                stats[m] = (float(col.mean()), float(col.std(ddof=0)))
        if stats:
            result[jname][rep] = stats
    return result


# ===========================================================================
# Slowdown computation
# ===========================================================================

def _compute_slowdowns(
    entries:  list[tuple[dict, pd.DataFrame]],
    baseline: dict[tuple[str, int], float],
    metric:   str,
    agg:      str,
) -> dict[str, dict[int, float]] | None:
    """
    Returns { job_name -> { repetition -> σ_j } } or None if data is missing.
    σ_j = T0(strategy, gpus) / T(job | C)
    """
    agg_fn       = _agg_fn(agg)
    result:       dict[str, dict[int, float]] = defaultdict(dict)
    missing_keys: set[tuple[str, int]]        = set()

    for meta, df in entries:
        jname    = meta["job_name"]
        rep      = meta["repetition"]
        strategy = meta.get("strategy")
        gpus     = meta.get("gpus")
        if not strategy:
            strategy = jname.split('_')[0]
        if not gpus:
            gpus = int(jname.split('_')[1][1:])

        if strategy is None or gpus is None:
            print(f"  [slowdown] missing 'strategy'/'gpus' for job {jname}, rep {rep} — skipping.")
            continue

        key = (strategy, int(gpus))
        t0  = baseline.get(key)
        if t0 is None:
            missing_keys.add(key)
            continue

        if metric not in df.columns:
            continue
        col = pd.to_numeric(df[metric], errors="coerce").dropna()
        if col.empty:
            continue

        t_conc = agg_fn(col)
        if t_conc <= 0:
            continue

        result[jname][rep] = t0 / t_conc

    for key in missing_keys:
        print(f"  [slowdown] WARNING: no baseline for {key} — those jobs excluded from metrics.")

    return result if result else None


# ===========================================================================
# Aggregate slowdown metrics
# ===========================================================================

def _aggregate_slowdowns(
    slowdowns: dict[str, dict[int, float]],
    entries:   list[tuple[dict, pd.DataFrame]],
) -> dict[str, float | dict]:
    """
    Returns:
      'per_job'      : { job_name -> mean_σ over reps }
      'mean'         : σ̄
      'gpu_weighted' : σ̃
      'worst_case'   : σ_max
      'per_strategy' : { strategy -> mean_σ }
    """
    job_meta: dict[str, tuple[str, int]] = {}
    for meta, _ in entries:
        jname    = meta["job_name"]
        strategy = meta.get("strategy", "unknown")
        gpus     = int(meta.get("gpus", 0))
        job_meta[jname] = (strategy, gpus)

    total_gpus = sum(v[1] for v in job_meta.values())

    per_job: dict[str, float] = {
        jname: float(np.mean(list(reps.values())))
        for jname, reps in slowdowns.items() if reps
    }
    if not per_job:
        return {}

    mean_sigma   = float(np.mean(list(per_job.values())))
    worst_case   = float(max(per_job.values()))
    gpu_weighted = sum(
        (job_meta.get(jn, ("", 0))[1] / total_gpus) * s if total_gpus > 0 else s
        for jn, s in per_job.items()
    )

    by_strategy: dict[str, list[float]] = defaultdict(list)
    for jname, sigma in per_job.items():
        by_strategy[job_meta.get(jname, ("unknown", 0))[0]].append(sigma)
    per_strategy = {s: float(np.mean(vs)) for s, vs in by_strategy.items()}

    return {
        "per_job":      per_job,
        "mean":         mean_sigma,
        "gpu_weighted": gpu_weighted,
        "worst_case":   worst_case,
        "per_strategy": per_strategy,
    }


# ===========================================================================
# Standard performance plot
# ===========================================================================

def _plot_performance(
    sbm_job_id: str,
    tag: str,
    entries: list[tuple[dict, pd.DataFrame]],
) -> Path | None:
    agg = _aggregate_runs(entries)
    if not agg:
        print(f"  [skip] {tag}: no aggregatable data")
        return None

    job_names = sorted(agg.keys())
    colour    = {jn: _PALETTE[i % len(_PALETTE)] for i, jn in enumerate(job_names)}

    present_metrics = [
        m for m in METRICS
        if any(m in stats for reps in agg.values() for stats in reps.values())
    ]
    if not present_metrics:
        print(f"  [skip] {tag}: no recognised metrics")
        return None

    n_metrics = len(present_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4), squeeze=False)
    fig.suptitle(
        f"Concurrent run performance\nsbm_job={sbm_job_id}  tag={tag}",
        fontsize=10, y=1.02,
    )

    for col_idx, metric in enumerate(present_metrics):
        ax = axes[0][col_idx]
        for jname in job_names:
            reps_dict  = agg[jname]
            xs_present = sorted(r for r in reps_dict if metric in reps_dict[r])
            if not xs_present:
                continue
            ys    = np.array([reps_dict[r][metric][0] for r in xs_present])
            yerrs = np.array([reps_dict[r][metric][1] for r in xs_present])
            c     = colour[jname]
            ax.plot(xs_present, ys, marker="o", linewidth=1.5, markersize=4, label=jname, color=c)
            ax.fill_between(xs_present, ys - yerrs, ys + yerrs, alpha=0.15, color=c)

        ax.set_xlabel("Repetition", fontsize=9)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9)
        ax.set_title(metric, fontsize=9)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(True, linewidth=0.4, alpha=0.6)
        ax.legend(fontsize=8, title="job", title_fontsize=8)

    fig.tight_layout()
    out_path = OUT_DIR / f"{tag}_performance.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ===========================================================================
# Interference / slowdown plots
# ===========================================================================

def _plot_slowdown_timeline(
    sbm_job_id: str,
    tag:        str,
    slowdowns:  dict[str, dict[int, float]],
) -> Path:
    job_names = sorted(slowdowns.keys())
    colour    = {jn: _PALETTE[i % len(_PALETTE)] for i, jn in enumerate(job_names)}

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle(f"Slowdown timeline\nsbm_job={sbm_job_id}  tag={tag}", fontsize=10)

    for jname in job_names:
        reps = slowdowns[jname]
        xs   = sorted(reps.keys())
        ys   = [reps[r] for r in xs]
        ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=4,
                label=jname, color=colour[jname])

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", label="No slowdown (σ=1)")
    ax.set_xlabel("Repetition", fontsize=9)
    ax.set_ylabel("Slowdown  σ_j  (higher = worse)", fontsize=9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, linewidth=0.4, alpha=0.6)
    ax.legend(fontsize=8, title="job", title_fontsize=8)

    fig.tight_layout()
    out_path = OUT_DIR / f"{tag}_slowdown_timeline.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_slowdown_summary(
    sbm_job_id:  str,
    tag:         str,
    agg_metrics: dict,
) -> Path:
    per_job      = agg_metrics["per_job"]
    per_strategy = agg_metrics["per_strategy"]
    job_names    = sorted(per_job.keys())
    colour       = {jn: _PALETTE[i % len(_PALETTE)] for i, jn in enumerate(job_names)}
    strat_names  = sorted(per_strategy.keys())

    fig, (ax_jobs, ax_summary) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Interference summary\nsbm_job={sbm_job_id}  tag={tag}", fontsize=10)

    # Left: per-job σ
    bars = ax_jobs.bar(
        range(len(job_names)),
        [per_job[jn] for jn in job_names],
        color=[colour[jn] for jn in job_names],
        edgecolor="black", linewidth=0.5,
    )
    ax_jobs.axhline(1.0, color="black", linewidth=0.8, linestyle="--", label="No slowdown")
    ax_jobs.set_xticks(range(len(job_names)))
    ax_jobs.set_xticklabels(job_names, rotation=20, ha="right", fontsize=8)
    ax_jobs.set_ylabel("Mean slowdown  σ_j", fontsize=9)
    ax_jobs.set_title("Per-job slowdown", fontsize=9)
    ax_jobs.legend(fontsize=8)
    ax_jobs.grid(True, axis="y", linewidth=0.4, alpha=0.6)
    for bar in bars:
        h = bar.get_height()
        ax_jobs.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                     f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    # Right: aggregate scalars + per-strategy
    scalar_labels = ["Mean σ̄", "GPU-weighted σ̃", "Worst-case σ_max"]
    scalar_values = [agg_metrics["mean"], agg_metrics["gpu_weighted"], agg_metrics["worst_case"]]
    all_labels    = scalar_labels + [f"{s}\n(strategy)" for s in strat_names]
    all_values    = scalar_values + [per_strategy[s] for s in strat_names]
    bar_colours   = (
        ["#4C72B0", "#DD8452", "#55A868"]
        + [_PALETTE[(3 + i) % len(_PALETTE)] for i in range(len(strat_names))]
    )

    bars2 = ax_summary.bar(range(len(all_labels)), all_values,
                           color=bar_colours, edgecolor="black", linewidth=0.5)
    ax_summary.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax_summary.set_xticks(range(len(all_labels)))
    ax_summary.set_xticklabels(all_labels, rotation=15, ha="right", fontsize=8)
    ax_summary.set_ylabel("Slowdown σ", fontsize=9)
    ax_summary.set_title("Aggregate interference metrics", fontsize=9)
    ax_summary.grid(True, axis="y", linewidth=0.4, alpha=0.6)
    for bar in bars2:
        h = bar.get_height()
        ax_summary.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    out_path = OUT_DIR / f"{tag}_slowdown_summary.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_slowdown_heatmap(
    groups:   dict[str, list[tuple[dict, pd.DataFrame]]],
    baseline: dict[tuple[str, int], float],
    metric:   str,
    agg:      str,
) -> Path | None:
    rows: list[dict] = []
    row_tags:  list[str] = []

    for sbm_job_id, entries in sorted(groups.items()):
        slowdowns = _compute_slowdowns(entries, baseline, metric, agg)
        if not slowdowns:
            continue
        row_tags.append(sbm_job_id)
        rows.append({jn: float(np.mean(list(reps.values()))) for jn, reps in slowdowns.items()})

    if not rows:
        return None

    df  = pd.DataFrame(rows, index=row_tags).sort_index(axis=1).fillna(np.nan)
    fig, ax = plt.subplots(figsize=(max(6, len(df.columns) * 1.2), max(4, len(df) * 0.5 + 1)))
    im  = ax.imshow(df.values, aspect="auto", cmap="RdYlGn_r", vmin=1.0)

    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=7)
    ax.set_xlabel("Job name (slot)", fontsize=9)
    ax.set_ylabel("sbm_job_id", fontsize=9)
    ax.set_title("Slowdown heatmap  (σ_j, mean over repetitions)", fontsize=10)
    fig.colorbar(im, ax=ax, pad=0.01).set_label("σ_j", fontsize=8)

    for r in range(len(df.index)):
        for c in range(len(df.columns)):
            v = df.values[r, c]
            if not np.isnan(v):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6, color="black")

    fig.tight_layout()
    out_path = OUT_DIR / "all_jobs_slowdown_heatmap.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ===========================================================================
# Summary printer
# ===========================================================================

def _print_summary(
    tag:       str,
    entries:   list[tuple[dict, pd.DataFrame]],
    perf_path: Path | None,
    slow_path: Path | None = None,
    summ_path: Path | None = None,
) -> None:
    job_names  = sorted({e[0]["job_name"] for e in entries})
    rep_counts = {
        jn: len({e[0]["repetition"] for e in entries if e[0]["job_name"] == jn})
        for jn in job_names
    }
    total_rows = sum(len(df) for _, df in entries)
    outs = " | ".join(filter(None, [
        f"perf→{perf_path}"     if perf_path else None,
        f"timeline→{slow_path}" if slow_path else None,
        f"summary→{summ_path}"  if summ_path else None,
    ]))
    print(
        f"  {tag:<40}  "
        f"jobs={len(job_names)}  "
        f"reps/job=[{', '.join(f'{jn}:{n}' for jn, n in rep_counts.items())}]  "
        f"rows={total_rows}  {outs or '→ skipped'}"
    )


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser(
        description="Plot concurrent training runs and (optionally) interference metrics."
    )
    parser.add_argument(
        "parquet_files", nargs="+",
        help="Concurrent-run parquet file(s).",
    )
    parser.add_argument(
        "--baseline", "-b", metavar="PARQUET", nargs="+",
        help="Baseline parquet file(s). Multiple files are merged before building the T0 table.",
    )
    parser.add_argument(
        "--metric", default=SLOWDOWN_METRIC,
        help=f"Measurement column to use for slowdown (default: {SLOWDOWN_METRIC}).",
    )
    parser.add_argument(
        "--agg", default=SLOWDOWN_AGG,
        choices=["mean", "median", "max", "min"],
        help=f"How to reduce per-row values to a scalar (default: {SLOWDOWN_AGG}).",
    )
    parser.add_argument(
        "--out-dir", default=str(OUT_DIR),
        help="Output directory for plots.",
    )
    args = parser.parse_args()

    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metric: str = args.metric
    agg:    str = args.agg

    # ------------------------------------------------------------------
    # Load concurrent data
    # ------------------------------------------------------------------
    mapping: list[tuple[dict, dict[str, pd.DataFrame]]] = []
    for p in [Path(f) for f in args.parquet_files]:
        if not p.exists():
            print(f"WARNING: {p} not found, skipping.")
            continue
        chunk = _load_parquet(p)
        mapping.extend(chunk)
        print(f"Loaded {len(chunk)} run(s) from {p}")

    if not mapping:
        print("No data loaded.")
        sys.exit(1)

    groups = _group_by_sbm_job(mapping)
    print(f"\n{len(groups)} sbatchman job(s) found.")

    # ------------------------------------------------------------------
    # Load baseline (optional)
    # ------------------------------------------------------------------
    baseline: dict[tuple[str, int], float] | None = None
    if args.baseline:
        baseline_mapping: list[tuple[dict, dict[str, pd.DataFrame]]] = []
        for bp in [Path(f) for f in args.baseline]:
            if not bp.exists():
                print(f"WARNING: baseline file '{bp}' not found — skipping.")
                continue
            chunk = _load_parquet(bp)
            baseline_mapping.extend(chunk)
            print(f"Loaded {len(chunk)} baseline run(s) from {bp}")

        if not baseline_mapping:
            print("WARNING: no baseline data loaded — interference metrics disabled.")
        else:
            print(f"\nBuilding T0 table from {len(baseline_mapping)} run(s)  [metric={metric}, agg={agg}]")
            baseline, counts = _build_baseline_table_from_mapping(baseline_mapping, metric, agg)
            _print_baseline_table(baseline, metric, counts)

    # ------------------------------------------------------------------
    # Per-sbatchman-job plots + debug output
    # ------------------------------------------------------------------
    print(f"Generating plots → {OUT_DIR}/\n")
    skipped = 0

    for sbm_job_id, entries in sorted(groups.items()):
        tag = sbm_job_id  # FIXME: replace with human-readable tag if available

        perf_path = _plot_performance(sbm_job_id, tag, entries)

        slow_path = summ_path = None
        slowdowns = None

        if baseline is not None:
            slowdowns = _compute_slowdowns(entries, baseline, metric, agg)

            if slowdowns:
                agg_metrics = _aggregate_slowdowns(slowdowns, entries)
                if agg_metrics:
                    slow_path = _plot_slowdown_timeline(sbm_job_id, tag, slowdowns)
                    summ_path = _plot_slowdown_summary(sbm_job_id, tag, agg_metrics)

        # ── Debug tables ──────────────────────────────────────────────
        _print_concurrent_job_stats(
            sbm_job_id, tag, entries,
            metric=metric, agg=agg,
            slowdowns=slowdowns,
            baseline=baseline,
        )

        if slowdowns:
            agg_metrics = _aggregate_slowdowns(slowdowns, entries)
            if agg_metrics:
                strat_str = "  ".join(
                    f"{s}={v:.3f}" for s, v in sorted(agg_metrics["per_strategy"].items())
                )
                print(
                    f"  Aggregates:  "
                    f"σ̄={agg_metrics['mean']:.3f}  "
                    f"σ̃={agg_metrics['gpu_weighted']:.3f}  "
                    f"σ_max={agg_metrics['worst_case']:.3f}  "
                    f"per-strategy: {strat_str}\n"
                )
        # ─────────────────────────────────────────────────────────────

        _print_summary(tag, entries, perf_path, slow_path, summ_path)
        if perf_path is None:
            skipped += 1

    # ------------------------------------------------------------------
    # Cross-job heatmap
    # ------------------------------------------------------------------
    if baseline is not None and len(groups) > 1:
        print("\nGenerating cross-job slowdown heatmap...")
        hmap_path = _plot_slowdown_heatmap(groups, baseline, metric, agg)
        if hmap_path:
            print(f"  Heatmap saved to {hmap_path}")

    total = len(groups)
    print(f"\nDone. {total - skipped} plot(s) written, {skipped} skipped.")


if __name__ == "__main__":
    main()