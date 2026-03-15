"""
plot_runs.py
============
Reads the Parquet file produced by aggregate_runs.py and generates one
performance plot per sbatchman job, saved as <tag>.pdf under OUT_DIR.

Layout of each plot
-------------------
Three metrics are shown: runtime, commtime, throughput.
Within one sbatchman job there are N concurrent jobs (job_1, job_2, …).
Each job was re-spawned R times (repetitions 0, 1, 2, …).

For each metric, one subplot is produced.  Inside each subplot:
  - X axis  : repetition index
  - One line / box per job_name, so it is easy to compare concurrent jobs
    across the repetition timeline.
  - Each point is the mean over the MPI ranks (rows) of that run;
    a shaded band shows ±1 std across ranks.

Usage
-----
    python plot_runs.py results/DLNetBenchConcurrent_<cluster>_data.parquet
    python plot_runs.py results/*.parquet          # multiple files
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent / "common"))
import import_export

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUT_DIR = Path("plots") / "concurrent"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["throughput", "runtime", "commtime"]

METRIC_LABELS = {
    "throughput": "Throughput (img/s)",
    "runtime":    "Runtime (s)",
    "commtime":   "Comm. time (s)",
}

# Colour palette — one colour per job_name, assigned on first encounter
_PALETTE = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_parquet(path: Path) -> list[tuple[dict, dict[str, pd.DataFrame]]]:
    mapping, _ = import_export.read_multiple_from_parquet(path)
    return mapping


def _group_by_sbm_job(
    mapping: list[tuple[dict, dict[str, pd.DataFrame]]]
) -> dict[str, list[tuple[dict, pd.DataFrame]]]:
    """
    Returns { sbm_job_id -> [(meta, df), ...] } sorted by repetition.
    """
    groups: dict[str, list[tuple[dict, pd.DataFrame]]] = defaultdict(list)
    for meta, dfs in mapping:
        df = dfs.get("measurements")
        if df is None or df.empty:
            continue
        groups[str(meta["sbm_job_id"])].append((meta, df))

    # sort each group by (job_name, repetition) for deterministic x-axis order
    for jid in groups:
        groups[jid].sort(key=lambda t: (t[0]["job_name"], t[0]["repetition"]))

    return groups


def _aggregate_runs(
    entries: list[tuple[dict, pd.DataFrame]]
) -> dict[str, dict[int, dict[str, tuple[float, float]]]]:
    """
    Returns { job_name -> { repetition -> { metric -> (mean, std) } } }.
    The mean/std are computed over the rows of the measurements DataFrame
    (i.e. over MPI ranks / iterations within a single run).
    """
    result: dict[str, dict[int, dict[str, tuple[float, float]]]] = defaultdict(dict)
    for meta, df in entries:
        jname = meta["job_name"]
        rep   = meta["repetition"]
        stats: dict[str, tuple[float, float]] = {}
        for metric in METRICS:
            if metric in df.columns:
                col = pd.to_numeric(df[metric], errors="coerce").dropna()
                stats[metric] = (float(col.mean()), float(col.std(ddof=0)))
        if stats:
            result[jname][rep] = stats
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_job(
    sbm_job_id: str,
    tag: str,
    entries: list[tuple[dict, pd.DataFrame]],
) -> Path:
    agg = _aggregate_runs(entries)
    if not agg:
        print(f"  [skip] {tag}: no aggregatable data")
        return None

    job_names = sorted(agg.keys())
    colour = {jn: _PALETTE[i % len(_PALETTE)] for i, jn in enumerate(job_names)}

    # Determine which metrics are actually present in this job's data
    present_metrics = [
        m for m in METRICS
        if any(
            m in stats
            for reps in agg.values()
            for stats in reps.values()
        )
    ]
    if not present_metrics:
        print(f"  [skip] {tag}: no recognised metrics")
        return None

    n_metrics = len(present_metrics)
    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(5 * n_metrics, 4),
        squeeze=False,
    )
    fig.suptitle(f"Concurrent run performance\nsbm_job={sbm_job_id}  tag={tag}",
                 fontsize=10, y=1.02)

    for col_idx, metric in enumerate(present_metrics):
        ax = axes[0][col_idx]

        for jname in job_names:
            reps_dict = agg[jname]
            xs = sorted(reps_dict.keys())
            if not xs:
                continue

            ys    = np.array([reps_dict[r][metric][0] for r in xs if metric in reps_dict[r]])
            yerrs = np.array([reps_dict[r][metric][1] for r in xs if metric in reps_dict[r]])
            xs_present = [r for r in xs if metric in reps_dict[r]]

            if len(xs_present) == 0:
                continue

            c = colour[jname]
            ax.plot(xs_present, ys, marker="o", linewidth=1.5, markersize=4,
                    label=jname, color=c)
            ax.fill_between(xs_present,
                            ys - yerrs, ys + yerrs,
                            alpha=0.15, color=c)

        ax.set_xlabel("Repetition", fontsize=9)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9)
        ax.set_title(metric, fontsize=9)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(True, linewidth=0.4, alpha=0.6)
        ax.legend(fontsize=8, title="job", title_fontsize=8)

    fig.tight_layout()

    out_path = OUT_DIR / f"{tag}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _print_summary(
    tag: str,
    entries: list[tuple[dict, pd.DataFrame]],
    out_path: Path | None,
) -> None:
    job_names = sorted({e[0]["job_name"] for e in entries})
    rep_counts = {
        jn: len({e[0]["repetition"] for e in entries if e[0]["job_name"] == jn})
        for jn in job_names
    }
    total_rows = sum(len(df) for _, df in entries)
    status = f"→ {out_path}" if out_path else "→ skipped"
    print(
        f"  {tag:<40}  "
        f"jobs={len(job_names)}  "
        f"reps/job=[{', '.join(f'{jn}:{n}' for jn, n in rep_counts.items())}]  "
        f"total_rows={total_rows}  {status}"
    )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print("Usage: python plot_runs.py <file.parquet> [<file2.parquet> ...]")
        sys.exit(1)

    # Load all parquet files, merging into one flat mapping
    mapping: list[tuple[dict, dict[str, pd.DataFrame]]] = []
    for p in paths:
        if not p.exists():
            print(f"WARNING: {p} not found, skipping.")
            continue
        chunk = _load_parquet(p)
        mapping.extend(chunk)
        print(f"Loaded {len(chunk)} run(s) from {p}")

    if not mapping:
        print("No data loaded.")
        sys.exit(1)

    groups  = _group_by_sbm_job(mapping)

    print(f"\nGenerating plots for {len(groups)} sbatchman job(s) → {OUT_DIR}/\n")

    skipped = 0
    for sbm_job_id, entries in sorted(groups.items()):
        tag      = sbm_job_id # FIXME
        out_path = _plot_job(sbm_job_id, tag, entries)
        _print_summary(tag, entries, out_path)
        if out_path is None:
            skipped += 1

    print(f"\nDone. {len(groups) - skipped} plot(s) written, {skipped} skipped.")


if __name__ == "__main__":
    main()