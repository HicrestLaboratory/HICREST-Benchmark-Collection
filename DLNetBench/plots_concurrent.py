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
For each sbatchman job in the concurrent data, three scalar summaries are
computed from per-job slowdown  σ_j = T0(strategy, gpus, placement) / T(job | C):

  - Mean slowdown          σ̄  = mean(σ_j)
  - Worst-case slowdown    σ_max = max(σ_j)
  - Per-strategy histogram : one bar-plot bin per strategy, aggregated
                             with the chosen aggregate function

The metric and its aggregation function are fully configurable via
SLOWDOWN_METRIC and SLOWDOWN_AGG at the top of this file, or at runtime
via --metric and --agg.

Baseline lookup key
--------------------
Baselines are keyed by (strategy, gpus, placement_class) where
placement_class is extracted from the ``sbm_tag`` field using the same
``class-<placement>_rep`` regex that the baseline plotting script uses.

If a concurrent run's exact (strategy, gpus, placement_class) triple is not
found in the baseline table, the script emits **one** warning per missing key
and falls back to the nearest available baseline for the same (strategy, gpus)
pair (if any exists).  When no (strategy, gpus) match exists at all, those
jobs are excluded from interference metrics and a warning is printed.

Placement visualisation (requires --show-placements)
-----------------------------------------------------
For every sbatchman job / repetition a SVG is produced showing which nodes
each concurrent job was placed on.  The SVGs are written alongside the other
plots with the naming convention::

    <tag>_rep<N>_placement.svg

The feature requires each run's metadata to contain:
  - ``system``   : cluster name (e.g. "alps")
  - ``nodelist`` : nodes allocated to that job (comma/space separated)

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

    # With placement SVGs
    python plot_concurrent.py results/concurrent.parquet \\
        --show-placements
"""

from __future__ import annotations

import sys
import re
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Callable

import matplotlib

import command_map
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent / "common"))
import import_export
try:
    from JobPlacer.cli_wrapper import JobPlacer
except ImportError:
    JobPlacer = None


# ===========================================================================
# ❶  USER-CONFIGURABLE SECTION
#    Change these to switch how slowdown is extracted and aggregated.
# ===========================================================================

# Which column in dfs['main'] to use as the throughput proxy.
# Must be a higher-is-better metric (slowdown = T0 / T).
SLOWDOWN_METRIC: str = "throughput_mean"

# How to reduce per-rank/iteration rows to a single scalar for one run.
# Options: "mean" | "median" | "max" | "min"
SLOWDOWN_AGG: str = "mean"


# ===========================================================================
# Config
# ===========================================================================

OUT_DIR = Path("plots") / "concurrent"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["throughput_mean", "runtime_mean", "commtime"]

METRIC_LABELS = {
    "throughput_mean": "Throughput (samples/s)",
    "runtime_mean":    "Runtime (s)",
    "commtime":        "Comm. time (s)",
}

_PALETTE = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Regex that extracts the placement class from an sbm_tag.
# Matches tags of the form:  ...class-<placement>_rep...
# Identical to the pattern used in the baseline plotting script.
_CLASS_TAG_RE = re.compile(r'class-([^_]+(?:_[^_]+)*)_rep')

# ---------------------------------------------------------------------------
# Visual-encoding pools
# ---------------------------------------------------------------------------

_COLOR_POOL: list[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#9edae5", "#dbdb8d", "#c7c7c7",
]
_MARKER_POOL: list[str] = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p", "H", "8"]
_LINESTYLE_POOL: list[str] = ["-", "--", "-.", ":",
                               (0, (3, 1, 1, 1)),
                               (0, (5, 1)),
                               (0, (1, 1)),
                               (0, (3, 1, 1, 1, 1, 1))]


# ===========================================================================
# Placement-class extraction
# ===========================================================================

# Mapping from concurrent job_name placement labels to baseline placement
# class names.  Sourced from experiments_generator.PLACEMENT_DEFS.
_LABEL_TO_PLACEMENT_CLASS: dict[str, str] = {
    "intra-l1":               "INTRA_L1_RANDOM",
    "intra-group":            "INTRA_GROUP_RANDOM",
    "inter-group":            "INTER_GROUP_RANDOM",
    "intra-group-same-l1-2":  "INTRA_GROUP_SAME_L1_2",
    "intra-group-same-l1-4":  "INTRA_GROUP_SAME_L1_4",
    "inter-group-same-l1-2":  "INTER_GROUP_SAME_L1_2",
    "inter-group-same-l1-4":  "INTER_GROUP_SAME_L1_4",
}


def _extract_placement_class(sbm_tag: str | None) -> str:
    """
    Return the placement class encoded in *sbm_tag*, or ``""`` if absent.

    The tag format mirrors the baseline plotting script::

        ..._class-<placement_class>_rep<N>_...

    Examples
    --------
    >>> _extract_placement_class("ddp_g8_class-spread_rep0")
    'spread'
    >>> _extract_placement_class("ddp_g8_class-pack_tight_rep1")
    'pack_tight'
    >>> _extract_placement_class("ddp_g8_rep0")
    ''
    """
    if not sbm_tag:
        return ""
    m = _CLASS_TAG_RE.search(sbm_tag)
    return m.group(1) if m else ""


def _extract_placement_class_with_fallback(
    sbm_tag: str | None,
    job_name: str | None = None,
) -> str:
    """
    Like :func:`_extract_placement_class`, but when the *sbm_tag* does not
    contain a ``class-<placement>_rep`` pattern, falls back to extracting
    the placement label from *job_name* and mapping it to the canonical
    baseline class name via :data:`_LABEL_TO_PLACEMENT_CLASS`.

    The *job_name* format is ``<strategy>_g<gpus>_n<nodes>_<label>_<uid>``.
    """
    result = _extract_placement_class(sbm_tag)
    if result:
        return result
    if job_name:
        parts = job_name.split('_')
        if len(parts) > 3:
            label = parts[3]
            return _LABEL_TO_PLACEMENT_CLASS.get(label, label)
    return ""


def _parse_job_name(job_name: str) -> tuple[str, str, str, str]:
    """
    Parse ``<strategy>_g<gpus>_n<nodes>_<placement>_<uid>`` into
    ``(strategy, gpus, placement, uid)``.

    Falls back gracefully when the name does not match the expected format.
    """
    parts = job_name.split('_')
    strategy  = parts[0] if len(parts) > 0 else ""
    gpus      = parts[1] if len(parts) > 1 else ""
    placement = parts[3] if len(parts) > 3 else ""
    uid       = "_".join(parts[4:]) if len(parts) > 4 else ""
    return strategy, gpus, placement, uid


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
    for row in rows[:50]:
        print(pad + "  ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))
    print('...')
    print()


def _print_baseline_table(
    baseline: dict[tuple[str, int, str], float],
    metric: str,
    counts: dict[tuple[str, int, str], int],
) -> None:
    """Print a summary table of all T0 baseline values."""
    if not baseline:
        print("  (no baseline entries)\n")
        return
    rows = [
        [strategy, str(gpus), placement or "(none)", f"{t0:.4f}",
         str(counts.get((strategy, gpus, placement), "?"))]
        for (strategy, gpus, placement), t0 in sorted(baseline.items())
    ]
    _table(
        rows,
        header=["Strategy", "GPUs", "Placement", f"T0 ({metric})", "Runs averaged"],
        title=" Baseline T0 summary ",
    )


def _print_concurrent_job_stats(
    sbm_job_id: str,
    tag: str,
    entries: list[tuple[dict, pd.DataFrame]],
    metric: str,
    agg: str,
    slowdowns: dict[str, dict[int, float]] | None,
    baseline: dict[tuple[str, int, str], float] | None,
) -> None:
    """
    Print two tables for one sbatchman job.

    Table 1 — per (job_name, repetition) row showing strategy/gpus/placement,
               metric stats, T0, σ_j.
    Table 2 — per job_name summary (σ aggregated over repetitions).
    """
    agg_fn = _agg_fn(agg)

    detail_rows = []
    for meta, df in sorted(entries, key=lambda t: (t[0]["job_name"], t[0]["repetition"])):
        jname     = meta["job_name"]
        rep       = meta["repetition"]
        strategy  = meta.get("strategy") or jname.split('_')[0]
        gpus      = meta.get("gpus") or int(jname.split('_')[1][1:])
        placement = _extract_placement_class_with_fallback(
            meta.get("sbm_tag", ""), jname)

        if metric in df.columns:
            col    = pd.to_numeric(df[metric], errors="coerce").dropna()
            n      = len(col)
            mean_v = col.mean()
            std_v  = col.std(ddof=0)
            agg_v  = agg_fn(col)
        else:
            n = mean_v = std_v = agg_v = float("nan")

        t0 = float("nan")
        if baseline is not None:
            t0_val = _get_baseline_with_fallback(baseline, strategy, int(gpus), placement)
            t0 = t0_val if t0_val is not None else float("nan")

        sigma = (
            slowdowns[jname][rep]
            if slowdowns and jname in slowdowns and rep in slowdowns[jname]
            else float("nan")
        )

        detail_rows.append([
            jname,
            str(rep),
            f"{strategy}/{gpus}g/{placement or '—'}",
            str(n),
            f"{mean_v:.4f}" if not np.isnan(mean_v) else "—",
            f"{std_v:.4f}"  if not np.isnan(std_v)  else "—",
            f"{agg_v:.4f}"  if not np.isnan(agg_v)  else "—",
            f"{t0:.4f}"     if not np.isnan(t0)      else "—",
            f"{sigma:.4f}"  if not np.isnan(sigma)   else "—",
        ])

    _table(
        detail_rows,
        header=["job_name", "rep", "strategy/gpus/placement", "n_rows",
                f"{metric} mean", f"{metric} std", f"{metric} ({agg})",
                "T0", "σ_j"],
        title=f" sbm_job={sbm_job_id} / tag={tag} — per-repetition detail ",
    )

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
# Baseline fallback helper
# ===========================================================================

# Module-level set to track which missing (strategy, gpus, placement) triples
# have already triggered a warning, so each distinct missing key only produces
# one warning per script run.
_baseline_fallback_warned: set[tuple[str, int, str]] = set()


def _get_baseline_with_fallback(
    baseline:  dict[tuple[str, int, str], float],
    strategy:  str,
    gpus:      int,
    placement: str,
    _warned:   set[tuple[str, int, str]] | None = None,
) -> float | None:
    """
    Look up T0 for (strategy, gpus, placement).

    Resolution order
    ----------------
    1. Exact match  (strategy, gpus, placement)
    2. If not found: warn once per missing key, then return the T0 for the
       alphabetically-first available placement with the same (strategy, gpus).
    3. If still not found (no (strategy, gpus) match at all): return None.
       The caller is responsible for skipping the job and printing a final
       "excluded from metrics" notice.

    Parameters
    ----------
    _warned:
        External set for deduplicating warnings.  When None the module-level
        ``_baseline_fallback_warned`` set is used.
    """
    warned = _warned if _warned is not None else _baseline_fallback_warned

    exact = (strategy, gpus, placement)
    if exact in baseline:
        return baseline[exact]

    candidates = {
        k: v for k, v in baseline.items()
        if k[0] == strategy and k[1] == gpus
    }
    if not candidates:
        return None

    fallback_key = min(candidates.keys(), key=lambda k: k[2])

    if exact not in warned:
        available = sorted(k[2] for k in candidates)
        print(
            f"  [baseline WARNING] no T0 for "
            f"(strategy={strategy!r}, gpus={gpus}, placement={placement!r}). "
            f"Available placements for this strategy/gpu: {available}. "
            f"Falling back to placement={fallback_key[2]!r}."
        )
        warned.add(exact)

    return candidates[fallback_key]


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
        df = dfs.get("main")
        if df is None or df.empty:
            continue
        groups[str(meta["sbm_job_id"])].append((meta, df))

    for jid in groups:
        groups[jid].sort(key=lambda t: (t[0]["job_name"], t[0]["repetition"]))

    return groups


# FIXME TUNE exclude_first_n
def _build_baseline_table_from_mapping(
    mapping: list[tuple[dict, dict[str, pd.DataFrame]]],
    metric: str,
    agg: str,
    exclude_first_n: int = 1,
    cyclic_warmup: bool = True,
    variance_warn_threshold: float = 0.05,
) -> tuple[dict[tuple[str, int, str], float], dict[tuple[str, int, str], int]]:
    """
    Build { (strategy, gpus, placement_class) -> T0 } from an already-loaded mapping.

    The placement class is extracted from each run's ``sbm_tag`` using the same
    ``class-<placement>_rep`` regex as the baseline plotting script.

    Multiple runs with the same key are averaged together.

    Returns (baseline, counts) where counts[key] is how many rows
    contributed to that T0 value.
    """
    agg_fn      = _agg_fn(agg)
    accumulator: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    counts:      dict[tuple[str, int, str], int]         = defaultdict(int)

    for meta, dfs in mapping:
        df = dfs.get("main")
        if df is None or df.empty:
            continue

        strategy  = meta.get("strategy")
        gpus      = meta.get("gpus")
        uid       = meta.get("sbm_tag", "")
        placement = _extract_placement_class(uid)

        if not strategy and uid:
            strategy = uid.split('_')[0]
        if not gpus and uid:
            try:
                gpus = int(uid.split('_')[1][1:])
            except Exception:
                gpus = None

        current_metric = metric
        if current_metric not in df.columns:
            current_metric = current_metric.split('_')[0]

        if current_metric not in df.columns:
            print(
                f"  [baseline] metric '{metric}' not found in job "
                f"{meta.get('sbm_job_id')} "
                f"({strategy=} / {gpus=} / placement={placement!r}), skipping."
            )
            continue

        if strategy is None or gpus is None:
            print(f"  [baseline] missing 'strategy' or 'gpus' in meta {meta}, skipping.")
            continue

        col   = pd.to_numeric(df[current_metric], errors="coerce").dropna()
        nruns = command_map._STRATEGIES_NUM_RUNS[strategy]
        nruns = nruns[0] + nruns[1]

        if col.empty:
            continue

        # Warmup removal
        if cyclic_warmup:
            kept_idx  = []
            total_len = len(col)
            for start in range(0, total_len, int(nruns)):
                chunk_idx = list(range(start, min(start + int(nruns), total_len)))
                if len(chunk_idx) > exclude_first_n:
                    kept_idx.extend(chunk_idx[exclude_first_n:])
                else:
                    print(
                        f"  [baseline] cannot remove {exclude_first_n} warmup runs for chunk "
                        f"starting at {start} "
                        f"({strategy=} / {gpus=} / placement={placement!r} / {nruns=}), "
                        f"not enough values."
                    )
            col = col.iloc[sorted(kept_idx)]
        else:
            if len(col) > exclude_first_n:
                col = col.iloc[exclude_first_n:]
            else:
                print(
                    f"  [baseline] cannot remove {exclude_first_n} warmup runs for "
                    f"{strategy=} / {gpus=} / placement={placement!r} / {nruns=}, "
                    f"not enough values."
                )

        if col.empty:
            continue

        # Variance check
        if len(col) > 1:
            mean = col.mean()
            std  = col.std()
            if mean != 0:
                rel_std = std / mean
                if rel_std > variance_warn_threshold:
                    print(
                        f"  [warning] high variance after warmup removal "
                        f"({rel_std:.2%}) for "
                        f"{strategy=} / {gpus=} / placement={placement!r} / {uid=}"
                    )

        key = (strategy, int(gpus), placement)
        counts[key] += len(col)
        accumulator[key].append(agg_fn(col))

    baseline: dict[tuple[str, int, str], float] = {}
    for key, vals in accumulator.items():
        baseline[key] = float(np.mean(vals))

    return baseline, counts


def _build_baseline_table(
    path: Path,
    metric: str,
    agg: str,
) -> tuple[dict[tuple[str, int, str], float], dict[tuple[str, int, str], int]]:
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
    baseline: dict[tuple[str, int, str], float],
    metric:   str,
    agg:      str,
    cyclic_warmup:           bool  = True,
    exclude_first_n:         int   = 0,
    variance_warn_threshold: float = 0.05,
) -> dict[str, dict[int, float]] | None:
    """
    Returns { job_name -> { repetition -> σ_j } } or None if data is missing.

    σ_j = T0(strategy, gpus, placement_class) / T(job | C)

    The placement class is extracted from each run's ``sbm_tag`` using the same
    regex as the baseline script.  Missing exact keys fall back via
    ``_get_baseline_with_fallback`` — one warning per unique missing key.
    """
    agg_fn      = _agg_fn(agg)
    result:      dict[str, dict[int, float]] = defaultdict(dict)
    no_baseline: set[tuple[str, int, str]]   = set()

    for meta, df in entries:
        jname     = meta["job_name"]
        rep       = meta["repetition"]
        strategy  = meta.get("strategy")
        gpus      = meta.get("gpus")
        nruns     = meta.get("nruns")
        placement = _extract_placement_class_with_fallback(
            meta.get("sbm_tag", ""), jname)

        if not strategy:
            strategy = jname.split('_')[0]
        if not gpus:
            try:
                gpus = int(jname.split('_')[1][1:])
            except Exception:
                gpus = None
        if not nruns:
            try:
                nruns = int(jname.split('_')[2][1:])
            except Exception:
                nruns = None

        if strategy is None or gpus is None:
            print(
                f"  [slowdown] missing 'strategy'/'gpus' for job {jname}, rep {rep} — skipping."
            )
            continue

        t0 = _get_baseline_with_fallback(baseline, strategy, int(gpus), placement)
        if t0 is None:
            no_baseline.add((strategy, int(gpus), placement))
            continue

        current_metric = metric
        if current_metric not in df.columns:
            current_metric = current_metric.split('_')[0]
        if current_metric not in df.columns:
            continue

        col = pd.to_numeric(df[current_metric], errors="coerce").dropna()
        if col.empty:
            continue

        # Warmup removal
        if cyclic_warmup and nruns:
            kept_idx  = []
            total_len = len(col)
            for start in range(0, total_len, int(nruns)):
                chunk_idx = list(range(start, min(start + int(nruns), total_len)))
                if len(chunk_idx) > exclude_first_n:
                    kept_idx.extend(chunk_idx[exclude_first_n:])
                else:
                    print(
                        f"  [slowdown] cannot remove {exclude_first_n} warmup runs for chunk "
                        f"starting at {start} "
                        f"({strategy=} / {gpus=} / placement={placement!r} / {nruns=}), "
                        f"not enough values."
                    )
            col = col.iloc[sorted(kept_idx)]
        else:
            if len(col) > exclude_first_n:
                col = col.iloc[exclude_first_n:]
            else:
                print(
                    f"  [slowdown] cannot remove {exclude_first_n} warmup runs for {jname}, "
                    f"not enough values."
                )

        if col.empty:
            continue

        # Variance check
        if len(col) > 1:
            mean = col.mean()
            std  = col.std()
            if mean != 0:
                rel_std = std / mean
                if rel_std > variance_warn_threshold:
                    print(
                        f"  [slowdown WARNING] high variance after warmup removal "
                        f"({rel_std:.2%}) for "
                        f"{jname=} / {strategy=} / {gpus=} / "
                        f"placement={placement!r} / {nruns=} / {rep=}"
                    )

        t_conc = agg_fn(col)
        if t_conc <= 0:
            continue

        result[jname][rep] = t0 / t_conc

    for key in no_baseline:
        print(
            f"  [slowdown] WARNING: no baseline (even via fallback) for "
            f"(strategy={key[0]!r}, gpus={key[1]}, placement={key[2]!r}) "
            f"— those jobs excluded from interference metrics."
        )

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
      'per_job_std'  : { job_name -> std_σ over reps }
      'mean'         : σ̄
      'worst_case'   : σ_max
      'per_strategy' : { strategy -> mean_σ }
      'per_group'    : { (strategy, gpus, placement) -> {'mean': float, 'std': float,
                          'job_names': list[str]} }
    """
    job_meta: dict[str, tuple[str, int, str]] = {}
    for meta, _ in entries:
        jname     = meta["job_name"]
        strategy  = meta.get("strategy", "unknown")
        gpus      = int(meta.get("gpus", 0))
        placement = _extract_placement_class_with_fallback(
            meta.get("sbm_tag", ""), jname)
        job_meta[jname] = (strategy, gpus, placement)

    per_job: dict[str, float] = {
        jname: float(np.mean(list(reps.values())))
        for jname, reps in slowdowns.items() if reps
    }
    per_job_std: dict[str, float] = {
        jname: float(np.std(list(reps.values()), ddof=0)) if len(reps) > 1 else 0.0
        for jname, reps in slowdowns.items() if reps
    }
    if not per_job:
        return {}

    mean_sigma = float(np.mean(list(per_job.values())))
    worst_case = float(max(per_job.values()))

    by_strategy: dict[str, list[float]] = defaultdict(list)
    for jname, sigma in per_job.items():
        by_strategy[job_meta.get(jname, ("unknown", 0, ""))[0]].append(sigma)
    per_strategy = {s: float(np.mean(vs)) for s, vs in by_strategy.items()}

    # Group by (strategy, gpus, placement) — jobs sharing those three attrs
    # get the same colour in the bar chart and are summarised together.
    by_group: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for jname, sigma in per_job.items():
        key = job_meta.get(jname, ("unknown", 0, ""))
        by_group[key].append(sigma)

    per_group: dict[tuple[str, int, str], dict] = {}
    for gkey, vals in by_group.items():
        per_group[gkey] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0,
            "job_names": [jn for jn in per_job if job_meta.get(jn) == gkey],
        }

    return {
        "per_job":      per_job,
        "per_job_std":  per_job_std,
        "mean":         mean_sigma,
        "worst_case":   worst_case,
        "per_strategy": per_strategy,
        "per_group":    per_group,
    }


# ===========================================================================
# Slowdown distribution helpers
# ===========================================================================

def _compute_adaptive_bins(
    all_pct: list[float],
) -> list[tuple[float, float, str]]:
    """
    Compute adaptive histogram bins from a list of percentage slowdown values
    (i.e. (σ - 1) * 100).

    Strategy
    --------
    * Start with the range of the data.
    * Always include a "no-slowdown" bin straddling 0 (negative = speedup).
    * Use a fixed set of candidate breakpoints and keep those that divide
      the data into non-trivially-small bins (>=5 % of total).
    * Return a list of (lo_pct, hi_pct, label) tuples.

    The labels use "%" notation with a leading sign so they read naturally
    on a bar-chart axis, e.g. ``"[-5 %, +5 %)"`` or ``">= 20 %"``.
    """
    if not all_pct:
        return [(-5.0, 5.0, "~0 %"), (5.0, float("inf"), "> 5 %")]

    lo = min(all_pct)
    hi = max(all_pct)
    n  = len(all_pct)

    # Candidate breakpoints (percentage slowdown).
    candidate_breaks = [-50, -20, -10, -5, 0, 2, 5, 10, 15, 20, 30, 50, 100, 200]
    # Filter to range that covers the data, with a little headroom.
    breaks = [b for b in candidate_breaks if lo - 5 <= b <= hi + 5]

    # Always have -inf as left edge and +inf as right edge.
    edges: list[float] = sorted(set(breaks))
    if not edges or edges[0] > lo:
        edges = [lo - 1] + edges
    if not edges or edges[-1] < hi:
        edges = edges + [hi + 1]

    # Merge bins that are too small (< 5 % of total runs).
    min_count = max(1, int(0.05 * n))
    merged: list[float] = [edges[0]]
    for e in edges[1:]:
        count_in_new = sum(1 for v in all_pct if merged[-1] <= v < e)
        if count_in_new >= min_count or e == edges[-1]:
            merged.append(e)
        # else: skip this edge — merge with next

    # Extend last bin to +inf and first bin to -inf
    bins: list[tuple[float, float, str]] = []
    for i in range(len(merged) - 1):
        lo_b = merged[i]
        hi_b = merged[i + 1]

        if i == 0:
            lo_label = f"< {hi_b:+.0f} %"
        else:
            lo_label = f"{lo_b:+.0f} %"

        if i == len(merged) - 2:
            hi_label = f">= {lo_b:+.0f} %"
            label = hi_label
        else:
            label = f"[{lo_b:+.0f}, {hi_b:+.0f} %)"

        bins.append((
            -float("inf") if i == 0 else lo_b,
            float("inf")  if i == len(merged) - 2 else hi_b,
            label,
        ))

    return bins if bins else [(-float("inf"), float("inf"), "all")]


def _build_distribution_panel(
    ax: plt.Axes,
    all_pct: list[float],
    title: str = "Slowdown distribution",
) -> None:
    """
    Draw an adaptive histogram of slowdown percentages on *ax*.

    Each bar is labelled with its count and percentage of total runs.
    Bars are coloured green (speedup / negligible) -> red (heavy slowdown).
    """
    bins = _compute_adaptive_bins(all_pct)
    n    = len(all_pct)

    counts = []
    labels = []
    for lo_b, hi_b, label in bins:
        cnt = sum(1 for v in all_pct if lo_b <= v < hi_b)
        counts.append(cnt)
        labels.append(label)

    # Colour: map bin midpoint to a green-yellow-red ramp.
    def _bin_colour(lo_b: float, hi_b: float) -> str:
        mid = (
            (lo_b + hi_b) / 2
            if not np.isinf(lo_b) and not np.isinf(hi_b)
            else (hi_b if np.isinf(lo_b) else lo_b)
        )
        # Clamp to [-20, +50] for colour mapping.
        t = np.clip((mid + 20) / 70, 0, 1)   # 0 = green (-20 %), 1 = red (+50 %)
        r = t
        g = 1 - t * 0.8
        return (r, g, 0.2)

    colours = [_bin_colour(lo_b, hi_b) for lo_b, hi_b, _ in bins]

    x_pos = np.arange(len(counts))
    bars  = ax.bar(x_pos, counts, color=colours, edgecolor="black", linewidth=0.6)

    for bar, cnt in zip(bars, counts):
        if cnt == 0:
            continue
        pct_of_total = 100.0 * cnt / n if n else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{cnt}\n({pct_of_total:.0f}%)",
            ha="center", va="bottom", fontsize=7,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("# runs", fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)


# ===========================================================================
# Visual encoding helpers
# ===========================================================================

def _build_job_visuals(
    job_names: list[str],
) -> dict[str, tuple[str, str, str]]:
    """
    Return ``{ job_name -> (color, marker, linestyle) }``.

    Jobs that share the same (strategy, gpus, placement) triple get the same
    colour; their marker / linestyle varies by uid index.
    """
    color_key_order: dict[tuple[str, str, str], int] = {}
    uid_order:       dict[str, int]                  = {}

    for jname in sorted(job_names):
        strategy, gpus, placement, uid = _parse_job_name(jname)
        ck = (strategy, gpus, placement)
        if ck not in color_key_order:
            color_key_order[ck] = len(color_key_order)
        if uid not in uid_order:
            uid_order[uid] = len(uid_order)

    visuals: dict[str, tuple[str, str, str]] = {}
    for jname in job_names:
        strategy, gpus, placement, uid = _parse_job_name(jname)
        ck        = (strategy, gpus, placement)
        color     = _COLOR_POOL[color_key_order[ck] % len(_COLOR_POOL)]
        uid_idx   = uid_order[uid]
        marker    = _MARKER_POOL[uid_idx % len(_MARKER_POOL)]
        linestyle = _LINESTYLE_POOL[uid_idx % len(_LINESTYLE_POOL)]
        visuals[jname] = (color, marker, linestyle)
    return visuals


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
    n_jobs    = len(job_names)
    visuals   = _build_job_visuals(job_names)

    present_metrics = [
        m for m in METRICS
        if any(m in stats for reps in agg.values() for stats in reps.values())
    ]
    if not present_metrics:
        print(f"  [skip] {tag}: no recognised metrics")
        return None

    n_metrics = len(present_metrics)

    legend_ncol  = min(8, n_jobs)
    legend_rows  = (n_jobs + legend_ncol - 1) // legend_ncol
    legend_h_in  = max(0.7, legend_rows * 0.30 + 0.4)
    subplot_h_in = 4.0
    fig_h        = subplot_h_in + legend_h_in

    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(max(6 * n_metrics, 8), fig_h),
        squeeze=False,
    )
    fig.suptitle(
        f"Concurrent run performance\nsbm_job={sbm_job_id}  tag={tag}",
        fontsize=10, y=1.01,
    )

    legend_handles: list = []
    legend_labels:  list[str] = []

    for col_idx, metric in enumerate(present_metrics):
        ax = axes[0][col_idx]
        for jname in job_names:
            reps_dict  = agg[jname]
            xs_present = sorted(r for r in reps_dict if metric in reps_dict[r])
            if not xs_present:
                continue
            ys    = np.array([reps_dict[r][metric][0] for r in xs_present])
            yerrs = np.array([reps_dict[r][metric][1] for r in xs_present])
            color, marker, linestyle = visuals[jname]
            line, = ax.plot(
                xs_present, ys,
                marker=marker, linestyle=linestyle,
                linewidth=1.5, markersize=5,
                color=color, label=jname,
            )
            ax.fill_between(xs_present, ys - yerrs, ys + yerrs, alpha=0.12, color=color)

            if col_idx == 0:
                legend_handles.append(line)
                legend_labels.append(jname)

        ax.set_xlabel("Repetition", fontsize=9)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9)
        ax.set_title(metric, fontsize=9)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(True, linewidth=0.4, alpha=0.6)

    legend_frac = legend_h_in / fig_h
    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=legend_ncol,
            fontsize=8,
            title="job",
            title_fontsize=8,
            frameon=True,
            borderaxespad=0.3,
        )

    fig.tight_layout()
    fig.subplots_adjust(bottom=legend_frac + 0.02)
    out_path = OUT_DIR / f"{tag}_performance.png"
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
    n_jobs    = len(job_names)
    visuals   = _build_job_visuals(job_names)

    legend_ncol = min(8, n_jobs)
    legend_rows = (n_jobs + legend_ncol - 1) // legend_ncol
    legend_h_in = max(0.7, legend_rows * 0.30 + 0.4)
    fig_h       = 4.0 + legend_h_in

    fig, ax = plt.subplots(figsize=(max(7, n_jobs * 0.35 + 4), fig_h))
    fig.suptitle(f"Slowdown timeline\nsbm_job={sbm_job_id}  tag={tag}", fontsize=10)

    handles: list = []
    labels:  list[str] = []
    for jname in job_names:
        reps  = slowdowns[jname]
        xs    = sorted(reps.keys())
        ys    = [reps[r] for r in xs]
        color, marker, linestyle = visuals[jname]
        line, = ax.plot(
            xs, ys,
            marker=marker, linestyle=linestyle,
            linewidth=1.5, markersize=5,
            color=color, label=jname,
        )
        handles.append(line)
        labels.append(jname)

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", label="No slowdown (sigma=1)")
    ax.set_xlabel("Repetition", fontsize=9)
    ax.set_ylabel("Slowdown  sigma_j  (higher = worse)", fontsize=9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, linewidth=0.4, alpha=0.6)

    legend_frac = legend_h_in / fig_h
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=legend_ncol,
        fontsize=8,
        title="job",
        title_fontsize=8,
        frameon=True,
        borderaxespad=0.3,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=legend_frac + 0.02)
    out_path = OUT_DIR / f"{tag}_slowdown_timeline.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_slowdown_summary(
    sbm_job_id:  str,
    tag:         str,
    agg_metrics: dict,
    slowdowns:   dict[str, dict[int, float]] | None = None,
) -> Path:
    """
    Two-panel interference summary:

    Panel 1 (left)  — Per-job bar chart.
        * Bars coloured by (strategy, gpus, placement) group.
        * Error bars show std of sigma over repetitions.
        * Bar label inside the bar, shown as signed percentage (+12 %, -2 %).

    Panel 2 (right) — Adaptive slowdown-percentage histogram with grouped bars.
        * Bins are computed from all per-repetition slowdown values.
        * Within each bin, one bar per (strategy, gpus, placement) group,
          coloured consistently with panel 1.
        * Each bar counts the repetitions belonging to that group that fall
          in the bin.
    """
    per_job     = agg_metrics["per_job"]
    per_job_std = agg_metrics.get("per_job_std", {})
    job_names   = sorted(per_job.keys())
    n_jobs      = len(job_names)

    # ------------------------------------------------------------------ #
    # Derive group ordering and colours — stable across both panels       #
    # ------------------------------------------------------------------ #
    group_keys_ordered: list[tuple[str, str, str]] = []
    for jn in sorted(job_names):
        strategy, gpus, placement, _ = _parse_job_name(jn)
        ck = (strategy, gpus, placement)
        if ck not in group_keys_ordered:
            group_keys_ordered.append(ck)

    group_color: dict[tuple, str] = {
        ck: _COLOR_POOL[i % len(_COLOR_POOL)]
        for i, ck in enumerate(group_keys_ordered)
    }

    def _group_key(jname: str) -> tuple[str, str, str]:
        s, g, p, _ = _parse_job_name(jname)
        return (s, g, p)

    # ------------------------------------------------------------------ #
    # Build per-group per-rep slowdown pct lists for panel 2             #
    # { group_key -> [pct_value, ...] }                                   #
    # ------------------------------------------------------------------ #
    group_pct: dict[tuple, list[float]] = defaultdict(list)
    all_pct:   list[float] = []

    if slowdowns:
        for jname, reps in slowdowns.items():
            gk = _group_key(jname)
            for sigma in reps.values():
                pct = (sigma - 1.0) * 100.0
                group_pct[gk].append(pct)
                all_pct.append(pct)
    elif per_job:
        for jname, sigma in per_job.items():
            gk  = _group_key(jname)
            pct = (sigma - 1.0) * 100.0
            group_pct[gk].append(pct)
            all_pct.append(pct)

    # ------------------------------------------------------------------ #
    # Figure layout: 1 x 2                                                #
    # ------------------------------------------------------------------ #
    fig_w = max(10, n_jobs * 0.5 + 7)
    fig, (ax_jobs, ax_hist) = plt.subplots(2, 1, figsize=(fig_w, 18))
    fig.suptitle(f"Interference summary\nsbm_job={sbm_job_id}  tag={tag}", fontsize=11)

    # ==================================================================== #
    # Panel 1 — per-job bars                                                #
    # ==================================================================== #
    bar_colors = [group_color.get(_group_key(jn), "#888888") for jn in job_names]
    job_means  = [per_job[jn]               for jn in job_names]
    job_stds   = [per_job_std.get(jn, 0.0) for jn in job_names]

    bars = ax_jobs.bar(
        range(n_jobs),
        job_means,
        yerr=job_stds,
        color=bar_colors,
        edgecolor="black", linewidth=0.5,
        capsize=3, error_kw={"elinewidth": 1.0, "ecolor": "black"},
    )
    ax_jobs.axhline(1.0, color="black", linewidth=0.8, linestyle="--", label="sigma = 1")

    for bar, mean_v in zip(bars, job_means):
        pct       = (mean_v - 1.0) * 100.0
        sign      = "+" if pct >= 0 else ""
        label_txt = f"{sign}{pct:.0f}%"
        bar_h     = bar.get_height()
        y_ref     = ax_jobs.get_ylim()[0]
        label_y   = y_ref + 0.60 * (bar_h - y_ref) if bar_h > y_ref else bar_h * 0.5
        ax_jobs.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            label_txt,
            ha="center", va="center",
            fontsize=max(6, 9 - n_jobs // 12),
            color="white",
            fontweight="bold",
        )

    ax_jobs.set_xticks(range(n_jobs))
    ax_jobs.set_xticklabels(
        job_names, rotation=35, ha="right",
        fontsize=max(14, 8 - n_jobs // 10),
    )
    ax_jobs.set_ylabel("Mean slowdown  sigma_j", fontsize=18)
    ax_jobs.set_title("Per-job slowdown  (error bar = std over repetitions)", fontsize=18)
    ax_jobs.legend(fontsize=12)
    ax_jobs.grid(True, axis="y", linewidth=0.5, alpha=0.8)

    # ==================================================================== #
    # Panel 2 — adaptive histogram with one grouped bar per group per bin  #
    # ==================================================================== #
    if all_pct:
        bins          = _compute_adaptive_bins(all_pct)
        n_bins        = len(bins)
        active_groups = [gk for gk in group_keys_ordered if group_pct.get(gk)]
        n_active      = len(active_groups)

        total_width = 0.8   # fraction of one bin slot used by all bars together
        bar_width   = total_width / max(n_active, 1)
        offsets     = np.linspace(
            -total_width / 2 + bar_width / 2,
            total_width / 2 - bar_width / 2,
            n_active,
        )

        bin_centers    = np.arange(n_bins)
        legend_handles: list = []
        legend_labels:  list[str] = []

        for g_idx, gk in enumerate(active_groups):
            pct_vals = group_pct[gk]
            color    = group_color.get(gk, "#888888")
            counts   = [
                sum(1 for v in pct_vals if lo_b <= v < hi_b)
                for lo_b, hi_b, _ in bins
            ]

            x_pos    = bin_centers + offsets[g_idx]
            grp_bars = ax_hist.bar(
                x_pos,
                counts,
                width=bar_width * 0.92,   # tiny gap between adjacent group bars
                color=color,
                edgecolor="black", linewidth=0.4,
                label=f"{gk[0]} / {gk[1]} / {gk[2] or '-'}",
            )
            legend_handles.append(grp_bars[0])
            legend_labels.append(f"{gk[0]} / {gk[1]} / {gk[2] or '-'}")

            # Count label above each non-zero bar
            for bar, cnt in zip(grp_bars, counts):
                if cnt == 0:
                    continue
                ax_hist.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.15,
                    str(cnt),
                    ha="center", va="bottom",
                    fontsize=max(14, 8 - n_active),
                )
                total = len(pct_vals)
                pct_val = (cnt / total * 100.0) if total > 0 else 0.0
                ax_hist.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    f"{pct_val:.0f}%",
                    ha="center", va="center",
                    fontsize=max(14, 8 - n_active),
                    color="white",
                    fontweight="bold",
                )

        bin_labels = [label for _, _, label in bins]
        ax_hist.set_xticks(bin_centers)
        ax_hist.set_xticklabels(bin_labels, rotation=30, ha="right", fontsize=18)
        ax_hist.set_ylabel("# repetitions", fontsize=20)
        ax_hist.set_title(
            "Slowdown distribution — grouped by (strategy / GPUs / placement)",
            fontsize=18,
        )
        ax_hist.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax_hist.grid(True, axis="y", linewidth=0.6, alpha=0.8)

        if legend_handles:
            ax_hist.legend(
                legend_handles, legend_labels,
                fontsize=14,
                loc="upper right",
            )
    else:
        ax_hist.set_visible(False)

    fig.tight_layout()
    out_path = OUT_DIR / f"{tag}_slowdown_summary.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_slowdown_heatmap(
    groups:   dict[str, list[tuple[dict, pd.DataFrame]]],
    baseline: dict[tuple[str, int, str], float],
    metric:   str,
    agg:      str,
) -> Path | None:
    rows: list[dict] = []
    row_tags: list[str] = []

    for sbm_job_id, entries in sorted(groups.items()):
        slowdowns = _compute_slowdowns(entries, baseline, metric, agg)
        if not slowdowns:
            continue
        row_tags.append(sbm_job_id)
        rows.append({jn: float(np.mean(list(reps.values()))) for jn, reps in slowdowns.items()})

    if not rows:
        return None

    df      = pd.DataFrame(rows, index=row_tags).sort_index(axis=1).fillna(np.nan)
    n_cols  = len(df.columns)
    n_rows  = len(df.index)

    fig_w = max(8, n_cols * 1.0 + 2)
    fig_h = max(4, n_rows * 0.55 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im  = ax.imshow(df.values, aspect="auto", cmap="RdYlGn_r", vmin=1.0)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(df.columns, rotation=45, ha="right",
                       fontsize=max(6, 10 - n_cols // 8))
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(df.index, fontsize=max(6, 10 - n_rows // 8))
    ax.set_xlabel("Job name (slot)", fontsize=9)
    ax.set_ylabel("sbm_job_id", fontsize=9)
    ax.set_title("Slowdown heatmap  (sigma_j, mean over repetitions)", fontsize=10)
    fig.colorbar(im, ax=ax, pad=0.01).set_label("sigma_j", fontsize=8)

    cell_fontsize = max(5, 10 - n_cols // 8)
    for r in range(n_rows):
        for c in range(n_cols):
            v = df.values[r, c]
            if not np.isnan(v):
                ax.text(c, r, f"{(v*100)-100:.1f}%", ha="center", va="center",
                        fontsize=cell_fontsize, color="black")

    fig.tight_layout()
    out_path = OUT_DIR / "all_jobs_slowdown_heatmap.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ===========================================================================
# Placement visualisation
# ===========================================================================

def _plot_placements(
    sbm_job_id: str,
    tag:        str,
    entries:    list[tuple[dict, pd.DataFrame]],
) -> list[Path]:
    """
    Create one SVG for the first repetition showing the node placement of each job.
    """
    first_rep   = min(meta["repetition"] for meta, _ in entries)
    rep_entries = [(meta, df) for meta, df in entries if meta["repetition"] == first_rep]

    jobs:   dict[str, list[str]] = {}
    system: str | None = None

    for meta, _ in rep_entries:
        jname   = meta["job_name"]
        jnodes  = meta.get("resources") or meta.get("nodelist") or meta.get("nodes") or ""
        jsystem = meta.get("cluster") or ""

        if jsystem and system is None:
            system = jsystem
        if jnodes:
            node_list = [n.strip() for n in jnodes.replace(",", " ").split() if n.strip()]
            jobs[jname] = node_list

    if not system:
        print(f"  [placements] {tag} rep={first_rep}: 'cluster' missing in metadata — skipping.")
        return []
    if not jobs:
        print(f"  [placements] {tag} rep={first_rep}: no nodelist data found — skipping.")
        return []

    try:
        base = Path(__file__).parent.parent / "common" / "JobPlacer"
        kwargs: dict = dict(
            system        = system,
            topology_file = str(base / f"{system}_topo.txt"),
            sinfo_file    = str(base / f"{system}_sinfo.txt"),
            verbose       = False,
        )
        toml_path = base / "systems" / f"{system.upper()}.toml"
        if toml_path.exists():
            kwargs["topology_toml_file"] = str(toml_path)

        placer = JobPlacer(**kwargs)
    except Exception as exc:
        print(f"  [placements] {tag} rep={first_rep}: failed to create JobPlacer for "
              f"system '{system}' — {exc}")
        return []

    out_path = OUT_DIR / f"{tag}_rep{first_rep}_placement.svg"
    try:
        placer.visualize(jobs=jobs, out_svg=out_path)
        print(f"  [placements] {tag} rep={first_rep}: saved -> {out_path}")
        return [out_path]
    except Exception as exc:
        print(f"  [placements] {tag} rep={first_rep}: visualize() failed — {exc}")
        return []


# ===========================================================================
# Summary printer
# ===========================================================================

def _print_summary(
    tag:       str,
    entries:   list[tuple[dict, pd.DataFrame]],
    perf_path: Path | None,
    slow_path: Path | None = None,
    summ_path: Path | None = None,
    plac_paths: list[Path] | None = None,
) -> None:
    job_names  = sorted({e[0]["job_name"] for e in entries})
    rep_counts = {
        jn: len({e[0]["repetition"] for e in entries if e[0]["job_name"] == jn})
        for jn in job_names
    }
    total_rows = sum(len(df) for _, df in entries)
    parts = [
        f"perf->{perf_path}"     if perf_path  else None,
        f"timeline->{slow_path}" if slow_path  else None,
        f"summary->{summ_path}"  if summ_path  else None,
    ]
    if plac_paths:
        parts.append(f"placements({len(plac_paths)})->{OUT_DIR}")
    outs = " | ".join(filter(None, parts))
    print(
        f"  {tag:<40}  "
        f"jobs={len(job_names)}  "
        f"reps/job=[{', '.join(f'{jn}:{n}' for jn, n in rep_counts.items())}]  "
        f"rows={total_rows}  {outs or '-> skipped'}"
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
    parser.add_argument(
        "--show-placements", action="store_true", default=False,
        help=(
            "Produce one SVG per repetition showing node placements, "
            "using JobPlacer.  Requires 'system' and 'nodelist' fields "
            "in the run metadata."
        ),
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
    baseline: dict[tuple[str, int, str], float] | None = None
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
            print(
                f"\nBuilding T0 table from {len(baseline_mapping)} run(s)  "
                f"[metric={metric}, agg={agg}]"
            )
            baseline, counts = _build_baseline_table_from_mapping(baseline_mapping, metric, agg)
            _print_baseline_table(baseline, metric, counts)

    # ------------------------------------------------------------------
    # Per-sbatchman-job plots + debug output
    # ------------------------------------------------------------------
    print(f"Generating plots -> {OUT_DIR}/\n")
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
                    summ_path = _plot_slowdown_summary(
                        sbm_job_id, tag, agg_metrics, slowdowns=slowdowns
                    )

        plac_paths: list[Path] = []
        if args.show_placements:
            plac_paths = _plot_placements(sbm_job_id, tag, entries)

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
                    f"sigma_bar={agg_metrics['mean']:.3f}  "
                    f"sigma_max={agg_metrics['worst_case']:.3f}  "
                    f"per-strategy: {strat_str}\n"
                )

        _print_summary(tag, entries, perf_path, slow_path, summ_path, plac_paths)
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