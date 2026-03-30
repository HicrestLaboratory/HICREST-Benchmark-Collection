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
computed from per-job slowdown  σ_j = T0(strategy, model, gpus, placement) / T(job | C):

  - Mean slowdown          σ̄  = mean(σ_j)
  - Worst-case slowdown    σ_max = max(σ_j)
  - Per-strategy histogram : one bar-plot bin per strategy, aggregated
                             with the chosen aggregate function

The metric and its aggregation function are fully configurable via
SLOWDOWN_METRIC and SLOWDOWN_AGG at the top of this file, or at runtime
via --metric and --agg.

Baseline lookup key
--------------------
Baselines are keyed by RunKey(strategy, model, gpus, placement_class) where
placement_class is extracted from the ``sbm_tag`` field using the same
``class-<placement>_rep`` regex that the baseline plotting script uses.

If a concurrent run's exact RunKey is not found in the baseline table, the
script emits **one** warning per missing key and falls back to the nearest
available baseline for the same (strategy, model, gpus) triple (if any
exists).  When no such match exists at all, those jobs are excluded from
interference metrics and a warning is printed.

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
from typing import Callable, NamedTuple

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
# ===========================================================================

# Which column in dfs['main'] to use as the throughput proxy.
# Must be a higher-is-better metric (slowdown = T0 / T).
SLOWDOWN_METRIC: str = "throughput_mean"

# How to reduce per-rank/iteration rows to a single scalar for one run.
# Options: "mean" | "median" | "max" | "min"
SLOWDOWN_AGG: str = "mean"


# ===========================================================================
# ❷  RUN IDENTITY
# ===========================================================================

class RunKey(NamedTuple):
    """Canonical identifier for a training run configuration."""
    strategy:  str
    model:     str
    gpus:      int
    placement: str

    def display(self) -> str:
        """Human-readable label used in plot titles and table columns."""
        return f"{self.strategy}/{self.model}/{self.gpus}g/{self.placement or '—'}"

    def short(self) -> str:
        """Compact label used in legend entries and bar-chart x-ticks."""
        return f"{self.strategy}\n{self.model}\n{self.gpus}g\n{self.placement or '—'}"


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
_MARKER_POOL:    list[str] = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p", "H", "8"]
_LINESTYLE_POOL: list[str] = [
    "-", "--", "-.", ":",
    (0, (3, 1, 1, 1)),
    (0, (5, 1)),
    (0, (1, 1)),
    (0, (3, 1, 1, 1, 1, 1)),
]


# ===========================================================================
# Placement-class extraction
# ===========================================================================

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
    """Return the placement class encoded in *sbm_tag*, or ``""`` if absent."""
    if not sbm_tag:
        return "ERR"
    m = _CLASS_TAG_RE.search(sbm_tag)
    return m.group(1) if m else "na"


# ===========================================================================
# Job-name parsing
# ===========================================================================

class ParsedJobName(NamedTuple):
    strategy:  str
    model:     str
    gpus:      str   # kept as str; callers cast to int when needed
    placement: str
    uid:       str


def _parse_job_name(job_name: str) -> ParsedJobName:
    """
    Parse ``<strategy>_<model>_g<gpus>_n<nodes>_<placement>_<uid>`` into a
    :class:`ParsedJobName`.  Falls back gracefully on short names.
    """
    parts = job_name.split('_')
    return ParsedJobName(
        strategy  = parts[0] if len(parts) > 0 else "",
        model     = parts[1] if len(parts) > 1 else "",
        gpus      = parts[2] if len(parts) > 2 else "",
        placement = parts[4] if len(parts) > 4 else "",
        uid       = "_".join(parts[5:]) if len(parts) > 5 else "",
    )


def _run_key_from_job_name(job_name: str) -> RunKey:
    """Build a :class:`RunKey` directly from a job_name string."""
    p = _parse_job_name(job_name)
    try:
        gpus = int(p.gpus.lstrip("g"))
    except ValueError:
        gpus = 0
    return RunKey(strategy=p.strategy, model=p.model, gpus=gpus, placement=p.placement)


def _run_key_from_meta_baseline(meta: dict) -> RunKey:
    plac = _extract_placement_class(meta['sbm_tag'])
    return RunKey(
        strategy=meta['strategy'],
        model=meta.get('model', command_map.get_default_model(meta['strategy'])),
        gpus=meta['gpus'],
        placement=_LABEL_TO_PLACEMENT_CLASS.get(plac, plac)
    )
    

def _run_key_from_meta(meta: dict) -> RunKey:
    """
    Build a :class:`RunKey` from a run's metadata dict, falling back to
    job_name parsing for any missing fields.
    """    
    jname     = str(meta.get("job_name", ""))
    parsed    = _parse_job_name(jname)
    strategy  = parsed.strategy or meta.get("strategy") or parsed.strategy
    model     = parsed.model or meta.get("model") or command_map.get_default_model(strategy)
    placement = parsed.placement

    raw_gpus = meta.get("gpus") or parsed.gpus
    try:
        gpus = int(str(raw_gpus).lstrip("g"))
    except (ValueError, AttributeError):
        gpus = 0

    return RunKey(strategy=strategy, model=model, gpus=gpus, placement=placement)


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


def _resolve_metric_column(df: pd.DataFrame, metric: str) -> str | None:
    """
    Return the actual column name to use for *metric*.

    Tries the full name first (e.g. ``throughput_mean``), then the prefix
    (e.g. ``throughput``).  Returns ``None`` when neither is present.
    """
    if metric in df.columns:
        return metric
    prefix = metric.split('_')[0]
    if prefix in df.columns:
        return prefix
    return None


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
    baseline: dict[RunKey, float],
    metric:   str,
    counts:   dict[RunKey, int],
) -> None:
    """Print a summary table of all T0 baseline values."""
    if not baseline:
        print("  (no baseline entries)\n")
        return
    rows = [
        [k.strategy, k.model, str(k.gpus), k.placement or "(none)",
         f"{t0:.4f}", str(counts.get(k, "?"))]
        for k, t0 in sorted(baseline.items())
    ]
    _table(
        rows,
        header=["Strategy", "Model", "GPUs", "Placement", f"T0 ({metric})", "Values averaged"],
        title=" Baseline T0 summary ",
    )


def _print_concurrent_job_stats(
    sbm_job_id: str,
    tag:        str,
    entries:    list[tuple[dict, pd.DataFrame]],
    metric:     str,
    agg:        str,
    slowdowns:  dict[str, dict[int, float]] | None,
    baseline:   dict[RunKey, float] | None,
) -> None:
    """Print per-repetition detail and per-job σ summary tables."""
    agg_fn = _agg_fn(agg)

    detail_rows = []
    for meta, df in sorted(entries, key=lambda t: (t[0]["job_name"], t[0]["repetition"])):
        jname = meta["job_name"]
        rep   = meta["repetition"]
        rkey  = _run_key_from_meta(meta)

        col_name = _resolve_metric_column(df, metric)
        if col_name is not None:
            col    = pd.to_numeric(df[col_name], errors="coerce").dropna()
            n      = len(col)
            mean_v = col.mean()
            std_v  = col.std(ddof=0)
            agg_v  = agg_fn(col)
        else:
            n = mean_v = std_v = agg_v = float("nan")

        t0 = float("nan")
        if baseline is not None:
            t0_val = _get_baseline_with_fallback(baseline, rkey)
            t0     = t0_val if t0_val is not None else float("nan")

        sigma = (
            slowdowns[jname][rep]
            if slowdowns and jname in slowdowns and rep in slowdowns[jname]
            else float("nan")
        )

        detail_rows.append([
            jname, str(rep), rkey.display(),
            str(n),
            f"{mean_v:.4f}" if not np.isnan(mean_v) else "—",
            f"{std_v:.4f}"  if not np.isnan(std_v)  else "—",
            f"{agg_v:.4f}"  if not np.isnan(agg_v)  else "—",
            f"{t0:.4f}"     if not np.isnan(t0)      else "—",
            f"{sigma:.4f}"  if not np.isnan(sigma)   else "—",
        ])

    _table(
        detail_rows,
        header=["job_name", "rep", "strategy/model/gpus/placement", "n_rows",
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
                jname, str(len(vals)),
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

_baseline_fallback_warned: set[RunKey] = set()


def _get_baseline_with_fallback(
    baseline: dict[RunKey, float],
    key:      RunKey,
    _warned:  set[RunKey] | None = None,
) -> float | None:
    """
    Look up T0 for *key*.

    Resolution order
    ----------------
    1. Exact match on the full RunKey.
    2. Warn once, then fall back to the alphabetically-first available
       placement for the same (strategy, model, gpus).
    3. Return None when no (strategy, model, gpus) match exists at all.
    """
    warned = _warned if _warned is not None else _baseline_fallback_warned

    # Normalise "na" placement to empty string
    if key.placement.lower() == 'na':
        key = key._replace(placement='')

    if key in baseline:
        return baseline[key]

    candidates = {
        k: v for k, v in baseline.items()
        if k.strategy == key.strategy and k.model == key.model and k.gpus == key.gpus
    }
    if not candidates:
        return None

    fallback_key = min(candidates.keys(), key=lambda k: k.placement)

    if key not in warned:
        available = sorted(k.placement for k in candidates)
        print(
            f"  [baseline WARNING] no T0 for {key!r}. "
            f"Available placements for (strategy={key.strategy!r}, model={key.model!r}, "
            f"gpus={key.gpus}): {available}. "
            f"Falling back to placement={fallback_key.placement!r}."
        )
        warned.add(key)

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


# ===========================================================================
# Baseline table construction
# ===========================================================================

# FIXME TUNE exclude_first_n
def _build_baseline_table_from_mapping(
    mapping:                  list[tuple[dict, dict[str, pd.DataFrame]]],
    metric:                   str,
    agg:                      str,
    exclude_first_n:          int   = 1,
    cyclic_warmup:            bool  = True,
    variance_warn_threshold:  float = 0.05,
) -> tuple[dict[RunKey, float], dict[RunKey, int]]:
    """
    Build ``{ RunKey -> T0 }`` from an already-loaded mapping.

    Multiple runs with the same key are kept as the max value (best baseline).

    Returns ``(baseline, counts)`` where ``counts[key]`` is how many data
    rows contributed to that T0 value.
    """
    agg_fn      = _agg_fn(agg)
    accumulator: dict[RunKey, list[float]] = defaultdict(list)
    counts:      dict[RunKey, int]         = defaultdict(int)

    for meta, dfs in mapping:
        df = dfs.get("main")
        if df is None or df.empty:
            continue

        rkey = _run_key_from_meta_baseline(meta)
        
        if not rkey.strategy or not rkey.gpus:
            print(f"  [baseline] missing 'strategy' or 'gpus' in meta {meta}, skipping.")
            continue

        col_name = _resolve_metric_column(df, metric)
        if col_name is None:
            print(
                f"  [baseline] metric '{metric}' not found in job "
                f"{meta.get('sbm_job_id')} ({rkey!r}), skipping."
            )
            continue

        col   = pd.to_numeric(df[col_name], errors="coerce").dropna()
        nruns = command_map._STRATEGIES_NUM_RUNS[rkey.strategy]
        nruns = nruns[0] + nruns[1]

        if col.empty:
            continue

        col = _remove_warmup(col, nruns, exclude_first_n, cyclic_warmup, label=str(rkey))

        if col.empty:
            continue

        _warn_high_variance(col, variance_warn_threshold, label=str(rkey))

        counts[rkey] += len(col)
        accumulator[rkey].append(agg_fn(col))

    baseline: dict[RunKey, float] = {}
    for key, vals in accumulator.items():
        if len(vals) > 1:
            print(f"[WARN] Multiple baseline candidates for {key}:")
            for i, v in enumerate(vals):
                print(f"  - candidate[{i}] = {v:.6f}")
            best = max(vals)
            print(f"  -> keeping max value {best:.6f}")
        baseline[key] = float(max(vals))

    return baseline, counts


# ===========================================================================
# Warmup / variance helpers  (extracted for reuse)
# ===========================================================================

def _remove_warmup(
    col:             pd.Series,
    nruns:           int,
    exclude_first_n: int,
    cyclic:          bool,
    label:           str = "",
) -> pd.Series:
    """
    Return *col* with the first *exclude_first_n* rows of each cycle removed.

    When *cyclic* is False a single leading slice is removed instead.
    """
    if cyclic:
        kept_idx: list[int] = []
        total_len = len(col)
        for start in range(0, total_len, int(nruns)):
            chunk_idx = list(range(start, min(start + int(nruns), total_len)))
            if len(chunk_idx) > exclude_first_n:
                kept_idx.extend(chunk_idx[exclude_first_n:])
            else:
                print(
                    f"  [warmup] cannot remove {exclude_first_n} warmup run(s) "
                    f"for chunk starting at {start} ({label}), not enough values."
                )
        return col.iloc[sorted(kept_idx)]
    else:
        if len(col) > exclude_first_n:
            return col.iloc[exclude_first_n:]
        print(
            f"  [warmup] cannot remove {exclude_first_n} warmup run(s) "
            f"for {label}, not enough values."
        )
        return col


def _warn_high_variance(
    col:       pd.Series,
    threshold: float,
    label:     str = "",
) -> None:
    """Print a warning when the relative standard deviation of *col* exceeds *threshold*."""
    if len(col) <= 1:
        return
    mean = col.mean()
    if mean == 0:
        return
    rel_std = col.std() / mean
    if rel_std > threshold:
        print(
            f"  [variance WARNING] high relative std ({rel_std:.2%}) for {label}"
        )


# ===========================================================================
# Per-run aggregation for concurrent data
# ===========================================================================

def _aggregate_runs(
    entries: list[tuple[dict, pd.DataFrame]],
) -> dict[str, dict[int, dict[str, tuple[float, float]]]]:
    """
    Returns ``{ job_name -> { repetition -> { metric -> (mean, std) } } }``.
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
    entries:                 list[tuple[dict, pd.DataFrame]],
    baseline:                dict[RunKey, float],
    metric:                  str,
    agg:                     str,
    cyclic_warmup:           bool  = True,
    exclude_first_n:         int   = 0,
    variance_warn_threshold: float = 0.05,
) -> dict[str, dict[int, float]] | None:
    """
    Returns ``{ job_name -> { repetition -> σ_j } }`` or ``None`` if no data.

    ``σ_j = T0(RunKey) / T(job | concurrent)``
    """
    agg_fn      = _agg_fn(agg)
    result:      dict[str, dict[int, float]]  = defaultdict(dict)
    no_baseline: set[RunKey]                  = set()

    for meta, df in entries:
        jname = meta["job_name"]
        rep   = meta["repetition"]
        rkey  = _run_key_from_meta(meta)
        # nruns = meta.get("nruns")
        
        # print(meta)
        # print(jname)
        # print(rep)
        # print(rkey)
        # print()
        
        # if not nruns:
        #     try:
        #         nruns = int(jname.split('_')[3][1:])
        #     except Exception:
        #         nruns = None

        if not rkey.strategy or not rkey.gpus:
            print(f"  [slowdown] missing strategy/gpus for job {jname}, rep {rep} — skipping.")
            continue

        t0 = _get_baseline_with_fallback(baseline, rkey)
        if t0 is None:
            no_baseline.add(rkey)
            continue

        col_name = _resolve_metric_column(df, metric)
        if col_name is None:
            continue

        col = pd.to_numeric(df[col_name], errors="coerce").dropna()
        if col.empty:
            continue

        # HERE WE DO NOT HAVE WARMUP (already excluded)
        # if nruns:
        #     col = _remove_warmup(
        #         col, int(nruns), exclude_first_n, cyclic_warmup,
        #         label=f"{jname} rep={rep}",
        #     )
        # elif len(col) > exclude_first_n:
        #     col = col.iloc[exclude_first_n:]

        if col.empty:
            continue

        _warn_high_variance(col, variance_warn_threshold, label=f"{jname} rep={rep} {rkey!r}")

        t_conc = agg_fn(col)
        if t_conc <= 0:
            continue

        result[jname][rep] = t0 / t_conc

    for key in no_baseline:
        print(
            f"  [slowdown] WARNING: no baseline (even via fallback) for {key!r} "
            f"— those jobs excluded from interference metrics."
        )
    
    return result if result else None


# ===========================================================================
# Aggregate slowdown metrics
# ===========================================================================

def _aggregate_slowdowns(
    slowdowns: dict[str, dict[int, float]],
    entries:   list[tuple[dict, pd.DataFrame]],
) -> dict:
    """
    Returns a dict with keys:
      ``per_job``      → { job_name -> mean_σ over reps }
      ``per_job_std``  → { job_name -> std_σ over reps }
      ``mean``         → σ̄
      ``worst_case``   → σ_max
      ``per_strategy`` → { strategy -> mean_σ }
      ``per_group``    → { RunKey -> {'mean', 'std', 'job_names'} }
    """
    job_rkey: dict[str, RunKey] = {
        meta["job_name"]: _run_key_from_meta(meta)
        for meta, _ in entries
    }

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
        by_strategy[job_rkey[jname].strategy].append(sigma)
    per_strategy = {s: float(np.mean(vs)) for s, vs in by_strategy.items()}

    by_group: dict[RunKey, list[float]] = defaultdict(list)
    for jname, sigma in per_job.items():
        by_group[job_rkey[jname]].append(sigma)

    per_group: dict[RunKey, dict] = {
        rk: {
            "mean":      float(np.mean(vals)),
            "std":       float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0,
            "job_names": [jn for jn in per_job if job_rkey.get(jn) == rk],
        }
        for rk, vals in by_group.items()
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
    (i.e. ``(σ - 1) * 100``).
    """
    if not all_pct:
        return [(-5.0, 5.0, "~0 %"), (5.0, float("inf"), "> 5 %")]

    lo = min(all_pct)
    hi = max(all_pct)
    n  = len(all_pct)

    candidate_breaks = [-50, -20, -10, -5, 0, 2, 5, 10, 15, 20, 30, 50, 100, 200]
    breaks = [b for b in candidate_breaks if lo - 5 <= b <= hi + 5]

    edges: list[float] = sorted(set(breaks))
    if not edges or edges[0] > lo:
        edges = [lo - 1] + edges
    if not edges or edges[-1] < hi:
        edges = edges + [hi + 1]

    min_count = max(1, int(0.05 * n))
    merged: list[float] = [edges[0]]
    for e in edges[1:]:
        count_in_new = sum(1 for v in all_pct if merged[-1] <= v < e)
        if count_in_new >= min_count or e == edges[-1]:
            merged.append(e)

    bins: list[tuple[float, float, str]] = []
    for i in range(len(merged) - 1):
        lo_b = merged[i]
        hi_b = merged[i + 1]
        if i == len(merged) - 2:
            label = f">= {lo_b:+.0f} %"
        else:
            label = f"[{lo_b:+.0f}, {hi_b:+.0f} %)"

        bins.append((
            -float("inf") if i == 0 else lo_b,
            float("inf")  if i == len(merged) - 2 else hi_b,
            label,
        ))

    return bins if bins else [(-float("inf"), float("inf"), "all")]


def _build_distribution_panel(
    ax:     plt.Axes,
    all_pct: list[float],
    title:  str = "Slowdown distribution",
) -> None:
    """Draw an adaptive histogram of slowdown percentages on *ax*."""
    bins = _compute_adaptive_bins(all_pct)
    n    = len(all_pct)

    counts = [
        sum(1 for v in all_pct if lo_b <= v < hi_b)
        for lo_b, hi_b, _ in bins
    ]
    labels = [label for _, _, label in bins]

    def _bin_colour(lo_b: float, hi_b: float) -> tuple:
        mid = (
            (lo_b + hi_b) / 2
            if not np.isinf(lo_b) and not np.isinf(hi_b)
            else (hi_b if np.isinf(lo_b) else lo_b)
        )
        t = np.clip((mid + 20) / 70, 0, 1)
        return (t, 1 - t * 0.8, 0.2)

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

    Jobs that share the same :class:`RunKey` (minus uid) get the same colour;
    their marker / linestyle varies by uid index.
    """
    color_key_order: dict[RunKey, int] = {}
    uid_order:       dict[str, int]    = {}

    for jname in sorted(job_names):
        p  = _parse_job_name(jname)
        rk = RunKey(p.strategy, p.model, int(p.gpus.lstrip("g") or 0), p.placement)
        if rk not in color_key_order:
            color_key_order[rk] = len(color_key_order)
        if p.uid not in uid_order:
            uid_order[p.uid] = len(uid_order)

    visuals: dict[str, tuple[str, str, str]] = {}
    for jname in job_names:
        p   = _parse_job_name(jname)
        rk  = RunKey(p.strategy, p.model, int(p.gpus.lstrip("g") or 0), p.placement)
        color     = _COLOR_POOL[color_key_order[rk] % len(_COLOR_POOL)]
        uid_idx   = uid_order[p.uid]
        marker    = _MARKER_POOL[uid_idx % len(_MARKER_POOL)]
        linestyle = _LINESTYLE_POOL[uid_idx % len(_LINESTYLE_POOL)]
        visuals[jname] = (color, marker, linestyle)
    return visuals


# ===========================================================================
# Standard performance plot
# ===========================================================================

def _plot_performance(
    sbm_job_id: str,
    tag:        str,
    entries:    list[tuple[dict, pd.DataFrame]],
) -> Path | None:
    agg_data  = _aggregate_runs(entries)
    if not agg_data:
        print(f"  [skip] {tag}: no aggregatable data")
        return None

    job_names = sorted(agg_data.keys())
    n_jobs    = len(job_names)
    visuals   = _build_job_visuals(job_names)

    present_metrics = [
        m for m in METRICS
        if any(m in stats for reps in agg_data.values() for stats in reps.values())
    ]
    if not present_metrics:
        print(f"  [skip] {tag}: no recognised metrics")
        return None

    n_metrics    = len(present_metrics)
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
            reps_dict  = agg_data[jname]
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
    job_names   = sorted(slowdowns.keys())
    n_jobs      = len(job_names)
    visuals     = _build_job_visuals(job_names)
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

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", label="No slowdown (σ=1)")
    ax.set_xlabel("Repetition", fontsize=9)
    ax.set_ylabel("Slowdown  σ_j  (higher = worse)", fontsize=9)
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
    Two-panel interference summary (stacked vertically):

    Panel 1 — Per-job bar chart with error bars and signed-% labels.
    Panel 2 — Adaptive slowdown-% histogram, grouped by RunKey.
    """
    per_job     = agg_metrics["per_job"]
    per_job_std = agg_metrics.get("per_job_std", {})
    job_names   = sorted(per_job.keys())
    n_jobs      = len(job_names)

    # Stable group ordering and colours
    group_keys_ordered: list[RunKey] = []
    for jn in sorted(job_names):
        rk = _run_key_from_job_name(jn)
        if rk not in group_keys_ordered:
            group_keys_ordered.append(rk)

    group_color: dict[RunKey, str] = {
        rk: _COLOR_POOL[i % len(_COLOR_POOL)]
        for i, rk in enumerate(group_keys_ordered)
    }

    # Per-group per-rep pct lists for panel 2
    group_pct: dict[RunKey, list[float]] = defaultdict(list)
    all_pct:   list[float] = []

    if slowdowns:
        for jname, reps in slowdowns.items():
            rk = _run_key_from_job_name(jname)
            for sigma in reps.values():
                pct = (sigma - 1.0) * 100.0
                group_pct[rk].append(pct)
                all_pct.append(pct)
    elif per_job:
        for jname, sigma in per_job.items():
            rk  = _run_key_from_job_name(jname)
            pct = (sigma - 1.0) * 100.0
            group_pct[rk].append(pct)
            all_pct.append(pct)

    fig_w = max(10, n_jobs * 0.5 + 7)
    fig, (ax_jobs, ax_hist) = plt.subplots(2, 1, figsize=(fig_w, 18))
    fig.suptitle(f"Interference summary\nsbm_job={sbm_job_id}  tag={tag}", fontsize=11)

    # --- Panel 1: per-job bars ---
    bar_colors = [group_color.get(_run_key_from_job_name(jn), "#888888") for jn in job_names]
    job_means  = [per_job[jn]               for jn in job_names]
    job_stds   = [per_job_std.get(jn, 0.0) for jn in job_names]

    bars = ax_jobs.bar(
        range(n_jobs), job_means,
        yerr=job_stds,
        color=bar_colors, edgecolor="black", linewidth=0.5,
        capsize=3, error_kw={"elinewidth": 1.0, "ecolor": "black"},
    )
    ax_jobs.axhline(1.0, color="black", linewidth=0.8, linestyle="--", label="σ = 1")

    for bar, mean_v in zip(bars, job_means):
        pct       = (mean_v - 1.0) * 100.0
        sign      = "+" if pct >= 0 else ""
        label_txt = f"{sign}{pct:.0f}%"
        bar_h     = bar.get_height()
        y_ref     = ax_jobs.get_ylim()[0]
        label_y   = y_ref + 0.60 * (bar_h - y_ref) if bar_h > y_ref else bar_h * 0.5
        ax_jobs.text(
            bar.get_x() + bar.get_width() / 2, label_y,
            label_txt,
            ha="center", va="center",
            fontsize=max(6, 9 - n_jobs // 12),
            color="white", fontweight="bold",
        )

    ax_jobs.set_xticks(range(n_jobs))
    ax_jobs.set_xticklabels(
        job_names, rotation=35, ha="right",
        fontsize=max(14, 8 - n_jobs // 10),
    )
    ax_jobs.set_ylabel("Mean slowdown  σ_j", fontsize=18)
    ax_jobs.set_title("Per-job slowdown  (error bar = std over repetitions)", fontsize=18)
    ax_jobs.legend(fontsize=12)
    ax_jobs.grid(True, axis="y", linewidth=0.5, alpha=0.8)

    # --- Panel 2: adaptive histogram grouped by RunKey ---
    if all_pct:
        bins          = _compute_adaptive_bins(all_pct)
        n_bins        = len(bins)
        active_groups = [rk for rk in group_keys_ordered if group_pct.get(rk)]
        n_active      = len(active_groups)

        total_width = 0.8
        bar_width   = total_width / max(n_active, 1)
        offsets     = np.linspace(
            -total_width / 2 + bar_width / 2,
            total_width / 2 - bar_width / 2,
            n_active,
        )

        bin_centers    = np.arange(n_bins)
        legend_handles: list = []
        legend_labels:  list[str] = []

        for g_idx, rk in enumerate(active_groups):
            pct_vals = group_pct[rk]
            color    = group_color.get(rk, "#888888")
            counts   = [
                sum(1 for v in pct_vals if lo_b <= v < hi_b)
                for lo_b, hi_b, _ in bins
            ]

            x_pos    = bin_centers + offsets[g_idx]
            grp_bars = ax_hist.bar(
                x_pos, counts,
                width=bar_width * 0.92,
                color=color, edgecolor="black", linewidth=0.4,
                label=rk.display(),
            )
            legend_handles.append(grp_bars[0])
            legend_labels.append(rk.display())

            total = len(pct_vals)
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
                pct_val = (cnt / total * 100.0) if total > 0 else 0.0
                ax_hist.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    f"{pct_val:.0f}%",
                    ha="center", va="center",
                    fontsize=max(14, 8 - n_active),
                    color="white", fontweight="bold",
                )

        bin_labels = [label for _, _, label in bins]
        ax_hist.set_xticks(bin_centers)
        ax_hist.set_xticklabels(bin_labels, rotation=30, ha="right", fontsize=18)
        ax_hist.set_ylabel("# repetitions", fontsize=20)
        ax_hist.set_title(
            "Slowdown distribution — grouped by (strategy / model / GPUs / placement)",
            fontsize=18,
        )
        ax_hist.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax_hist.grid(True, axis="y", linewidth=0.6, alpha=0.8)

        if legend_handles:
            ax_hist.legend(legend_handles, legend_labels, fontsize=14, loc="upper right")
    else:
        ax_hist.set_visible(False)

    fig.tight_layout()
    out_path = OUT_DIR / f"{tag}_slowdown_summary.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ===========================================================================
# Cross-experiment analysis plots
# ===========================================================================

SLOWDOWN_THRESHOLD_PCT: float = 5.0
CAP_THRESHOLD_PCT:      float = 200.0


def _collect_cross_experiment_data(
    groups:   dict[str, list[tuple[dict, pd.DataFrame]]],
    baseline: dict[RunKey, float],
    metric:   str,
    agg:      str,
    cap:      float | None = None,
) -> dict[RunKey, list[float]]:
    """
    Aggregate all per-repetition σ values across every sbatchman job,
    keyed by :class:`RunKey`.

    Returns ``{ RunKey -> [sigma_pct, ...] }`` where
    ``sigma_pct = (σ - 1) * 100``.  Values above *cap* are dropped (with a
    warning).
    """
    result:   dict[RunKey, list[float]] = defaultdict(list)
    n_capped: dict[RunKey, int]         = defaultdict(int)

    for sbm_job_id, entries in groups.items():
        slowdowns = _compute_slowdowns(entries, baseline, metric, agg)
        if not slowdowns:
            continue
        for jname, reps in slowdowns.items():
            rk = _run_key_from_job_name(jname)
            for sigma in reps.values():
                pct = (sigma - 1.0) * 100.0
                if cap is not None and pct > cap:
                    n_capped[rk] += 1
                else:
                    result[rk].append(pct)

    if n_capped:
        total_removed = sum(n_capped.values())
        details = "  ".join(
            f"{rk.display()}:{n}" for rk, n in sorted(n_capped.items())
        )
        print(
            f"  [cap] removed {total_removed} outlier value(s) "
            f"with slowdown_pct > {cap:.1f}%:  {details}"
        )

    return dict(result)


def _plot_ecdf_by_placement(
    groups:    dict[str, list[tuple[dict, pd.DataFrame]]],
    baseline:  dict[RunKey, float],
    metric:    str,
    agg:       str,
    threshold: float = SLOWDOWN_THRESHOLD_PCT,
    cap:       float | None = None,
) -> Path | None:
    """
    Empirical CDF of slowdown percentage, faceted by placement class.

    One subplot per placement; one eCDF curve per (strategy, model, gpus) triple.
    """
    cross = _collect_cross_experiment_data(groups, baseline, metric, agg, cap=cap)
    if not cross:
        return None

    placements = sorted({rk.placement for rk in cross})
    # (strategy, model, gpus) combos for colour/marker assignment
    smg_pairs: list[tuple[str, str, int]] = sorted({
        (rk.strategy, rk.model, rk.gpus) for rk in cross
    })

    if not placements:
        return None

    smg_color:  dict[tuple, str] = {
        smg: _COLOR_POOL[i % len(_COLOR_POOL)] for i, smg in enumerate(smg_pairs)
    }
    smg_marker: dict[tuple, str] = {
        smg: _MARKER_POOL[i % len(_MARKER_POOL)] for i, smg in enumerate(smg_pairs)
    }

    n_placements = len(placements)
    n_cols       = min(3, n_placements)
    n_rows       = (n_placements + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7 * n_cols, 5 * n_rows),
        squeeze=False,
    )
    fig.suptitle(
        f"eCDF of slowdown (>= {threshold:+.0f}%) — faceted by placement\n"
        f"one curve per (strategy, model, GPUs)",
        fontsize=13,
    )

    legend_handles: list = []
    legend_labels:  list[str] = []
    legend_built = False

    for p_idx, placement in enumerate(placements):
        row = p_idx // n_cols
        col = p_idx % n_cols
        ax  = axes[row][col]

        ax.axvline(threshold, color="grey", linewidth=1.0, linestyle=":",
                   label=f"threshold ({threshold:+.0f}%)")
        ax.axhline(1.0, color="lightgrey", linewidth=0.6, linestyle="--")

        has_data = False
        for smg in smg_pairs:
            rk       = RunKey(strategy=smg[0], model=smg[1], gpus=smg[2], placement=placement)
            pct_all  = cross.get(rk, [])
            pct_filt = [v for v in pct_all if v >= threshold]
            if not pct_filt:
                continue

            has_data = True
            xs = sorted(pct_filt)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            xs = [threshold] + xs
            ys = np.concatenate([[0.0], ys])

            color  = smg_color[smg]
            marker = smg_marker[smg]
            line, = ax.plot(
                xs, ys,
                color=color, marker=marker,
                markevery=max(1, len(xs) // 10),
                markersize=4, linewidth=1.6,
                label=f"{smg[0]} / {smg[1]} / {smg[2]}g",
            )
            if not legend_built:
                legend_handles.append(line)
                legend_labels.append(f"{smg[0]} / {smg[1]} / {smg[2]}g")

        if not has_data:
            ax.text(0.5, 0.5, "no data above threshold",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="grey")

        legend_built = True
        ax.set_title(f"placement: {placement or '(none)'}", fontsize=11)
        ax.set_xlabel("Slowdown (%)", fontsize=10)
        ax.set_ylabel("Empirical CDF", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linewidth=0.4, alpha=0.6)
        ax.tick_params(labelsize=9)

    for p_idx in range(n_placements, n_rows * n_cols):
        axes[p_idx // n_cols][p_idx % n_cols].set_visible(False)

    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=min(6, len(legend_handles)),
            fontsize=10,
            title="strategy / model / GPUs",
            title_fontsize=10,
            frameon=True,
        )

    fig.tight_layout()
    fig.subplots_adjust(bottom=max(0.08, 0.04 * ((len(legend_handles) + 5) // 6)))
    out_path = OUT_DIR / "cross_experiment_ecdf_by_placement.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_exceedance(
    groups:    dict[str, list[tuple[dict, pd.DataFrame]]],
    baseline:  dict[RunKey, float],
    metric:    str,
    agg:       str,
    threshold: float = SLOWDOWN_THRESHOLD_PCT,
    cap:       float | None = None,
) -> Path | None:
    """
    Two-panel exceedance summary across all experiments.

    Panel 1 — Exceedance rate: fraction of reps with slowdown_pct >= threshold.
    Panel 2 — Conditional mean excess: mean slowdown of exceeding reps only.

    Groups are sorted by placement then (strategy, model, gpus) within each
    placement block.
    """
    cross = _collect_cross_experiment_data(groups, baseline, metric, agg, cap=cap)
    if not cross:
        return None

    # Sort: primary = placement, secondary = (strategy, model, gpus)
    all_rkeys = sorted(cross.keys(), key=lambda rk: (rk.placement, rk.strategy, rk.model, rk.gpus))
    if not all_rkeys:
        return None

    smg_pairs = sorted({(rk.strategy, rk.model, rk.gpus) for rk in cross})
    smg_color: dict[tuple, str] = {
        smg: _COLOR_POOL[i % len(_COLOR_POOL)] for i, smg in enumerate(smg_pairs)
    }

    exceedance_rate: list[float] = []
    mean_excess:     list[float] = []
    bar_colors:      list[str]   = []
    x_labels:        list[str]   = []

    for rk in all_rkeys:
        pct_all      = cross[rk]
        pct_exceeded = [v for v in pct_all if v >= threshold]
        rate         = len(pct_exceeded) / len(pct_all) if pct_all else 0.0
        excess       = float(np.mean(pct_exceeded)) if pct_exceeded else float("nan")

        exceedance_rate.append(rate * 100.0)
        mean_excess.append(excess)
        bar_colors.append(smg_color.get((rk.strategy, rk.model, rk.gpus), "#888888"))
        x_labels.append(rk.short())

    n_groups = len(all_rkeys)
    x_pos    = np.arange(n_groups)

    fig, (ax_rate, ax_excess) = plt.subplots(
        2, 1, figsize=(max(10, n_groups * 0.9 + 3), 12)
    )
    fig.suptitle(
        f"Cross-experiment exceedance analysis  (threshold >= {threshold:+.0f}%)\n"
        f"groups sorted by placement",
        fontsize=13,
    )

    # --- Panel 1 ---
    bars1 = ax_rate.bar(
        x_pos, exceedance_rate,
        color=bar_colors, edgecolor="black", linewidth=0.5,
    )
    ax_rate.set_ylabel("Exceedance rate (%)", fontsize=13)
    ax_rate.set_title(f"Fraction of reps with slowdown >= {threshold:+.0f}%", fontsize=12)
    ax_rate.set_xticks(x_pos)
    ax_rate.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=9)
    ax_rate.set_ylim(0, 110)
    ax_rate.grid(True, axis="y", linewidth=0.5, alpha=0.7)

    for bar, val in zip(bars1, exceedance_rate):
        if val == 0:
            continue
        ax_rate.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.0f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # Placement block shading (helper extracted below)
    placement_blocks: dict[str, list[int]] = defaultdict(list)
    for i, rk in enumerate(all_rkeys):
        placement_blocks[rk.placement].append(i)

    _shade_placement_blocks(ax_rate, placement_blocks, y_label=107)

    # --- Panel 2 ---
    mean_excess_plot = [v if not np.isnan(v) else 0.0 for v in mean_excess]
    bars2 = ax_excess.bar(
        x_pos, mean_excess_plot,
        color=bar_colors, edgecolor="black", linewidth=0.5,
    )
    ax_excess.axhline(threshold, color="grey", linewidth=1.0, linestyle=":",
                      label=f"threshold ({threshold:+.0f}%)")
    ax_excess.set_ylabel("Mean slowdown of exceeding reps (%)", fontsize=13)
    ax_excess.set_title("Conditional mean excess  (only reps above threshold)", fontsize=12)
    ax_excess.set_xticks(x_pos)
    ax_excess.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=9)
    ax_excess.grid(True, axis="y", linewidth=0.5, alpha=0.7)
    ax_excess.legend(fontsize=10)

    for bar, val, raw in zip(bars2, mean_excess_plot, mean_excess):
        if np.isnan(raw) or raw == 0:
            ax_excess.text(
                bar.get_x() + bar.get_width() / 2,
                threshold * 0.3,
                "none",
                ha="center", va="bottom", fontsize=8, color="grey",
            )
            continue
        ax_excess.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    y_sep2 = ax_excess.get_ylim()[1] * 0.97 if ax_excess.get_ylim()[1] > 0 else threshold * 2
    _shade_placement_blocks(ax_excess, placement_blocks, y_label=y_sep2, label_va="top")

    # Shared colour legend for (strategy, model, gpus)
    sg_handles = [plt.Rectangle((0, 0), 1, 1, color=smg_color[smg]) for smg in smg_pairs]
    sg_labels  = [f"{smg[0]} / {smg[1]} / {smg[2]}g" for smg in smg_pairs]
    fig.legend(
        sg_handles, sg_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(15, len(sg_handles)),
        fontsize=10,
        frameon=True,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=max(0.08, 0.04 * ((len(sg_handles) + 5) // 6)))
    out_path = OUT_DIR / "cross_experiment_exceedance.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _shade_placement_blocks(
    ax:               plt.Axes,
    placement_blocks: dict[str, list[int]],
    y_label:          float,
    label_va:         str = "bottom",
) -> None:
    """Draw shaded placement-block bands and labels on *ax*."""
    for plac, indices in sorted(placement_blocks.items(), key=lambda kv: kv[1][0]):
        lo = indices[0] - 0.5
        hi = indices[-1] + 0.5
        ax.axvspan(lo, hi, alpha=0.06, color="steelblue")
        ax.text(
            (lo + hi) / 2, y_label,
            plac or "(none)",
            ha="center", va=label_va, fontsize=8, style="italic", color="steelblue",
        )
        if lo > -0.5:
            ax.axvline(lo, color="steelblue", linewidth=0.8, linestyle="--", alpha=0.5)


def _plot_slowdown_heatmap(
    groups:   dict[str, list[tuple[dict, pd.DataFrame]]],
    baseline: dict[RunKey, float],
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

    df     = pd.DataFrame(rows, index=row_tags).sort_index(axis=1).fillna(np.nan)
    n_cols = len(df.columns)
    n_rows = len(df.index)

    fig_w = max(8, n_cols * 1.0 + 2)
    fig_h = max(4, n_rows * 0.55 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(df.values, aspect="auto", cmap="RdYlGn_r", vmin=1.0)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(df.columns, rotation=45, ha="right",
                       fontsize=max(6, 10 - n_cols // 8))
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(df.index, fontsize=max(6, 10 - n_rows // 8))
    ax.set_xlabel("Job name (slot)", fontsize=9)
    ax.set_ylabel("sbm_job_id", fontsize=9)
    ax.set_title("Slowdown heatmap  (σ_j, mean over repetitions)", fontsize=10)
    fig.colorbar(im, ax=ax, pad=0.01).set_label("σ_j", fontsize=8)

    cell_fontsize = max(5, 10 - n_cols // 8)
    for r in range(n_rows):
        for c in range(n_cols):
            v = df.values[r, c]
            if not np.isnan(v):
                ax.text(c, r, f"{(v * 100) - 100:.1f}%",
                        ha="center", va="center",
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
    """Create one SVG for the first repetition showing node placements."""
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
        base   = Path(__file__).parent.parent / "common" / "JobPlacer"
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
        print(f"  [placements] {tag} rep={first_rep}: failed to create JobPlacer — {exc}")
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
    tag:        str,
    entries:    list[tuple[dict, pd.DataFrame]],
    perf_path:  Path | None,
    slow_path:  Path | None      = None,
    summ_path:  Path | None      = None,
    plac_paths: list[Path] | None = None,
) -> None:
    job_names  = sorted({e[0]["job_name"] for e in entries})
    rep_counts = {
        jn: len({e[0]["repetition"] for e in entries if e[0]["job_name"] == jn})
        for jn in job_names
    }
    total_rows = sum(len(df) for _, df in entries)
    parts = [
        f"perf->{perf_path}"     if perf_path else None,
        f"timeline->{slow_path}" if slow_path else None,
        f"summary->{summ_path}"  if summ_path else None,
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
        help="Baseline parquet file(s). Multiple files are merged.",
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
        "--gpu-models", default=None, nargs="+",
        choices=["GB300", "GB200", "B200", "H100", "H200", "A100", "GH200"],
        help=f"GPU models to include (default: ALL available).",
    )
    parser.add_argument(
        "--out-dir", default=str(OUT_DIR),
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--show-placements", action="store_true", default=False,
        help="Produce one SVG per repetition showing node placements via JobPlacer.",
    )
    parser.add_argument(
        "--threshold", type=float, default=SLOWDOWN_THRESHOLD_PCT,
        help=(
            f"Minimum slowdown %% to include in cross-experiment eCDF and "
            f"exceedance plots (default: {SLOWDOWN_THRESHOLD_PCT})."
        ),
    )
    parser.add_argument(
        "--cap", type=float, default=CAP_THRESHOLD_PCT, metavar="PCT",
        help=(
            "Drop cross-experiment slowdown values strictly above this %% before "
            "plotting.  Removed values are reported in a warning."
        ),
    )
    args = parser.parse_args()

    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metric: str = args.metric
    agg:    str = args.agg

    # --- Load concurrent data ---
    mapping_full: list[tuple[dict, dict[str, pd.DataFrame]]] = []
    for p in [Path(f) for f in args.parquet_files]:
        if not p.exists():
            print(f"WARNING: {p} not found, skipping.")
            continue
        chunk = _load_parquet(p)
        mapping_full.extend(chunk)
        print(f"Loaded {len(chunk)} run(s) from {p}")
        
    # Filtering
    mapping: list[tuple[dict, dict[str, pd.DataFrame]]] = []
    if args.gpu_models:
        for meta, dicts in mapping_full:
            gpu_model = str(meta['sbm_tag']).split('_')[-1]
            if gpu_model in args.gpu_models:
                mapping.append((meta, dicts))
    else:
        mapping = mapping_full

    if not mapping:
        print("No data loaded.")
        sys.exit(1)    

    groups = _group_by_sbm_job(mapping)
    print(f"\n{len(groups)} sbatchman job(s) found.")

    # --- Load baseline (optional) ---
    baseline: dict[RunKey, float] | None = None
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
            baseline, counts = _build_baseline_table_from_mapping(
                baseline_mapping, metric, agg
            )
            _print_baseline_table(baseline, metric, counts)

    # --- Per-sbatchman-job plots ---
    print(f"Generating plots -> {OUT_DIR}/\n")
    skipped = 0

    for sbm_job_id, entries in sorted(groups.items()):
        tag = f'{sbm_job_id}_{entries[0][0].get("sbm_tag", sbm_job_id)}'

        # Enrich metadata with model derived from the app command
        for meta, _ in entries:
            meta['model'] = command_map.get_model_from_command(meta['app'])
            name_parts    = str(meta['job_name']).split('_')
            meta['job_name'] = '_'.join([name_parts[0], meta['model']] + name_parts[1:])
            # if 'FSDP' in meta['job_name']:
            #     print(f"{meta['job_name']}   {meta['model']}   {meta['app']}")
                
        # continue

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
                    f"σ̄={agg_metrics['mean']:.3f}  "
                    f"σ_max={agg_metrics['worst_case']:.3f}  "
                    f"per-strategy: {strat_str}\n"
                )

        _print_summary(tag, entries, perf_path, slow_path, summ_path, plac_paths)
        if perf_path is None:
            skipped += 1
            
    
    # --- Cross-job plots ---
    if baseline is not None and len(groups) > 1:
        print("\nGenerating cross-job slowdown heatmap...")
        hmap_path = _plot_slowdown_heatmap(groups, baseline, metric, agg)
        if hmap_path:
            print(f"  Heatmap saved to {hmap_path}")

        print("\nGenerating cross-experiment eCDF by placement...")
        ecdf_path = _plot_ecdf_by_placement(
            groups, baseline, metric, agg,
            threshold=args.threshold, cap=args.cap,
        )
        if ecdf_path:
            print(f"  eCDF plot saved to {ecdf_path}")
        else:
            print("  eCDF plot skipped (no data above threshold).")

        print("\nGenerating cross-experiment exceedance analysis...")
        exc_path = _plot_exceedance(
            groups, baseline, metric, agg,
            threshold=args.threshold, cap=args.cap,
        )
        if exc_path:
            print(f"  Exceedance plot saved to {exc_path}")
        else:
            print("  Exceedance plot skipped (no data above threshold).")

    total = len(groups)
    print(f"\nDone. {total - skipped} plot(s) written, {skipped} skipped.")


if __name__ == "__main__":
    main()