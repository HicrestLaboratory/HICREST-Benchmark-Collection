#!/usr/bin/env python3
"""
plot_baselines.py — Scaling and time-breakdown plots for DLNetBench baselines.

Replaces the old parquet-based workflow: data is read directly from raw
ccutils stdout files and SbatchMan metadata via parse_results.parse_baselines.

Produced plot sets (one per system, saved under --output-dir):
  <output_dir>/
      <sys>_<strategy>_all_scaling[_aggr].png
      <sys>_<strategy>_all_breakdown[_aggr].png
      <sys>_global_all_scaling[_aggr].png
      <sys>_comm_pct_table.tex
  <output_dir>/per_system/
      <sys>_<strategy>_on_<system>_scaling.png
      <sys>_<strategy>_on_<system>_breakdown.png
  <output_dir>/cross_system/   (only when multiple systems are requested)
      <sys+strategy>_xsystem_*.png
"""

import os
import re
import glob
import sys
import yaml
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib

from data_types import PLACEMENT_ORDER, STRATEGY_ORDER, SYSTEM_NAMES_MAP, SYSTEM_ORDER, Model, Placement, Strategy

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def get_model_from_command(_cmd): return None
def get_default_model(_strategy): return "unknown"

# ============================================================================
#  Style helpers  (unchanged from original)
# ============================================================================

# --- utilities ---------------------------------------------------------------
from data_types import ensure_model, ensure_placement, ensure_strategy


# --- placement ---------------------------------------------------------------

def _placement_linestyles(placements: List[Union[str, Placement]]) -> Dict[str, str]:
    ps = {ensure_placement(p) for p in placements}
    return {str(p): p.linestyle() for p in PLACEMENT_ORDER if p in ps}

def _placement_markers(placements: List[Union[str, Placement]]) -> Dict[str, str]:
    ps = {ensure_placement(p) for p in placements}
    return {str(p): p.marker() for p in PLACEMENT_ORDER if p in ps}


# --- strategy ----------------------------------------------------------------

def _strategy_linestyles(strategies: List[Union[str, Strategy]]) -> Dict[str, str]:
    ss = {ensure_strategy(s) for s in strategies}
    return {str(s): s.linestyle() for s in STRATEGY_ORDER if s in ss}

def _strategy_markers(strategies: List[Union[str, Strategy]]) -> Dict[str, str]:
    ss = {ensure_strategy(s) for s in strategies}
    return {str(s): s.marker() for s in STRATEGY_ORDER if s in ss}

def _strategy_colors(strategies: List[Union[str, Strategy]]) -> Dict[str, str]:
    ss = {ensure_strategy(s) for s in strategies}
    return {str(s): s.color() for s in STRATEGY_ORDER if s in ss}

def _strategy_styles(strategies: List[Union[str, Strategy]]) -> Dict[str, dict]:
    ss = {ensure_strategy(s) for s in strategies}
    return {
        str(s): {
            "color": s.color(),
            "marker": s.marker(),
            "linestyle": s.linestyle(),
        }
        for s in STRATEGY_ORDER if s in ss
    }


# --- model -------------------------------------------------------------------

def _model_markers(models: List[Union[str, Model]]) -> Dict[str, str]:
    ms = {ensure_model(m) for m in models}
    return {str(m): m.marker() for m in sorted(ms, key=lambda x: x.value)}

def _model_colors(models: List[Union[str, Model]]) -> Dict[str, str]:
    ms = {ensure_model(m) for m in models}
    return {str(m): m.color() for m in sorted(ms, key=lambda x: x.value)}

def _model_linestyles(models: List[Union[str, Model]]) -> Dict[str, str]:
    # Models don't define linestyles → fallback (clean, deterministic)
    styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]
    ms = sorted({ensure_model(m) for m in models}, key=lambda x: x.value)
    return {str(m): styles[i % len(styles)] for i, m in enumerate(ms)}

def _system_linestyles(systems: List[str]) -> Dict[str, str]:
    styles = ['-', '--', '-.', ':']
    return {c: styles[i % len(styles)] for i, c in enumerate(sorted(set(systems)))}

def _system_colors(systems: List[str]) -> Dict[str, str]:
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return {c: palette[i % len(palette)] for i, c in enumerate(sorted(set(systems)))}

def _lighten_color(color, amount: float):
    import matplotlib.colors as mc, colorsys
    try:
        c = mc.cnames.get(color, color)
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], min(1.0, 1 - amount * (1 - c[1])), c[2])
    except Exception:
        return color

def _system_placement_colors(
    system: str, placements: List[str], base_color: str
) -> Dict[str, tuple]:
    import matplotlib.colors as mc, colorsys
    lightness_levels = [0.35, 0.50, 0.65, 0.75, 0.45, 0.60]
    c = mc.to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(*c)
    return {
        tag: colorsys.hls_to_rgb(h, lightness_levels[i % len(lightness_levels)], s)
        for i, tag in enumerate(sorted(placements))
    }


# ============================================================================
#  Data loading from raw files
# ============================================================================

# Warm-up skip logic: mirrors min_throughput_across_ranks but keeps iteration
# arrays so we can compute per-iteration runtime / comm stats.
def _apply_warmup_skip(values: list, skip_first: int = 1) -> list:
    """Drop warm-up iterations from a per-iteration list."""
    skip = 3 if len(values) >= 6 else skip_first
    usable = values[skip:] if len(values) > skip else values[-1:]
    return usable


# Candidate keys for the communication / barrier time column, in priority order.
# parse_stdout_throughputs stores these under the "extra" sub-dict if they are
# not in _RANK_KEYS; we try each name in turn.
_BARRIER_KEY_CANDIDATES = [
    "barrier",
    "barrier_time",
    "dp_comm_time",
    "commtime",
    "comm_time",
]


def _get_barrier(rank_record: dict) -> Optional[list]:
    """
    Extract the per-iteration communication / barrier time list from a rank
    record.  Searches both the top-level record and the 'extra' sub-dict.
    Returns None when no candidate key is found.
    """
    extra = rank_record.get("extra", {})
    for key in _BARRIER_KEY_CANDIDATES:
        if key in rank_record and rank_record[key] is not None:
            v = rank_record[key]
            return v if isinstance(v, list) else [v]
        if key in extra and extra[key] is not None:
            v = extra[key]
            return v if isinstance(v, list) else [v]
    return None


def _build_baseline_records(
    backup_dir: str,
    system_name: str,
    skip_first: int = 1,
) -> List[dict]:
    """
    Walk SbatchMan experiment directories for *system_name* and return a list
    of flat per-job dicts suitable for building a summary DataFrame.

    Each dict contains:
        system, system, gpu_model, strategy, strategy, placement,
        model, gpus, nodes,
        throughputs        (list[float], post-warmup, bottleneck rank)
        iteration_times    (list[float] | None)
        barrier_times      (list[float] | None)
        memory_allocated   (float | None)

    The bottleneck rank is the rank whose median throughput is lowest (same
    criterion as min_throughput_across_ranks).
    """
    records = []
    sbm_system_name = system_name
    if system_name =='dgxA100':
        sbm_system_name = 'baldo'
        
    meta_pattern = os.path.join(backup_dir, "SbatchMan", "experiments", sbm_system_name, "*", "baseline_*", "*", "metadata.yaml")

    for meta_path in glob.glob(meta_pattern):
        with open(meta_path) as f:
            meta = yaml.safe_load(f)

        if meta.get("status") != "COMPLETED":
            continue

        v = meta.get("variables", {})
        if system_name == "nvl72" and v.get("gpu_model", "").upper() != "GB300":
            continue

        strategy  = v.get("strategy")
        nodes     = v.get("nodes")
        gpus      = v.get("gpus", nodes * GPUS_PER_NODE_MAP[system_name])
        placement = v.get("placement_class") or v.get("placement")
        if not all([strategy, gpus, placement]):
            continue

        stdout = os.path.join(os.path.dirname(meta_path), "stdout.log")
        if not os.path.isfile(stdout):
            continue

        model = v.get("model")
        if not model:
            model = get_model_from_command(meta.get("command", ""))
        if not model:
            model = parse_model_from_stdout(stdout)
        if not model:
            model = get_default_model(strategy)
        if not model:
            model = "unknown"

        rank_records = parse_stdout_throughputs(stdout)
        if not rank_records:
            continue

        # --- identify the bottleneck rank (lowest median throughput) ---
        def _rank_median(rr):
            tp = rr["throughputs"]
            skip = 3 if len(tp) >= 6 else skip_first
            usable = tp[skip:] if len(tp) > skip else tp[-1:]
            return float(np.median(usable)) if usable else float("inf")

        bottleneck_rank = min(rank_records, key=lambda rid: _rank_median(rank_records[rid]))
        rr = rank_records[bottleneck_rank]

        # --- extract post-warmup iteration arrays ---
        tp_raw  = rr["throughputs"]
        it_raw  = rr.get("iteration_times") or []
        bar_raw = _get_barrier(rr) or []

        skip = 3 if len(tp_raw) >= 6 else skip_first
        tp_usable  = tp_raw[skip:]  if len(tp_raw)  > skip else tp_raw[-1:]
        it_usable  = it_raw[skip:]  if len(it_raw)  > skip else (it_raw[-1:] if it_raw else [])
        bar_usable = bar_raw[skip:] if len(bar_raw) > skip else (bar_raw[-1:] if bar_raw else [])

        # gpu_model: prefer rank-level metadata, fall back to yaml variable
        gpu_model = rr.get("gpu_model") or v.get("gpu_model", "unknown")

        records.append({
            "system":          system_name,
            "system":         system_name,        # display system = system name
            "gpu_model":       gpu_model,
            "strategy":        strategy,
            "strategy":   strategy,            # will be refined if placement encodes it
            "placement":       placement,
            "model":      model,
            "gpus":            int(gpus),
            "nodes":           int(nodes),
            "throughputs":     tp_usable,
            "iteration_times": it_usable if it_usable else None,
            "barrier_times":   bar_usable if bar_usable else None,
            "memory_allocated": rr.get("memory_allocated"),
        })

    return records


# ============================================================================
#  Summary DataFrame construction
# ============================================================================

def build_summary(records: List[dict]) -> pd.DataFrame:
    """
    Aggregate per-iteration arrays into a one-row-per-job summary DataFrame.

    Columns
    -------
    system, system, gpu_model, strategy, strategy, placement,
    model, gpus, nodes,
    throughput_mean, throughput_std,
    runtime_mean,
    barrier_mean,    (NaN when no barrier data available)
    compute_mean,    (runtime - barrier; NaN when barrier unavailable)
    comm_pct,        (barrier / runtime * 100; NaN when unavailable)
    compute_pct,     (compute / runtime * 100; NaN when unavailable)
    memory_allocated
    """
    rows = []
    for r in records:
        tp  = np.asarray(r["throughputs"],     dtype=float)
        it  = np.asarray(r["iteration_times"], dtype=float) if r["iteration_times"] else None
        bar = np.asarray(r["barrier_times"],   dtype=float) if r["barrier_times"]   else None

        throughput_mean = float(tp.mean()) if len(tp) else float("nan")
        throughput_std  = float(tp.std())  if len(tp) else float("nan")

        runtime_mean = float(it.mean())  if it  is not None and len(it)  else float("nan")
        barrier_mean = float(bar.mean()) if bar is not None and len(bar) else float("nan")

        if it is not None and bar is not None and len(it) and len(bar):
            # Align lengths in case iteration_times and barrier_times differ
            n = min(len(it), len(bar))
            compute = it[:n] - bar[:n]
            compute_mean = float(compute.mean())
            comm_pct     = float((bar[:n] / it[:n] * 100).mean())
            compute_pct  = float((compute   / it[:n] * 100).mean())
        else:
            compute_mean = float("nan")
            comm_pct     = float("nan")
            compute_pct  = float("nan")

        rows.append({
            "system":          r["system"],
            "system":         r["system"],
            "gpu_model":       r["gpu_model"],
            "strategy":        r["strategy"],
            "strategy":   r["strategy"],
            "placement":       r["placement"],
            "model":      r["model"],
            "gpus":            r["gpus"],
            "nodes":           r["nodes"],
            "throughput_mean": throughput_mean,
            "throughput_std":  throughput_std,
            "runtime_mean":    runtime_mean,
            "barrier_mean":    barrier_mean,
            "compute_mean":    compute_mean,
            "comm_pct":        comm_pct,
            "compute_pct":     compute_pct,
            "memory_allocated": r.get("memory_allocated"),
        })

    return pd.DataFrame(rows)


# ============================================================================
#  Aggregation helper  (unchanged semantics from original)
# ============================================================================

def aggregate_placements(summary: pd.DataFrame, agg_type: str = "geomean") -> pd.DataFrame:
    """
    Collapse all placements (placements) of the same
    (strategy, model, system, gpus, gpu_model) into a single row
    by averaging throughput and time metrics.

    model is kept as a groupby key so that different models that share
    the same strategy are never merged together.
    """
    agg = (
        summary
        .groupby(
            ["strategy", "model", "system", "gpus"],
            as_index=False,
        )
        .agg(
            system         =("system",          "first"),
            nodes          =("gpus",            "first"),
            throughput_median=("throughput_median", agg_type),
            throughput_std =("throughput_std",  agg_type),
            runtime_mean   =("runtime_mean",    agg_type),
            barrier_mean   =("barrier_mean",    agg_type),
            compute_mean   =("compute_mean",    agg_type),
            comm_pct       =("comm_pct",        agg_type),
            compute_pct    =("compute_pct",     agg_type),
        )
    )
    agg["strategy"]  = agg["strategy"]
    agg["placement"] = ""
    return agg


# ============================================================================
#  Plot helpers
# ============================================================================

def _save_or_show(fig, output_file: Optional[str], label: str = "Plot"):
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"  {label} saved to: {output_file}")
        plt.close(fig)
    else:
        plt.show()


def _make_label(
    strategy: str,
    model: str,
    placement: str,
    system: str,
    include_system: bool = True,
) -> str:
    """Human-readable series label including model name."""
    place = ensure_placement(placement).display(new_line=False, short=True)
    # system_disp = f"{system}/{gpu_model}" if gpu_model and gpu_model != "unknown" else system
    system_disp = system
    strategy = ensure_strategy(strategy)
    model = ensure_model(model)
    parts = [strategy.short(), model.short()]
    # if place and place != 'N/A':
    parts.append(place)
    if include_system:
        parts.append(system_disp)
    return " - ".join(parts)


def _dedup_gpus(grp: pd.DataFrame, label: str, metric: str) -> pd.DataFrame:
    """Keep the highest-throughput row when the same (GPU count, model) appears twice."""
    if grp["gpus"].duplicated().any():
        dup_counts = grp["gpus"].value_counts()
        for gpu, count in dup_counts[dup_counts > 1].items():
            best = grp[grp["gpus"] == gpu][f"throughput_{metric}"].max()
            print(f"[WARN] Duplicate GPU entries for {label}: gpus={gpu} "
                  f"({count} rows) → keeping max {best:.3f}")
        grp = grp.loc[grp.groupby("gpus")[f"throughput_{metric}"].idxmax()]
    return grp.sort_values("gpus")


# ============================================================================
#  Plot 1: Scaling
# ============================================================================

def plot_scaling(
    summary: pd.DataFrame,
    metric: str,
    strategies: Optional[List[str]] = None,
    systems: Optional[List[str]] = None,
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 7),
    title: str = "Scaling: Throughput vs Number of GPUs",
    show_ideal: bool = True,
    plot_efficiency: bool = False,
):
    df = summary.copy()
    if strategies:
        df = df[df["strategy"].isin(strategies)]
    if systems:
        df = df[df["system"].isin(systems)]

    fig, ax = plt.subplots(figsize=figsize)
    if plot_efficiency:
        fig_eff, ax_eff = plt.subplots(figsize=figsize)

    clust_color  = _system_colors(df["system"].unique())
    place_ls     = _placement_linestyles(df["placement"].unique())
    base_markers = _strategy_markers(df["strategy"].unique())

    min_eff = 1.0        

    for (system, strategy, model), grp in df.groupby(
        ["system", "strategy", "model"]
    ):
        strategy = grp["strategy"].iloc[0]
        placement  = grp["placement"].iloc[0]
        gpu_model  = grp["gpu_model"].iloc[0]

        color  = clust_color[system]
        ls     = place_ls.get(placement, "-")
        marker = base_markers.get(strategy, "o")
        label  = _make_label(strategy, model, placement, system, gpu_model)

        grp = _dedup_gpus(grp, label, metric)

        ax.errorbar(
            grp["gpus"], grp[f"throughput_{metric}"],
            yerr=grp["throughput_std"],
            label=label, color=color, marker=marker, linestyle=ls,
            linewidth=1, markersize=5, capsize=2,
        )

        if show_ideal:
            base_row = grp.iloc[0]
            gpus_range = np.array(sorted(df["gpus"].unique()))
            gpus_range = gpus_range[gpus_range >= base_row["gpus"]]
            ideal = base_row[f"throughput_{metric}"] * (gpus_range / base_row["gpus"])
            ax.plot(gpus_range, ideal, color=color, linestyle=":", linewidth=1, alpha=0.4)

        if plot_efficiency:
            g0  = grp.iloc[0]["gpus"]
            T0  = grp.iloc[0][f"throughput_{metric}"]
            eff = (grp[f"throughput_{metric}"] / T0) * (g0 / grp["gpus"])
            min_eff = min(min_eff, eff.min())
            ax_eff.plot(grp["gpus"], eff, label=label,
                        color=color, marker=marker, linestyle=ls,
                        linewidth=1, markersize=5)

    if show_ideal:
        ax.plot([], [], color="gray", linestyle=":", linewidth=1,
                alpha=0.7, label="Ideal scaling")

    all_gpus = sorted(df["gpus"].unique())
    for _ax in [ax]:
        _ax.set_xscale("log", base=2)
        _ax.set_yscale("log", base=2)
        _ax.set_xticks(all_gpus)
        _ax.set_xticklabels([str(g) for g in all_gpus])
        _ax.set_xlabel("Number of GPUs", fontsize=12)
        _ax.set_ylabel("Throughput (samples/s)", fontsize=12)
        _ax.set_title(title, fontsize=14, fontweight="bold")
        _ax.legend(fontsize=9, loc="center left",
                   bbox_to_anchor=(1.01, 0.5), borderaxespad=0)
        _ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 0.78, 1])
    _save_or_show(fig, output_file, "Scaling plot")

    if plot_efficiency:
        ax_eff.axhline(1.0, linestyle=":", linewidth=1, alpha=0.7,
                       color="gray", label="Ideal efficiency")
        ax_eff.set_xscale("log", base=2)
        ax_eff.set_xticks(all_gpus)
        ax_eff.set_xticklabels([str(g) for g in all_gpus])
        ax_eff.set_yticks(
            list(np.linspace(min_eff, 1.0, 10)),
            labels=[f"{int(e*100)}%" for e in np.linspace(min_eff, 1.0, 10)],
        )
        ax_eff.set_xlabel("Number of GPUs", fontsize=12)
        ax_eff.set_ylabel("Parallel Efficiency", fontsize=12)
        ax_eff.set_title("Scaling Efficiency", fontsize=14, fontweight="bold")
        ax_eff.legend(fontsize=8, loc="center left",
                      bbox_to_anchor=(1.01, 0.5), borderaxespad=0)
        ax_eff.grid(True, alpha=0.3)
        fig_eff.tight_layout(rect=[0, 0, 0.78, 1])

        eff_file = output_file.replace(".png", "_efficiency.png") if output_file else None
        _save_or_show(fig_eff, eff_file, "Efficiency plot")
        return (fig, ax), (fig_eff, ax_eff)

    return fig, ax, (None, None)


# ============================================================================
#  Plot 2: Time breakdown
# ============================================================================

def plot_breakdown(
    summary: pd.DataFrame,
    strategies: Optional[List[str]] = None,
    systems: Optional[List[str]] = None,
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 9),
    title: str = "Time Breakdown: Compute vs Communication (%)",
):
    df = summary.copy()
    if strategies:
        df = df[df["strategy"].isin(strategies)]
    if systems:
        df = df[df["system"].isin(systems)]

    # Drop rows where comm/compute data is unavailable
    has_breakdown = df["comm_pct"].notna() & df["compute_pct"].notna()
    if not has_breakdown.any():
        print(f"  [SKIP] No barrier/comm data available for breakdown plot: {title}")
        return None, None
    df = df[has_breakdown]

    fig, ax = plt.subplots(figsize=figsize)

    clust_color  = _system_colors(df["system"].unique())
    place_colors: Dict[Tuple[str, str], tuple] = {}
    for system in df["system"].unique():
        tags = sorted(df[df["system"] == system]["placement"].unique())
        shades = _system_placement_colors(system, tags, clust_color[system])
        for tag, shade in shades.items():
            place_colors[(system, tag)] = shade

    all_gpus = sorted(df["gpus"].unique())
    combos   = sorted(df.groupby(["strategy", "system", "model"]).groups.keys())
    n_combos = len(combos)

    group_width = 0.75
    gap         = 0.04
    bar_width   = (group_width - gap * (n_combos - 1)) / n_combos
    x           = np.arange(len(all_gpus))

    legend_added: set = set()

    for i, (strategy, system, model) in enumerate(combos):
        grp = df[
            (df["strategy"] == strategy) &
            (df["system"]  == system)  &
            (df["model"] == model)
        ].copy()
        strategy = grp["strategy"].iloc[0]
        placement  = grp["placement"].iloc[0]
        gpu_model  = grp["gpu_model"].iloc[0]

        color = place_colors.get((system, placement), clust_color[system])
        label_base = _make_label(strategy, model, placement, system, gpu_model)

        if grp["gpus"].duplicated().any():
            grp = (
                grp.groupby("gpus", as_index=False)
                .agg(compute_pct=("compute_pct", "mean"),
                     comm_pct   =("comm_pct",    "mean"))
            )

        grp = grp.set_index("gpus").reindex(all_gpus).fillna(0)
        offset = x - group_width / 2 + i * (bar_width + gap) + bar_width / 2

        ax.bar(offset, grp["compute_pct"].values, width=bar_width,
               color=color, alpha=0.95,
               label=f"{label_base} - Comp" if label_base not in legend_added
               else "_nolegend_")
        ax.bar(offset, grp["comm_pct"].values, width=bar_width,
               bottom=grp["compute_pct"].values, color=color, alpha=0.45,
               label=f"{label_base} - Comm" if label_base not in legend_added
               else "_nolegend_")
        legend_added.add(label_base)

    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in all_gpus])
    ax.set_xlabel("Number of GPUs", fontsize=12)
    ax.set_ylabel("Time (%)", fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="center left",
              bbox_to_anchor=(1.01, 0.5), borderaxespad=0)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout(rect=[0, 0, 0.78, 1])
    _save_or_show(fig, output_file, "Breakdown plot")
    return fig, ax


# ============================================================================
#  Comm-pct LaTeX table  (unchanged semantics from original)
# ============================================================================

def generate_comm_pct_table(
    summary: pd.DataFrame,
    output_file: Optional[str] = None,
    gpus: Optional[int] = None,
) -> str:
    """
    Build a LaTeX table of communication-time percentages.

    Rows    : (strategy, model)
    Columns : system
    Cells   : min–max range of comm_pct across placements (placements).
    """
    df = summary.copy()
    if gpus is not None:
        df = df[df["gpus"] == gpus]

    df["row_key"] = df["strategy"] + " / " + df["model"]

    systems = sorted(df["system"].unique())
    row_keys = sorted(df["row_key"].unique())

    def _cell(sub: pd.DataFrame) -> str:
        per_placement = sub.groupby("placement")["comm_pct"].mean()
        # Drop NaN placements (jobs without barrier data)
        per_placement = per_placement.dropna()
        if per_placement.empty:
            return "---"
        lo, hi = per_placement.min(), per_placement.max()
        if len(per_placement) == 1 or abs(hi - lo) < 0.05:
            return f"{lo:.1f}\\%"
        return f"{lo:.1f}--{hi:.1f}\\%"

    cells: Dict[Tuple[str, str], str] = {}
    for (row_key, system), grp in df.groupby(["row_key", "system"]):
        cells[(row_key, system)] = _cell(grp)

    def _esc(s: str) -> str:
        return s.replace("_", r"\_")

    col_spec   = "ll" + "c" * len(systems)
    header_cols = " & ".join(
        [r"\textbf{Strategy / Model}"] +
        [r"\textbf{" + _esc(c) + "}" for c in systems]
    )

    rows_tex = [
        _esc(rk) + " & " + " & ".join(cells.get((rk, c), "---") for c in systems) + r" \\"
        for rk in row_keys
    ]

    gpu_note = f"GPU count: {gpus}" if gpus is not None else "all GPU counts"
    caption  = (
        f"Communication time (\\%) by strategy, model and system ({gpu_note}). "
        f"Cells show the min--max range across placements."
    )

    lines = (
        [r"\begin{table}[ht]", r"  \centering",
         r"  \caption{" + caption + "}", r"  \label{tab:comm_pct}",
         r"  \begin{tabular}{" + col_spec + "}", r"    \toprule",
         f"    {header_cols} \\\\", r"    \midrule"]
        + [f"    {r}" for r in rows_tex]
        + [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    )
    tex = "\n".join(lines) + "\n"

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text(tex)
        print(f"  Comm-pct table saved to: {output_file}")

    return tex


# ============================================================================
#  Per-system orchestration + CLI
# ============================================================================

def process_system(
    system_name: str,
    backup_dir: str,
    skip_first: int,
    output_dir: str,
    prefix: str = "",
    no_ideal: bool = False,
    aggregate_placements_flag: bool = False,
    only_all: bool = False,
    table_gpus: Optional[int] = None,
    strategyegies_filter: Optional[List[str]] = None,
):
    pfx = f"{prefix}_" if prefix else ""
    output_dir = Path(output_dir)
    per_system_dir   = output_dir / "per_system"
    cross_system_dir = output_dir / "cross_system"

    print(f"\n{'='*60}")
    print(f"  System     : {system_name}")
    print(f"  Backup dir : {backup_dir}")
    print(f"  Skip first : {skip_first} iterations")
    print(f"{'='*60}\n")

    print("Loading baseline records ...")
    records = _build_baseline_records(backup_dir, system_name, skip_first)
    print(f"  {len(records)} baseline jobs loaded\n")
    if not records:
        print("  Nothing to plot.")
        return

    summary = build_summary(records)
    print(f"Summary ({len(summary)} jobs):")
    print(summary[["system", "gpu_model", "strategy", "placement",
                   "gpus", "throughput_mean", "comm_pct", "compute_pct"]
                  ].to_string(index=False))
    print()

    systems        = sorted(summary["system"].unique())
    strategyegies = sorted(summary["strategy"].unique())
    if strategyegies_filter:
        strategyegies = [s for s in strategyegies if s in strategyegies_filter]
    all_strategies  = sorted(summary["strategy"].unique())
    all_models      = sorted(summary["model"].unique())

    # 1) Per-system, per-base-strategy, per-model: compare placements
    if not only_all:
        print("[Per-system placement comparison]")
        for system in systems:
            for strategy in strategyegies:
                for model in all_models:
                    strats_here = summary[
                        (summary["system"]       == system) &
                        (summary["strategy"] == strategy) &
                        (summary["model"]    == model)
                    ]["strategy"].unique().tolist()
                    if not strats_here:
                        continue

                    tag = f"{pfx}{system_name}_{strategy}_{model}_on_{system}"
                    print(f"  {tag}  placements={strats_here}")

                    plot_scaling(
                        summary, strategies=strats_here, systems=[system],
                        output_file=str(per_system_dir / f"{tag}_scaling.png"),
                        title=f"Scaling — {strategy} / {model} placements on {system}",
                        show_ideal=not no_ideal,
                    )
                    plot_breakdown(
                        summary, strategies=strats_here, systems=[system],
                        output_file=str(per_system_dir / f"{tag}_breakdown.png"),
                        title=f"Time Breakdown — {strategy} / {model} placements on {system}",
                    )

        # 2) Cross-system per strategy+placement+model
        if not only_all:
            if len(systems) > 1:
                print("\n[Cross-system comparison per strategy+placement+model]")
                for strategy in all_strategies:
                    base = summary[summary["strategy"] == strategy]["strategy"].iloc[0]
                    if strategyegies_filter and base not in strategyegies_filter:
                        continue
                    for model in all_models:
                        clust_here = summary[
                            (summary["strategy"]   == strategy) &
                            (summary["model"] == model)
                        ]["system"].unique().tolist()
                        if not clust_here:
                            continue
                        print(f"  {strategy} / {model}  systems={clust_here}")
                        plot_scaling(
                            summary, strategies=[strategy], systems=clust_here,
                            output_file=str(cross_system_dir / f"{pfx}{system_name}_{strategy}_{model}_xsystem_scaling.png"),
                            title=f"Scaling — {strategy} / {model} across systems",
                            show_ideal=not no_ideal,
                        )
                        plot_breakdown(
                            summary, strategies=[strategy], systems=clust_here,
                            output_file=str(cross_system_dir / f"{pfx}{system_name}_{strategy}_{model}_xsystem_breakdown.png"),
                            title=f"Time Breakdown — {strategy} / {model} across systems",
                        )
    if not only_all:
        # 3) Cross-system overview per base-strategy, per-model
        print("\n[Cross-system + cross-placement overview per base strategy / model]")
        for strategy in strategyegies:
            for model in all_models:
                sub = summary[
                    (summary["strategy"] == strategy) &
                    (summary["model"]    == model)
                ]
                if sub.empty:
                    continue
                plot_summary = aggregate_placements(sub) if aggregate_placements_flag else sub
                agg_label    = " (placements averaged)" if aggregate_placements_flag else ""
                strats_here  = plot_summary["strategy"].unique().tolist()
                mode_tag     = "aggr" if aggregate_placements_flag else "all"
                print(f"  {strategy} / {model}  [{mode_tag}] strategies={strats_here}")

                plot_scaling(
                    plot_summary, strategies=strats_here,
                    output_file=str(output_dir / f"{pfx}{system_name}_{strategy}_{model}_{mode_tag}_scaling.png"),
                    title=f"Scaling — {strategy} / {model} (all systems{agg_label})",
                    show_ideal=not no_ideal,
                    plot_efficiency=True,
                )
                plot_breakdown(
                    plot_summary, strategies=strats_here,
                    output_file=str(output_dir / f"{pfx}{system_name}_{strategy}_{model}_{mode_tag}_breakdown.png"),
                    title=f"Time Breakdown — {strategy} / {model} (all systems{agg_label})",
                )

    # 4) Comm-pct LaTeX table
    print("\n[Comm-pct LaTeX table]")
    generate_comm_pct_table(
        summary,
        output_file=str(output_dir / f"{pfx}{system_name}_comm_pct_table.tex"),
        gpus=table_gpus,
    )

    return summary




# ============================================================================
#  Global plots  (all systems combined)
# ============================================================================

def plot_global(
    combined: pd.DataFrame,
    output_dir: Path,
    pfx: str = "",
    no_ideal: bool = False,
    aggregate_placements_flag: bool = False,
):
    """
    Single-figure global overview across all systems.

    Produces scaling (+ efficiency) and breakdown plots with every system,
    strategy, model, and placement on one figure each.
    """
    print("\n[Global overview — all systems combined]")

    plot_summary = aggregate_placements(combined) if aggregate_placements_flag else combined
    agg_label    = " (placements averaged)" if aggregate_placements_flag else ""
    mode_tag     = "aggr" if aggregate_placements_flag else "all"
    all_strats   = plot_summary["strategy"].unique().tolist()
    systems      = sorted(combined["system"].unique())
    sys_label    = "+".join(systems)

    print(f"  [{mode_tag}] systems={systems}  strategies={sorted(all_strats)}")

    plot_scaling(
        plot_summary, strategies=all_strats,
        output_file=str(output_dir / f"{pfx}global_{sys_label}_{mode_tag}_scaling.png"),
        title=f"Scaling — all strategies / all models ({sys_label}{agg_label})",
        show_ideal=not no_ideal,
        plot_efficiency=True,
    )
    plot_breakdown(
        plot_summary, strategies=all_strats,
        output_file=str(output_dir / f"{pfx}global_{sys_label}_{mode_tag}_breakdown.png"),
        title=f"Time Breakdown — all strategies / all models ({sys_label}{agg_label})",
    )


def plot_global_faceted(
    combined: pd.DataFrame,
    output_dir: Path,
    metric: str,
    pfx: str = "",
    no_ideal: bool = False,
    aggregate_placements_flag: bool = False,
    n_rows: int = 1,
    plot_type: str = "scaling",   # "scaling" | "breakdown" | "efficiency"
    figsize_per_cell: Tuple[int, int] = (8, 6),
):
    """
    One subplot per system, laid out in a grid of *n_rows* rows.

    Parameters
    ----------
    combined : pd.DataFrame
        Concatenated summary from all systems.
    output_dir : Path
        Directory where the figure is saved.
    pfx : str
        Filename prefix.
    no_ideal : bool
        Suppress ideal-scaling reference lines.
    aggregate_placements_flag : bool
        Average across placements before plotting each panel.
    n_rows : int
        Number of rows in the subplot grid.  n_cols is derived automatically
        as ceil(n_systems / n_rows).  Set n_rows=1 for a single row of panels,
        n_rows=n_systems for a single column, or anything in between.
    plot_type : str
        What to draw in each panel:
          "scaling"    — throughput vs GPUs (log/log)
          "efficiency" — parallel efficiency vs GPUs
          "breakdown"  — stacked bar of compute vs comm %
    figsize_per_cell : (width, height)
        Size of each individual subplot cell in inches.  The total figure size
        is derived as (n_cols * width, n_rows * height).
    """
    import math
    from matplotlib.ticker import FuncFormatter

    systems = [s for s in SYSTEM_ORDER if s in combined["system"].unique()]
    n_sys   = len(systems)
    if n_sys == 0:
        print("  [SKIP] No systems in combined summary.")
        return

    n_rows  = max(1, min(n_rows, n_sys))          # clamp to [1, n_systems]
    n_cols  = math.ceil(n_sys / n_rows)

    mode_tag  = "aggr" if aggregate_placements_flag else "all"

    fig_w = n_cols * figsize_per_cell[0]
    fig_h = n_rows * figsize_per_cell[1]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    # Flatten axes to a 1-D list for easy indexed access; hide surplus panels.
    axes_flat = [axes[r][c] for r in range(n_rows) for c in range(n_cols)]
    for ax in axes_flat[n_sys:]:
        ax.set_visible(False)

    # Global legend storage
    global_handles = []
    global_labels  = []
    
    def format_throughput(throughput: float, *_):
        if throughput < 100:
            return str(int(throughput))
        if throughput < 1e3:
            return f"{(throughput / 1e3):.1f}K"
        if throughput < 1e6:
            return f"{int(throughput / 1e3)}K"
        if throughput < 1e9:
            return f"{int(throughput / 1e6)}M"
        return f"{int(throughput / 1e9)}B"
    
    def format_gpus(gpus: int, *_):
        if gpus == 224:
            return ''
        if gpus > 512:
            return f'{int(gpus / 1e3)}K'
        return str(gpus)

    for idx, system_name in enumerate(systems):
        ax = axes_flat[idx]
        sub = combined[combined["system"] == system_name].copy()

        plot_summary = aggregate_placements(sub) if aggregate_placements_flag else sub
        # plot_summary = sub
        # print(plot_summary)

        # Local GPU domain per subplot
        gpus_local = sorted(plot_summary["gpus"].unique())

        clust_color     = _system_colors(plot_summary["system"].unique())
        place_ls        = _placement_linestyles(plot_summary["placement"].unique())
        model_ls        = _model_linestyles(plot_summary["model"].unique())
        place_markers   = _placement_markers(plot_summary["placement"].unique())
        base_markers    = _strategy_markers(plot_summary["strategy"].unique())
        base_color      = _strategy_colors(plot_summary["strategy"].unique())

        if plot_type in ("scaling", "efficiency"):
            min_eff = 1.0

            for (strategy, system, model, placement), grp in plot_summary.groupby(
                ["strategy", "system", "model", "placement"]
            ):
                color    = base_color[strategy]
                ls       = model_ls[model]
                marker   = place_markers.get(placement, "o")
                
                label  = _make_label(strategy, model, placement, system, include_system=False)

                grp = _dedup_gpus(grp, label, metric)

                if plot_type == "scaling":
                    ax.errorbar(
                        grp["gpus"], grp[f"throughput_{metric}"],
                        yerr=grp["throughput_std"],
                        label=label, color=color, marker=marker, linestyle=ls,
                        linewidth=2, markersize=4, capsize=3,
                    )
                    if not no_ideal:
                        base_row   = grp.iloc[0]
                        gpus_range = np.array([g for g in grp["gpus"].unique()
                                               if g >= base_row["gpus"]])
                        ideal = base_row[f"throughput_{metric}"] * (gpus_range / base_row["gpus"])
                        ax.plot(gpus_range, ideal, color=color,
                                linestyle=":", linewidth=2, alpha=0.35)

                else:  # efficiency
                    g0  = grp.iloc[0]["gpus"]
                    T0  = grp.iloc[0][f"throughput_{metric}"]
                    eff = (grp[f"throughput_{metric}"] / T0) * (g0 / grp["gpus"])
                    min_eff = min(min_eff, float(eff.min()))
                    ax.plot(grp["gpus"], eff, label=label,
                            color=color, marker=marker, linestyle=ls,
                            linewidth=1, markersize=4)

            if plot_type == "scaling":
                ax.set_xscale("log", base=2)
                ax.set_yscale("log", base=2)
                if idx == 0:
                    ax.set_ylabel("Throughput (samples/s)", fontsize=22)
                if not no_ideal:
                    ax.plot([], [], color="gray", linestyle=":", linewidth=2,
                            alpha=0.9, label="Ideal")
                ax.yaxis.set_major_formatter(FuncFormatter(format_throughput))
                ax.yaxis.set_tick_params(labelsize=20)
            else:
                ax.axhline(1.0, linestyle=":", linewidth=1, alpha=0.6,
                           color="gray", label="Ideal")
                ax.set_xscale("log", base=2)
                ax.set_ylabel("Parallel Efficiency", fontsize=9)

            
            # Optional custom formatter
            ax.set_xticks(gpus_local)
            ax.xaxis.set_major_formatter(FuncFormatter(format_gpus))
            ax.xaxis.set_tick_params(labelsize=20)
            # ax.set_xticklabels([str(g) if g != 224 else '' for g in gpus_local], fontsize=16)

        elif plot_type == "breakdown":
            has_breakdown = plot_summary["comm_pct"].notna() & plot_summary["compute_pct"].notna()
            if not has_breakdown.any():
                ax.text(0.5, 0.5, "No barrier data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")
            else:
                df_b = plot_summary[has_breakdown].copy()
                combos   = sorted(df_b.groupby(["strategy", "system", "model"]).groups.keys())
                n_combos = len(combos)
                group_width = 0.75
                gap         = 0.04
                bar_width   = (group_width - gap * (n_combos - 1)) / max(n_combos, 1)

                gpus_local = sorted(df_b["gpus"].unique())
                x_pos      = np.arange(len(gpus_local))

                clust_color_b  = _system_colors(df_b["system"].unique())
                place_colors_b: Dict[Tuple[str, str], tuple] = {}
                for cl in df_b["system"].unique():
                    tags   = sorted(df_b[df_b["system"] == cl]["placement"].unique())
                    shades = _system_placement_colors(cl, tags, clust_color_b[cl])
                    place_colors_b.update({(cl, t): s for t, s in shades.items()})

                legend_added: set = set()
                for bi, (strategy, system, model) in enumerate(combos):
                    grp = df_b[
                        (df_b["strategy"]   == strategy) &
                        (df_b["system"]    == system)  &
                        (df_b["model"] == model)
                    ].copy()
                    strategy = grp["strategy"].iloc[0]
                    placement  = grp["placement"].iloc[0]
                    gpu_model  = grp["gpu_model"].iloc[0]
                    color      = place_colors_b.get((system, placement), clust_color_b[system])
                    lbl        = _make_label(strategy, model, placement, system, gpu_model)

                    if grp["gpus"].duplicated().any():
                        grp = grp.groupby("gpus", as_index=False).agg(
                            compute_pct=("compute_pct", "mean"),
                            comm_pct   =("comm_pct",    "mean"),
                        )
                    grp = grp.set_index("gpus").reindex(gpus_local).fillna(0)

                    offset = (x_pos - group_width / 2
                              + bi * (bar_width + gap) + bar_width / 2)

                    ax.bar(offset, grp["compute_pct"].values, width=bar_width,
                           color=color, alpha=0.95,
                           label=f"{lbl} - Comp" if lbl not in legend_added else "_nolegend_")
                    ax.bar(offset, grp["comm_pct"].values, width=bar_width,
                           bottom=grp["compute_pct"].values, color=color, alpha=0.45,
                           label=f"{lbl} - Comm" if lbl not in legend_added else "_nolegend_")
                    legend_added.add(lbl)

                ax.set_xticks(x_pos)
                ax.set_xticklabels([str(g) for g in gpus_local], fontsize=7)
                ax.set_ylim(0, 110)
                ax.set_ylabel("Time (%)", fontsize=9)

        ax.set_title(SYSTEM_NAMES_MAP.get(system_name, system_name), fontsize=24, fontweight="bold")
        ax.set_xlabel("GPUs", fontsize=20)
        ax.grid(True, alpha=0.45)

        # Collect legend entries
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            for h, l in zip(handles, labels):
                if l not in global_labels:
                    global_handles.append(h)
                    global_labels.append(l)
    

    # One global legend at the bottom
    if global_labels:
        sorted_pairs = sorted(zip(global_handles, global_labels), key=lambda x: x[1].split('-'))
        global_handles, global_labels = zip(*sorted_pairs)
        fig.legend(
            list(global_handles),
            list(global_labels),
            loc="lower center",
            # nrows=2,
            ncol=min(11, max(1, len(global_labels))),
            fontsize=16,
            frameon=False,
        )

    fig.tight_layout(rect=[0, 0.18, 1, 1.0])

    out = str(output_dir / f"{pfx}global_{mode_tag}_{metric}_{plot_type}_faceted.png")
    _save_or_show(fig, out, f"Faceted {plot_type} plot")


def main():
    parser = ArgumentParser(
        description="Plot DLNetBench baseline results from raw stdout/yaml files."
    )
    parser.add_argument(
        "--systems", nargs="+", default=list(SYSTEMS.keys()),
        help=f"Systems to process (default: {list(SYSTEMS.keys())})",
    )
    parser.add_argument(
        "--backup-dirs", nargs="+", default=None,
        help=(
            "Raw data directories, one per system (in the same order as --systems). "
            "Overrides the built-in SYSTEMS paths."
        ),
    )
    parser.add_argument("--output-dir", default="plots/baselines",
                        help="Output directory for plots")
    parser.add_argument("--prefix", default="",
                        help="Optional filename prefix")
    parser.add_argument("--skip-first", type=int, default=1,
                        help="Warm-up iterations to skip per rank (default: 1)")
    parser.add_argument("--no-ideal", action="store_true",
                        help="Suppress ideal-scaling lines")
    parser.add_argument("--strategies", nargs="*",
                        help="Filter to specific base strategies")
    parser.add_argument("--aggregate-placements", action="store_true",
                        help="Average across placements before plotting")
    parser.add_argument("--only-all", action="store_true",
                        help="Produce only cross-system overview plots")
    parser.add_argument("--table-gpus", type=int, default=None,
                        help="GPU count to filter for the comm-pct table")
    parser.add_argument(
        "--grid-rows", type=int, default=1,
        help=(
            "Number of rows in the faceted global plot (one panel per system). "
            "n_cols is derived automatically as ceil(n_systems / grid-rows). "
            "Use 1 for a single row, or match --systems count for a single column. "
            "(default: 1)"
        ),
    )
    parser.add_argument(
        "--cell-size", type=int, nargs=2, default=[8, 6],
        metavar=("W", "H"),
        help="Width and height in inches of each panel cell in the faceted plot (default: 8 6)",
    )
    args = parser.parse_args()

    output_dir  = Path(args.output_dir)
    pfx         = f"{args.prefix}_" if args.prefix else ""
    backup_dirs = args.backup_dirs or [SYSTEMS.get(s) for s in args.systems]

    # --- per-system plots; collect summaries for the combined global step ---
    all_summaries: List[pd.DataFrame] = []

    for system_name, backup_dir in zip(args.systems, backup_dirs):
        if backup_dir is None:
            print(f"ERROR: no backup dir known for system '{system_name}'.")
            continue
        if not os.path.isdir(backup_dir):
            print(f"WARNING: backup dir not found for {system_name}: {backup_dir}")
            continue

        summary = process_system(
            system_name               = system_name,
            backup_dir                = backup_dir,
            skip_first                = args.skip_first,
            output_dir                = args.output_dir,
            prefix                    = args.prefix,
            no_ideal                  = args.no_ideal,
            aggregate_placements_flag = args.aggregate_placements,
            only_all                  = args.only_all,
            table_gpus                = args.table_gpus,
            strategyegies_filter    = args.strategies,
        )
        if summary is not None and not summary.empty:
            all_summaries.append(summary)

    # --- global plots combining all systems ---
    if len(all_summaries) == 0:
        print("\nNo data loaded; skipping global plots.")
    else:
        combined = pd.concat(all_summaries, ignore_index=True)

        plot_global(
            combined,
            output_dir                = output_dir,
            pfx                       = pfx,
            no_ideal                  = args.no_ideal,
            aggregate_placements_flag = args.aggregate_placements,
        )

        for plot_type in ("scaling", "efficiency", "breakdown"):
            plot_global_faceted(
                combined,
                output_dir                = output_dir,
                pfx                       = pfx,
                no_ideal                  = args.no_ideal,
                aggregate_placements_flag = args.aggregate_placements,
                n_rows                    = args.grid_rows,
                plot_type                 = plot_type,
                figsize_per_cell          = tuple(args.cell_size),
            )

    print(f"\nDone.")


if __name__ == "__main__":
    main()