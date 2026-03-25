#!/usr/bin/env python3
"""
Plotting functions for distributed training analysis.
Supports scaling and time breakdown plots, per-cluster and cross-cluster comparisons.

Data format (new):
  - import_export.read_multiple_from_parquet returns (pairs, meta_df)
  - meta_df columns: sbm_job_id, sbm_tag, cluster, tot_runtime, strategy, gpus, nodes
  - pairs: list of (meta_dict, dfs) where dfs['main'] has:
      runtime, commtime, throughput  (per-iteration rows)
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from argparse import ArgumentParser
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent / "common"))
import import_export


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(parquet_files: List[str]) -> Tuple[pd.DataFrame, List[Tuple[dict, dict]]]:
    """
    Load all parquet files and return (meta_df, pairs).

    meta_df has one row per job; pairs[i] = (meta_dict, dfs) for job i,
    where dfs['main'] is the per-iteration DataFrame.
    """
    pairs, meta_df = import_export.read_multiple_from_parquet(parquet_files)
    return meta_df, pairs


def build_summary(meta_df: pd.DataFrame, pairs: List[Tuple[dict, dict]]) -> pd.DataFrame:
    """
    Aggregate per-iteration measurements into a per-job summary DataFrame.

    Returns a DataFrame with one row per job, columns:
        sbm_job_id, sbm_tag, cluster, strategy, gpus, nodes,
        throughput_mean, throughput_std,
        commtime_mean, runtime_mean,
        comm_pct_mean   (commtime / runtime * 100)
    """
    records = []
    for meta, dfs in pairs:
        if 'main' not in dfs:
            dfs['main'] = dfs['measurements']
        meas = dfs['main'].copy()

        # Skip the first iteration (warm-up) if there are enough rows
        if len(meas) > 1:
            meas = meas.iloc[1:]

        print(meas.columns)
        
        barrier_col_name = 'barrier'
        if barrier_col_name not in meas.columns:
            barrier_col_name = 'barrier_time'
        if barrier_col_name not in meas.columns:
            barrier_col_name = 'dp_comm_time'
        if barrier_col_name not in meas.columns:
            barrier_col_name = 'commtime'
            
        compute_time = meas['runtime'] - meas[barrier_col_name]

        records.append({
            'sbm_job_id':      meta['sbm_job_id'],
            'sbm_tag':         meta['sbm_tag'],
            'cluster':         meta['cluster'],
            'gpu_model':       meta['gpu_model'],
            'strategy':        meta['strategy'],
            'gpus':            meta['gpus'],
            'nodes':           meta['nodes'],
            'throughput_mean': meas['throughput'].mean(),
            'throughput_std':  meas['throughput'].std(),
            'runtime_mean':    meas['runtime'].mean(),
            'barrier_mean':    meas[barrier_col_name].mean(),
            'compute_mean':    compute_time.mean(),
            'comm_pct':        (meas[barrier_col_name] / meas['runtime'] * 100).mean(),
            'compute_pct':     (compute_time     / meas['runtime'] * 100).mean(),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Color / style helpers
# ---------------------------------------------------------------------------

def _strategy_styles(strategies: List[str]) -> Dict[str, dict]:
    """Assign a consistent color+marker per strategy."""
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['o', 's', '^', 'D', 'v', 'P', '*']
    return {
        s: {'color': palette[i % len(palette)], 'marker': markers[i % len(markers)]}
        for i, s in enumerate(sorted(set(strategies)))
    }


def _cluster_linestyles(clusters: List[str]) -> Dict[str, str]:
    """Assign a consistent linestyle per cluster."""
    styles = ['-', '--', '-.', ':']
    return {c: styles[i % len(styles)] for i, c in enumerate(sorted(set(clusters)))}

def _cluster_colors(clusters: List[str]) -> Dict[str, str]:
    """Assign a consistent color per cluster."""
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return {c: palette[i % len(palette)] for i, c in enumerate(sorted(set(clusters)))}


# ---------------------------------------------------------------------------
# 1) Scaling plot
# ---------------------------------------------------------------------------

def plot_scaling(
    summary: pd.DataFrame,
    strategies: Optional[List[str]] = None,
    clusters: Optional[List[str]] = None,
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Scaling: Throughput vs Number of GPUs",
    show_ideal: bool = True,
):
    """
    Plot throughput (samples/s) vs GPU count.

    - One line per (strategy, cluster) combination.
    - Strategies are distinguished by color+marker; clusters by linestyle.
    - Optionally draws ideal-scaling reference lines (one per strategy,
      anchored at the smallest GPU count for that strategy).

    Parameters
    ----------
    summary     : DataFrame from build_summary()
    strategies  : subset of strategies to plot (None = all)
    clusters    : subset of clusters to plot (None = all)
    output_file : save path; None => interactive display
    figsize     : figure size
    title       : plot title
    show_ideal  : whether to draw dashed ideal-scaling lines
    """
    df = summary.copy()
    if strategies:
        df = df[df['strategy'].isin(strategies)]
    if clusters:
        df = df[df['cluster'].isin(clusters)]

    fig, ax = plt.subplots(figsize=figsize)

    strat_styles = _strategy_styles(df['strategy'].unique())
    clust_ls     = _cluster_linestyles(df['cluster'].unique())
    clust_color  = _cluster_colors(df['cluster'].unique())

    # Track what we've added to the legend to avoid duplicates
    legend_handles = {}

    for (strategy, cluster), grp in df.groupby(['strategy', 'cluster']):
        grp = grp.sort_values('gpus')
        style = strat_styles[strategy]
        ls    = clust_ls[cluster]
        color = clust_color[cluster]
        label = f"{strategy} — {cluster}"

        line = ax.errorbar(
            grp['gpus'],
            grp['throughput_mean'],
            yerr=grp['throughput_std'],
            label=label,
            color=color, # style['color'],
            marker=style['marker'],
            linestyle=ls,
            linewidth=2,
            markersize=8,
            capsize=4,
        )
        legend_handles[label] = line

        # Ideal scaling anchor: smallest GPU count for this (strategy, cluster)
        if show_ideal:
            base = grp.iloc[0]
            gpus_range = np.array(sorted(df['gpus'].unique()))
            gpus_range = gpus_range[gpus_range >= base['gpus']]
            ideal = base['throughput_mean'] * (gpus_range / base['gpus'])
            ax.plot(
                gpus_range, ideal,
                color=style['color'],
                linestyle=':',
                linewidth=1,
                alpha=0.5,
            )

    # Proxy for ideal scaling in legend
    if show_ideal:
        ax.plot([], [], color='gray', linestyle=':', linewidth=1, alpha=0.7,
                label='Ideal scaling')

    all_gpus = sorted(df['gpus'].unique())
    ax.set_xticks(all_gpus)
    ax.set_xticklabels([str(g) for g in all_gpus])
    ax.set_xlabel('Number of GPUs', fontsize=12)
    ax.set_ylabel('Throughput (samples/s)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, output_file, "Scaling plot")
    return fig, ax


# ---------------------------------------------------------------------------
# 2) Breakdown plot
# ---------------------------------------------------------------------------

def plot_breakdown(
    summary: pd.DataFrame,
    strategies: Optional[List[str]] = None,
    clusters: Optional[List[str]] = None,
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Time Breakdown: Compute vs Communication (%)",
):
    """
    Grouped stacked bar chart of compute% vs comm% per GPU count.

    Each (strategy, cluster) pair gets its own bar group per x-tick (GPU count).
    Compute is the solid fill; communication is hatched on top.

    Parameters
    ----------
    summary     : DataFrame from build_summary()
    strategies  : subset of strategies to plot (None = all)
    clusters    : subset of clusters to plot (None = all)
    output_file : save path; None => interactive display
    figsize     : figure size
    title       : plot title
    """
    df = summary.copy()
    if strategies:
        df = df[df['strategy'].isin(strategies)]
    if clusters:
        df = df[df['cluster'].isin(clusters)]

    fig, ax = plt.subplots(figsize=figsize)

    strat_styles = _strategy_styles(df['strategy'].unique())
    clust_ls     = _cluster_linestyles(df['cluster'].unique())  # unused visually but kept for consistency

    all_gpus   = sorted(df['gpus'].unique())
    combos     = sorted(df.groupby(['strategy', 'cluster']).groups.keys())
    n_combos   = len(combos)

    group_width = 0.7
    bar_width   = group_width / n_combos
    x           = np.arange(len(all_gpus))

    legend_added = set()

    for i, (strategy, cluster) in enumerate(combos):
        grp = df[(df['strategy'] == strategy) & (df['cluster'] == cluster)]
        style = strat_styles[strategy]
        color = style['color']
        label_base = f"{strategy} — {cluster}"

        # Reindex to all gpu counts (fill missing with 0)
        grp = grp.set_index('gpus').reindex(all_gpus).fillna(0)
        compute_pcts = grp['compute_pct'].values
        comm_pcts    = grp['comm_pct'].values

        offsets = x - group_width / 2 + i * bar_width + bar_width / 2

        # Compute bar
        lbl_compute = f"{label_base} — Compute" if label_base not in legend_added else "_nolegend_"
        ax.bar(offsets, compute_pcts, width=bar_width,
               color=color, alpha=0.85, label=lbl_compute)

        # Comm bar stacked on top
        lbl_comm = f"{label_base} — Comm" if label_base not in legend_added else "_nolegend_"
        ax.bar(offsets, comm_pcts, width=bar_width,
               bottom=compute_pcts, color=color, alpha=0.4,
               hatch='//', label=lbl_comm)

        legend_added.add(label_base)

    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in all_gpus])
    ax.set_xlabel('Number of GPUs', fontsize=12)
    ax.set_ylabel('Time (%)', fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.035))
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    _save_or_show(fig, output_file, "Breakdown plot")
    return fig, ax


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _save_or_show(fig, output_file: Optional[str], label: str = "Plot"):
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  {label} saved to: {output_file}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main: generate per-cluster + cross-cluster plots
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser(description="Plot distributed training results from parquet files")
    parser.add_argument("parquet_files", nargs="+", help="Parquet file(s) to plot")
    parser.add_argument("--output-dir", default="plots/baselines", help="Output directory for plots")
    parser.add_argument("--prefix", default="", help="Optional prefix for output filenames")
    parser.add_argument("--no-ideal", action="store_true", help="Suppress ideal-scaling lines")
    parser.add_argument("--strategies", nargs="*", help="Filter to specific strategies")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pfx = f"{args.prefix}_" if args.prefix else ""

    # ------------------------------------------------------------------
    print("Loading data...")
    meta_df, pairs = load_data(args.parquet_files)
    meta_df['cluster'] = meta_df['cluster'] + meta_df['gpu_model']
    
    for p, _ in pairs:
        p['cluster'] = p['cluster'] # + p['gpu_model']
    print(meta_df.to_string(index=False))

    summary = build_summary(meta_df, pairs)
    print(f"\nSummary ({len(summary)} jobs):")
    print(summary[['sbm_tag', 'cluster', 'gpu_model', 'strategy', 'gpus',
                    'throughput_mean', 'comm_pct', 'compute_pct']].to_string(index=False))

    strategies = args.strategies  # None = all
    clusters   = sorted(summary['cluster'].unique())

    all_strategies = sorted(summary['strategy'].unique()) if not strategies else strategies

    # ------------------------------------------------------------------
    # Per-strategy plots (all clusters overlaid)
    # ------------------------------------------------------------------
    for strategy in all_strategies:
        print(f"\n[Strategy: {strategy}]")

        plot_scaling(
            summary,
            strategies=[strategy],
            output_file=str(output_dir / f"{pfx}{strategy}_scaling.png"),
            title=f"Scaling — {strategy}",
            show_ideal=not args.no_ideal,
        )

        plot_breakdown(
            summary,
            strategies=[strategy],
            output_file=str(output_dir / f"{pfx}{strategy}_breakdown.png"),
            title=f"Time Breakdown — {strategy}",
        )

    # ------------------------------------------------------------------
    # Per-strategy + per-cluster plots
    # ------------------------------------------------------------------
    for strategy in all_strategies:
        for cluster in clusters:
            subset = summary[(summary['strategy'] == strategy) & (summary['cluster'] == cluster)]
            if subset.empty:
                continue
            print(f"\n[Strategy: {strategy} | Cluster: {cluster}]")

            plot_scaling(
                summary,
                strategies=[strategy],
                clusters=[cluster],
                output_file=str(output_dir / f"{pfx}{strategy}_{cluster}_scaling.png"),
                title=f"Scaling — {strategy} on {cluster}",
                show_ideal=not args.no_ideal,
            )

            plot_breakdown(
                summary,
                strategies=[strategy],
                clusters=[cluster],
                output_file=str(output_dir / f"{pfx}{strategy}_{cluster}_breakdown.png"),
                title=f"Time Breakdown — {strategy} on {cluster}",
            )

    # ------------------------------------------------------------------
    # Cross-cluster comparison plots (only when >1 cluster present)
    # ------------------------------------------------------------------
    print("\n[Cross-cluster comparison — all strategies]")

    plot_scaling(
        summary,
        strategies=strategies,
        output_file=str(output_dir / f"{pfx}all_scaling.png"),
        title="Scaling — All Strategies, All Clusters",
        show_ideal=not args.no_ideal,
    )

    plot_breakdown(
        summary,
        strategies=strategies,
        output_file=str(output_dir / f"{pfx}all_breakdown.png"),
        title="Time Breakdown — All Strategies, All Clusters",
    )

    print(f"\nDone. All plots saved to: {output_dir}/")


if __name__ == "__main__":
    main()