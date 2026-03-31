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

import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
from argparse import ArgumentParser
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent / "common"))
import import_export
from command_map import _STRATEGY_MODELS_MAP, get_default_model

PLACEMENT_SHORT_NAME_MAP = {
    "INTRA_L1_RANDOM"       :    "L1",             
    "INTRA_GROUP_RANDOM"    :    "IG",          
    "INTER_GROUP_RANDOM"    :    "EG",          
    "INTRA_GROUP_SAME_L1_2" :    "IG-L1-2",
    "INTRA_GROUP_SAME_L1_4" :    "IG-L1-4",
    "INTER_GROUP_SAME_L1_2" :    "EG-L1-2",
    "INTER_GROUP_SAME_L1_4" :    "EG-L1-4",
    "NA"                    :    "N/A"
}


def _placement_linestyles(class_tags: List[str]) -> Dict[str, str]:
    """Assign a consistent linestyle per placement (class_tag)."""
    styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]
    return {t: styles[i % len(styles)] for i, t in enumerate(sorted(set(class_tags)))}


def _base_strategy_markers(base_strategies: List[str]) -> Dict[str, str]:
    """Assign a consistent marker per base strategy."""
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '8']
    return {s: markers[i % len(markers)] for i, s in enumerate(sorted(set(base_strategies)))}

def _placement_markers(class_tags: List[str]) -> Dict[str, str]:
    """Assign a consistent marker per placement (class_tag)."""
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '8']
    return {t: markers[i % len(markers)] for i, t in enumerate(sorted(set(class_tags)))}


def _base_strategy_linestyles(base_strategies: List[str]) -> Dict[str, str]:
    styles = ['-', '--', '-.', ':']
    return {s: styles[i % len(styles)] for i, s in enumerate(sorted(set(base_strategies)))}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


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
        # TODO double-check
        if len(meas) > 1:
            n_ranks = int(meta['gpus'])
            chunk_size = len(meas) // n_ranks
            meas = pd.concat([
                meas.iloc[i * chunk_size + 1 : (i + 1) * chunk_size]
                for i in range(n_ranks)
            ]).reset_index(drop=True)

        print([int(t) for t in meas['throughput'].values])
        
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
# Aggregation helper
# ---------------------------------------------------------------------------

def aggregate_placements(summary: pd.DataFrame, agg_type: str = 'mean') -> pd.DataFrame:
    """
    Collapse all placements (class_tags) of the same (base_strategy, cluster, gpus)
    into a single row by averaging throughput and time-breakdown metrics.

    The resulting 'strategy' and 'class_tag' are set to the base_strategy so that
    downstream plotting treats it as a single series.
    """
    if 'base_strategy' not in summary.columns:
        summary = summary.copy()
        summary[['base_strategy', 'class_tag']] = summary['strategy'].apply(
            lambda s: pd.Series(split_strategy(s))
        )

    agg = (
        summary
        .groupby(['base_strategy', 'cluster', 'gpus', 'gpu_model'], as_index=False)
        .agg(
            sbm_job_id     =('sbm_job_id',      'first'),
            sbm_tag        =('sbm_tag',         'first'),
            nodes          =('nodes',           'first'),
            throughput_mean=('throughput_mean', agg_type),
            throughput_std =('throughput_std',  agg_type),
            runtime_mean   =('runtime_mean',    agg_type),
            barrier_mean   =('barrier_mean',    agg_type),
            compute_mean   =('compute_mean',    agg_type),
            comm_pct       =('comm_pct',        agg_type),
            compute_pct    =('compute_pct',     agg_type),
        )
    )

    # Make strategy == base_strategy and class_tag empty so styles are consistent
    agg['strategy']  = agg['base_strategy']
    agg['class_tag'] = ''

    return agg


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
    figsize: Tuple[int, int] = (20, 7),
    title: str = "Scaling: Throughput vs Number of GPUs",
    show_ideal: bool = True,
    plot_efficiency: bool = False,
):
    df = summary.copy()
    if strategies:
        df = df[df['strategy'].isin(strategies)]
    if clusters:
        df = df[df['cluster'].isin(clusters)]
 
    if 'base_strategy' not in df.columns:
        df[['base_strategy', 'class_tag']] = df['strategy'].apply(
            lambda s: pd.Series(split_strategy(s))
        )
 
    fig, ax = plt.subplots(figsize=figsize)
 
    # Optional second figure
    if plot_efficiency:
        fig_eff, ax_eff = plt.subplots(figsize=figsize)
 
    clust_color    = _cluster_colors(df['cluster'].unique())
    place_ls       = _placement_linestyles(df['class_tag'].unique())
    base_markers   = _base_strategy_markers(df['base_strategy'].unique())
 
    min_eff = 1.0
    for (strategy, cluster), grp in df.groupby(['strategy', 'cluster']):
        grp = grp.sort_values('gpus')
        base_strat = grp['base_strategy'].iloc[0]
        class_tag  = grp['class_tag'].iloc[0]
 
        color  = clust_color[cluster]
        ls     = place_ls[class_tag]
        marker = base_markers[base_strat]
        label  = f"{base_strat} - {PLACEMENT_SHORT_NAME_MAP[class_tag]} - {cluster}" if class_tag else f"{base_strat} — {cluster}"

        # --- HANDLE DUPLICATES ---
        if grp['gpus'].duplicated().any():
            dup_counts = grp['gpus'].value_counts()
            dups = dup_counts[dup_counts > 1]
 
            print(f"[WARN] Duplicate GPU entries detected for {label}:")
            for gpu, count in dups.items():
                sub = grp[grp['gpus'] == gpu]
                best_idx = sub['throughput_mean'].idxmax()
                best_val = sub.loc[best_idx, 'throughput_mean']
                print(f"  - gpus={gpu}: {count} rows → keeping max throughput {best_val:.3f}")
 
            grp = grp.loc[
                grp.groupby('gpus')['throughput_mean'].idxmax()
            ].sort_values('gpus')
 
        # --- MAIN SCALING PLOT ---
        ax.errorbar(
            grp['gpus'], grp['throughput_mean'],
            yerr=grp['throughput_std'],
            label=label,
            color=color, marker=marker, linestyle=ls,
            linewidth=1, markersize=2, capsize=2,
        )
 
        if show_ideal:
            base = grp.iloc[0]
            gpus_range = np.array(sorted(df['gpus'].unique()))
            gpus_range = gpus_range[gpus_range >= base['gpus']]
            ideal = base['throughput_mean'] * (gpus_range / base['gpus'])
            ax.plot(gpus_range, ideal, color=color, linestyle=':', linewidth=1, alpha=0.4)
 
        # --- EFFICIENCY PLOT ---
        if plot_efficiency:
            base = grp.iloc[0]
            g0   = base['gpus']
            T0   = base['throughput_mean']
 
            eff = (grp['throughput_mean'] / T0) * (g0 / grp['gpus'])
            min_eff = min(min_eff, eff.min())
 
            ax_eff.plot(
                grp['gpus'], eff,
                label=label,
                color=color, marker=marker, linestyle=ls,
                linewidth=1, markersize=2,
            )
 
    # --- FINALIZE MAIN PLOT ---
    if show_ideal:
        ax.plot([], [], color='gray', linestyle=':', linewidth=1, alpha=0.7, label='Ideal scaling')
 
    all_gpus = sorted(df['gpus'].unique())
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_xticks(all_gpus)
    ax.set_xticklabels([str(g) for g in all_gpus])
    ax.set_xlabel('Number of GPUs', fontsize=12)
    ax.set_ylabel('Throughput (samples/s)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.01, 0.5), borderaxespad=0)
    ax.grid(True, alpha=0.3)
 
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    _save_or_show(fig, output_file, "Scaling plot")
 
    # --- FINALIZE EFFICIENCY PLOT ---
    if plot_efficiency:
        ax_eff.axhline(1.0, linestyle=':', linewidth=1, alpha=0.7, color='gray', label='Ideal efficiency')
 
        ax_eff.set_xscale('log', base=2)
        ax_eff.set_yscale('log', base=2)
        ax_eff.set_xticks(all_gpus)
        ax_eff.set_xticklabels([str(g) for g in all_gpus])
        ax_eff.set_yticks(
            [e for e in np.linspace(min_eff, 1.0, 10)],
            labels=[f'{int(e*100.0)}%' for e in np.linspace(min_eff, 1.0, 10)]
        )
        ax_eff.set_xlabel('Number of GPUs', fontsize=12)
        ax_eff.set_ylabel('Parallel Efficiency', fontsize=12)
        ax_eff.set_title("Scaling Efficiency", fontsize=14, fontweight='bold')
        ax_eff.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, 0.5), borderaxespad=0)
        ax_eff.grid(True, alpha=0.3)
 
        fig_eff.tight_layout(rect=[0, 0, 0.78, 1])
 
        eff_file = None
        if output_file:
            eff_file = output_file.replace(".png", "_efficiency.png")
 
        _save_or_show(fig_eff, eff_file, "Efficiency plot")
 
        return (fig, ax), (fig_eff, ax_eff)
 
    return fig, ax, (None, None)


# ---------------------------------------------------------------------------
# 2) Breakdown plot
# ---------------------------------------------------------------------------

def _lighten_color(color, amount: float):
    """Lighten a color by blending it toward white. amount in [0, 1], 0=original, 1=white."""
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames.get(color, color)
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], min(1.0, 1 - amount * (1 - c[1])), c[2])
    except Exception:
        return color
    
    
def _cluster_placement_colors(cluster: str, class_tags: List[str], base_color: str) -> Dict[str, tuple]:
    """
    Generate perceptually distinct shades for each placement of a cluster.
    Uses fixed lightness levels rather than progressively lightening toward white.
    """
    import matplotlib.colors as mc
    import colorsys

    # Fixed lightness levels — dark to medium, all clearly visible
    lightness_levels = [0.35, 0.50, 0.65, 0.75, 0.45, 0.60]

    c = mc.to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(*c)

    result = {}
    for i, tag in enumerate(sorted(class_tags)):
        new_l = lightness_levels[i % len(lightness_levels)]
        result[tag] = colorsys.hls_to_rgb(h, new_l, s)
    return result


def plot_breakdown(
    summary: pd.DataFrame,
    strategies: Optional[List[str]] = None,
    clusters: Optional[List[str]] = None,
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 9),
    title: str = "Time Breakdown: Compute vs Communication (%)",
):
    df = summary.copy()
    if strategies:
        df = df[df['strategy'].isin(strategies)]
    if clusters:
        df = df[df['cluster'].isin(clusters)]

    if 'base_strategy' not in df.columns:
        df[['base_strategy', 'class_tag']] = df['strategy'].apply(
            lambda s: pd.Series(split_strategy(s))
        )

    fig, ax = plt.subplots(figsize=figsize)

    clust_color = _cluster_colors(df['cluster'].unique())
    
    place_colors = {}
    for cluster in df['cluster'].unique():
        tags_in_cluster = sorted(df[df['cluster'] == cluster]['class_tag'].unique())
        shades = _cluster_placement_colors(cluster, tags_in_cluster, clust_color[cluster])
        for tag, shade in shades.items():
            place_colors[(cluster, tag)] = shade

    place_list   = sorted(df['class_tag'].unique())
    n_places     = len(place_list)
    lighten_steps = [i / max(n_places, 1) * 0.55 for i in range(n_places)]
    place_lighten = {t: lighten_steps[i] for i, t in enumerate(place_list)}

    all_gpus = sorted(df['gpus'].unique())
    combos   = sorted(df.groupby(['strategy', 'cluster']).groups.keys())
    n_combos = len(combos)

    # Wider groups with a small gap between bars in the same group
    group_width = 0.75
    gap         = 0.04
    bar_width   = (group_width - gap * (n_combos - 1)) / n_combos
    x           = np.arange(len(all_gpus))

    legend_added = set()

    for i, (strategy, cluster) in enumerate(combos):
        grp = df[(df['strategy'] == strategy) & (df['cluster'] == cluster)]
        base_strat = grp['base_strategy'].iloc[0]
        class_tag  = grp['class_tag'].iloc[0]

        color = place_colors.get((cluster, class_tag), clust_color[cluster])

        label_base = (
            f"{base_strat} - {PLACEMENT_SHORT_NAME_MAP[class_tag]} - {cluster}" if class_tag
            else f"{base_strat} - {cluster}"
        )

        # --- HANDLE DUPLICATES ---
        if grp['gpus'].duplicated().any():
            dup_counts = grp['gpus'].value_counts()
            dups = dup_counts[dup_counts > 1]

            print(f"[WARN] Duplicate GPU entries detected for {label_base}:")
            for gpu, count in dups.items():
                print(f"  - gpus={gpu}: {count} rows merged")

            grp = (
                grp.groupby('gpus', as_index=False)
                .agg({
                    'compute_pct': 'mean',
                    'comm_pct': 'mean'
                })
            )

        grp = grp.set_index('gpus').reindex(all_gpus).fillna(0)

        offset = x - group_width / 2 + i * (bar_width + gap) + bar_width / 2

        compute_vals = grp['compute_pct'].values
        comm_vals    = grp['comm_pct'].values

        ax.bar(offset, compute_vals, width=bar_width,
            color=color, alpha=0.95,
            label=f"{label_base} - Comp" if label_base not in legend_added else "_nolegend_")

        ax.bar(offset, comm_vals, width=bar_width,
            bottom=compute_vals, color=color, alpha=0.45,
            label=f"{label_base} - Comm" if label_base not in legend_added else "_nolegend_")

        legend_added.add(label_base)

    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in all_gpus])
    ax.set_xlabel('Number of GPUs', fontsize=12)
    ax.set_ylabel('Time (%)', fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Legend on the right, outside the plot area
    ax.legend(
        fontsize=8,
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0,
    )
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout(rect=[0, 0, 0.78, 1])  # leave room for the right-side legend
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

def split_strategy(s: str):
    """Split 'ddp_INTRA_GROUP_RANDOM' -> ('ddp', 'INTRA_GROUP_RANDOM').
    Splits at the first underscore followed by an uppercase letter.
    Falls back to (s, '') if no uppercase segment is found.
    """
    m = re.match(r'^(.*?)_([A-Z][A-Z0-9_]*)$', s)
    if m:
        return m.group(1), m.group(2)
    return s, ''

def main():
    parser = ArgumentParser(description="Plot distributed training results from parquet files")
    parser.add_argument("parquet_files", nargs="+", help="Parquet file(s) to plot")
    parser.add_argument("--output-dir", default="plots/baselines", help="Output directory for plots")
    parser.add_argument("--prefix", default="", help="Optional prefix for output filenames")
    parser.add_argument("--no-ideal", action="store_true", help="Suppress ideal-scaling lines")
    parser.add_argument("--strategies", nargs="*", help="Filter to specific base strategies")
    parser.add_argument(
        "--aggregate-placements", action="store_true",
        help=(
            "In the all-placements × all-clusters overview plots, average across placements "
            "instead of drawing one line/bar-group per placement."
        ),
    )
    parser.add_argument("--only-all", action="store_true", help=("Produce only *all* plots"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pfx = f"{args.prefix}_" if args.prefix else ""

    # Subdirectory layout:
    #   output_dir/                  ← *all* overview plots (cross-placement × cross-cluster)
    #   output_dir/per_cluster/      ← per-cluster, per-base-strategy placement comparisons
    #   output_dir/cross_cluster/    ← cross-cluster comparison per strategy+placement
    per_cluster_dir  = output_dir / "per_cluster"
    cross_cluster_dir = output_dir / "cross_cluster"

    # ------------------------------------------------------------------
    print("Loading data...")
    
    def meta_transform(meta):
        match = re.search(r'class-([^_]+(?:_[^_]+)*)_rep', meta.get('sbm_tag', ''))
        placement_class = match.group(1) if match else "NA"
        if str(meta['strategy']).endswith('orig'):
            meta['strategy'] = str(meta['strategy']).strip('orig')
        if not meta.get('model'):
            meta['model'] = get_default_model(meta["strategy"])
        meta['strategy'] = f'{meta["strategy"]}_{meta['model']}_{placement_class}'
        meta['cluster'] = meta['cluster'] + meta['gpu_model']
        return meta
        
    pairs, meta_df = import_export.read_multiple_from_parquet(args.parquet_files, meta_transform)

    print(meta_df.to_string(index=False))

    summary = build_summary(meta_df, pairs)

    # Decompose strategy back into base + class_tag for grouping
    summary[['base_strategy', 'class_tag']] = summary['strategy'].apply(
        lambda s: pd.Series(split_strategy(s))
    )

    print(f"\nSummary ({len(summary)} jobs):")
    print(summary[['sbm_tag', 'cluster', 'gpu_model', 'strategy', 'gpus',
                    'throughput_mean', 'comm_pct', 'compute_pct']].to_string(index=False))

    clusters        = sorted(summary['cluster'].unique())
    base_strategies = sorted(summary['base_strategy'].unique())
    if args.strategies:
        base_strategies = [s for s in base_strategies if s in args.strategies]
    all_strategies = sorted(summary['strategy'].unique())

    if not args.only_all:
        # ------------------------------------------------------------------
        # 1) Per-cluster, per-base-strategy: compare placements (class_tags)
        #    → saved under output_dir/per_cluster/
        # ------------------------------------------------------------------
        print("\n[Per-cluster placement comparison]")
        for cluster in clusters:
            for base_strat in base_strategies:
                strats_here = summary[
                    (summary['cluster'] == cluster) &
                    (summary['base_strategy'] == base_strat)
                ]['strategy'].unique().tolist()
                if not strats_here:
                    continue

                tag = f"{base_strat}_on_{cluster}"
                print(f"  {tag}  placements={strats_here}")

                plot_scaling(
                    summary, strategies=strats_here, clusters=[cluster],
                    output_file=str(per_cluster_dir / f"{pfx}{tag}_scaling.png"),
                    title=f"Scaling — {base_strat} placements on {cluster}",
                    show_ideal=not args.no_ideal,
                )
                plot_breakdown(
                    summary, strategies=strats_here, clusters=[cluster],
                    output_file=str(per_cluster_dir / f"{pfx}{tag}_breakdown.png"),
                    title=f"Time Breakdown — {base_strat} placements on {cluster}",
                )

        # ------------------------------------------------------------------
        # 2) Cross-cluster, per strategy+placement: compare systems
        #    → saved under output_dir/cross_cluster/
        # ------------------------------------------------------------------
        if len(clusters) > 1:
            print("\n[Cross-cluster comparison per strategy+placement]")
            for strategy in all_strategies:
                base = summary[summary['strategy'] == strategy]['base_strategy'].iloc[0]
                if args.strategies and base not in args.strategies:
                    continue
                clust_here = summary[summary['strategy'] == strategy]['cluster'].unique().tolist()
                if not clust_here:
                    continue

                print(f"  {strategy}  clusters={clust_here}")

                plot_scaling(
                    summary, strategies=[strategy], clusters=clust_here,
                    output_file=str(cross_cluster_dir / f"{pfx}{strategy}_xcluster_scaling.png"),
                    title=f"Scaling — {strategy} across clusters",
                    show_ideal=not args.no_ideal
                )
                plot_breakdown(
                    summary, strategies=[strategy], clusters=clust_here,
                    output_file=str(cross_cluster_dir / f"{pfx}{strategy}_xcluster_breakdown.png"),
                    title=f"Time Breakdown — {strategy} across clusters",
                )

    # ------------------------------------------------------------------
    # 3) Cross-cluster overview per base-strategy: all placements × all clusters
    #    → saved directly in output_dir/
    #
    #    With --aggregate-placements: average across class_tags first,
    #    producing one line/bar-group per (base_strategy, cluster).
    # ------------------------------------------------------------------
    print("\n[Cross-cluster + cross-placement overview per base strategy]")
    for base_strat in base_strategies:
        sub = summary[summary['base_strategy'] == base_strat]

        if args.aggregate_placements:
            plot_summary = aggregate_placements(sub)
            agg_label = " (placements averaged)"
        else:
            plot_summary = sub
            agg_label = ""

        strats_here = plot_summary['strategy'].unique().tolist()
        mode_tag = "agg" if args.aggregate_placements else "all"
        print(f"  {base_strat}  [{mode_tag}] strategies={strats_here}")

        plot_scaling(
            plot_summary, strategies=strats_here,
            output_file=str(output_dir / f"{pfx}{base_strat}_all_scaling{'_aggr' if args.aggregate_placements else ''}.png"),
            title=f"Scaling — {base_strat} (all clusters{agg_label})",
            show_ideal=not args.no_ideal,
            plot_efficiency=True,
        )
        plot_breakdown(
            plot_summary, strategies=strats_here,
            output_file=str(output_dir / f"{pfx}{base_strat}_all_breakdown{'_aggr' if args.aggregate_placements else ''}.png"),
            title=f"Time Breakdown — {base_strat} (all clusters{agg_label})",
        )

    # ------------------------------------------------------------------
    # 4) Global overview: all strategies × all placements × all clusters
    #    → saved directly in output_dir/
    #
    #    With --aggregate-placements: average across class_tags per
    #    (base_strategy, cluster) before plotting.
    # ------------------------------------------------------------------
    print("\n[Global overview: all strategies × all placements × all clusters]")

    if args.aggregate_placements:
        global_summary = aggregate_placements(summary)
        agg_label = " (placements averaged)"
    else:
        global_summary = summary
        agg_label = ""

    all_strats_global = global_summary['strategy'].unique().tolist()
    mode_tag = "agg" if args.aggregate_placements else "all"
    print(f"  [{mode_tag}] strategies={sorted(all_strats_global)}")

    plot_scaling(
        global_summary,
        strategies=all_strats_global,
        output_file=str(output_dir / f"{pfx}global_all_scaling{'_aggr' if args.aggregate_placements else ''}.png"),
        title=f"Scaling — all strategies, all clusters{agg_label}",
        show_ideal=not args.no_ideal,
        plot_efficiency=True,
    )

    print(f"\nDone.")
    print(f"  Overview plots  → {output_dir}/")
    print(f"  Per-cluster     → {per_cluster_dir}/")
    if len(clusters) > 1:
        print(f"  Cross-cluster   → {cross_cluster_dir}/")


if __name__ == "__main__":
    main()