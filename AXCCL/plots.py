from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.append(str(Path(__file__).parent.parent / "common"))
import import_export
from utils.plots import format_bytes, parse_bytes, create_color_map, create_linestyle_map
from constants.plots import *
from constants.systems import compute_theoretical_bandwidth


# ==============================
# Matplotlib config
# ==============================

plt.rc('axes', titlesize=FONT_AXES - 2)
plt.rc('axes', labelsize=FONT_AXES - 2)
plt.rc('xtick', labelsize=FONT_TICKS)
plt.rc('ytick', labelsize=FONT_TICKS)
plt.rc('legend', fontsize=FONT_LEGEND + 1)
plt.rc('figure', titlesize=FONT_TITLE - 10)


def format_bytes_tick(x, _pos):
    return format_bytes(int(x), binary=True, precision=0)


# ==============================
# Tag parsing
# ==============================

def parse_implementation(impl: str) -> str:
    parts = impl.split('__')
    return impl if len(parts) <= 1 else parts[-1]


TOPOLOGY_MAP = {
    'same-l1': 'same-l1',
    'same-group': 'same-group',
    'inter-group': 'inter-group',
    'distance_2.0': 'same-l1',
    'distance_4.0': 'same-group',
    'distance_5.0': 'inter-group',
    '2.0': 'same-l1',
    '4.0': 'same-group',
    '5.0': 'inter-group',
    'unknown': '?',
}


def get_topology_from_tag(tag: str) -> str:
    return TOPOLOGY_MAP[tag.split('__')[-1]]


def get_primitive_from_tag(tag: str) -> str:
    return tag.split('__')[0]


COMM_TYPE_MAP = {
    'gpu_buff': 'G2G',
    'cpu_buff': 'C2C',
    'gpu': 'G2G',
    'cpu': 'C2C',
    'host2dev': 'C2G',
    'dev2host': 'G2C',
}


def get_comm_type(tag: str) -> str:
    parts = tag.split('__')
    comm_type = parts[1]
    if comm_type == 'hybrid':
        direction = parts[2].split('_')[0]
        comm_type = direction
    return COMM_TYPE_MAP[comm_type]


def natural_sort_key(s):
    import re
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]


# ==========================================================
# Build statistics dataframe from dfs['raw']
# ==========================================================

def build_statistics_dataframe(meta_df_dict_pairs, gib_to_gbps):
    rows = []

    for meta, dfs in meta_df_dict_pairs:

        raw = dfs['raw'].copy()

        raw['bandwidth'] = raw['bandwidth_GiB_s'] * gib_to_gbps
        raw['latency'] = raw['time_s'] * 1e6

        grouped = raw.groupby('transfer_size_B')

        stats = grouped.agg({
            'bandwidth': ['mean', 'std', 'min', 'max'],
            'latency': ['mean', 'std', 'min', 'max']
        }).reset_index()

        stats.columns = [
            'transfer_size_B',
            'mean_bandwidth', 'std_bandwidth', 'min_bandwidth', 'max_bandwidth',
            'mean_latency', 'std_latency', 'min_latency', 'max_latency'
        ]

        for _, row in stats.iterrows():
            rows.append({
                'cluster': meta['cluster'],
                'primitive': meta['primitive'],
                'implementation': meta['implementation'],
                'comm_type': meta['comm_type'],
                'topology': meta['topology'],
                'peering': meta.get('peering', 'na'),
                **row.to_dict()
            })

    return pd.DataFrame(rows)


# ==========================================================
# Variability detection
# ==========================================================

def detect_high_variability(stats_df, metric='bandwidth', threshold=0.15):
    std_col = f'std_{metric}'
    mean_col = f'mean_{metric}'

    stats_df['cv'] = stats_df[std_col] / stats_df[mean_col]

    high = stats_df[stats_df['cv'] > threshold]

    if not high.empty:
        print("\n⚠️ High variability detected")

        # Coefficient of variation
        # CV = sigma / mu

        for _, row in high.iterrows():
            print(
                f"  {row['cluster']:<8} | {row['primitive']:<8} | "
                f"{row['implementation']:<11} | {row['comm_type']:<5} | "
                f"{row['topology']:<12} | "
                f"size={format_bytes(row['transfer_size_B'], True, 0):<7} | "
                f"CV={row['cv']:.2f}"
            )

        print()


# ==========================================================
# Grouped plotting
# ==========================================================

def plot_grouped_experiments(
        stats_df,
        peering,
        group_by='implementation',
        metric='bandwidth',
        outdir=Path('plots'),
        cluster=None,
        primitive=None,
        show_std=True,
        show_minmax=False,
        same_y_lim=True,
):
    filtered = stats_df.copy()

    filtered = filtered[filtered['peering'] == peering]
    if cluster:
        filtered = filtered[filtered['cluster'] == cluster]
    if primitive:
        filtered = filtered[filtered['primitive'] == primitive]

    if filtered.empty:
        return

    group_values = sorted(filtered[group_by].unique(), key=natural_sort_key)

    n_groups = len(group_values)
    n_cols = min(2, n_groups)
    n_rows = int(np.ceil(n_groups / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(10*n_cols, 8*n_rows))

    axes = np.array(axes).flatten()

    ylabel = 'Goodput (Gb/s)' if metric == 'bandwidth' else 'Latency (μs)'
    mean_col = f'mean_{metric}'
    std_col = f'std_{metric}'
    min_col = f'min_{metric}'
    max_col = f'max_{metric}'

    color_map = create_color_map(filtered['topology'].unique())
    linestyle_map = create_linestyle_map(filtered['comm_type'].unique())

    # Determine global y-limits
    all_y = filtered[mean_col].values
    y_lim = None
    if same_y_lim and len(all_y) > 0:
        y_min = np.min(all_y)
        y_max = np.max(all_y)
        padding = 0.05 * y_max
        y_lim = (y_min, y_max + padding)

    for idx, group_val in enumerate(group_values):

        ax = axes[idx]
        df_group = filtered[filtered[group_by] == group_val]

        combinations = df_group[['comm_type', 'topology']].drop_duplicates()

        for _, combo in combinations.iterrows():

            df_line = df_group[
                (df_group['comm_type'] == combo['comm_type']) &
                (df_group['topology'] == combo['topology'])
            ].sort_values('transfer_size_B')

            x = df_line['transfer_size_B']
            y = df_line[mean_col]
            std = df_line[std_col]
            ymin = df_line[min_col]
            ymax = df_line[max_col]

            color = color_map[combo['topology']]
            linestyle = linestyle_map[combo['comm_type']]

            label = f"{combo['comm_type']} | {combo['topology']}"

            ax.plot(x, y,
                    marker='o',
                    linewidth=2,
                    markersize=4,
                    color=color,
                    linestyle=linestyle,
                    label=label)

            if show_std:
                ax.fill_between(x, y - std, y + std,
                                alpha=0.25, color=color)

            if show_minmax:
                ax.fill_between(x, ymin, ymax,
                                alpha=0.12, color=color)
            
            if metric == 'bandwidth':
                theoretical_bw = compute_theoretical_bandwidth(
                    cluster,
                    combo['comm_type'],
                    combo['topology']
                )
                if theoretical_bw is not None:
                    if isinstance(y_lim, tuple):
                        y_lim_min, y_lim_max = y_lim
                        y_lim = (y_lim_min, max(y_lim_max, theoretical_bw))
                    ax.axhline(
                        theoretical_bw,
                        linestyle='--',
                        linewidth=1.5,
                        alpha=0.7,
                        color=color
                    )
                    ax.text(x.iloc[0], theoretical_bw, 'Expected Goodput',
                            fontsize=9, alpha=0.7, color=color,
                            verticalalignment='bottom', horizontalalignment='left')

        ax.set_xscale('log', base=2)
        ax.set_xlabel('Message Size')
        if idx % n_cols == 0:
            ax.set_ylabel(ylabel)
        ax.set_title(group_val)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(FuncFormatter(format_bytes_tick))

        if y_lim:
            ax.set_ylim(y_lim)

        ax.legend()

    for idx in range(n_groups, len(axes)):
        axes[idx].set_visible(False)

    cluster_str = cluster if cluster else 'all_clusters'
    primitive_str = primitive if primitive else 'all_primitives'

    fig.suptitle(
        f'System: {cluster_str.capitalize()} - {primitive_str} - Peering: {peering} - Metric: {ylabel}',
        y=0.995
    )

    plt.tight_layout()

    filename = f"{cluster_str}_{primitive_str}_peering-{peering}_grouped-by-{group_by}_{metric}.png"
    filepath = outdir / filename

    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {filepath}")


# ==========================================================
# Main
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description='Plot microbenchmark results'
    )

    parser.add_argument("parquet_files", type=Path, nargs="+")
    parser.add_argument("--outdir", type=Path, default=Path("plots"))
    parser.add_argument("--metric", choices=['bandwidth', 'latency'],
                        default='bandwidth')
    parser.add_argument("--group-by",
                        choices=['implementation', 'comm_type', 'topology'],
                        default='implementation')
    parser.add_argument("--show-std", "-std", action="store_true")
    parser.add_argument("--show-minmax", "-minmax", action="store_true")
    parser.add_argument("--min-size", type=str, default=None,
                    help="Minimum transfer size (e.g., 128MiB)")
    parser.add_argument("--max-size", type=str, default=None,
                        help="Maximum transfer size (e.g., 2GiB)")

    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    min_size_B = parse_bytes(args.min_size, binary=False) if args.min_size else None
    max_size_B = parse_bytes(args.max_size, binary=False) if args.max_size else None

    meta_df_dict_pairs, meta_df = import_export.read_multiple_from_parquet(
        args.parquet_files
    )

    if meta_df is None:
        raise Exception("meta_df is None")

    # Derived metadata
    meta_df['primitive'] = meta_df['tag'].map(get_primitive_from_tag)
    meta_df['topology'] = meta_df['tag'].map(get_topology_from_tag)
    meta_df['comm_type'] = meta_df['tag'].map(get_comm_type)
    meta_df['implementation'] = meta_df['implementation'].map(parse_implementation)

    # Apply to meta_df_dict_pairs too
    for meta, _ in meta_df_dict_pairs:
        tag = meta['tag']
        meta['primitive'] = get_primitive_from_tag(tag)
        meta['topology'] = get_topology_from_tag(tag)
        meta['comm_type'] = get_comm_type(tag)
        meta['implementation'] = parse_implementation(meta['implementation'])

    GIB_TO_GBPS = 8 * (2**30) / 1e9

    # Build statistics dataframe
    stats_df = build_statistics_dataframe(meta_df_dict_pairs, GIB_TO_GBPS)
    
    if min_size_B is not None:
        stats_df = stats_df[stats_df['transfer_size_B'] >= min_size_B]
    if max_size_B is not None:
        stats_df = stats_df[stats_df['transfer_size_B'] <= max_size_B]

    # Detect unstable experiments
    detect_high_variability(stats_df, metric=args.metric, threshold=0.15)

    clusters = sorted(stats_df['cluster'].unique(), key=natural_sort_key)
    primitives = sorted(stats_df['primitive'].unique(), key=natural_sort_key)
    peerings = sorted(stats_df['peering'].unique(), key=natural_sort_key)
    
    print('DATA')
    print(meta_df.columns)
    display_df = stats_df.copy()
    display_df['experiment'] = (
        display_df['cluster'] + '|' +
        display_df['primitive'] + '|' +
        display_df['implementation'] + '|' +
        display_df['topology'] + '|' +
        display_df['peering'] + '|' +
        display_df['comm_type']
    )
    display_df['size'] = display_df['transfer_size_B'].apply(
        lambda x: format_bytes(int(x), binary=True, precision=0)
    )
    display_df = display_df[['experiment', 'size'] + [col for col in display_df.columns if col not in ['experiment', 'size']]]
    print(display_df.drop(columns=['cluster', 'primitive', 'implementation', 'topology', 'transfer_size_B', 'comm_type', 'peering']))

    for cluster in clusters:
        for peering in peerings:
            for primitive in primitives:
                plot_grouped_experiments(
                    stats_df,
                    peering,
                    group_by=args.group_by,
                    metric=args.metric,
                    outdir=args.outdir,
                    cluster=cluster,
                    primitive=primitive,
                    show_std=args.show_std,
                    show_minmax=args.show_minmax,
                )

    print(f"\nAll plots saved to '{args.outdir}'")


if __name__ == "__main__":
    main()