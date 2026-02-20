from pathlib import Path
import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

sys.path.append(str(Path(__file__).parent.parent / "common"))
import import_export
from utils.plots import format_bytes, create_color_map, create_linestyle_map
from constants.plots import *
from constants.systems import SPECS_LEONARDO

plt.rc('axes', titlesize=FONT_AXES - 2)
plt.rc('axes', labelsize=FONT_AXES - 2)
plt.rc('xtick', labelsize=FONT_TICKS)
plt.rc('ytick', labelsize=FONT_TICKS)
plt.rc('legend', fontsize=FONT_LEGEND + 1)
plt.rc('figure', titlesize=FONT_TITLE - 10)

def format_bytes_tick(x, pos):
    return format_bytes(int(x), binary=True, precision=0)

def parse_implementation(impl: str) -> str:
    parts = impl.split('_')
    return impl if len(parts) <= 1 else parts[-1]

TOPOLOGY_MAP = {
    'same-l1': 'same-l1',
    'same-group': 'same-group',
    'inter-group': 'inter-group',
    
    'distance_2.0': 'same-l1',
    'distance_4.0': 'same-group',
    'distance_5.0': 'inter-group',
}

def get_topology_from_tag(tag: str) -> str:
    return TOPOLOGY_MAP[tag.split('__')[-1]]

def get_primitive_from_tag(tag: str) -> str:
    return tag.split('__')[-0]

def get_comm_type_from_tag(tag: str) -> str:
    return tag.split('__')[1]

def get_comm_type_direction_from_tag(tag: str) -> str:
    return tag.split('__')[2].split('_')[0] if tag.split('__')[1] == 'hybrid_buff' else ''

COMM_TYPE_MAP = {
    'gpu_buff': 'G2G',
    'cpu_buff': 'C2C',
    'host2dev': 'C2G',
    'dev2host': 'G2C',
}

def get_comm_type(tag: str) -> str:
    comm_type = get_comm_type_from_tag(tag)
    if comm_type == 'hybrid_buff':
        comm_type = get_comm_type_direction_from_tag(tag)
    return COMM_TYPE_MAP[comm_type]


def natural_sort_key(s):
    """Generate key for natural sorting (handles numbers in strings)."""
    import re
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split('([0-9]+)', str(s))]


def plot_single_experiment(df, meta, metric='bandwidth', outdir=Path('plots')):
    """Create a single plot for one experiment."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = df['transfer_size_B']
    if metric == 'bandwidth':
        y = df['bandwidth']
        ylabel = 'Goodput (Gb/s)'
    else:  # latency
        y = df['time_s'] * 1e6  # Convert to microseconds
        ylabel = 'Latency (μs)'
    
    ax.plot(x, y, marker='o', linewidth=2, markersize=6)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Message Size')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, which='both')
    
    # Format x-axis tick labels
    ax.xaxis.set_major_formatter(FuncFormatter(format_bytes_tick))
    
    title = f"{meta['cluster']} - {meta['implementation']} - {meta['comm_type']}\n{meta['topology']}"
    ax.set_title(title)
    
    plt.tight_layout()
    
    # Create filename
    filename = f"{meta['cluster']}_{meta['implementation']}_{meta['comm_type']}_{meta['topology']}_{metric}.png"
    filepath = outdir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filepath}")

ALL_VARS = ['implementation', 'comm_type', 'topology']

def plot_grouped_experiments(meta_df_dict_pairs, meta_df, group_by='implementation', 
                             metric='bandwidth', 
                             outdir=Path('plots'), cluster=None, primitive=None,
                             y_log=False, same_y_lim=True,
                             flip_color_linestyle=False,
                             max_message_size=None):
    """
    Create multiple subplots grouped by one variable with lines for combinations of the other two.
    
    Args:
        group_by: Variable to create separate subplots for ('implementation', 'comm_type', or 'topology')
        metric: 'bandwidth' or 'latency'
    """
    
    # Determine which variables to combine for lines
    line_vars = [v for v in ALL_VARS if v != group_by and (not primitive or v != primitive)]
    
    # Filter (if specified)
    filtered_df = meta_df
    if cluster:
        filtered_df = meta_df[meta_df['cluster'] == cluster]
    if primitive:
        filtered_df = filtered_df[filtered_df['primitive'] == primitive]
    
    # Get unique values for grouping - sorted naturally
    group_values = sorted(filtered_df[group_by].unique(), key=natural_sort_key)
    n_groups = len(group_values)
    
    # Create subplots
    n_cols = min(2, n_groups)
    n_rows = int(np.ceil(n_groups / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10*n_cols, 8*n_rows))
    if n_groups == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Set up metric
    if metric == 'bandwidth':
        ylabel = 'Goodput (Gb/s)'
        col_name = 'bandwidth'
    else:  # latency
        ylabel = 'Latency (μs)'
        col_name = 'time_s'
    
    # Color map for lines
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    comm_type_color_map = create_color_map(filtered_df['comm_type'].unique())
    comm_type_linestyle_map = create_linestyle_map(filtered_df['comm_type'].unique())
    topology_linestyle_map = create_linestyle_map(filtered_df['topology'].unique())
    topology_color_map = create_color_map(filtered_df['topology'].unique())
    
    # First pass: collect all data to determine global y-axis range
    all_y_values = []
    plot_data = {}  # Store data for second pass
    
    roofline = None
    if cluster and str(cluster).lower() =='leonardo':
        roofline = {
            # TODO same node bw_gpu_gpu and hybrid
            'G2G': {
                'same-l1': min(SPECS_LEONARDO['bw_gpu_nic'], SPECS_LEONARDO['bw_nic_l1']),
                'same-group': min(SPECS_LEONARDO['bw_gpu_nic'], SPECS_LEONARDO['bw_nic_l1'], SPECS_LEONARDO['bw_l1_l2']),
                'inter-group': min(SPECS_LEONARDO['bw_gpu_nic'], SPECS_LEONARDO['bw_nic_l1'], SPECS_LEONARDO['bw_l1_l2'], SPECS_LEONARDO['bw_l2_l2']),
            },
            'C2C': {
                'same-l1': min(SPECS_LEONARDO['bw_cpu_nic'], SPECS_LEONARDO['bw_nic_l1']),
                'same-group': min(SPECS_LEONARDO['bw_cpu_nic'], SPECS_LEONARDO['bw_nic_l1'], SPECS_LEONARDO['bw_l1_l2']),
                'inter-group': min(SPECS_LEONARDO['bw_cpu_nic'], SPECS_LEONARDO['bw_nic_l1'], SPECS_LEONARDO['bw_l1_l2'], SPECS_LEONARDO['bw_l2_l2']),
            },
        }
        
    for idx, group_val in enumerate(group_values):
        # Get all experiments for this group
        group_meta = filtered_df[filtered_df[group_by] == group_val]
        
        # Get unique combinations of the other two variables - sorted naturally
        line_combinations = group_meta[line_vars].drop_duplicates().values.tolist()
        line_combinations = [tuple(combo) for combo in line_combinations]
        line_combinations = sorted(line_combinations, key=lambda x: [natural_sort_key(str(v)) for v in x])
        
        plot_data[group_val] = {'line_combinations': line_combinations, 'data': []}
        
        # Collect data for each line
        for line_idx, line_combo in enumerate(line_combinations):
            # Build query filters
            query_filters = [(group_by, group_val)]
            if cluster:
                query_filters.append(('cluster', cluster))
            if primitive:
                query_filters.append(('primitive', primitive))
            for var_name, var_val in zip(line_vars, line_combo):
                query_filters.append((var_name, var_val))
            
            for meta, dfs in import_export.query_meta_df_dict_pairs(meta_df_dict_pairs, query_filters):
                df = dfs['avg']
                if max_message_size:
                    df = df[df['transfer_size_B'] <= max_message_size]
                x = df['transfer_size_B']
                
                if metric == 'bandwidth':
                    y = df[col_name]
                else:
                    y = df[col_name] * 1e6  # Convert to microseconds
                
                all_y_values.extend(y.values)
                
                # Store data for plotting
                label = ' | '.join(str(v) for v in line_combo)
                color = None
                linestyle = None
                comm_type = line_combo[line_vars.index('comm_type')]
                topology = line_combo[line_vars.index('topology')]
                if flip_color_linestyle:
                    color = comm_type_color_map[comm_type]
                    linestyle = topology_linestyle_map[topology]
                else:
                    color = topology_color_map[topology]
                    linestyle = comm_type_linestyle_map[comm_type]
                    
                plot_data[group_val]['data'].append({
                    'x': x,
                    'y': y,
                    'label': label,
                    'color_idx': line_idx,
                    'color': color,
                    'linestyle': linestyle,
                    'comm_type': comm_type,
                    'topology': topology,
                    'primitive': primitive if primitive else line_combo[line_vars.index('primitive')],
                })
    
    # Calculate global y-axis range with some padding
    if all_y_values:
        y_min = min(all_y_values)
        y_max = max(all_y_values)
        if y_log:
            # Add padding in log space
            log_range = np.log10(y_max) - np.log10(y_min)
            padding = 0.1 * log_range
            y_lim = (10**(np.log10(y_min) - padding), 10**(np.log10(y_max) + padding))
        else:
            padding = 0.05 * y_max
            y_lim = (y_min, y_max + padding)
    else:
        y_lim = None
        
    print(plot_data.keys())
    for k in plot_data.keys():
        print(plot_data[k]['line_combinations'])
        print([(d['comm_type'], d['topology']) for d in plot_data[k]['data']])
    print()
    
    # Second pass: plot with consistent y-axis
    for idx, group_val in enumerate(group_values):
        ax = axes[idx]
        
        # Plot each line
        for line_data in plot_data[group_val]['data']:
            ax.plot(line_data['x'], line_data['y'], 
                   marker='o', linewidth=2, markersize=4,
                   color=colors[line_data['color_idx'] % len(colors)] if line_data['color'] is None else line_data['color'],
                   linestyle=line_data['linestyle'] if line_data['linestyle'] is not None else '-',
                   label=line_data['label'])
            
            # Plot roofline (if available)
            if metric == 'bandwidth' and roofline:
                y = roofline.get(line_data['comm_type'])
                #FIXME
                print(f'TOTO Adding roof for {line_data["primitive"]} {line_data["comm_type"]}: {y}')
                y = y.get(line_data['topology']) if y else y
                if y and y_lim and y <= y_lim[1]:
                    ax.axhline(y=y, color=line_data['color'], linestyle=line_data['linestyle'])
                    ax.text(line_data['x'][0]/2, y, 'Max Bandwidth', fontsize=8, verticalalignment='center', backgroundcolor='white')
        
        ax.set_xscale('log', base=2)
        if y_log: ax.set_yscale('log')
        if idx // n_cols == n_rows - 1: ax.set_xlabel('Message Size')
        if idx % n_rows == 0: ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_title(f"{group_val}")
        
        # Format x-axis tick labels
        ax.xaxis.set_major_formatter(FuncFormatter(format_bytes_tick))
        
        # Set consistent y-axis range
        if same_y_lim and y_lim:
            ax.set_ylim(y_lim)
        
        # Sort legend entries naturally
        handles, labels = ax.get_legend_handles_labels()
        # Sort by natural order
        sorted_pairs = sorted(zip(labels, handles), key=lambda x: natural_sort_key(x[0]))
        sorted_labels, sorted_handles = zip(*sorted_pairs) if sorted_pairs else ([], [])
        ax.legend(sorted_handles, sorted_labels, loc='best')
    
    # Hide unused subplots
    for idx in range(n_groups, len(axes)):
        axes[idx].set_visible(False)
    
    cluster_str = cluster if cluster else 'all_clusters'
    primitive_str = primitive if primitive else 'all_primitives'
    fig.suptitle(f'{cluster_str.capitalize()} - {primitive_str} - {'Goodput' if metric == 'bandwidth' else metric.capitalize()}', y=0.995)
    plt.tight_layout()
    
    # Save
    filename = f"{cluster_str}_{primitive_str}_grouped_by_{group_by}_{metric}.png"
    filepath = outdir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Plot microbenchmark results')
    parser.add_argument("parquet_files", type=Path, nargs="+")
    parser.add_argument("--outdir", type=Path, default=Path("plots"))
    parser.add_argument("--metric", type=str, default="bandwidth", 
                       choices=['bandwidth', 'latency'],
                       help="Metric to plot: bandwidth or latency")
    parser.add_argument("--plot-mode", type=str, default="grouped",
                       choices=['individual', 'grouped', 'both'],
                       help="Plot mode: individual plots, grouped plots, or both")
    parser.add_argument("--group-by", type=str, default="implementation",
                       choices=['implementation', 'comm_type', 'topology'],
                       help="For grouped plots: variable to group subplots by (lines will show combinations of the other two)")

    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    # Read data
    meta_df_dict_pairs, meta_df = import_export.read_multiple_from_parquet(
        args.parquet_files
    )
    import_export.describe_pairs_content(meta_df_dict_pairs)

    if meta_df is None:
        raise Exception("meta_df is None")

    # Add derived columns
    meta_df['primitive'] = meta_df['tag'].map(get_primitive_from_tag)
    meta_df['topology'] = meta_df['tag'].map(get_topology_from_tag)
    meta_df['comm_type'] = meta_df['tag'].map(get_comm_type)
    meta_df['implementation'] = meta_df['implementation'].map(parse_implementation)
    
    print('\nMetadata Dataframe')
    print(meta_df.sort_values('tag'))
    print(f"\nGenerating plots in {args.outdir}")
    print(f"Metric: {args.metric}")
    print(f"Plot mode: {args.plot_mode}\n")
    
    GIB_TO_GBPS = 8 * (2**30) / 1e9
    
    clusters = sorted(meta_df['cluster'].unique(), key=natural_sort_key)
    primitives = sorted(meta_df['primitive'].unique(), key=natural_sort_key)
    for meta, dfs in meta_df_dict_pairs:
        tag = meta['tag']
        meta['implementation'] = parse_implementation(meta['implementation'])
        meta['primitive'] = get_primitive_from_tag(tag)
        meta['topology'] = get_topology_from_tag(tag)
        meta['comm_type'] = get_comm_type(tag)
        # Convert bandwidths Gb/s
        for df in dfs.values():
            df['bandwidth'] = df['bandwidth_GiB_s'] * GIB_TO_GBPS
    
    # Generate individual plots
    if args.plot_mode in ['individual', 'both']:
        print("Generating individual plots...")
        for cluster in clusters:
            print(f'  Cluster: {cluster}')
            for meta, dfs in import_export.query_meta_df_dict_pairs(
                meta_df_dict_pairs, [('cluster', cluster)]
            ):
                tag = meta['tag']
                meta['topology'] = get_topology_from_tag(tag)
                meta['comm_type'] = get_comm_type_from_tag(tag) + get_comm_type_direction_from_tag(tag)
                
                plot_single_experiment(dfs['avg'], meta, args.metric, args.outdir)
        print()
        
    # FIXME delete
    meta_df = meta_df[meta_df['comm_type'].isin(['C2C', 'G2G'])]
    meta_df = meta_df[meta_df['topology'].isin(['same-l1', 'same-group', 'inter-group'])]
    # meta_df = meta_df[meta_df['implementation'].isin(['CudaAware'])]
    flip_color_linestyle = False
    same_y_lim = True
    max_message_size = None # 1*(1024**2)
    
    # Generate grouped plots
    if args.plot_mode in ['grouped', 'both']:
        print("Generating grouped plots...")
        for system in clusters:
            for primitive in primitives:
                print(f'  System: {system}, Primitive: {primitive}')
                plot_grouped_experiments(
                    meta_df_dict_pairs, meta_df, 
                    group_by=args.group_by,
                    metric=args.metric,
                    outdir=args.outdir,
                    cluster=system,
                    primitive=primitive,
                    flip_color_linestyle=flip_color_linestyle,
                    same_y_lim=same_y_lim,
                    max_message_size=max_message_size,
                )
    
    print(f"\nAll plots saved to '{args.outdir}'")


if __name__ == "__main__":
    main()