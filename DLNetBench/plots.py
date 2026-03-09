#!/usr/bin/env python3
"""
Plotting functions for DP (Data Parallelism) analysis.
Includes scaling plots and time breakdown plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def plot_dp_scaling(
    parquet_file: str,
    tag_labels: Optional[Dict[str, str]] = None,
    tags_to_plot: Optional[List[str]] = None,
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "DP Scaling: Throughput vs World Size"
):
    """
    Plot scaling performance (samples/s) for DP strategy across different experiments.
    
    Parameters:
    -----------
    parquet_file : str
        Path to parquet file containing data from multiple job_tags
        Example: "results/dp/cluster1.parquet"
    
    tag_labels : Optional[Dict[str, str]]
        Dictionary mapping job_tag to display label
        Example: {"exp1": "Baseline", "exp2": "Optimized"}
        If None, uses job_tag as label
    
    tags_to_plot : Optional[List[str]]
        List of specific job_tags to plot. If None, plots all tags in the file.
        Example: ["exp1", "exp2"]
    
    output_file : Optional[str]
        Path to save the plot. If None, displays interactively.
    
    figsize : Tuple[int, int]
        Figure size (width, height)
    
    title : str
        Plot title
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if tag_labels is None:
        tag_labels = {}
    
    # Read the parquet file
    df = pd.read_parquet(parquet_file)
    
    # Get unique job_tags
    all_tags = df['job_tag'].unique()
    
    # Filter tags if specified
    if tags_to_plot is not None:
        all_tags = [tag for tag in all_tags if tag in tags_to_plot]
    
    # Store data for ideal scaling calculation
    all_world_sizes = set()
    
    # Plot each job_tag
    for job_tag in sorted(all_tags):
        tag_df = df[df['job_tag'] == job_tag].copy()
        
        # Calculate throughput (samples/s) for each run
        # samples_per_run = local_batch_size (since DP replicates the batch)
        tag_df['throughput'] = (tag_df['local_batch_size'] * tag_df['world_size']) / tag_df['runtime']
        
        # Group by world_size and calculate mean throughput
        throughput_by_ws = tag_df.groupby('world_size').agg({
            'throughput': ['mean', 'std', 'count']
        }).reset_index()
        
        throughput_by_ws.columns = ['world_size', 'throughput_mean', 'throughput_std', 'count']
        
        # Sort by world_size
        throughput_by_ws = throughput_by_ws.sort_values('world_size')
        
        # Store for scaling
        all_world_sizes.update(throughput_by_ws['world_size'].values)
        
        # Get label
        label = tag_labels.get(job_tag, job_tag)
        
        # Plot with error bars
        ax.errorbar(
            throughput_by_ws['world_size'],
            throughput_by_ws['throughput_mean'],
            # yerr=throughput_by_ws['throughput_std'],
            marker='o',
            linewidth=2,
            markersize=8,
            capsize=5,
            label=label
        )
    
    ax.set_xlabel('World Size (Number of GPUs)', fontsize=12)
    ax.set_ylabel('Throughput (samples/s)', fontsize=12)
    ax.set_xticks(sorted(all_world_sizes))
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Scaling plot saved to: {output_file}")
    else:
        plt.show()
    
    return fig, ax


def plot_dp_breakdown(
    parquet_file: str,
    tag_labels: Optional[Dict[str, str]] = None,
    tag_colors: Optional[Dict[str, str]] = None,
    tags_to_plot: Optional[List[str]] = None,
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (6, 6),
    title: str = "DP Time Breakdown: Compute vs Communication (%)"
):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    
    tag_labels = tag_labels or {}
    tag_colors = tag_colors or {}
    
    df = pd.read_parquet(parquet_file)
    all_tags = df['job_tag'].unique()
    if tags_to_plot:
        all_tags = [tag for tag in all_tags if tag in tags_to_plot]

    # Compute time definition
    df['compute_time'] = df['runtime'] - df['barrier_time']  # Compute time is total runtime minus barrier time
    df['barrier_time'] = df['barrier_time']  # Convert barrier time to seconds

    job_tags_ordered = sorted(all_tags)
    world_sizes = sorted(df['world_size'].unique())
    n_tags = len(job_tags_ordered)
    
    total_width = 0.05
    bar_width = total_width / n_tags
    x = np.arange(len(world_sizes))
    
    for idx, job_tag in enumerate(job_tags_ordered):
        tag_df = df[df['job_tag'] == job_tag].copy()
        if tag_df.empty:
            continue

        # Aggregate per world_size
        breakdown = tag_df.groupby('world_size').agg(
            compute_time=('compute_time', 'mean'),
            barrier_time=('barrier_time', 'mean'),
            runtime=('runtime', 'mean')
        ).reindex(world_sizes).fillna(0)
        
        # Compute percentages
        compute_pct = (breakdown['compute_time'] / breakdown['runtime']) * 100
        barrier_pct = (breakdown['barrier_time'] / breakdown['runtime']) * 100
        
        # Center bars for multiple tags
        positions = x - total_width/2 + idx*bar_width + bar_width/2
        
        ax.bar(
            positions,
            compute_pct,
            width=bar_width,
            label=f"{tag_labels.get(job_tag, job_tag)} - Compute",
            color=tag_colors.get(job_tag, None),
            alpha=0.8
        )
        print(job_tag)
        ax.bar(
            positions,
            barrier_pct,
            width=bar_width,
            bottom=compute_pct,
            label=f"{tag_labels.get(job_tag, job_tag)} - Barrier",
            color=tag_colors.get(job_tag, None),
            alpha=0.4,
            hatch='//'
        )
    
    ax.set_xlabel('World Size (Number of GPUs)', fontsize=12)
    ax.set_ylabel('Time (%)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(world_sizes)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Breakdown plot saved to: {output_file}")
    else:
        plt.show()
    
    return fig, ax


# Example usage
if __name__ == "__main__":
    import sys
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Plot DP scaling and breakdown from parquet files")
    parser.add_argument("parquet_files", nargs="+", help="Parquet file(s) to plot")
    parser.add_argument("--output-dir", default=".", help="Output directory for plots")
    parser.add_argument("--prefix", default="", help="Prefix for output filenames")
    
    args = parser.parse_args()

    tag_labels = {
        "dp_vit_h_16_bfloat16_nccl_distance_unknown": "Leonardo-test"
    }
    
    # Process each parquet file
    for parquet_file in args.parquet_files:
        file_path = Path(parquet_file)
        base_name = file_path.stem
        
        if args.prefix:
            output_prefix = f"{args.prefix}_{base_name}"
        else:
            output_prefix = base_name
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing: {parquet_file}")
        
        # Plot 1: Scaling
        print("  Creating scaling plot...")
        plot_dp_scaling(
            parquet_file=parquet_file,
            output_file=str(output_dir / f"{output_prefix}_scaling.png"), 
            tag_labels=tag_labels
        )
        
        # Plot 2: Breakdown (percentage)
        print("  Creating breakdown plot...")
        plot_dp_breakdown(
            parquet_file=parquet_file,
            output_file=str(output_dir / f"{output_prefix}_breakdown.png"),
            tag_labels=tag_labels
        )
    
    print(f"\nAll plots saved to: {output_dir}")