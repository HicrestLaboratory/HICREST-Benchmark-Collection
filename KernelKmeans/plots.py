import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent / "common"))
from import_export import read_multiple_from_csv
from utils.plots import create_color_map, create_linestyle_map, create_marker_map


def prepare(df):
    """
    Aggregate data across iterations.
    Groups by experiment params (excluding timers and per-run metrics).
    Computes geometric mean for all metrics (throughput and timers).
    """
    if "total" not in df.columns:
        raise ValueError("Missing 'total' column (required for performance)")
    
    # Calculate throughput per row
    df["throughput"] = (
        df["n"] * df["k"] * df["d"]
    ) / (df["total"] / 1000.0)
 
    # Columns to exclude from grouping
    exclude_cols = {"iteration", "total", "throughput", "job_id", "seed", "runs", 
                    "avg_score", "min_score", "max_score", "maxiter", "total_mem_bytes"}
    
    timer_cols = ["argmin_assign", "distances_compute", "init", "score_compute", "v_matrix_update"]
    
    # Group columns = everything except timers and excluded columns
    group_cols = list(set([c for c in df.columns if c not in exclude_cols]) - set(timer_cols))
    
    # Initialize result dataframe
    result_rows = []
    
    for group_vals, group_data in df.groupby(group_cols):
        row_dict = dict(zip(group_cols, group_vals) if isinstance(group_vals, tuple) else {group_cols[0]: group_vals})
        
        # Compute geometric mean for throughput
        throughputs = group_data["throughput"].values
        throughputs = throughputs[np.isfinite(throughputs)]
        if len(throughputs) > 0:
            row_dict["throughput_mean"] = np.exp(np.mean(np.log(throughputs)))
        else:
            row_dict["throughput_mean"] = np.nan
        
        # Compute geometric mean for each timer
        for timer in timer_cols:
            timer_vals = group_data[timer].values
            # Filter out NaN and non-positive values (can't take log of negative/zero)
            timer_vals = timer_vals[np.isfinite(timer_vals) & (timer_vals > 0)]
            if len(timer_vals) > 0:
                row_dict[timer] = np.exp(np.mean(np.log(timer_vals)))
            else:
                row_dict[timer] = np.nan
        
        result_rows.append(row_dict)
    
    agg = pd.DataFrame(result_rows)
    return agg


def plot_strong_scaling(df, outdir):
    """
    Single figure per (n, d, k):
    - Rows: kernel
    - Columns: sorted (impl, compiler, cluster, board)
    - Each subplot: throughput curves vs CPUs
    """
    required = ["throughput_mean", "cpus", "compiler", "impl", "board", "cluster", "kernel"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    groups = df.groupby(["n", "d", "k"])

    for (n, d, k), sub in groups:
        # Get all unique configs: (impl, compiler, cluster, board)
        configs_df = sub[["impl", "compiler", "cluster", "board"]].drop_duplicates()
        configs = [tuple(row) for row in configs_df.values]
        configs = sorted(configs)
        kernels = sorted(sub["kernel"].unique())
        
        n_configs = len(configs)
        n_kernels = len(kernels)
        
        # Create figure: rows=kernels, cols=configs
        fig, axes = plt.subplots(
            n_kernels, n_configs,
            figsize=(4 * n_configs, 3.5 * n_kernels),
            squeeze=False
        )
        
        # Get compiler/impl for color/linestyle mapping
        compilers = sorted(sub["compiler"].unique())
        impls = sorted(sub["impl"].unique())
        color_map = create_color_map(compilers)
        linestyle_map = create_linestyle_map(impls)
        
        for kernel_idx, kernel in enumerate(kernels):
            kernel_data = sub[sub["kernel"] == kernel]
            
            for config_idx, (impl, compiler, cluster, board) in enumerate(configs):
                ax = axes[kernel_idx, config_idx]
                
                # Filter to this config
                mask = (
                    (kernel_data["impl"] == impl) &
                    (kernel_data["compiler"] == compiler) &
                    (kernel_data["cluster"] == cluster) &
                    (kernel_data["board"] == board)
                )
                g = kernel_data[mask].sort_values("cpus")
                
                if len(g) > 0:
                    x = g["cpus"].values
                    y = g["throughput_mean"].values
                    
                    color = color_map[compiler]
                    linestyle = linestyle_map[impl]
                    
                    # Plot throughput
                    ax.plot(
                        x, y,
                        color=color,
                        linestyle=linestyle,
                        marker="o",
                        linewidth=2,
                        markersize=6,
                        label=f"{compiler}|{impl}"
                    )
                    
                    # Ideal scaling line
                    if len(x) > 1:
                        baseline_idx = 0
                        y0 = y[baseline_idx]
                        x0 = x[baseline_idx]
                        xs = np.array(sorted(x))
                        ideal = y0 * (xs / x0)
                        
                        ax.plot(
                            xs, ideal,
                            color=color,
                            linestyle=":",
                            alpha=0.5,
                            linewidth=1.5
                        )
                
                # Formatting
                ax.set_xscale("log", base=2)
                ax.set_yscale("log", base=2)
                
                # Set x-axis to show actual numeric values, not powers
                if len(g) > 0:
                    cpu_ticks = sorted(g["cpus"].unique())
                    ax.set_xticks(cpu_ticks)
                    ax.set_xticklabels([str(int(c)) for c in cpu_ticks], rotation=45, ha="right")
                
                ax.grid(True, which="both", linestyle="--", alpha=0.3)
                
                # Title: config info
                title = f"{impl}|{compiler}\n{cluster}@{board}"
                ax.set_title(title, fontsize=9, fontweight="bold")
                
                # Labels only on edges
                if config_idx == 0:
                    ax.set_ylabel("Throughput (ops/sec)", fontsize=9)
                else:
                    ax.set_ylabel("")
                
                if kernel_idx == n_kernels - 1:
                    ax.set_xlabel("CPUs", fontsize=9)
                else:
                    ax.set_xlabel("")
                
                # Row label: kernel
                if config_idx == n_configs - 1:
                    ax.text(
                        1.15, 0.5,
                        kernel,
                        transform=ax.transAxes,
                        fontsize=10,
                        fontweight="bold",
                        rotation=270,
                        va="center"
                    )
        
        fig.suptitle(f"Strong Scaling (n={n}, d={d}, k={k})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        fname = f"strong_scaling_n{n}_d{d}_k{k}.png"
        plt.savefig(outdir / fname, dpi=150, bbox_inches="tight")
        plt.close()


def plot_runtime_breakdown(df, outdir):
    """
    Stacked bar plots for runtime breakdown.
    Single figure per (n, d, k):
    - Rows: kernel
    - Columns: sorted (impl, compiler, cluster, board)
    - Each subplot: stacked bars showing timer breakdown per CPU count
    """
    timers = [
        "argmin_assign",
        "distances_compute",
        "score_compute",
        "v_matrix_update",
    ]

    required = ["cpus", "compiler", "impl", "cluster", "board", "kernel"] + timers
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    groups = df.groupby(["n", "d", "k"])

    for (n, d, k), sub in groups:
        # Get all unique configs
        configs_df = sub[["impl", "compiler", "cluster", "board"]].drop_duplicates()
        configs = [tuple(row) for row in configs_df.values]
        configs = sorted(configs)
        kernels = sorted(sub["kernel"].unique())
        
        n_configs = len(configs)
        n_kernels = len(kernels)
        
        fig, axes = plt.subplots(
            n_kernels, n_configs,
            figsize=(4 * n_configs, 3.5 * n_kernels),
            squeeze=False
        )
        
        for kernel_idx, kernel in enumerate(kernels):
            kernel_data = sub[sub["kernel"] == kernel]
            
            for config_idx, (impl, compiler, cluster, board) in enumerate(configs):
                ax = axes[kernel_idx, config_idx]
                
                # Filter to this config
                mask = (
                    (kernel_data["impl"] == impl) &
                    (kernel_data["compiler"] == compiler) &
                    (kernel_data["cluster"] == cluster) &
                    (kernel_data["board"] == board)
                )
                g = kernel_data[mask].sort_values("cpus")
                
                if len(g) > 0:
                    x = np.arange(len(g))
                    cpu_labels = [str(int(c)) for c in g["cpus"].values]
                    
                    # Stack the timers
                    bottom = np.zeros(len(g))
                    colors = sns.color_palette("husl", len(timers))
                    
                    for timer_idx, timer in enumerate(timers):
                        values = g[timer].values
                        ax.bar(
                            x, values,
                            bottom=bottom,
                            label=timer,
                            color=colors[timer_idx],
                            alpha=0.85
                        )
                        bottom += values
                    
                    ax.set_xticks(x)
                    ax.set_xticklabels(cpu_labels, rotation=45, ha="right")
                    ax.set_ylabel("Runtime (ms)", fontsize=9)
                    
                    # Title: config info
                    title = f"{impl}|{compiler}\n{cluster}@{board}"
                    ax.set_title(title, fontsize=9, fontweight="bold")
                    
                    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.3)
                    
                    # Legend only on first subplot
                    if kernel_idx == 0 and config_idx == 0:
                        ax.legend(fontsize=7, loc="upper left")
                
                # Row label: kernel
                if config_idx == n_configs - 1:
                    ax.text(
                        1.15, 0.5,
                        kernel,
                        transform=ax.transAxes,
                        fontsize=10,
                        fontweight="bold",
                        rotation=270,
                        va="center"
                    )
        
        fig.suptitle(f"Runtime Breakdown (n={n}, d={d}, k={k})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        fname = f"breakdown_n{n}_d{d}_k{k}.png"
        plt.savefig(outdir / fname, dpi=150, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", nargs="+", help="Input CSV files")
    parser.add_argument("--outdir", default="plots", help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    df = read_multiple_from_csv(args.csvs)
    df = prepare(df)
    
    with pd.option_context("display.max_rows", None): # , "display.max_columns", None
        print(df)

    plot_strong_scaling(df, outdir)
    plot_runtime_breakdown(df, outdir)

    print(f"Plots saved to {outdir}")


if __name__ == "__main__":
    main()