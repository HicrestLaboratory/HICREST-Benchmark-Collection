import argparse
from collections import Counter
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent / "common"))
from constants.plots import *
from constants.systems import *
from utils.utils import query_meta_df_dict_pairs
from utils.plots import (
    create_color_map,
    create_linestyle_map,
    create_marker_map,
    format_bytes,
)
import import_export

plt.rc("axes", titlesize=FONT_AXES - 6)
plt.rc("axes", labelsize=FONT_AXES - 6)
plt.rc("xtick", labelsize=FONT_TICKS - 4)
plt.rc("ytick", labelsize=FONT_TICKS - 2)
plt.rc("legend", fontsize=FONT_LEGEND - 4)
plt.rc("figure", titlesize=FONT_TITLE)

def plot_gflops_scaling(df, outdir="results"):
    """Plot GFLOP/s scaling across nodes."""
    plt.figure(figsize=(10, 7))
    cluster_color_map = create_color_map(df.sort_values("cluster")["cluster"].unique())
    gridX_marker_map = create_marker_map(df.sort_values("global_nx")["global_nx"].unique())

    for cluster, grp_cluster in df.groupby("cluster"):
        partition_linestyles = create_linestyle_map(grp_cluster["partition"].unique())

        for partition, grp_cluster_partition in grp_cluster.groupby("partition"):
            for grid, grp in grp_cluster_partition.groupby("global_nx"):

                grp_sorted = grp.sort_values("nodes")

                plt.plot(
                    grp_sorted["gpus"],
                    grp_sorted["gflops"],
                    marker=gridX_marker_map[grid],
                    label=f"{cluster}-{partition}-gridX{grid}",
                    color=cluster_color_map[cluster],
                    linestyle=partition_linestyles[partition],
                )

    plt.xticks(sorted(df["gpus"].unique()))
    plt.xlabel("GPUs")
    plt.ylabel("GFLOPs")
    plt.title("HPCG Scaling")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    path = Path(outdir) / "HPCG_Scaling_GFLOPs.png"
    plt.savefig(path, dpi=200)
    print(f"Plot saved to {path.resolve().absolute()}")
    plt.close()

def plot_runtime_scaling(df, outdir="results"):
    """Plot runtime scaling across nodes."""
    plt.figure(figsize=(10, 7))
    cluster_color_map = create_color_map(df.sort_values("cluster")["cluster"].unique())
    gridX_marker_map = create_marker_map(df.sort_values("global_nx")["global_nx"].unique())

    for cluster, grp_cluster in df.groupby("cluster"):
        partition_linestyles = create_linestyle_map(grp_cluster["partition"].unique())

        for partition, grp_cluster_partition in grp_cluster.groupby("partition"):
            for grid, grp in grp_cluster_partition.groupby("global_nx"):

                grp_sorted = grp.sort_values("nodes")

                plt.plot(
                    grp_sorted["gpus"],
                    grp_sorted["time_tot"],
                    marker=gridX_marker_map[grid],
                    label=f"{cluster}-{partition}-gridX{grid}",
                    color=cluster_color_map[cluster],
                    linestyle=partition_linestyles[partition],
                )

    plt.xticks(sorted(df["gpus"].unique()))
    plt.xlabel("GPUs")
    plt.ylabel("Runtime [s]")
    plt.title("HPCG Scaling")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    path = Path(outdir) / "HPCG_Scaling_Runtime.png"
    plt.savefig(path, dpi=200)
    print(f"Plot saved to {path.resolve().absolute()}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison scaling plots from HPCG CSVs"
    )
    parser.add_argument("csv_files", nargs="+", help="Input CSV files")
    parser.add_argument("--outdir", default="plots", help="Output directory for plots")
    args = parser.parse_args()

    # Create output directory
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    meta_df = import_export.read_multiple_from_csv(args.csv_files)

    if meta_df is None:
        raise Exception("meta_df is None")

    required = {
        "cluster",
        "partition",
        "nodes",
        "threads",
        "time_tot",
        "gflops",
        "gflops_opt",
        "mem",
        "global_nx",
        "global_ny",
        "global_nz",
        "num_equations",
        "final_result_valid",
    }
    if not required.issubset(meta_df.columns):
        raise ValueError(f"CSV must contain columns: {', '.join(required)}")

    # Map names and prepare dataframe
    meta_df["cluster"] = meta_df["cluster"].map(CLUSTER_NAMES_MAP)
    meta_df["partition"] = (
        meta_df["partition"]
        .map(PARTITION_NAMES_MAP)
        .fillna(meta_df["cluster"].map(DEFAULT_PARTITION_NAMES_MAP))
    )
    meta_df["cluster_partition"] = (
        meta_df["cluster"].astype(str) + "-" + meta_df["partition"].astype(str)
    )
    meta_df["grid"] = (
        meta_df["global_nx"].astype(str)
        + "x"
        + meta_df["global_ny"].astype(str)
        + "x"
        + meta_df["global_nz"].astype(str)
    )
    meta_df["gpus"] = meta_df["nodes"] * 4  # FIXME set this based on cluster

    print("Metedata df:")
    print(meta_df.sort_values(["cluster", "partition", "nodes"]))
    print("\nGenerating plots...")

    # Generate all plots
    plot_gflops_scaling(meta_df, args.outdir)
    plot_runtime_scaling(meta_df, args.outdir)

    # grids = sorted(meta_df["grid"].unique())
    # nodes_list = sorted(meta_df["nodes"].astype(int).unique())
    # cluster_partitions = sorted(meta_df["cluster_partition"].unique())

    # for nodes in nodes_list:
    #     # select matching experiments
    #     pairs = query_meta_df_dict_pairs(
    #         meta_dfs_pairs,
    #         [("nodes", nodes)], # ("grid", grid)
    #     )

    #     # Ensure ALL grid sizes match
    #     grid = None
    #     for meta, _ in pairs:
    #         if grid is None:
    #             grid = meta['grid']
    #         else:
    #             if meta['grid'] != grid:
    #                 warnings.warn(f'Found NON MATCHING grids in {nodes}-node experiments (prev grid: {grid}, curr meta: {meta})\nSkipping {nodes}-node experiments...')
    #                 continue

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
