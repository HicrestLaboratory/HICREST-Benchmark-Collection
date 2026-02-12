#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))
from py_utils.constants.machines import *
from py_utils.constants.plots import *
from py_utils.utils.plots import create_color_map, create_linestyle_map

plt.rc('axes', titlesize=FONT_AXES - 2)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_AXES - 2)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONT_TICKS)   # fontsize of the tick labels
plt.rc('ytick', labelsize=FONT_TICKS)   # fontsize of the tick labels
plt.rc('legend', fontsize=FONT_LEGEND + 3)  # legend fontsize
plt.rc('figure', titlesize=FONT_TITLE)  # fontsize of the figure title

def main():
  parser = argparse.ArgumentParser(description="Generate comparative scaling plots from HPL benchmark CSV files.")
  parser.add_argument("csv_files", nargs="+", help="Paths to CSV files with HPL results")
  parser.add_argument("--out", default="results", help="Directory for output plot files")
  args = parser.parse_args()

  # Read all CSVs and concatenate
  dfs = [pd.read_csv(f) for f in args.csv_files]
  df = pd.concat(dfs, ignore_index=True)

  # Ensure correct dtypes
  df["nodes"] = df["nodes"].astype(int)
  df["cpus"] = df["cpus"].astype(int)

  # Map names
  df['cluster'] = df['cluster'].map(CLUSTER_NAMES_MAP)
  df['partition'] = df['partition'].map(PARTITION_NAMES_MAP)

  print(df)

  # Plot scaling of Time
  # plt.figure(figsize=(8,6))
  # for partition, grp in df.groupby("partition"):
  #   grp_sorted = grp.sort_values("nodes")
  #   plt.plot(grp_sorted["nodes"], grp_sorted["Time"], marker="o", label=partition)
  # plt.xticks(df["nodes"].unique())
  # plt.xlabel("Nodes")
  # plt.ylabel("Time (s)")
  # plt.title("HPL Scaling - Runtime")
  # plt.legend()
  # plt.grid(True)
  # path = Path(args.out) / "hpl_scaling_time.png"
  # path.parent.mkdir(parents=True, exist_ok=True)
  # plt.savefig(path, dpi=150, bbox_inches="tight")
  # print(f"Plot saved as {path.absolute()}")

  # Plot scaling of Gflops
  plt.figure(figsize=(10, 7))
  
  cluster_color_map = create_color_map(df.sort_values('cluster')["cluster"].unique())

  for cluster, grp_cluster in df.groupby("cluster"):
    partition_linestyles = create_linestyle_map(grp_cluster["partition"].unique())

    for partition, grp_cluster_partition in grp_cluster.groupby("partition"):
      grp_sorted = grp_cluster_partition.sort_values(["nodes",'partition'])
      plt.plot(
        grp_sorted["nodes"],
        grp_sorted["Gflops"],
        marker="o",
        label=f'{cluster}-{partition}',
        color=cluster_color_map[cluster],
        linestyle=partition_linestyles[partition],
      )
  plt.xticks(df["nodes"].unique())
  plt.xlabel("Nodes")
  plt.ylabel("GFLOPs")
  plt.title("HPL Scaling")
  plt.legend()
  plt.tight_layout()
  plt.grid(True)
  path = Path(args.out) / "HPL_Scaling_GFLOPs.png"
  plt.savefig(path, dpi=200, bbox_inches="tight")
  print(f"Plot saved as {path.absolute()}")


if __name__ == "__main__":
  main()
