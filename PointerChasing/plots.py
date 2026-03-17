import itertools
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import sbatchman as sbm

sys.path.append(str(Path(__file__).parent.parent / "common"))
from py_utils.cli import get_basic_cli_parser, load_csv_files
from py_utils.utils.plots import add_zoom_inset
from py_utils.constants import *

FONT_TITLE -= 12
FONT_AXES -= 7
FONT_LEGEND -= 5
FONT_TICKS -= 2
plt.rc('axes', titlesize=FONT_AXES)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_AXES)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONT_TICKS)   # fontsize of the tick labels
plt.rc('ytick', labelsize=FONT_TICKS)   # fontsize of the tick labels
plt.rc('legend', fontsize=FONT_LEGEND)  # legend fontsize
plt.rc('figure', titlesize=FONT_TITLE)  # fontsize of the figure title

def human_readable_bytes(x, _):
  if x == 0:
    return "0"
  units = ['B', 'KiB', 'MiB', 'GiB']
  i = 0
  while x >= 1024 and i < len(units) - 1:
    x /= 1024.0
    i += 1
  if x >= 10:
    return f"{int(x)}{units[i]}"
  else:
    return f"{x:.0f}{units[i]}"
  

def parse_random_chase(path, hw_name):
  data = np.genfromtxt(path, skip_header=2, usecols=(0, 1))
  return pd.DataFrame(data, columns=['x', 'y']).assign(program='random-chase', hw=hw_name)


def parse_linear_chase(path, hw_name):
  data = np.genfromtxt(path, skip_header=2, usecols=(0, 1))
  return pd.DataFrame(data, columns=['x', 'y']).assign(program='linear-chase', hw=hw_name)


def parse_fused_linear_chase(path, hw_name):
  try:
    raw = np.genfromtxt(path, skip_header=4)
  except ValueError:
    raw = np.genfromtxt(path, skip_header=4, skip_footer=1)

  stride = raw[:, 0]
  dfs = []
  for fuse in range(8):  # fuse factors 1-8
    y = raw[:, fuse + 1]
    df = pd.DataFrame({
      'x': stride,
      'y': y,
      'fuse': fuse + 1,
      'program': 'fused-linear-chase',
      'hw': hw_name
    })
    dfs.append(df)
  return pd.concat(dfs, ignore_index=True)


PARSERS = {
  'random-chase': parse_random_chase,
  'linear-chase': parse_linear_chase,
  'fused-linear-chase': parse_fused_linear_chase
}


def generate_dataframe_from_jobs(jobs):
  dfs = []
  for job in jobs:
    prog = job.tag
    hw = job.config_name
    if prog not in PARSERS:
      print(f"Skipping unrecognized program: {prog}")
      continue
    df = PARSERS[prog](job.get_stdout_path(), hw)
    dfs.append(df)
  return pd.concat(dfs, ignore_index=True)


def plot_random(df, dst: Path, hws_color_map, hws_linestyle_map, hws_marker_map, fuse_color_map):
  fig, ax = plt.subplots(figsize=(11, 4), dpi=300)
  
  occupied_cache_size_text_pos = []
  for hw_name, group in df.groupby("hw"):
    ax.plot(group['x'], group['y'], marker="o", markersize=3, linewidth=1, label=BOARD_NAMES_MAP.get(hw_name,hw_name), color=hws_color_map[hw_name])
    l1, l2, l3 = CACHE_SIZES.get(hw_name, ((0,''), (0,''), (0,'')))
    for pos, name in l1, l2, l3:
      if pos > 0:
        ax.axvline(x=pos, linestyle="--", color=hws_color_map[hw_name], linewidth=0.9)
        occupied = any([abs(1-(pos/p))<.05 for p in occupied_cache_size_text_pos])
        ax.text(
          pos*(1.03 if occupied else 0.78),
          df['y'].max()*1.04,
          f'{BOARD_SHORT_NAMES_MAP.get(hw_name,hw_name)} {name}',
          rotation=270,
          color=hws_color_map[hw_name],
          horizontalalignment='left',
          verticalalignment='top',
        )
        occupied_cache_size_text_pos.append(pos)

  ax.set_xscale("log", base=2)
  ax.xaxis.set_major_formatter(FuncFormatter(human_readable_bytes))
  ax.set_xlabel("Memory Size") # [Bytes]
  ax.set_ylabel("Avg Access Time [ns]")
  if SET_FIG_TITLE:
    ax.set_title("Random Chase - Memory Latency vs Memory Size", fontsize=FONT_TITLE)
  ax.grid(True, linestyle=":", alpha=0.8)
  ax.legend()
  
  zoom_ax = add_zoom_inset(
    ax,
    zoom_region=(800, 9*1024, 1., 3.),
    inset_position=(0.01, 0.18, 0.25, 0.45),  # x0, y0, width, height (ALL in percentage wrt ax size)
    rect_kwargs={'edgecolor': 'purple', 'linestyle': '-.', 'linewidth': 1},
    rect_region=(900, -8, 8*1024, 20),
    zoom_ax_kwargs={
      'grid': True,
      'set_xticks': [i*1024 for i in [2,4,8]],
      'set_xticklabels': dict(labels=[f'{i}Kib' for i in [2,4,8]], fontsize=12),
      'set_yticks': [1., 1.5, 2.5],
      'set_yticklabels': dict(labels=['1', '1.5', '2.5'], fontsize=12),
    },
  )
  zoom_ax.yaxis.tick_right()
  for dir in ['top', 'right', 'bottom', 'left']:
    zoom_ax.spines[dir].set_linestyle("-.")
    zoom_ax.spines[dir].set_edgecolor("purple")
    zoom_ax.spines[dir].set_linewidth(1.5)
  
  fig.tight_layout()
  fig.savefig(dst)
  print(f'Plot saved to {dst}')
  plt.close(fig)


def plot_linear(df, dst: Path, hws_color_map, hws_linestyle_map, hws_marker_map, fuse_color_map):
  fig, ax = plt.subplots(figsize=(11, 4), dpi=300)
  for hw_name, group in df.groupby("hw"):
    ax.plot(group['x'], group['y'], marker="o", markersize=3, linewidth=1, label=BOARD_NAMES_MAP.get(hw_name,hw_name), color=hws_color_map[hw_name])

  ax.set_xlabel("Stride [Bytes]")
  ax.set_ylabel("Avg Access Time [ns]")
  if SET_FIG_TITLE:
    ax.set_title("Linear Chase - Memory Latency vs Stride", fontsize=FONT_TITLE)
  ax.grid(True, linestyle="-", alpha=0.6)
  ax.legend()
  fig.tight_layout()
  fig.savefig(dst)
  print(f'Plot saved to {dst}')
  plt.close(fig)


def plot_fused(df: pd.DataFrame, dst: Path, hws_color_map, hws_linestyle_map, hws_marker_map, fuse_color_map):
  fig, ax = plt.subplots(figsize=(14, 5), dpi=300)
  
  ## !! FILTER !!
  df = df[df['x']<=80]
  df = df[df['hw'].isin(['baldo', 'pioneer'])]

  for (hw_name, fuse), group in df.groupby(["hw", "fuse"]):
    label = f"{BOARD_NAMES_MAP.get(hw_name, hw_name)} - Fuse {int(fuse)}"
    ax.plot(
      group['x'], group['y'],
      marker=hws_marker_map.get(hw_name, 'o'),
      markersize=5,
      linewidth=1.5,
      label=label,
      linestyle=hws_linestyle_map.get(hw_name, '-'),
      color=fuse_color_map.get(fuse, '#000000')
    )

  ax.set_xlabel("Stride [Bytes]")
  ax.set_ylabel("Access Speed [GiB/s]")
  if SET_FIG_TITLE:
    ax.set_title("Fused Linear Chase - Memory Access Speed vs Stride and Fuse", fontsize=FONT_TITLE + 5)
  ax.grid(True, linestyle="-", alpha=0.8)

  # Improve legend
  ax.legend(
    ncol=2,# Change to 3 or more if 2 is still too wide
    loc='best',
    # loc='upper center',
    # bbox_to_anchor=(0.5, -0.1),  # Move below the plot
    # title="Board - Fuse"
  )

  fig.tight_layout()
  # fig.subplots_adjust(bottom=0.2)  # Make space for bottom legend
  fig.savefig(dst)
  print(f'Plot saved to {dst}')
  plt.close(fig)


PLOTTERS = {
  'random-chase': plot_random,
  'linear-chase': plot_linear,
  'fused-linear-chase': plot_fused
}


def main():
  parser = get_basic_cli_parser()
  args = parser.parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)

  if args.csv:
    print(f"Reading data from CSV file(s): {args.csv}")
    df = load_csv_files(args.csv)
  else:
    print("Generating data SbatchMan from jobs...")
    jobs = sbm.jobs_list(from_active=True, from_archived=False, status=[sbm.Status.COMPLETED, sbm.Status.TIMEOUT])
    df = generate_dataframe_from_jobs(jobs)
    path = args.output_dir / f'{sbm.get_cluster_name()}_pointer_chasing.csv'
    df.to_csv(path, index=False)
    print(f"Saved dataframe to CSV: {path}")
    
  hws = sorted(df['hw'].unique())
  hws_color_map = dict(zip(hws, COLORS_CYCLE))
  fuse_color_map = dict(zip(sorted(df['fuse'].unique()), COLORS_CYCLE))
  hws_linestyle_map = dict(zip(hws, LINESTYLES_CYCLE))
  hws_marker_map = dict(zip(hws, itertools.cycle(MARKERS_LIST)))

  for program, plot_func in PLOTTERS.items():
    df_subset = df[df['program'] == program]
    if df_subset.empty:
      continue

    # Combined plot over all hardware configs
    plot_func(df_subset, args.output_dir / f"combined_{program}.png", hws_color_map, hws_linestyle_map, hws_marker_map, fuse_color_map)

    # Individual plots per hardware
    for hw, hw_df in df_subset.groupby("hw"):
      plot_func(hw_df, args.output_dir / f"{hw}_{program}.png", hws_color_map, hws_linestyle_map, hws_marker_map, fuse_color_map)

  print("✅ All plots generated.")


if __name__ == "__main__":
  main()
