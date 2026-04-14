"""
plot_stream.py — Visualise STREAM benchmark results from **files** *or* **sbm jobs**.

* Two sub-commands:
  * **files** - ORIGINAL workflow (one or more plain-text STREAM outputs).
  * **sbm**   - Pulls completed jobs via ``sbm.jobs_list`` and parses their
    captured *stdout* on the fly (see example below).
* A shared plotting backend, so both paths end up on the same figure.

Example - classic file mode
───────────────────────────
```bash
python plot_stream.py files A100_1c.txt A100_32c.txt -H A100 A100
```

Example - sbm mode (no filenames needed!)
─────────────────────────────────────────
```bash
python plot_stream.py sbm -s COMPLETE
```
This walks over
```python
jobs = sbm.jobs_list(from_active=True, from_archived=True,
                     status=["COMPLETE"])
```
then, for every job whose ``config_name`` looks like
``HWName_16cpus`` (regex ``(\\w+)_(\\d+)cpus``), it extracts the STREAM metrics
straight from ``job.get_stdout()``.

Dependencies
────────────
Python ≥3.9, pandas, matplotlib, and of course your in-house ``sbm`` package.
Install the PyPI bits with: ``pip install pandas matplotlib``.
"""

import argparse
import os
from pathlib import Path
import re
import sys
from typing import List, Union

import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path('results')
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(Path(__file__).parent.parent / "common"))
from utils.cli import load_csv_files
from utils.plots import add_zoom_inset
from constants.plots import *
from constants.systems import BOARD_NAMES_MAP

# ──────────────────────────────────────────────────────────────────────────────
# Regex patterns and constants
# ──────────────────────────────────────────────────────────────────────────────

FUNCTIONS = ["Copy", "Scale", "Add", "Triad"]
_RATE_RE = re.compile(rf"^({'|'.join(FUNCTIONS)}):\s+([0-9]+(?:\.[0-9]+)?)")
_THREADS_RE = re.compile(r"Number of Threads counted\s*=\s*(\d+)")
_JOB_RE = re.compile(r"(\w+)_(\d+)cpus")  # captures hw and core count

FONT_LEGEND += 6
plt.rc('axes', titlesize=FONT_AXES)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_AXES)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONT_TICKS)   # fontsize of the tick labels
plt.rc('ytick', labelsize=FONT_TICKS)   # fontsize of the tick labels
plt.rc('legend', fontsize=FONT_LEGEND)  # legend fontsize
plt.rc('figure', titlesize=FONT_TITLE)  # fontsize of the figure title

LOG_SCALE = True

# ──────────────────────────────────────────────────────────────────────────────
# Low-level parsing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_metrics(lines: List[str]) -> dict[str, float]:
  """Return a dict with keys 'cores', 'Copy', 'Scale', 'Add', 'Triad'."""
  metrics: dict[str, float] = {}
  for line in lines:
    if "cores" not in metrics:
      mt = _THREADS_RE.search(line)
      if mt:
        metrics["cores"] = int(mt.group(1))

    mk = _RATE_RE.match(line.strip())
    if mk:
      func, rate = mk.groups()
      metrics[func] = float(rate)

    if len(metrics) == 1 + len(FUNCTIONS):
      break  # early exit once all fields are present

  missing = [k for k in (FUNCTIONS + ["cores"]) if k not in metrics]
  if missing:
    raise ValueError(f"missing fields: {', '.join(missing)}")
  return metrics


def _parse_stream_text(text: str) -> dict[str, float]:
  return _parse_metrics(text.splitlines())


def _parse_single_stream_file(path: str) -> dict[str, float]:
  with open(path, "r", encoding="utf-8", errors="ignore") as fh:
    return _parse_metrics(fh)

# ──────────────────────────────────────────────────────────────────────────────
# DataFrame builders
# ──────────────────────────────────────────────────────────────────────────────

def _infer_hardware_labels(paths: List[str], manual: Union[List[str], None]) -> List[str]:
  if manual is None:
    return [os.path.basename(p).split("_")[0] for p in paths]
  if len(manual) != len(paths):
    raise ValueError("--hardware/-H labels must match number of input files")
  return manual


def _build_dataframe_from_files(paths: List[str], hw_labels: List[str]) -> pd.DataFrame:
  rows = []
  for path, hw in zip(paths, hw_labels):
    parsed = _parse_single_stream_file(path)
    for func in FUNCTIONS:
      rows.append({
        "hardware": hw,
        "cores": parsed["cores"],
        "function": func,
        "bandwidth_MBps": parsed[func],
      })
  return pd.DataFrame(rows).sort_values(["function", "hardware", "cores"])


def _build_dataframe_from_jobs(status: List[str]) -> pd.DataFrame:
  import sbatchman as sbm

  jobs = sbm.jobs_list(from_active=True, from_archived=True, status=status)
  rows: list[dict] = []
  for job in jobs:
    vars = job.variables or {}
    if 'system' in vars and 'ncpus' in vars:
      hw = vars['system']
      cores_str = vars['ncpus']
    else:
      m = _JOB_RE.match(job.config_name)
      if not m:
        continue  # skip unrelated jobs
      hw, cores_str = m.groups()
    cores = int(cores_str)
    stdout = job.get_stdout()
    try:
      parsed = _parse_stream_text(stdout)
    except ValueError as exc:
      print(f"[WARN] job {job.id}: {exc}", file=sys.stderr)
      continue
    for func in FUNCTIONS:
      rows.append({
        "hardware": hw,
        "cores": cores,
        "function": func,
        "bandwidth_MBps": parsed[func],
      })

  if not rows:
    raise RuntimeError("No valid STREAM outputs found via sbm.")
  return pd.DataFrame(rows).sort_values(["function", "hardware", "cores"])


def _plot(df: pd.DataFrame, cores: Union[List[int], None]) -> None:
  hws_color_map = dict(zip(sorted(df['hardware'].unique()), COLORS_CYCLE))

  fig, axes = plt.subplots(2, 2, figsize=(17, 11.5), sharey=False)
  axes = axes.flatten()

  for idx, func in enumerate(FUNCTIONS):
    ax = axes[idx]
    func_df = df[df["function"] == func]
    if cores:
      func_df = func_df[func_df['cores'].isin(cores)]

    func_df['bandwidth_GBps'] = func_df['bandwidth_MBps'] / 1e3

    for j, (hw, group) in enumerate(func_df.groupby("hardware", sort=False)):
      group_sorted = group.sort_values("cores")
      ax.plot(
        group_sorted["cores"],
        group_sorted["bandwidth_GBps"],
        color=hws_color_map.get(hw, hw),
        label=BOARD_NAMES_MAP.get(hw, hw),
        marker=MARKERS_LIST[j % len(MARKERS_LIST)],
        linewidth=1.8,
      )
    if LOG_SCALE:
      ax.set_xscale('log', base=2)
    cores_ticks = list(sorted(func_df['cores'].unique()))
    ax.set_xticks(cores_ticks)
    ax.set_xticklabels([str(c) for c in cores_ticks])
    ax.set_title(func, fontsize=FONT_TITLE)
    if idx >= 2: ax.set_xlabel("CPU cores")
    if idx % 2 == 0: ax.set_ylabel("Bandwidth [GB/s]")
    ax.grid(True, linestyle="-", alpha=0.8)
    # ax.legend(loc='best')
    
    zoom_cores_limit = 8
    max_y = func_df[func_df['cores'] <= zoom_cores_limit]['bandwidth_GBps'].max()
    min_y = func_df[func_df['cores'] <= zoom_cores_limit]['bandwidth_GBps'].min()
    
    if idx == 0:
      zoom_ax = add_zoom_inset(
        ax,
        zoom_region=(0.85, 8.5, -1., max_y*1.05),
        inset_position=(0.02, 0.36, 0.68, 0.6),  # x0, y0, width, height (ALL in percentage wrt ax size)
        rect_kwargs={'edgecolor': 'purple', 'linestyle': '-.', 'linewidth': 1},
        zoom_ax_kwargs={'grid': True, 'set_xticks': [2**p for p in range(8) if 2**p <= zoom_cores_limit]}
      )
      zoom_ax.yaxis.tick_right()
      for dir in ['top', 'right', 'bottom', 'left']:
        zoom_ax.spines[dir].set_linestyle("-.")
        zoom_ax.spines[dir].set_edgecolor("purple")
        zoom_ax.spines[dir].set_linewidth(1.5)

    ## Add zoom
    if not LOG_SCALE:
      zoom_ax = add_zoom_inset(
        ax,
        zoom_region=(0.0, 9.0, min_y*0.95, max_y*1.05),
        inset_position=(0.33, 0.1, 0.6, 0.4),  # x0, y0, width, height (ALL in percentage wrt ax size)
        rect_kwargs={'edgecolor': 'purple', 'linestyle': '-.', 'linewidth': 1},
        zoom_ax_kwargs={'grid': True, 'set_xticks': [2**p for p in range(8) if 2**p <= zoom_cores_limit]}
      )
      zoom_ax.yaxis.tick_right()
      for dir in ['top', 'right', 'bottom', 'left']:
        zoom_ax.spines[dir].set_linestyle("-.")
        zoom_ax.spines[dir].set_edgecolor("purple")
        zoom_ax.spines[dir].set_linewidth(1.5)

  if SET_FIG_TITLE:
    fig.suptitle("STREAM - Memory Bandwidth - Scaling", fontsize=17, y=0.97)
  fig.tight_layout() # (rect=[0, 0, 1, 0.95])
  
  # Add a single legend at the top
  legend_ax = fig.add_subplot(111, frameon=False)
  legend_ax.axis('off')
  func_df = df[df["function"] == 'Copy']
  func_df['bandwidth_GBps'] = func_df['bandwidth_MBps'] / 1e3
  for j, (hw, group) in enumerate(func_df.groupby("hardware", sort=False)):
    group_sorted = group.sort_values("cores")
    legend_ax.plot(
      group_sorted["cores"],
      group_sorted["bandwidth_GBps"],
      color=hws_color_map.get(hw, hw),
      label=BOARD_NAMES_MAP.get(hw, hw),
      marker=MARKERS_LIST[j % len(MARKERS_LIST)],
      linewidth=1.8,
    )
  handles, labels = legend_ax.get_legend_handles_labels()
  fig.delaxes(legend_ax)
  fig.legend(
    handles, labels,
    loc='upper center',
    ncol=len(labels),
    frameon=False
  )
  fig.tight_layout(rect=[0., 0., 1, 0.96])

  ## Save plot
  path = OUT_DIR / f'STREAM.png'
  fig.savefig(path, dpi=300)
  print(f"[ OK ] figure saved to {path.resolve().absolute()}")

# ──────────────────────────────────────────────────────────────────────────────
# Unified CLI (sub-commands: files / sbm)
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: Union[List[str], None] = None) -> None:
  parser = argparse.ArgumentParser(
    description="Plot STREAM results from plain files **or** directly from SbatchMan jobs.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  subparsers = parser.add_subparsers(dest="mode", required=True)

  # ── files sub-command ────────────────────────────────────────────────
  files_p = subparsers.add_parser("files", help="Parse one or more STREAM output files")
  files_p.add_argument("inputs", nargs="+", help="STREAM output text files")
  files_p.add_argument("-H", "--hardware", nargs="+", help="Hardware label per input file")
  files_p.add_argument("-c", "--cores", nargs='+', help="A filter for the number of cores", default=None)

  # ── df sub-command ────────────────────────────────────────────────
  df_p = subparsers.add_parser("df", help="Parse a CSV input file")
  df_p.add_argument("inputs", nargs="+", type=Path, help="Input CSV text file(s)")
  df_p.add_argument("-c", "--cores", nargs='+', help="A filter for the number of cores", default=None)

  # ── sbm sub-command ────────────────────────────────────────────────
  sbm_p = subparsers.add_parser("sbm", help="Pull STREAM outputs from sbm jobs")
  sbm_p.add_argument("-s", "--status", nargs="+", default=["COMPLETED"], help="Job status filter")
  sbm_p.add_argument("-c", "--cores", nargs='+', help="A filter for the number of cores", default=None)

  args = parser.parse_args(argv)
  cores = [int(c) for c in args.cores] if args.cores else None

  if args.mode == "sbm":
    df = _build_dataframe_from_jobs(args.status)
    path = OUT_DIR / "STREAM_data.csv"
    df.to_csv(path, index=False)
    print(f"Wrote CSV summary with {len(df)} rows to {path.resolve().absolute()}")
  elif args.mode == "files":
    labels = _infer_hardware_labels(args.inputs, args.hardware)
    df = _build_dataframe_from_files(args.inputs, labels)
  elif args.mode == "df":
    df = load_csv_files(args.inputs)

  _plot(df, cores)


if __name__ == "__main__":
  main()

# Run example: python3 plots.py sbm -s COMPLETED -o results/stream.png