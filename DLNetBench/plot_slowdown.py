#!/usr/bin/env python3
"""
Plot congestion impact of concurrent DLNetBench executions vs isolated baselines.

For each (strategy, nodes, placement) combination, this script:
  1. Parses per-rank throughputs from raw ccutils stdout files.
  2. Applies adaptive warm-up skipping: if a rank recorded >= 5 iterations
     the first 3 are dropped (DP strategies have slow warm-up); otherwise
     the default --skip-first value is used.
  3. Computes the median throughput per rank, then takes the minimum across
     all ranks (the bottleneck rank limits distributed training speed).
  4. Calculates slowdown = baseline_min / concurrent_min.
     Ratios > 1.0 indicate the concurrent run was slower than isolated
     (congestion-induced slowdown).

Data sources (under each system's backup dir):
  SbatchMan/experiments/<system>/   isolated baselines  (metadata.yaml + stdout.log)
  workerpool_out/                   concurrent runs       (*.stdout files)

Output plots (saved to --output-dir, one set per system):
  slowdown_<system>.png          2-panel bar chart: % slowdown + mean slowdown extent
  slowdown_<system>_violin.png   violin distribution of all ratios
  slowdown_<system>_boxplot.png  boxplot with outliers + mean diamond
  slowdown_<system>_strip.png    strip/jitter plot with individual points
"""

import argparse
import json
import os
import re
import sys
import glob
from collections import defaultdict
from command_map import get_model_from_command

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")                  # non-interactive backend (headless server)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch   # for legend colour swatches

# ============================================================================
#  Constants & Configuration
# ============================================================================

# Mapping from concurrent-run filename placement tags to the corresponding
# baseline placement class stored in SbatchMan metadata.yaml files.
# This lets us pair each concurrent run with the correct isolated baseline.
PLACEMENT_MAP = {
    "intra-l1":              "INTRA_L1_RANDOM",
    "intra-group":           "INTRA_GROUP_RANDOM",
    "inter-group":           "INTER_GROUP_RANDOM",
    "intra-group-same-l1-2": "INTRA_GROUP_SAME_L1_2",
    "inter-group-same-l1-2": "INTER_GROUP_SAME_L1_2",
    "intra-group-same-l1-4": "INTRA_GROUP_SAME_L1_4",
    "inter-group-same-l1-4": "INTER_GROUP_SAME_L1_4",
    "na": "na",
}

# Canonical ordering used to lay out bars within each placement group.
STRATEGY_ORDER = ["DP", "FSDP", "DP+PP", "DP+PP+TP", "DP+PP+Expert"]

# Short display names for strategies (used in tick labels and legend).
# "Expert" is abbreviated to "Exp." to avoid label collisions on the x-axis.
STRATEGY_DISPLAY = {
    "DP":           "D",
    "FSDP":         "FSDP",
    "DP+PP":        "D+P",
    "DP+PP+TP":     "D+P+T",
    "DP+PP+Expert": "D+P+E",
}

# Model used by each parallelism strategy (extracted from baseline commands).
# Each strategy uses exactly one model across all node/placement configurations.
STRATEGY_MODEL = {
    "vit-h":        "ViT-H",
    "llama3-8b":    "LaM-8",
    "llama3-70b":   "LaM-70",
    "minerva-7b":   "Minv",
    "mixtral-8x7b": "Mxt",
}

# Canonical ordering of placements — the primary sort key.
# Bars are grouped by placement; within each group they follow STRATEGY_ORDER.
PLACEMENT_ORDER = [
    "intra-l1",
    "intra-group",
    "intra-group-same-l1-2",
    "intra-group-same-l1-4",
    "inter-group",
    "inter-group-same-l1-2",
    "inter-group-same-l1-4",
]

# One distinct colour per parallelism strategy so that bars in different
# placement groups but using the same strategy share a colour.
STRATEGY_COLORS = {
    "DP":           "#1f77b4",   # blue
    "FSDP":         "#ff7f0e",   # orange
    "DP+PP":        "#2ca02c",   # green
    "DP+PP+TP":     "#d62728",   # red
    "DP+PP+Expert": "#9467bd",   # purple
}

# Human-readable placement names used as shared group labels on the x-axis.
PLACEMENT_DISPLAY = {
    "intra-l1":              "Intra L1",
    "intra-group":           "Intra Group",
    "inter-group":           "Inter Group",
    "intra-group-same-l1-2": "Intra Group\n2 Nodes/Switch",
    "inter-group-same-l1-2": "Inter Group\n2 Nodes/Switch",
    "intra-group-same-l1-4": "Intra Group\n4 Nodes/Switch",
    "inter-group-same-l1-4": "Inter Group\n4 Nodes/Switch",
}


def format_placement_label(placement):
    if placement == 'na' or not placement:
        return ''
    return PLACEMENT_DISPLAY.get(placement, placement)


def make_category_label(strategy, nodes, placement, model_name):
    placement_label = format_placement_label(placement)
    if not placement_label:
        return f'{strategy}\n{nodes}n\n{model_name}'
    return f'{strategy}\n{nodes}n / {placement_label}\n{model_name}'

# Leonardo cluster: each node has 4 GPUs (used to convert nodes -> GPUs).
GPUS_PER_NODE = 4

def format_gpus(gpus):
    '''Return a concise GPU label, e.g. 1024 -> '1K GPU'.'''
    if gpus >= 1024 and gpus % 1024 == 0:
        return f'{gpus // 1024}K GPU'
    if gpus >= 1000:
        return f'{gpus / 1024:.1f}K GPU'
    return f'{gpus} GPU'


# ============================================================================
#  Raw stdout parsing
# ============================================================================

# Regex for ccutils stdout format:
#   [[Rank N]]
#   { ... JSON with "throughputs": [...] ... }
#   [[END Rank N]]
_RANK_RE = re.compile(r"\[\[Rank (\d+)\]\]\n(.+?)\n\[\[END Rank \d+\]\]")


def parse_stdout_throughputs(filepath):
    """
    Parse a ccutils stdout file and extract throughput arrays per rank.

    Parameters
    ----------
    filepath : str
        Path to the stdout file.

    Returns
    -------
    dict
        {rank_id (int): [throughput_values (float)]} for every rank found.
    """
    ranks = {}
    with open(filepath, "r") as f:
        content = f.read()
    for m in _RANK_RE.finditer(content):
        rank_id = int(m.group(1))
        try:
            data = json.loads(m.group(2))
        except json.JSONDecodeError:
            continue
        tp = data.get("throughputs", [])
        if tp:
            ranks[rank_id] = tp
    return ranks


def parse_model_name_from_stdout(filepath):
    """Return model_name from stdout JSON metadata in the run file."""
    with open(filepath, 'r', errors='ignore') as f:
        for line in f:
            m = re.search(r'"model_name"\s*:\s*"([^"]+)"', line)
            if m:
                return m.group(1)
    return None


def min_throughput_across_ranks(ranks_throughputs, skip_first=1, adaptive_skip=True):
    """
    Compute a single representative throughput from multi-rank, multi-iteration
    data, targeting steady-state performance of the bottleneck rank.

    For each rank:
      - Adaptive warm-up removal: if the rank ran >= 5 iterations, skip the
        first 3 (DP strategies have a long warm-up ramp); otherwise use
        ``skip_first``.  Always keep at least the last iteration.
      - Take the **median** of the remaining iterations (robust to noise).

    Then return the **minimum** across all ranks — the bottleneck rank sets
    the effective distributed throughput.

    Parameters
    ----------
    ranks_throughputs : dict
        {rank_id: [throughputs]} as returned by parse_stdout_throughputs.
    skip_first : int
        Default iterations to discard (overridden to 3 when >= 5 are available
        and *adaptive_skip* is True).
    adaptive_skip : bool
        If True (default), use skip=3 for ranks with >= 5 iterations (tuned
        for baselines whose first iterations are warm-up).  Set to False for
        concurrent runs where throughput variability is the signal, not
        warm-up noise.

    Returns
    -------
    float or None
        Minimum per-rank median throughput, or None if no usable data.
    """
    medians = []
    for tp in ranks_throughputs.values():
        # Adaptive skip: DP baselines typically have 6 iterations where the
        # first 3 are warm-up; shorter runs (3-4 iters) use skip_first.
        # Concurrent runs should set adaptive_skip=False to keep all
        # post-warm-up iterations (variability there is the signal).
        skip = (3 if len(tp) >= 6 else skip_first) if adaptive_skip else skip_first
        usable = tp[skip:] if len(tp) > skip else tp[-1:]
        if usable:
            medians.append(float(np.median(usable)))
    return min(medians) if medians else None


# ============================================================================
#  Baseline parsing
# ============================================================================

def parse_baselines(backup_dir, skip_first=1, system_name="jupiter"):
    """
    Parse all isolated baseline experiments from SbatchMan output.

    Each baseline is a directory with:
      metadata.yaml  — experiment config (strategy, nodes, placement_class, status)
      stdout.log     — ccutils output with per-rank throughput JSON blocks

    Only COMPLETED experiments are included.

    Parameters
    ----------
    backup_dir : str
        Root directory containing SbatchMan/ and workerpool_out/.
    skip_first : int
        Iterations to skip per rank for warm-up removal.
    system_name : str
        System name (e.g. "jupiter", "leonardo") used to locate the
        correct SbatchMan subdirectory.

    Returns
    -------
    dict
        {(strategy, nodes, placement_class): min_throughput} per baseline.
    """
    baselines = {}
    if system_name == "nvl72":
        base = os.path.join(backup_dir, "SbatchMan", "experiments", "nvl72")
        # nvl72 has a simpler layout with experiment subfolders and metadata
        meta_pattern = os.path.join(base, "*", "baseline_*", "*", "metadata.yaml")
    else:
        base = os.path.join(backup_dir, "SbatchMan", "experiments", system_name)
        # SbatchMan directory layout:
        #   <system>_<set>/<experiment_name>/<YYYYMMDD_HHMMSS>/metadata.yaml
        meta_pattern = os.path.join(base, f"{system_name}_*", "*", "20*", "metadata.yaml")

    for meta_path in glob.glob(meta_pattern):
        with open(meta_path) as f:
            meta = yaml.safe_load(f)

        # Only use experiments that finished successfully
        if meta.get("status") != "COMPLETED":
            continue

        v = meta.get("variables", {})
        if system_name == "nvl72" and v.get("gpu_model", "").upper() != "GB300":
            continue
        strategy = v.get("strategy")
        nodes = v.get("nodes")
        placement = v.get("placement_class") or v.get("placement")
        if not all([strategy, nodes, placement]):
            continue

        stdout = os.path.join(os.path.dirname(meta_path), "stdout.log")
        if not os.path.isfile(stdout):
            continue

        model_name = v.get("model_name")
        if not model_name:
            model_name = get_model_from_command(meta.get("command", ""))
        if not model_name:
            model_name = parse_model_name_from_stdout(stdout)
        if not model_name:
            model_name = "unknown"

        ranks = parse_stdout_throughputs(stdout)
        t0 = min_throughput_across_ranks(ranks, skip_first, adaptive_skip=True)
        if t0 is not None:
            baselines[(strategy, int(nodes), placement, model_name)] = t0

    return baselines


# ============================================================================
#  Concurrent-run parsing
# ============================================================================

# Filename convention for concurrent runs:
#   <Strategy>_g<GPUs>_n<Nodes>_<Placement>[_<AppID>]_rep<N>_<worker>.stdout
# Example: DP_g8_n2_intra-l1_28_rep7_worker3.stdout
_CONC_RE = re.compile(
    r"((?:[A-Za-z]+\+)*[A-Za-z]+)_g(\d+)_n(\d+)_(.+?)_rep(\d+)_"
)


def parse_concurrent(backup_dir, skip_first=1, system_name="jupiter"):
    """
    Parse all concurrent-execution stdout files from the workerpool output.

    Filenames encode experiment parameters directly: strategy, GPU count,
    node count, placement, and repetition number.

    Parameters
    ----------
    backup_dir : str
        Root directory containing SbatchMan/ and workerpool_out/.
    skip_first : int
        Iterations to skip per rank for warm-up removal.

    Returns
    -------
    list of dict
        Each dict has keys: strategy, gpus, nodes, placement, app_id, rep,
        throughput.
    """
    results = []
    is_nvl72 = system_name == "nvl72"
    wp_dir = os.path.join(backup_dir, "workerpool_out")
    all_files = glob.glob(os.path.join(wp_dir, "*", "*.stdout"))
    total = len(all_files)

    for idx, filepath in enumerate(all_files, 1):
        if idx % 2000 == 0:
            print(f"  ... parsed {idx}/{total} concurrent files", file=sys.stderr)

        fname = os.path.basename(filepath)
        m = _CONC_RE.match(fname)
        if not m:
            continue

        if is_nvl72:
            is_gb300 = False
            with open(filepath, 'r', errors='ignore') as f:
                for line in f:
                    if '"GPU model":"GB300"' in line or '"GPU model": "GB300"' in line:
                        is_gb300 = True
                        break
            if not is_gb300:
                continue

        strategy, gpus, nodes, rest, rep = m.groups()
        # "rest" may contain both placement and a numeric app_id suffix:
        #   "intra-l1_28" -> placement="intra-l1", app_id=28
        app_id_m = re.search(r"_(\d+)$", rest)
        app_id = int(app_id_m.group(1)) if app_id_m else -1
        placement = re.sub(r"_\d+$", "", rest)

        model_name = parse_model_name_from_stdout(filepath)
        if not model_name:
            model_name = "unknown"

        ranks = parse_stdout_throughputs(filepath)
        t = min_throughput_across_ranks(ranks, skip_first, adaptive_skip=True)
        if t is not None:
            results.append({
                "strategy": strategy,
                "gpus": int(gpus),
                "nodes": int(nodes),
                "placement": placement,
                "model_name": model_name,
                "app_id": app_id,
                "rep": int(rep),
                "throughput": t,
            })

    return results


# ============================================================================
#  Slowdown computation
# ============================================================================

def compute_slowdowns(baselines, concurrent):
    """
    Match each concurrent run to its isolated baseline and compute the
    congestion impact (slowdown ratio).

    Matching uses (strategy, nodes) and maps the concurrent placement name
    to the baseline's placement class via PLACEMENT_MAP.

    Parameters
    ----------
    baselines : dict
        {(strategy, nodes, placement): throughput} from parse_baselines.
    concurrent : list of dict
        Concurrent run records from parse_concurrent.

    Returns
    -------
    dict
        {(strategy, nodes, placement): [ratio, ...]} per category.
        ratio = baseline_throughput / concurrent_throughput.
        Values > 1.0 indicate congestion-induced slowdown.
    """
    slowdowns = defaultdict(list)
    unmatched = set()
    skipped_na = 0

    for run in concurrent:
        placement = run["placement"]

        # Map the concurrent placement name to the baseline placement class
        baseline_placement = PLACEMENT_MAP.get(placement)
        if baseline_placement is None:
            # Unmapped placements (e.g. "na") are silently skipped
            skipped_na += 1
            continue

        model_name = run.get("model_name", "unknown")

        # Look up the isolated baseline throughput for this configuration
        bkey = (run["strategy"], run["nodes"], baseline_placement, model_name)
        t0 = baselines.get(bkey)
        if t0 is None or t0 == 0:
            unmatched.add(bkey)
            continue

        # ratio > 1.0 => concurrent was slower (congestion-induced slowdown)
        # ratio = baseline / concurrent: how many times slower the job ran
        sigma = t0 / run["throughput"]
        cat = (run["strategy"], run["nodes"], placement, model_name)
        slowdowns[cat].append(sigma)

    if skipped_na:
        print(f"  Skipped {skipped_na} concurrent runs with unmapped placement "
              f"(e.g. 'na')")
    if unmatched:
        print(f"  WARNING: {len(unmatched)} combos had no matching baseline:")
        for u in sorted(unmatched, key=str):
            print(f"    {u}")

    return slowdowns


# ============================================================================
#  Plotting helpers
# ============================================================================

def _sort_key(cat):
    """
    Sort key for (strategy, nodes, placement, model_name) tuples.

    Primary sort: placement index  (groups bars by allocation policy)
    Secondary:    strategy index   (consistent order within each group)
    Tertiary:     node count       (ascending)
    Quaternary:  model name    (stable alphabetical)
    """
    s, n, p, m = cat
    si = STRATEGY_ORDER.index(s) if s in STRATEGY_ORDER else 99
    pi = PLACEMENT_ORDER.index(p) if p in PLACEMENT_ORDER else 99
    return (pi, si, n, m)


def _placement_groups(categories):
    """
    Identify contiguous runs of bars that share the same placement.

    Since categories are sorted placement-first by _sort_key, consecutive
    entries with the same placement naturally form a group.

    Parameters
    ----------
    categories : list of (strategy, nodes, placement, model_name)
        Sorted category keys.

    Returns
    -------
    list of (placement_name, start_idx, end_idx)
        One entry per contiguous group.
    """
    if not categories:
        return []
    groups = []
    current = categories[0][2]
    start = 0
    for i, cat in enumerate(categories):
        if cat[2] != current:
            groups.append((current, start, i - 1))
            current = cat[2]
            start = i
    groups.append((current, start, len(categories) - 1))
    return groups


def _setup_grouped_xaxis(ax, categories, groups, y_offset=-0.36):
    """
    Create a two-level x-axis layout.

    Level 1  (tick labels):  short horizontal label per bar showing
             parallelism strategy and GPU count, e.g. "DP\\n8 GPUs".
    Level 2  (group labels): a single centred label below each placement
             group, connected by a thin horizontal bracket line.

    Vertical dashed lines are drawn between adjacent placement groups.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    categories : list of (strategy, nodes, placement, model_name)
        Sorted category keys (one per bar).
    groups : list of (placement, start_idx, end_idx)
        Placement groups from _placement_groups.
    y_offset : float
        Axes-fraction y-coordinate for the group label text (negative = below
        the axis).  Tune this if labels overlap tick marks.
    """
    x = np.arange(len(categories))

    # --- Level 1: per-bar tick labels (strategy + model + GPU count, horizontal) ---
    labels = []
    for cat in categories:
        strategy, nodes, _placement, m = cat
        gpus = nodes * GPUS_PER_NODE
        display_strategy = STRATEGY_DISPLAY.get(strategy, strategy)
        model_label = STRATEGY_MODEL.get(m, "unknown")
        labels.append(f"{display_strategy}\n{model_label}\n{format_gpus(gpus)}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, ha="center")

    # Blended transform: x in data coordinates, y in axes fraction [0..1]
    trans = ax.get_xaxis_transform()

    # --- Level 2: shared placement group labels with brackets ---
    for placement, start, end in groups:
        if placement == "na":
            continue
        center = (start + end) / 2.0
        display_name = PLACEMENT_DISPLAY.get(placement, placement)

        # Horizontal bracket line spanning the full group width
        bracket_y = y_offset + 0.01  # bracket even closer to label
        ax.plot([start - 0.15, end + 0.15], [bracket_y, bracket_y],  # even narrower bracket,
                transform=trans, color="black", lw=0.7, clip_on=False)

        # Centred placement name below the bracket
        ax.text(center, y_offset, display_name, transform=trans,
                ha="center", va="top", fontsize=11, fontweight="bold",  # match x-tick label size
                clip_on=False)

    # --- Vertical boundary lines between adjacent placement groups ---
    for i in range(len(groups) - 1):
        boundary_x = groups[i][2] + 0.5
        ax.axvline(boundary_x, color="grey", ls="--", lw=0.6, alpha=0.6)

    # The group labels replace the traditional x-axis label
    ax.set_xlabel("")


def _make_legend(ax, strats):
    """
    Add a compact colour-coded strategy legend (upper-right corner).

    Only unique strategies are shown, in first-appearance order.
    Uses STRATEGY_DISPLAY for short labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    strats : list of str
        Strategy name per bar (duplicates are deduplicated).
    """
    handles = []
    seen = set()
    for s in strats:
        if s not in seen:
            display = STRATEGY_DISPLAY.get(s, s)
            handles.append(Patch(facecolor=STRATEGY_COLORS.get(s, "#888"),
                                 label=display))
            seen.add(s)
    ax.legend(handles=handles, loc="upper right", fontsize=11,
              ncol=len(handles), framealpha=0.85)


# Y-axis clip threshold: values above this are visually clipped and annotated.
Y_CLIP = 2.0


def _annotate_clipped(ax, categories, slowdowns, y_clip=Y_CLIP):
    """
    For each category, count how many data points exceed y_clip and annotate
    them above the plot frame with the percentage and maximum value.

    Points above y_clip are still in the data but fall outside the visible
    y-axis range, so this annotation tells the reader what they're missing.
    Labels are placed just above the top of the axes frame so they don't
    collide with visible outlier points.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    categories : list of (strategy, nodes, placement)
    slowdowns : dict
    y_clip : float
        The y-axis upper limit.
    """
    for i, cat in enumerate(categories):
        vals = np.array(slowdowns[cat])
        above = vals[vals > y_clip]
        if len(above) > 0:
            pct = 100.0 * len(above) / len(vals)
            max_val = float(above.max())
            pct_str = f"{pct:.2f}%" if pct < 1 else f"{pct:.0f}%"
            ax.text(i, y_clip * 1.005, f"{pct_str}\n({max_val:.1f}x)",
                    ha="center", va="bottom", fontsize=11, color="#d62728",
                    fontweight="bold", zorder=5, clip_on=False)


# ============================================================================
#  Plot 1: Two-panel bar chart (% slowdown + mean extent)
# ============================================================================

def plot_slowdown(slowdowns, output_path):
    """
    Two-panel bar chart.

    Top panel:    percentage of concurrent runs that experienced meaningful
                  slowdown (ratio > 1.05, i.e. > 5% to filter noise).
    Bottom panel: mean slowdown ratio among only the slowed-down runs,
                  showing how severe the congestion impact is on average.

    Bars are coloured by parallelism strategy and grouped by placement.
    """
    categories = sorted(slowdowns.keys(), key=_sort_key)
    if not categories:
        print("No data to plot.")
        return

    groups = _placement_groups(categories)

    # ---------- per-category statistics ----------
    pct_slow = []       # percentage of runs above threshold (meaningful slowdown)
    mean_ratio = []     # mean ratio of the slowdown-only subset
    n_runs = []         # total runs in this category
    strats = []         # strategy per bar (for colouring)

    # Only ratios above this count as meaningful slowdown (filters <5% noise)
    threshold = 1.05

    for cat in categories:
        vals = np.array(slowdowns[cat])
        slow_vals = vals[vals > threshold]
        pct_slow.append(100.0 * len(slow_vals) / len(vals))
        mean_ratio.append(
            float(np.mean(slow_vals)) if len(slow_vals) > 0 else 1.0
        )
        n_runs.append(len(vals))
        strats.append(cat[0])

    colors = [STRATEGY_COLORS.get(s, "#888888") for s in strats]

    # ---------- figure layout ----------
    x = np.arange(len(categories))
    bw = 0.72                                # bar width
    fig_w = max(14, len(categories) * 0.7)   # scale width with bar count

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(fig_w, 9),
        sharex=True,                         # both panels share the x-axis
        gridspec_kw={"height_ratios": [1, 1.2], "hspace": 0.06},
    )

    # ---------- top panel: % of runs with slowdown ----------
    ax_top.bar(x, pct_slow, width=bw, color=colors,
               edgecolor="white", linewidth=0.5)
    ax_top.set_ylabel("Runs with slowdown (%)", fontsize=14)
    ax_top.set_ylim(0, 109)
    ax_top.yaxis.set_major_locator(mticker.MultipleLocator(25))
    ax_top.axhline(0, color="black", linewidth=0.4)

    # Vertical group boundaries on the top panel as well
    for i in range(len(groups) - 1):
        boundary_x = groups[i][2] + 0.5
        ax_top.axvline(boundary_x, color="grey", ls="--", lw=0.6, alpha=0.6)

    # ---------- bottom panel: mean slowdown ratio (slowdown runs only) ------
    bar_colors_bot = ["#d62728" if v > 1.0 else "#2ca02c" for v in mean_ratio]

    ax_bot.bar(x, mean_ratio, width=bw, color=bar_colors_bot,
               edgecolor="white", linewidth=0.5)
    ax_bot.axhline(1.0, color="black", ls="--", lw=1, alpha=0.8)
    ax_bot.set_ylabel("Congestion Impact (σ)", fontsize=14)
    y_max = max(1.25, max(mean_ratio) * 1.08)
    ax_bot.set_ylim(0.95, y_max)

    # ---------- grouped x-axis on the bottom panel ----------
    # y_offset is more negative to leave room under the 2-row tick labels
    _setup_grouped_xaxis(ax_bot, categories, groups, y_offset=-0.32)

    # ---------- legend ----------
    _make_legend(ax_top, strats)

    fig.subplots_adjust(bottom=0.15, top=0.97, hspace=0.06)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {output_path}")


# ============================================================================
#  Plot 2: Violin
# ============================================================================

def plot_slowdown_violin(slowdowns, output_path):
    """
    Violin plot: full distribution of slowdown ratios per category.

    Each violin's width is proportional to the local density of ratios.
    A black median line is overlaid inside each violin body.
    """
    categories = sorted(slowdowns.keys(), key=_sort_key)
    if not categories:
        print("No data to plot (violin).")
        return

    groups = _placement_groups(categories)
    data = [np.array(slowdowns[cat]) for cat in categories]
    strats = [cat[0] for cat in categories]
    colors = [STRATEGY_COLORS.get(s, "#888888") for s in strats]

    x = np.arange(len(categories))
    fig_w = max(14, len(categories) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, 4))

    # Draw violin bodies coloured by strategy
    parts = ax.violinplot(data, positions=x, showmedians=True,
                          showextrema=False, widths=0.7)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i])
        body.set_edgecolor("black")
        body.set_linewidth(0.5)
        body.set_alpha(0.75)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(0.8)

    # Horizontal reference at ratio = 1.0 (no slowdown)
    ax.axhline(1.0, color="black", ls="--", lw=1, alpha=0.8)

    # Clip y-axis and annotate categories with outliers above the clip
    ax.set_ylim(0.95, Y_CLIP)
    _annotate_clipped(ax, categories, slowdowns)

    _setup_grouped_xaxis(ax, categories, groups)
    ax.set_ylabel("Congestion Impact (σ)", fontsize=14)
    _make_legend(ax, strats)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {output_path}")


# ============================================================================
#  Plot 3: Boxplot
# ============================================================================

def plot_slowdown_boxplot(slowdowns, output_path, y_clip=1.2):
    """
    Box-and-whisker plot per category.

    Shows interquartile range (box), median (black line), mean (diamond),
    whiskers at 1.5 * IQR, and individual outlier points beyond the whiskers.
    """
    categories = sorted(slowdowns.keys(), key=_sort_key)
    if not categories:
        print("No data to plot (boxplot).")
        return

    groups = _placement_groups(categories)
    data = [np.array(slowdowns[cat]) for cat in categories]
    strats = [cat[0] for cat in categories]
    colors = [STRATEGY_COLORS.get(s, "#888888") for s in strats]

    x = np.arange(len(categories))
    fig_w = max(14, len(categories) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, 4))

    # Patch-based boxplot so we can fill each box with a strategy colour.
    # showmeans=True with diamond marker to distinguish mean from median.
    bp = ax.boxplot(data, positions=x, widths=0.6, patch_artist=True,
                    showfliers=True,
                    showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=4),
                    flierprops=dict(marker="o", markersize=2, alpha=0.4,
                                    markerfacecolor="grey"))
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(colors[i])
        box.set_alpha(0.75)
    for element in ("whiskers", "caps"):
        for line in bp[element]:
            line.set_color("black")
            line.set_linewidth(0.6)
    for line in bp["medians"]:
        line.set_color("black")
        line.set_linewidth(1.0)

    # Horizontal reference at ratio = 1.0 (no slowdown)
    ax.axhline(1.0, color="black", ls="--", lw=1, alpha=0.8)

    # Clip y-axis and annotate categories with outliers above the clip
    ax.set_ylim(0.95, y_clip)
    _annotate_clipped(ax, categories, slowdowns, y_clip=y_clip)

    _setup_grouped_xaxis(ax, categories, groups)
    ax.set_ylabel("Congestion Impact (σ)", fontsize=14)
    _make_legend(ax, strats)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {output_path}")


# ============================================================================
#  Plot 4: Strip / Jitter
# ============================================================================

def plot_slowdown_strip(slowdowns, output_path):
    """
    Strip (jitter) plot: every data point is shown as a small dot with
    random horizontal offset for readability.

    A solid median line is overlaid per category for quick comparison.
    This plot is most useful when sample sizes are small and you want to
    see every individual measurement.
    """
    categories = sorted(slowdowns.keys(), key=_sort_key)
    if not categories:
        print("No data to plot (strip).")
        return

    groups = _placement_groups(categories)
    strats = [cat[0] for cat in categories]
    colors = [STRATEGY_COLORS.get(s, "#888888") for s in strats]

    x = np.arange(len(categories))
    fig_w = max(14, len(categories) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, 4))

    # Scatter each category with reproducible horizontal jitter
    rng = np.random.default_rng(42)    # fixed seed for reproducibility
    for i, cat in enumerate(categories):
        vals = np.array(slowdowns[cat])
        jitter = rng.uniform(-0.25, 0.25, size=len(vals))
        ax.scatter(i + jitter, vals, s=3, alpha=0.35, color=colors[i],
                   edgecolors="none", zorder=2)

    # Reference line at ratio = 1.0
    ax.axhline(1.0, color="black", ls="--", lw=1, alpha=0.8)

    # Median lines per category
    for i, cat in enumerate(categories):
        vals = np.array(slowdowns[cat])
        med = float(np.median(vals))
        ax.plot([i - 0.3, i + 0.3], [med, med],
                color="black", lw=1.2, zorder=3)

    # Clip y-axis and annotate categories with outliers above the clip
    ax.set_ylim(0.95, Y_CLIP)
    _annotate_clipped(ax, categories, slowdowns)

    _setup_grouped_xaxis(ax, categories, groups)
    ax.set_ylabel("Congestion Impact (σ)", fontsize=14)
    _make_legend(ax, strats)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {output_path}")


# ============================================================================
#  Plot 5: Stacked boxplot (faceted rows by placement, uniform columns)
# ============================================================================

def plot_slowdown_boxplot_stacked(slowdowns, output_path):
    """
    Faceted boxplot: one row per placement, columns = strategy + GPU count.

    All rows share the same set of column positions (the union of all
    strategy+GPU combos that appear in any placement).  Positions that
    don't exist for a given placement are left blank.  This makes
    cross-placement comparison straightforward by vertical alignment.
    """
    categories = sorted(slowdowns.keys(), key=_sort_key)
    if not categories:
        print("No data to plot (boxplot_stacked).")
        return

    # --- Build uniform column grid ---
    # Columns = unique (strategy, nodes, model) combos, ordered by strategy/node/model
    col_keys_set = set()
    for s, n, p, m in categories:
        col_keys_set.add((s, n, m))
    col_keys = sorted(col_keys_set,
                      key=lambda snm: (STRATEGY_ORDER.index(snm[0])
                                        if snm[0] in STRATEGY_ORDER else 99,
                                        snm[1],
                                        snm[2]))

    col_idx = {k: i for i, k in enumerate(col_keys)}
    n_cols = len(col_keys)

    # --- Build row list (placements present in the data) ---
    row_placements = []
    seen = set()
    for s, n, p, m in categories:
        if p not in seen:
            row_placements.append(p)
            seen.add(p)
    # Reorder by PLACEMENT_ORDER
    row_placements = sorted(row_placements,
                            key=lambda p: PLACEMENT_ORDER.index(p)
                            if p in PLACEMENT_ORDER else 99)
    n_rows = len(row_placements)

    # --- Figure layout ---
    row_h = 1.1           # height per subplot row
    fig_w = max(7, n_cols * 0.85 + 2.5)  # width scales with column count
    fig_h = n_rows * row_h + 0.8         # total height

    fig, axes = plt.subplots(
        n_rows, 1, figsize=(fig_w, fig_h),
        sharex=True, sharey=True,
    )
    if n_rows == 1:
        axes = [axes]

    x_positions = np.arange(n_cols)

    for row_idx, placement in enumerate(row_placements):
        ax = axes[row_idx]

        # Collect data and colors for positions that exist in this placement
        data_by_col = {}
        for s, n, p, m in categories:
            if p != placement:
                continue
            ci = col_idx[(s, n, m)]
            data_by_col[ci] = np.array(slowdowns[(s, n, p, m)])

        # Draw boxplots only at the positions that have data
        if data_by_col:
            positions = sorted(data_by_col.keys())
            box_data = [data_by_col[p] for p in positions]
            box_colors = [STRATEGY_COLORS.get(col_keys[p][0], "#888")
                          for p in positions]

            bp = ax.boxplot(
                box_data, positions=positions, widths=0.6,
                patch_artist=True, showfliers=True, showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="white",
                               markeredgecolor="black", markersize=3),
                flierprops=dict(marker="o", markersize=1.5, alpha=0.35,
                                markerfacecolor="grey"),
            )
            for i, box in enumerate(bp["boxes"]):
                box.set_facecolor(box_colors[i])
                box.set_alpha(0.75)
            for element in ("whiskers", "caps"):
                for line in bp[element]:
                    line.set_color("black")
                    line.set_linewidth(0.5)
            for line in bp["medians"]:
                line.set_color("black")
                line.set_linewidth(0.8)

            # Annotate clipped outliers
            for ci in positions:
                vals = data_by_col[ci]
                above = vals[vals > Y_CLIP]
                if len(above) > 0:
                    pct = 100.0 * len(above) / len(vals)
                    max_val = float(above.max())
                    pct_str = f"{pct:.2f}%" if pct < 1 else f"{pct:.0f}%"
                    ax.text(ci, Y_CLIP * 1.005,
                            f"{pct_str}\n({max_val:.0f}x)",
                            ha="center", va="bottom", fontsize=7.5,
                            color="#d62728", fontweight="bold", zorder=5,
                            clip_on=False)

        # Reference line
        ax.axhline(1.0, color="black", ls="--", lw=0.6, alpha=0.7)

        # Row label (placement name) on the right side
        display_name = PLACEMENT_DISPLAY.get(placement, placement)
        ax.set_ylabel(display_name.replace("\n", " "), fontsize=10.5,
                       rotation=90, labelpad=8)

        ax.set_xlim(-0.6, n_cols - 0.4)
        ax.set_ylim(0.95, Y_CLIP)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(axis="y", alpha=0.2, lw=0.4)

    # --- Shared x-axis labels on the bottom row ---
    ax_bot = axes[-1]
    col_labels = []
    for s, n, m in col_keys:
        gpus = n * GPUS_PER_NODE
        ds = STRATEGY_DISPLAY.get(s, s)
        model_label = m or "unknown"
        col_labels.append(f"{ds}\n{model_label}\n{format_gpus(gpus)}")
    ax_bot.set_xticks(x_positions)
    ax_bot.set_xticklabels(col_labels, fontsize=8.5, ha="center")

    # --- Shared y-axis label ---
    fig.text(0.01, 0.5, "Congestion Impact (σ)", va="center",
             rotation="vertical", fontsize=13)

    # --- Legend ---
    _make_legend(axes[0], [k[0] for k in col_keys])

    fig.subplots_adjust(hspace=0.12, left=0.14, right=0.97, top=0.95, bottom=0.10)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {output_path}")


# ============================================================================
#  CLI entry point
# ============================================================================

# Per-system backup directory configuration.
# Maps system name to its backup directory (relative to the DLNetBench folder).
SYSTEMS = {
    "jupiter": os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "jupiter", "official", "backup",
    ),
    "leonardo": os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "leonardo", "official", "backups",
    ),
    "nvl72": os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "nvl72", "official", "backup",
    ),
}


def process_system(system_name, backup_dir, skip_first, output_dir):
    """Parse data and generate all plots for a single system."""
    print(f"{'='*60}")
    print(f"  System     : {system_name}")
    print(f"  Backup dir : {backup_dir}")
    print(f"  Skip first : {skip_first} iterations")
    print(f"{'='*60}")
    print()

    # -- Step 1: parse isolated baselines --
    print("Parsing baselines ...")
    baselines = parse_baselines(backup_dir, skip_first, system_name=system_name)
    print(f"  {len(baselines)} baseline T0 values\n")

    # -- Step 2: parse concurrent runs --
    print("Parsing concurrent runs ...")
    concurrent = parse_concurrent(backup_dir, skip_first, system_name=system_name)
    print(f"  {len(concurrent)} concurrent run throughputs\n")

    # -- Step 3: compute slowdown ratios --
    print("Computing slowdowns ...")
    slowdowns = compute_slowdowns(baselines, concurrent)
    print(f"  {len(slowdowns)} categories\n")

    # -- Step 4: summary table for quick terminal inspection --
    print(f"{'Category':50s} {'Runs':>5s}  {'Mean':>6s}  {'Median':>6s}  "
          f"{'%>1':>5s}")
    print("-" * 82)
    for cat in sorted(slowdowns.keys(), key=_sort_key):
        v = np.array(slowdowns[cat])
        gpus = cat[1] * GPUS_PER_NODE
        display_s = STRATEGY_DISPLAY.get(cat[0], cat[0])
        model_name = cat[3] if len(cat) > 3 else 'unknown'
        lbl = f"{display_s} / {format_gpus(gpus)} / {cat[2]} / {model_name}"
        print(f"{lbl:50s} {len(v):5d}  {v.mean():6.3f}  {np.median(v):6.3f}  "
              f"{100*np.mean(v>1):5.1f}%")
    print()

    # -- Step 5: generate all four plot variants --
    os.makedirs(output_dir, exist_ok=True)
    base_name = f"slowdown_{system_name}.png"
    output_path = os.path.join(output_dir, base_name)

    plot_slowdown(slowdowns, output_path)

    violin_path = output_path.replace(".png", "_violin.png")
    plot_slowdown_violin(slowdowns, violin_path)

    # System-specific boxplot y-clip: ensure every box's upper whisker is
    # visible.  Whisker = min(max, Q3 + 1.5*IQR).  If any category's whisker
    # exceeds 1.2, widen the clip to accommodate it.
    max_whisker = 1.0
    for v in slowdowns.values():
        arr = np.array(v)
        q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        whisker_hi = min(float(arr.max()), q3 + 1.5 * (q3 - q1))
        max_whisker = max(max_whisker, whisker_hi)
    boxplot_y_clip = max(1.2, round(max_whisker * 1.1, 1))

    box_path = output_path.replace(".png", "_boxplot.png")
    plot_slowdown_boxplot(slowdowns, box_path, y_clip=boxplot_y_clip)

    strip_path = output_path.replace(".png", "_strip.png")
    plot_slowdown_strip(slowdowns, strip_path)

    stacked_path = output_path.replace(".png", "_boxplot_stacked.png")
    plot_slowdown_boxplot_stacked(slowdowns, stacked_path)


def main():
    """Parse arguments, load data, compute slowdowns, generate all plots."""
    ap = argparse.ArgumentParser(
        description="Plot congestion impact of concurrent vs isolated baselines."
    )
    ap.add_argument(
        "--systems", nargs="+", default=list(SYSTEMS.keys()),
        help=f"Systems to process (default: {list(SYSTEMS.keys())})",
    )
    ap.add_argument(
        "--skip-first", type=int, default=1,
        help="Iterations to skip per rank before taking median (default: 1)",
    )
    ap.add_argument(
        "-o", "--output-dir",
        default=os.path.join(os.path.expanduser("~"), "work", "sc26_unitn", "plots"),
        help="Output directory for plots (default: ~/work/sc26_unitn/plots)",
    )
    args = ap.parse_args()

    for system_name in args.systems:
        backup_dir = SYSTEMS.get(system_name)
        if backup_dir is None:
            print(f"ERROR: unknown system '{system_name}'. "
                  f"Known systems: {list(SYSTEMS.keys())}")
            continue
        if not os.path.isdir(backup_dir):
            print(f"WARNING: backup dir not found for {system_name}: {backup_dir}")
            continue

        process_system(system_name, backup_dir, args.skip_first, args.output_dir)
        print()


if __name__ == "__main__":
    main()
