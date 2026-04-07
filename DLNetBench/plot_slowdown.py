#!/usr/bin/env python3
"""
plot_results.py — Congestion-impact visualisations for DLNetBench.

Reads parsed slowdown data produced by parse_results.py and generates
four plot variants per system:

  slowdown_<system>.png               2-panel bar chart: % slowdown + mean extent
  slowdown_<system>_violin.png        violin distribution of all ratios
  slowdown_<system>_boxplot.png       boxplot with outliers + mean diamond
  slowdown_<system>_strip.png         strip/jitter plot with individual points
  slowdown_<system>_boxplot_stacked   faceted boxplot (one row per placement)

Usage
-----
  python plot_results.py [--systems jupiter leonardo nvl72]
                         [--skip-first N]
                         [-o OUTPUT_DIR]
"""

import argparse
from collections import defaultdict
from math import ceil
import os
import pprint
from typing import Dict, List

import numpy as np
import matplotlib

from data_types import GPUS_PER_NODE_MAP, PLACEMENT_ORDER, STRATEGY_ORDER
from data_types import ensure_model, ensure_placement, ensure_strategy

matplotlib.use("Agg")                   # non-interactive backend (headless)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch    # for legend colour swatches
from matplotlib.ticker import FuncFormatter 

from parse_raw_data import (
    SYSTEMS,
    RunKey,
    build_baselines_dict,
    parse_baselines,
    parse_concurrent,
    compute_slowdowns,
)

# Y-axis clip threshold: values above this are visually clipped and annotated.
Y_CLIP = 2.0


# ============================================================================
#  Formatting helpers
# ============================================================================

def format_gpus(gpus):
    """Return a concise GPU label, e.g. 1024 -> '1K GPU'."""
    if gpus >= 1024 and gpus % 1024 == 0:
        return f"{gpus // 1024}K GPU"
    if gpus >= 1000:
        return f"{gpus / 1024:.1f}K GPU"
    return f"{gpus} GPU"


# ============================================================================
#  Sorting and grouping
# ============================================================================

# def _sort_concurrent(c: ConcurrentRun):
#     """
#     Sort key for  tuples.
#     """
#     s, g, p, m = run.strategy, run.gpus, run.placement_class, run.model
#     si = STRATEGY_ORDER.index(s) if s in STRATEGY_ORDER else 99
#     pi = PLACEMENT_ORDER.index(p) if p in PLACEMENT_ORDER else 99
#     return (pi, si, g, m)

def _sort_run(run: RunKey):
    """
    Sort key for (strategy, gpus, placement, model_name) tuples.
    """
    s, g, p, m = run.strategy, run.gpus, run.placement_class, run.model
    si = STRATEGY_ORDER.index(s) if s in STRATEGY_ORDER else 99
    pi = PLACEMENT_ORDER.index(p) if p in PLACEMENT_ORDER else 99
    return (pi, si, g, m)


def _placement_groups(categories: List[RunKey]):
    """
    Identify contiguous runs of bars that share the same placement.

    Since categories are sorted placement-first by _sort_run, consecutive
    entries with the same placement naturally form a group.

    Returns
    -------
    list of (placement_name, start_idx, end_idx)
    """
    if not categories:
        return []
    groups = []
    current = categories[0].placement_class
    start = 0
    for i, cat in enumerate(categories):
        if cat.placement_class != current:
            groups.append((current, start, i - 1))
            current = cat.placement_class
            start = i
    groups.append((current, start, len(categories) - 1))
    return groups


# ============================================================================
#  Shared axis helpers
# ============================================================================

def _setup_grouped_xaxis(ax, categories: List[RunKey], groups, system, y_offset=-0.36):
    """
    Create a two-level x-axis layout.

    Level 1 (tick labels):  short per-bar label — strategy, model, GPU count.
    Level 2 (group labels): centred placement name below each group, with a
                            thin horizontal bracket.

    Vertical dashed lines separate adjacent placement groups.
    """
    x = np.arange(len(categories))

    labels = []
    for cat in categories:
        gpus = cat.gpus
        display_strategy = cat.strategy.short()
        model_label = cat.model.short()
        labels.append(f"{display_strategy}\n{model_label}\n{format_gpus(gpus)}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, ha="center")

    trans = ax.get_xaxis_transform()   # x=data, y=axes fraction

    for placement, start, end in groups:
        if placement == "na":
            continue
        center = (start + end) / 2.0
        display_name = placement.display()

        bracket_y = y_offset + 0.01
        ax.plot(
            [start - 0.15, end + 0.15], [bracket_y, bracket_y],
            transform=trans, color="black", lw=0.7, clip_on=False,
        )
        ax.text(
            center, y_offset, display_name, transform=trans,
            ha="center", va="top", fontsize=11, fontweight="bold",
            clip_on=False,
        )

    for i in range(len(groups) - 1):
        boundary_x = groups[i][2] + 0.5
        ax.axvline(boundary_x, color="grey", ls="--", lw=0.6, alpha=0.6)

    ax.set_xlabel("")


def _make_legend(ax, strats):
    """Add a compact colour-coded strategy legend (upper-right corner)."""
    handles = []
    seen = set()
    for s in strats:
        if s not in seen:
            display = s.short()
            handles.append(
                Patch(facecolor=s.color(), label=display)
            )
            seen.add(s)
    ax.legend(
        handles=handles, loc="upper right", fontsize=11,
        ncol=len(handles), framealpha=0.85,
    )


def _annotate_clipped(ax, categories, slowdowns, y_clip=Y_CLIP):
    """
    Annotate bars/violins/points that exceed y_clip with percentage and max.

    Labels are placed just above the top of the axes frame.
    """
    for i, cat in enumerate(categories):
        vals = np.array(slowdowns[cat])
        above = vals[vals > y_clip]
        if len(above) > 0:
            pct = 100.0 * len(above) / len(vals)
            max_val = float(above.max()) / 100.0
            pct_str = f"{pct:.2f}%" if pct < 1 else f"{pct:.0f}%"
            ax.text(
                i, y_clip * 1.005, f"{pct_str}\n({max_val:.1f}x)",
                ha="center", va="bottom", fontsize=11, color="#d62728",
                fontweight="bold", zorder=5, clip_on=False,
            )


# ============================================================================
#  Plot 1: Two-panel bar chart (% slowdown + mean extent)
# ============================================================================

def plot_slowdown(slowdowns, system, output_path):
    """
    Two-panel bar chart.

    Top panel:    percentage of concurrent runs with meaningful slowdown
                  (ratio > 1.05, i.e. > 5% to filter noise).
    Bottom panel: mean slowdown ratio among only the slowed-down runs.
    """
    categories = sorted(slowdowns.keys(), key=_sort_run)
    if not categories:
        print("No data to plot.")
        return

    groups = _placement_groups(categories)

    threshold = 1.05
    pct_slow, mean_ratio, strats = [], [], []

    for cat in categories:
        vals = np.array(slowdowns[cat])
        slow_vals = vals[vals > threshold]
        pct_slow.append(100.0 * len(slow_vals) / len(vals))
        mean_ratio.append(float(np.mean(slow_vals)) if len(slow_vals) > 0 else 1.0)
        strats.append(cat[0])

    colors = [s.color() for s in strats]
    x = np.arange(len(categories))
    bw = 0.72
    fig_w = max(14, len(categories) * 0.7)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(fig_w, 9), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.2], "hspace": 0.06},
    )

    # --- top panel ---
    ax_top.bar(x, pct_slow, width=bw, color=colors, edgecolor="white", linewidth=0.5)
    ax_top.set_ylabel("Runs with slowdown (%)", fontsize=14)
    ax_top.set_ylim(0, 109)
    ax_top.yaxis.set_major_locator(mticker.MultipleLocator(25))
    ax_top.axhline(0, color="black", linewidth=0.4)
    for i in range(len(groups) - 1):
        ax_top.axvline(groups[i][2] + 0.5, color="grey", ls="--", lw=0.6, alpha=0.6)

    # --- bottom panel ---
    bar_colors_bot = ["#d62728" if v > 1.0 else "#2ca02c" for v in mean_ratio]
    ax_bot.bar(x, mean_ratio, width=bw, color=bar_colors_bot,
               edgecolor="white", linewidth=0.5)
    ax_bot.axhline(1.0, color="black", ls="--", lw=1, alpha=0.8)
    ax_bot.set_ylabel("Congestion Impact (σ)", fontsize=14)
    y_max = max(1.25, max(mean_ratio) * 1.08)
    ax_bot.set_ylim(0.95, y_max)

    _setup_grouped_xaxis(ax_bot, categories, groups, system, y_offset=-0.32)
    _make_legend(ax_top, strats)

    fig.subplots_adjust(bottom=0.15, top=0.97, hspace=0.06)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {output_path}")


# ============================================================================
#  Plot 2: Violin
# ============================================================================

def plot_slowdown_violin(slowdowns: Dict[RunKey, List[float]], system, output_path):
    """Violin plot: full distribution of slowdown ratios per category."""
    categories: List[RunKey] = list(sorted(slowdowns.keys(), key=_sort_run))
    if not categories:
        print("No data to plot (violin).")
        return

    groups = _placement_groups(categories)
    data = [np.array(slowdowns[cat]) for cat in categories]
    strats = [cat.strategy for cat in categories]
    colors = [s.color() for s in strats]

    x = np.arange(len(categories))
    fig_w = max(14, len(categories) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, 4))

    parts = ax.violinplot(data, positions=x, showmedians=True,
                          showextrema=False, widths=0.7)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i])
        body.set_edgecolor("black")
        body.set_linewidth(0.5)
        body.set_alpha(0.75)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(0.8)

    ax.axhline(1.0, color="black", ls="--", lw=1, alpha=0.8)
    ax.set_ylim(0.95, Y_CLIP)
    _annotate_clipped(ax, categories, slowdowns)
    _setup_grouped_xaxis(ax, categories, groups, system)
    ax.set_ylabel("Congestion Impact (σ)", fontsize=14)
    _make_legend(ax, strats)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {output_path}")


# ============================================================================
#  Plot 3: Boxplot
# ============================================================================

def plot_slowdown_boxplot(slowdowns: Dict[RunKey, np.ndarray], system, output_path, y_clip=1.2):
    """
    Box-and-whisker plot per category.

    Shows IQR (box), median (black line), mean (diamond), whiskers at
    1.5 * IQR, and individual outliers beyond the whiskers.
    """
    categories = sorted([k for k, v in slowdowns.items() if len(v) > 0], key=_sort_run)
    if not categories:
        print("No data to plot (boxplot).")
        return

    groups = _placement_groups(categories)
    data = [slowdowns[cat] for cat in categories]
    strats = [cat.strategy for cat in categories]
    colors = [ensure_strategy(s).color() for s in strats]

    x = np.arange(len(categories))
    fig_w = max(14, len(categories) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, 4))

    bp = ax.boxplot(
        data, positions=x, widths=0.6, patch_artist=True,
        showfliers=True, showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="black", markersize=4),
        flierprops=dict(marker="o", markersize=2, alpha=0.4,
                        markerfacecolor="grey"),
    )
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

    ax.axhline(1.0, color="black", ls="--", lw=1, alpha=0.8)
    ax.set_ylim(0.95, y_clip)
    _annotate_clipped(ax, categories, slowdowns, y_clip=y_clip)
    _setup_grouped_xaxis(ax, categories, groups, system)
    ax.set_ylabel("Slowdown %", fontsize=20)
    # _make_legend(ax, strats)
    ax.grid(True, alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output_path}")
    
# ============================================================================
#  IMPROVED Violin
# ============================================================================

def _find_clusters_kde(
    arr: np.ndarray,
    cluster_width: float = 0.01,
    min_cluster_frac: float = 0.05,   # NEW: merge clusters smaller than this fraction
    n_grid: int = 512,
) -> list[tuple[float, float, np.ndarray]]:
    """
    KDE-based cluster detection with post-merge of small/overlapping clusters.

    1. Estimate KDE, find local maxima, merge peaks closer than `cluster_width`.
    2. Assign each raw point to its nearest surviving peak.
    3. Post-merge pass: repeatedly merge adjacent clusters if their y-brackets
       overlap OR if either cluster contains fewer than `min_cluster_frac` of
       the total points. Repeats until stable.
    """
    from scipy.signal import find_peaks
    from scipy.stats import gaussian_kde

    if len(arr) < 2:
        return [(arr.min(), arr.max(), arr)]

    kde   = gaussian_kde(arr)
    grid  = np.linspace(arr.min(), arr.max(), n_grid)
    density = kde(grid)

    peak_idx, _ = find_peaks(density)
    if len(peak_idx) == 0:
        peak_idx = np.array([np.argmax(density)])

    # Merge KDE peaks closer than cluster_width
    merged: list[float] = []
    for p in sorted(grid[peak_idx]):
        if merged and (p - merged[-1]) < cluster_width:
            merged[-1] = (merged[-1] + p) / 2
        else:
            merged.append(p)
    merged_peaks = np.array(merged)

    # Assign raw points to nearest peak
    assignments = np.argmin(
        np.abs(arr[:, None] - merged_peaks[None, :]), axis=1
    )
    clusters: list[tuple[float, float, np.ndarray]] = []
    for ci in range(len(merged_peaks)):
        members = arr[assignments == ci]
        if len(members):
            clusters.append((members.min(), members.max(), members))

    # ── Post-merge: collapse overlapping or too-small adjacent clusters ───────
    min_count = max(1, int(min_cluster_frac * len(arr)))
    changed = True
    while changed and len(clusters) > 1:
        changed = False
        merged_clusters: list[tuple[float, float, np.ndarray]] = []
        i = 0
        while i < len(clusters):
            if i + 1 < len(clusters):
                lo_a, hi_a, mem_a = clusters[i]
                lo_b, hi_b, mem_b = clusters[i + 1]
                overlaps  = hi_a >= lo_b          # brackets touch or overlap
                too_small = len(mem_a) < min_count or len(mem_b) < min_count
                if overlaps or too_small:
                    combined = np.concatenate([mem_a, mem_b])
                    merged_clusters.append((combined.min(), combined.max(), combined))
                    i += 2
                    changed = True
                    continue
            merged_clusters.append(clusters[i])
            i += 1
        clusters = merged_clusters

    return clusters


def plot_slowdown_violinplot(
    slowdowns: Dict[RunKey, np.ndarray],
    system,
    output_path,
    y_clip: float = 1.2,
    threshold: float = 1.025,
    cluster_width: float = 0.01,
    min_cluster_frac: float = 0.05,
):
    """
    Violin plot per category.

    - Categories where NO value exceeds `threshold` are excluded.
    - Shows IQR box, median (black line), mean (white diamond), and whiskers
      at 1.5 × IQR, consistent with the original boxplot logic.
    - KDE-based cluster detection: peaks in the density estimate are found,
      then any two peaks closer than `cluster_width` are merged into one.
      Each surviving cluster gets a right-side bracket spanning the min–max
      of its assigned raw points, labelled "26.7% (8/30)".
    """
    categories = sorted(
        [k for k, v in slowdowns.items() if len(v) > 0 and np.any(v > threshold)],
        key=_sort_run,
    )
    if not categories:
        print("No data to plot (violinplot): all categories below threshold.")
        return

    groups  = _placement_groups(categories)
    data    = [slowdowns[cat] for cat in categories]
    strats  = [cat.strategy for cat in categories]
    colors  = [ensure_strategy(s).color() for s in strats]

    x       = np.arange(len(categories))
    fig_w   = max(11, len(categories) * 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    # ── Violins ───────────────────────────────────────────────────────────────
    vp = ax.violinplot(
        data, positions=x, widths=0.45,
        showmedians=False, showextrema=False, showmeans=False,
    )
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors[i])
        body.set_edgecolor("black")
        body.set_linewidth(0.6)
        body.set_alpha(0.75)

    # ── Per-category overlays ─────────────────────────────────────────────────
    BOX_W       = 0.06    # half-width of the IQR box
    BRACKET_X   = 0.28    # x offset of cluster bracket from violin centre
    BRACKET_CAP = 0.04    # half-width of bracket end-caps
    LABEL_PAD   = 0.015   # gap between bracket right edge and label

    for i, arr in enumerate(data):
        n_total          = len(arr)
        q1, med, q3      = np.percentile(arr, [25, 50, 75])
        iqr              = q3 - q1
        lo_whisk         = max(arr.min(), q1 - 1.5 * iqr)
        hi_whisk         = min(arr.max(), q3 + 1.5 * iqr)

        # IQR box
        ax.add_patch(plt.Rectangle(
            (x[i] - BOX_W, q1), 2 * BOX_W, iqr,
            linewidth=0.8, edgecolor="black",
            facecolor="white", alpha=0.6, zorder=3,
        ))

        # Whisker stems + caps
        for y0, y1 in [(lo_whisk, q1), (q3, hi_whisk)]:
            ax.plot([x[i], x[i]], [y0, y1], color="black", lw=0.8, zorder=3)
        for y_cap in (lo_whisk, hi_whisk):
            ax.plot([x[i] - BOX_W, x[i] + BOX_W], [y_cap, y_cap],
                    color="black", lw=0.8, zorder=3)

        # Median line
        ax.plot([x[i] - BOX_W, x[i] + BOX_W], [med, med],
                color="black", lw=1.2, zorder=4)

        # Mean diamond
        ax.plot(x[i], arr.mean(), marker="D", color="white",
                markeredgecolor="black", markersize=4, zorder=5)

        # ── KDE cluster brackets ──────────────────────────────────────────────
        clusters = _find_clusters_kde(arr, cluster_width=cluster_width, min_cluster_frac=min_cluster_frac)
        bx       = x[i] + BRACKET_X
        
        def fmt_nmembers(n: int) -> str:
            if n > 1e3:
                return f'{round(n/1e3)}k'
            return str(n)

        for lo, hi, members in clusters:
            # Clamp to visible range
            vis_lo = max(lo, 0.95)
            vis_hi = min(hi, y_clip)
            if vis_hi <= vis_lo:
                continue

            pct   = 100.0 * len(members) / n_total
            # label = f"{pct:.1f}%\n{fmt_nmembers(len(members))}/{fmt_nmembers(n_total)}"
            if len(members) == n_total:
                label = f"{fmt_nmembers(n_total)}"
            else:
                label = f"{fmt_nmembers(len(members))}/{fmt_nmembers(n_total)}"

            # Vertical bracket line — gray dashed
            ax.plot([bx, bx], [vis_lo, vis_hi],
                    color="gray", lw=0.8, ls="--", zorder=4,
                    solid_capstyle="butt")

            # End caps — gray solid (dashed caps look noisy at small scale)
            for y_end in (vis_lo, vis_hi):
                ax.plot([bx - BRACKET_CAP, bx + BRACKET_CAP], [y_end, y_end],
                        color="gray", lw=1.0, zorder=4)
            # End caps
            for y_end in (vis_lo, vis_hi):
                ax.plot([bx - BRACKET_CAP, bx + BRACKET_CAP], [y_end, y_end],
                        color="black", lw=1.0, zorder=4)

            # Label centred on the bracket
            ax.text(
                bx + BRACKET_CAP + LABEL_PAD, (vis_lo + vis_hi) / 2, label,
                fontsize=10, va="center", ha="left", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            )

    # ── Decorations ───────────────────────────────────────────────────────────
    ax.axhline(1.0, color="black", ls="--", lw=1, alpha=0.8)
    ax.set_ylim(0.95, y_clip)
    _annotate_clipped(ax, categories, slowdowns, y_clip=y_clip)
    _setup_grouped_xaxis(ax, categories, groups, system)
    ax.set_ylabel("Slowdown $\\sigma$", fontsize=20)
    ax.grid(True, alpha=0.5)
    def format_slowdown(s: float, *_):
        def fmt(x, suffix):
            return f"{int(x)}{suffix}" if x.is_integer() else f"{x:.1f}{suffix}"
        if s >= 100.0:
            return fmt(s / 100.0, "x")
        else:
            return fmt(s, "%")
    ax.yaxis.set_major_formatter(FuncFormatter(format_slowdown))
    ax.yaxis.set_tick_params(labelsize=18)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output_path}")


# ============================================================================
#  Plot 4: Strip / Jitter
# ============================================================================

def plot_slowdown_strip(slowdowns, system, output_path):
    """
    Strip (jitter) plot: every data point shown with random horizontal offset.

    A solid median line is overlaid per category for quick comparison.
    """
    categories = sorted(slowdowns.keys(), key=_sort_run)
    if not categories:
        print("No data to plot (strip).")
        return

    groups = _placement_groups(categories)
    strats = [cat[0] for cat in categories]
    colors = [s.color() for s in strats]

    x = np.arange(len(categories))
    fig_w = max(14, len(categories) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, 4))

    rng = np.random.default_rng(42)    # fixed seed for reproducibility
    for i, cat in enumerate(categories):
        vals = np.array(slowdowns[cat])
        jitter = rng.uniform(-0.25, 0.25, size=len(vals))
        ax.scatter(i + jitter, vals, s=3, alpha=0.35, color=colors[i],
                   edgecolors="none", zorder=2)

    ax.axhline(1.0, color="black", ls="--", lw=1, alpha=0.8)

    for i, cat in enumerate(categories):
        vals = np.array(slowdowns[cat])
        med = float(np.median(vals))
        ax.plot([i - 0.3, i + 0.3], [med, med], color="black", lw=1.2, zorder=3)

    ax.set_ylim(0.95, Y_CLIP)
    _annotate_clipped(ax, categories, slowdowns)
    _setup_grouped_xaxis(ax, categories, groups, system)
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

    All rows share the same column grid (the union of all strategy+GPU combos
    present in the data).  Missing positions are left blank so cross-placement
    comparison is straightforward by vertical alignment.
    """
    categories = sorted(slowdowns.keys(), key=_sort_run)
    if not categories:
        print("No data to plot (boxplot_stacked).")
        return

    # --- Build uniform column grid ---
    col_keys_set = {(s, n, m) for s, n, p, m in categories}
    col_keys = sorted(
        col_keys_set,
        key=lambda snm: (
            STRATEGY_ORDER.index(snm[0]) if snm[0] in STRATEGY_ORDER else 99,
            snm[1],
            snm[2],
        ),
    )
    col_idx = {k: i for i, k in enumerate(col_keys)}
    n_cols = len(col_keys)

    # --- Build row list (placements present in the data) ---
    seen_placements: set = set()
    row_placements = []
    for _s, _n, p, _m in categories:
        if p not in seen_placements:
            row_placements.append(p)
            seen_placements.add(p)
    row_placements = sorted(
        row_placements,
        key=lambda p: PLACEMENT_ORDER.index(p) if p in PLACEMENT_ORDER else 99,
    )
    n_rows = len(row_placements)

    # --- Figure layout ---
    row_h = 1.1
    fig_w = max(7, n_cols * 0.85 + 2.5)
    fig_h = n_rows * row_h + 0.8

    fig, axes = plt.subplots(
        n_rows, 1, figsize=(fig_w, fig_h), sharex=True, sharey=True,
    )
    if n_rows == 1:
        axes = [axes]

    x_positions = np.arange(n_cols)

    for row_idx, placement in enumerate(row_placements):
        ax = axes[row_idx]

        data_by_col = {}
        for s, n, p, m in categories:
            if p != placement:
                continue
            ci = col_idx[(s, n, m)]
            data_by_col[ci] = np.array(slowdowns[(s, n, p, m)])

        if data_by_col:
            positions = sorted(data_by_col.keys())
            box_data = [data_by_col[p] for p in positions]
            box_colors = [
                col_keys[p][0].color() for p in positions
            ]

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

            for ci in positions:
                vals = data_by_col[ci]
                above = vals[vals > Y_CLIP]
                if len(above) > 0:
                    pct = 100.0 * len(above) / len(vals)
                    max_val = float(above.max())
                    pct_str = f"{pct:.2f}%" if pct < 1 else f"{pct:.0f}%"
                    ax.text(
                        ci, Y_CLIP * 1.005, f"{pct_str}\n({max_val:.0f}x)",
                        ha="center", va="bottom", fontsize=7.5,
                        color="#d62728", fontweight="bold", zorder=5,
                        clip_on=False,
                    )

        ax.axhline(1.0, color="black", ls="--", lw=0.6, alpha=0.7)
        display_name = placement.short()
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
        gpus = n * GPUS_PER_NODE_MAP
        ds = s.short()
        col_labels.append(f"{ds}\n{m or 'unknown'}\n{format_gpus(gpus)}")
    ax_bot.set_xticks(x_positions)
    ax_bot.set_xticklabels(col_labels, fontsize=8.5, ha="center")

    fig.text(0.01, 0.5, "Congestion Impact (σ)", va="center",
             rotation="vertical", fontsize=13)
    _make_legend(axes[0], [k[0] for k in col_keys])

    fig.subplots_adjust(hspace=0.12, left=0.14, right=0.97, top=0.95, bottom=0.10)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {output_path}")


# ============================================================================
#  CLI entry point
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Plot congestion impact of concurrent vs isolated baselines."
    )
    ap.add_argument(
        "--systems", nargs="+", default=SYSTEMS,
        help=f"Systems to process (default: {SYSTEMS}",
    )
    ap.add_argument(
        "-o", "--output-dir", default="plots/slowdown_all",
        help="Output directory for plots (default: ./plots/slowdown_all)",
    )
    args = ap.parse_args()

    # baselines = parse_baselines(args.systems)
    # baselines_dict = build_baselines_dict(baselines)
    # print_baseline_table(baselines_dict)
    
    # concurrent = parse_concurrent(args.systems)
    # print('Loaded concurrent runs:')
    # for s, c in concurrent.items():
    #     print(f"{s} -> {len(c)}")

    # print("Computing slowdowns ...")
    # compute_slowdowns(baselines_dict, concurrent)
    
    # all_concurrent: List[ConcurrentRun] = []
    # for c in concurrent.values():
    #     all_concurrent.extend(c)

    # all_concurrent_slowdowns = defaultdict(list)
    # for c in all_concurrent:
    #     for key, s in c.slowdowns.items():
    #         all_concurrent_slowdowns[key].extend(s)
            
    # for system, sys_concurrent in concurrent.items():
    #     slowdowns: Dict[RunKey, List[float]] = defaultdict(list)
    #     for c in sys_concurrent:
    #         for key, s in c.slowdowns.items():
    #             slowdowns[key].extend(s)
                
    #     print(f'System: {system}')
    #     print(f"{'Category':50s} {'Runs':>5s}  {'Mean':>6s}  {'Median':>6s}  {'%>1':>5s}")
    #     print("-" * 82)
    #     for cat in sorted(slowdowns, key=_sort_run):
    #         v = np.array(slowdowns[cat])
    #         lbl = f"{STRATEGY_DISPLAY[cat.strategy]} / {format_gpus(cat.gpus)} / {cat.placement_class} / {cat.model}"
    #         print(
    #             f"{lbl:50s} {len(v):5d}  {v.mean():6.3f}  {np.median(v):6.3f}  "
    #             f"{100*np.mean(v>1):5.1f}%"
    #         )
    #     print()

    #     # --- generate plots ---
    #     os.makedirs(args.output_dir, exist_ok=True)
    #     base_path = os.path.join(args.output_dir, f"slowdown_{system}.png")

    #     plot_slowdown(slowdowns, system, base_path)
    #     plot_slowdown_violin(slowdowns, system, base_path.replace(".png", "_violin.png"))

    #     # Determine a y-clip that keeps all box upper-whiskers visible.
    #     max_whisker = 1.0
    #     for v in slowdowns.values():
    #         arr = np.array(v)
    #         q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    #         whisker_hi = min(float(arr.max()), q3 + 1.5 * (q3 - q1))
    #         max_whisker = max(max_whisker, whisker_hi)
    #     boxplot_y_clip = max(1.2, round(max_whisker * 1.1, 1))

    #     plot_slowdown_boxplot(slowdowns, system, base_path.replace(".png", "_boxplot.png"),
    #                         y_clip=boxplot_y_clip)
    #     plot_slowdown_strip(slowdowns, system, base_path.replace(".png", "_strip.png"))
    #     plot_slowdown_boxplot_stacked(slowdowns,
    #                                 base_path.replace(".png", "_boxplot_stacked.png"))


if __name__ == "__main__":
    main()