#!/usr/bin/env python3
"""
plot_baselines.py — Scaling plots for DLNetBench baselines.

Replaces the old parquet-based workflow: data is read directly from raw
ccutils stdout files and SbatchMan metadata via parse_results.parse_baselines.

Produced plot sets (one per system, saved under --output-dir):
  <output_dir>/
      <sys>_<strategy>_all_scaling[_aggr].png
      <sys>_global_all_scaling[_aggr].png
      <sys>_comm_pct_table.tex
  <output_dir>/per_system/
      <sys>_<strategy>_on_<system>_scaling.png
  <output_dir>/cross_system/   (only when multiple systems are requested)
      <sys+strategy>_xsystem_*.png

Point labels
------------
Text annotations are drawn near each plotted point.  Use the CLI flags
--label-filter, --label-gpu, and --label-every-nth to control which labels
are shown (see --help for details).
"""

import os
import re
import glob
import sys
import yaml
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib

from data_types import PLACEMENT_ORDER, STRATEGY_ORDER, SYSTEM_NAMES_MAP, SYSTEM_ORDER, Model, Placement, Strategy

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def get_model_from_command(_cmd): return None
def get_default_model(_strategy): return "unknown"

# ============================================================================
#  Style helpers
# ============================================================================

from data_types import ensure_model, ensure_placement, ensure_strategy


# --- placement ---------------------------------------------------------------

def _placement_linestyles(placements: List[Union[str, Placement]]) -> Dict[str, str]:
    ps = {ensure_placement(p) for p in placements}
    return {str(p): p.linestyle() for p in PLACEMENT_ORDER if p in ps}

def _placement_markers(placements: List[Union[str, Placement]]) -> Dict[str, str]:
    ps = {ensure_placement(p) for p in placements}
    return {str(p): p.marker() for p in PLACEMENT_ORDER if p in ps}


# --- strategy ----------------------------------------------------------------

def _strategy_linestyles(strategies: List[Union[str, Strategy]]) -> Dict[str, str]:
    ss = {ensure_strategy(s) for s in strategies}
    return {str(s): s.linestyle() for s in STRATEGY_ORDER if s in ss}

def _strategy_markers(strategies: List[Union[str, Strategy]]) -> Dict[str, str]:
    ss = {ensure_strategy(s) for s in strategies}
    return {str(s): s.marker() for s in STRATEGY_ORDER if s in ss}

def _strategy_colors(strategies: List[Union[str, Strategy]]) -> Dict[str, str]:
    ss = {ensure_strategy(s) for s in strategies}
    return {str(s): s.color() for s in STRATEGY_ORDER if s in ss}

def _strategy_styles(strategies: List[Union[str, Strategy]]) -> Dict[str, dict]:
    ss = {ensure_strategy(s) for s in strategies}
    return {
        str(s): {
            "color": s.color(),
            "marker": s.marker(),
            "linestyle": s.linestyle(),
        }
        for s in STRATEGY_ORDER if s in ss
    }


# --- model -------------------------------------------------------------------

def _model_markers(models: List[Union[str, Model]]) -> Dict[str, str]:
    ms = {ensure_model(m) for m in models}
    return {str(m): m.marker() for m in sorted(ms, key=lambda x: x.value)}

def _model_colors(models: List[Union[str, Model]]) -> Dict[str, str]:
    ms = {ensure_model(m) for m in models}
    return {str(m): m.color() for m in sorted(ms, key=lambda x: x.value)}

def _model_linestyles(models: List[Union[str, Model]]) -> Dict[str, str]:
    styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]
    ms = sorted({ensure_model(m) for m in models}, key=lambda x: x.value)
    return {str(m): styles[i % len(styles)] for i, m in enumerate(ms)}

def _system_linestyles(systems: List[str]) -> Dict[str, str]:
    styles = ['-', '--', '-.', ':']
    return {c: styles[i % len(styles)] for i, c in enumerate(sorted(set(systems)))}

def _system_colors(systems: List[str]) -> Dict[str, str]:
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return {c: palette[i % len(palette)] for i, c in enumerate(sorted(set(systems)))}

def _lighten_color(color, amount: float):
    import matplotlib.colors as mc, colorsys
    try:
        c = mc.cnames.get(color, color)
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], min(1.0, 1 - amount * (1 - c[1])), c[2])
    except Exception:
        return color

def _system_placement_colors(
    system: str, placements: List[str], base_color: str
) -> Dict[str, tuple]:
    import matplotlib.colors as mc, colorsys
    lightness_levels = [0.35, 0.50, 0.65, 0.75, 0.45, 0.60]
    c = mc.to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(*c)
    return {
        tag: colorsys.hls_to_rgb(h, lightness_levels[i % len(lightness_levels)], s)
        for i, tag in enumerate(sorted(placements))
    }


import itertools
import math

# ─── Overlap-aware annotation helpers ────────────────────────────────────────

# class OverlapHandler:
#     """Tracks bounding boxes of placed labels to prevent visual overlap."""

#     def __init__(self, padding: float = 2.0):
#         self.drawn_boxes: list = []
#         self.padding = padding

#     def _is_valid_box(self, box) -> bool:
#         return not (box.width <= 1 or box.height <= 1)

#     def _pad_box(self, box):
#         f = 1.0 + self.padding / 100.0
#         return box.expanded(f, f)

#     def overlaps(self, new_box) -> bool:
#         if not self._is_valid_box(new_box):
#             return False
#         padded = self._pad_box(new_box)
#         return any(padded.overlaps(b) for b in self.drawn_boxes)

#     def add(self, box) -> None:
#         if self._is_valid_box(box):
#             self.drawn_boxes.append(box)

#     def clear(self) -> None:
#         self.drawn_boxes.clear()


# def _get_renderer(fig):
#     """Ensure transforms are initialized, return renderer. Called ONCE per figure."""
#     fig.canvas.draw()  # must be called at least once so transforms are valid
#     return fig.canvas.get_renderer()


# def _candidate_offsets_pts(
#     y_steps=(12, -16, 22, -26, 32, -36, 44, -48),
#     x_steps=(0, 12, -12, 22, -22, 32, -32),
# ) -> list[tuple[int, int]]:
#     """
#     2-D candidates in (x_pts, y_pts) offset-points space,
#     sorted by distance from ideal position (0, +12).
#     """
#     import itertools, math
#     pairs = list(itertools.product(x_steps, y_steps))
#     pairs.sort(key=lambda xy: math.hypot(xy[0], xy[1] - 12))
#     return pairs

# _OFFSET_CANDIDATES = _candidate_offsets_pts()


# class OverlapHandler:
#     """
#     Tracks bounding boxes in DISPLAY (pixel) coordinates.
#     padding is in pixels, not percent — much more predictable.
#     """
#     def __init__(self, padding_px: float = 3.0):
#         self.drawn_boxes: list = []
#         self.padding_px = padding_px

#     def _is_valid(self, box) -> bool:
#         return box is not None and box.width > 1 and box.height > 1

#     def overlaps(self, new_box) -> bool:
#         if not self._is_valid(new_box):
#             return False
#         # Expand by padding_px on all sides
#         p = self.padding_px
#         expanded = new_box.expanded(
#             (new_box.width  + 2 * p) / new_box.width,
#             (new_box.height + 2 * p) / new_box.height,
#         )
#         return any(expanded.overlaps(b) for b in self.drawn_boxes)

#     def add(self, box) -> None:
#         if self._is_valid(box):
#             self.drawn_boxes.append(box)

#     def clear(self) -> None:
#         self.drawn_boxes.clear()


# def _should_show_label(label: str, label_filters) -> bool:
#     if not label_filters:
#         return True
#     for f in label_filters:
#         try:
#             if re.search(f, label, re.IGNORECASE):
#                 return True
#         except re.error:
#             if f.lower() in label.lower():
#                 return True
#     return False


# def annotate_points(
#     ax,
#     x_vals,
#     y_vals,
#     label: str,
#     color,
#     plot_type: str,
#     label_filters=None,
#     label_gpus=None,
#     label_every_nth: int = 1,
#     series_index: int = 0,
#     fontsize: int = 8,
#     overlap_handler: Optional[OverlapHandler] = None,
#     renderer=None,
# ):
#     if not _should_show_label(label, label_filters):
#         return

#     x_arr = np.asarray(x_vals)
#     y_arr = np.asarray(y_vals)

#     for i, (xv, yv) in enumerate(zip(x_arr, y_arr)):
#         if label_gpus is not None:
#             if int(xv) not in label_gpus:
#                 continue
#         elif i % label_every_nth != 0:
#             continue

#         if plot_type == "efficiency":
#             val_str = f"{int(yv * 100)}%"
#         elif yv >= 1e6:
#             val_str = f"{yv / 1e6:.1f}M"
#         elif yv >= 1e3:
#             val_str = f"{yv / 1e3:.1f}K"
#         else:
#             val_str = f"{yv:.0f}"

#         for x_off, y_off in _OFFSET_CANDIDATES:
#             ann = ax.annotate(
#                 val_str,
#                 xy=(xv, yv),
#                 xytext=(x_off, y_off),
#                 textcoords="offset points",
#                 fontsize=fontsize,
#                 color=color,
#                 ha="center",
#                 va="bottom" if y_off > 0 else "top",
#                 arrowprops=dict(arrowstyle="-", color=color, alpha=0.4, lw=0.8),
#                 bbox=dict(
#                     boxstyle="round,pad=0.15",
#                     fc="white",
#                     ec=color,
#                     alpha=0.7,
#                     lw=0.6,
#                 ),
#                 annotation_clip=False,  # ← NEVER clip; we guard with overlap_handler
#             )

#             if overlap_handler is None or renderer is None:
#                 break  # no collision detection, just place and move on

#             # Draw just this one annotation into the renderer so its
#             # transform is current, then immediately read its pixel bbox.
#             # This is cheap: we're drawing a single Text+BboxPatch, not
#             # the whole figure.
#             ann.draw(renderer)
#             bbox = ann.get_window_extent(renderer=renderer)

#             if bbox.width <= 1 or bbox.height <= 1:
#                 # bbox not ready (shouldn't happen after ann.draw())
#                 ann.remove()
#                 continue

#             if overlap_handler.overlaps(bbox):
#                 ann.remove()
#             else:
#                 overlap_handler.add(bbox)
#                 break  # placed successfully
#         # candidates exhausted → annotation simply omitted (already removed)

import re
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List


class OverlapHandler:
    """
    Tracks display-space bounding boxes of already-placed annotations
    so subsequent labels can be nudged to a free spot.
    """

    def __init__(self, padding_px: float = 2.0):
        self.padding_px = padding_px
        self._placed: list[tuple[float, float, float, float]] = []

    def _expand(self, box: tuple) -> tuple:
        p = self.padding_px
        x0, y0, x1, y1 = box
        return (x0 - p, y0 - p, x1 + p, y1 + p)

    def overlaps_any(self, box: tuple) -> bool:
        x0, y0, x1, y1 = self._expand(box)
        for bx0, by0, bx1, by1 in self._placed:
            if x0 < bx1 and x1 > bx0 and y0 < by1 and y1 > by0:
                return True
        return False

    def register(self, box: tuple):
        self._placed.append(box)


def _label_text(y_val: float, plot_type: str) -> str:
    """Format the annotation string for a single point."""
    if plot_type == "efficiency":
        return f"{y_val * 100:.0f}%"
    v = float(y_val)
    if v < 100:
        return str(int(v))
    if v < 1_000:
        return f"{v / 1_000:.1f}K"
    if v < 1_000_000:
        return f"{int(v / 1_000)}K"
    if v < 1_000_000_000:
        return f"{int(v / 1_000_000)}M"
    return f"{int(v / 1_000_000_000)}B"

STRATEGY_LABEL_OFFSETS: dict[Strategy, int] = {
    Strategy.DP:           2,  
    Strategy.FSDP:         40,   
    Strategy.DP_PP:       -1,   
    Strategy.DP_PP_TP:     2,   
    Strategy.DP_PP_EXPERT:-4,   
}

def annotate_points(
    ax: plt.Axes,
    x_vals,
    y_vals,
    *,
    label: str,
    color: str,
    plot_type: str,
    strategy: Optional[str] = None,
    label_filters: Optional[List[str]],
    label_gpus: Optional[List[int]],
    label_every_nth: int,
    series_index: int,
    fontsize: int,
    overlap_handler: OverlapHandler,
    renderer,
) -> None:
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)

    # ── series-level filter ───────────────────────────────────────────────
    if label_filters is not None:
        if not label_filters:
            return
        if not any(re.search(pat, label) for pat in label_filters):
            return

    # ── determine starting offset for this strategy ───────────────────────
    try:
        start_off = STRATEGY_LABEL_OFFSETS[Strategy(strategy)] if strategy else 0
    except ValueError:
        start_off = 0   # unknown strategy → fall back to on-marker

    MARKER_PX = 5
    MAX_R     = 10
    STEP      =  6

    def _offsets(start: int):
        """
        Yield offsets beginning at `start`, then alternating outward.
        If start > 0 we prefer going further up first; if start < 0, further down.
        If start == 0 we alternate symmetrically.
        """
        yield start
        sign = 1 if start >= 0 else -1
        for step in range(STEP, MAX_R + 1, STEP):
            yield start + sign * step          # preferred direction first
            yield start - sign * step          # opposite direction second

    # ── iterate over points ───────────────────────────────────────────────
    for pt_idx, (xv, yv) in enumerate(zip(x_arr, y_arr)):

        if label_gpus is not None and int(xv) not in label_gpus:
            continue

        if (pt_idx + series_index) % max(1, label_every_nth) != 0:
            continue

        val_str = _label_text(yv, plot_type)

        # Register the marker dot as occupied
        disp_pt = ax.transData.transform((xv, yv))
        overlap_handler.register((
            disp_pt[0] - MARKER_PX, disp_pt[1] - MARKER_PX,
            disp_pt[0] + MARKER_PX, disp_pt[1] + MARKER_PX,
        ))

        placed = False

        for y_off in _offsets(start_off):
            arrow = None if y_off == 0 else dict(
                arrowstyle="-", color=color, alpha=0.4, lw=0.8
            )
            va = "center" if y_off == 0 else ("bottom" if y_off > 0 else "top")

            ann = ax.annotate(
                val_str,
                xy=(xv, yv),
                xytext=(0, y_off),
                textcoords="offset points",
                fontsize=fontsize,
                color=color,
                ha="center",
                va=va,
                arrowprops=arrow,
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    fc="white",
                    ec=color,
                    alpha=0.7,
                    lw=0.6,
                ),
                annotation_clip=False,
            )

            try:
                bbox = ann.get_window_extent(renderer=renderer)
            except Exception:
                ann.remove()
                continue

            box = (bbox.x0, bbox.y0, bbox.x1, bbox.y1)

            if not overlap_handler.overlaps_any(box):
                overlap_handler.register(box)
                placed = True
                break
            else:
                ann.remove()

        # fall-back
        if not placed:
            fallback_off = start_off + (MAX_R if start_off >= 0 else -MAX_R)
            ann = ax.annotate(
                val_str,
                xy=(xv, yv),
                xytext=(0, fallback_off),
                textcoords="offset points",
                fontsize=fontsize,
                color=color,
                ha="center",
                va="bottom" if fallback_off >= 0 else "top",
                arrowprops=dict(arrowstyle="-", color=color, alpha=0.4, lw=0.8),
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    fc="white",
                    ec=color,
                    alpha=0.7,
                    lw=0.6,
                ),
                annotation_clip=False,
            )
            try:
                bbox = ann.get_window_extent(renderer=renderer)
                overlap_handler.register((bbox.x0, bbox.y0, bbox.x1, bbox.y1))
            except Exception:
                pass
        

# ============================================================================
#  Data loading from raw files
# ============================================================================

def _apply_warmup_skip(values: list, skip_first: int = 1) -> list:
    """Drop warm-up iterations from a per-iteration list."""
    skip = 3 if len(values) >= 6 else skip_first
    usable = values[skip:] if len(values) > skip else values[-1:]
    return usable


_BARRIER_KEY_CANDIDATES = [
    "barrier",
    "barrier_time",
    "dp_comm_time",
    "commtime",
    "comm_time",
]


def _get_barrier(rank_record: dict) -> Optional[list]:
    extra = rank_record.get("extra", {})
    for key in _BARRIER_KEY_CANDIDATES:
        if key in rank_record and rank_record[key] is not None:
            v = rank_record[key]
            return v if isinstance(v, list) else [v]
        if key in extra and extra[key] is not None:
            v = extra[key]
            return v if isinstance(v, list) else [v]
    return None


def _build_baseline_records(
    backup_dir: str,
    system_name: str,
    skip_first: int = 1,
) -> List[dict]:
    records = []
    sbm_system_name = system_name
    if system_name == 'dgxA100':
        sbm_system_name = 'baldo'

    meta_pattern = os.path.join(
        backup_dir, "SbatchMan", "experiments", sbm_system_name,
        "*", "baseline_*", "*", "metadata.yaml",
    )

    for meta_path in glob.glob(meta_pattern):
        with open(meta_path) as f:
            meta = yaml.safe_load(f)

        if meta.get("status") != "COMPLETED":
            continue

        v = meta.get("variables", {})
        if system_name == "nvl72" and v.get("gpu_model", "").upper() != "GB300":
            continue

        strategy  = v.get("strategy")
        nodes     = v.get("nodes")
        gpus      = v.get("gpus", nodes * GPUS_PER_NODE_MAP[system_name])
        placement = v.get("placement_class") or v.get("placement")
        if not all([strategy, gpus, placement]):
            continue

        stdout = os.path.join(os.path.dirname(meta_path), "stdout.log")
        if not os.path.isfile(stdout):
            continue

        model = v.get("model")
        if not model:
            model = get_model_from_command(meta.get("command", ""))
        if not model:
            model = parse_model_from_stdout(stdout)
        if not model:
            model = get_default_model(strategy)
        if not model:
            model = "unknown"

        rank_records = parse_stdout_throughputs(stdout)
        if not rank_records:
            continue

        def _rank_median(rr):
            tp = rr["throughputs"]
            skip = 3 if len(tp) >= 6 else skip_first
            usable = tp[skip:] if len(tp) > skip else tp[-1:]
            return float(np.median(usable)) if usable else float("inf")

        bottleneck_rank = min(rank_records, key=lambda rid: _rank_median(rank_records[rid]))
        rr = rank_records[bottleneck_rank]

        tp_raw  = rr["throughputs"]
        it_raw  = rr.get("iteration_times") or []
        bar_raw = _get_barrier(rr) or []

        skip = 3 if len(tp_raw) >= 6 else skip_first
        tp_usable  = tp_raw[skip:]  if len(tp_raw)  > skip else tp_raw[-1:]
        it_usable  = it_raw[skip:]  if len(it_raw)  > skip else (it_raw[-1:] if it_raw else [])
        bar_usable = bar_raw[skip:] if len(bar_raw) > skip else (bar_raw[-1:] if bar_raw else [])

        gpu_model = rr.get("gpu_model") or v.get("gpu_model", "unknown")

        records.append({
            "system":          system_name,
            "gpu_model":       gpu_model,
            "strategy":        strategy,
            "placement":       placement,
            "model":           model,
            "gpus":            int(gpus),
            "nodes":           int(nodes),
            "throughputs":     tp_usable,
            "iteration_times": it_usable if it_usable else None,
            "barrier_times":   bar_usable if bar_usable else None,
            "memory_allocated": rr.get("memory_allocated"),
        })

    return records


# ============================================================================
#  Summary DataFrame construction
# ============================================================================

def build_summary(records: List[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        tp  = np.asarray(r["throughputs"],     dtype=float)
        it  = np.asarray(r["iteration_times"], dtype=float) if r["iteration_times"] else None
        bar = np.asarray(r["barrier_times"],   dtype=float) if r["barrier_times"]   else None

        throughput_mean = float(tp.mean()) if len(tp) else float("nan")
        throughput_std  = float(tp.std())  if len(tp) else float("nan")

        runtime_mean = float(it.mean())  if it  is not None and len(it)  else float("nan")
        barrier_mean = float(bar.mean()) if bar is not None and len(bar) else float("nan")

        if it is not None and bar is not None and len(it) and len(bar):
            n = min(len(it), len(bar))
            compute = it[:n] - bar[:n]
            compute_mean = float(compute.mean())
            comm_pct     = float((bar[:n] / it[:n] * 100).mean())
            compute_pct  = float((compute   / it[:n] * 100).mean())
        else:
            compute_mean = float("nan")
            comm_pct     = float("nan")
            compute_pct  = float("nan")

        rows.append({
            "system":          r["system"],
            "gpu_model":       r["gpu_model"],
            "strategy":        r["strategy"],
            "placement":       r["placement"],
            "model":           r["model"],
            "gpus":            r["gpus"],
            "nodes":           r["nodes"],
            "throughput_mean": throughput_mean,
            "throughput_std":  throughput_std,
            "runtime_mean":    runtime_mean,
            "barrier_mean":    barrier_mean,
            "compute_mean":    compute_mean,
            "comm_pct":        comm_pct,
            "compute_pct":     compute_pct,
            "memory_allocated": r.get("memory_allocated"),
        })

    return pd.DataFrame(rows)


# ============================================================================
#  Aggregation helper
# ============================================================================

def aggregate_placements(summary: pd.DataFrame, agg_type: str = "geomean") -> pd.DataFrame:
    """
    Collapse all placements of the same
    (strategy, model, system, gpus, gpu_model) into a single row.
    """
    agg = (
        summary
        .groupby(
            ["strategy", "model", "system", "gpus"],
            as_index=False,
        )
        .agg(
            system           =("system",            "first"),
            nodes            =("gpus",              "first"),
            throughput_median=("throughput_median",  agg_type),
            throughput_std   =("throughput_std",     agg_type),
            runtime_mean     =("runtime_mean",       agg_type),
            barrier_mean     =("barrier_mean",       agg_type),
            compute_mean     =("compute_mean",       agg_type),
            comm_pct         =("comm_pct",           agg_type),
            compute_pct      =("compute_pct",        agg_type),
        )
    )
    agg["strategy"]  = agg["strategy"]
    agg["placement"] = ""
    return agg


# ============================================================================
#  Plot helpers
# ============================================================================

def _save_or_show(fig, output_file: Optional[str], label: str = "Plot"):
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output_file).with_suffix('.pdf'), dpi=300, bbox_inches="tight")
        print(f"  {label} saved to: {output_file}")
        plt.close(fig)
    else:
        plt.show()


def _make_label(
    strategy: str,
    model: str,
    placement: str,
    system: str,
    include_system: bool = True,
) -> str:
    """Human-readable series label including model name."""
    place = ensure_placement(placement).display(new_line=False, short=True)
    system_disp = system
    strategy = ensure_strategy(strategy)
    model = ensure_model(model)
    parts = [strategy.short(), model.short()]
    parts.append(place)
    if include_system:
        parts.append(system_disp)
    return " - ".join(parts)


def _dedup_gpus(grp: pd.DataFrame, label: str, metric: str) -> pd.DataFrame:
    """Keep the highest-throughput row when the same GPU count appears twice."""
    if grp["gpus"].duplicated().any():
        dup_counts = grp["gpus"].value_counts()
        for gpu, count in dup_counts[dup_counts > 1].items():
            best = grp[grp["gpus"] == gpu][f"throughput_{metric}"].max()
            print(f"[WARN] Duplicate GPU entries for {label}: gpus={gpu} "
                  f"({count} rows) → keeping max {best:.3f}")
        grp = grp.loc[grp.groupby("gpus")[f"throughput_{metric}"].idxmax()]
    return grp.sort_values("gpus")


# ============================================================================
#  Plot 1: Scaling
# ============================================================================

def plot_scaling(
    summary: pd.DataFrame,
    metric: str,
    strategies: Optional[List[str]] = None,
    systems: Optional[List[str]] = None,
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 7),
    title: str = "Scaling: Throughput vs Number of GPUs",
    show_ideal: bool = True,
    plot_efficiency: bool = False,
    label_filters: Optional[List[str]] = None,
    label_gpus: Optional[List[int]] = None,
    label_every_nth: int = 1,
    label_fontsize: int = 8,
):
    df = summary.copy()
    if strategies:
        df = df[df["strategy"].isin(strategies)]
    if systems:
        df = df[df["system"].isin(systems)]

    fig, ax = plt.subplots(figsize=figsize)
    if plot_efficiency:
        fig_eff, ax_eff = plt.subplots(figsize=figsize)

    clust_color  = _system_colors(df["system"].unique())
    place_ls     = _placement_linestyles(df["placement"].unique())
    base_markers = _strategy_markers(df["strategy"].unique())

    min_eff   = 1.0
    series_idx = 0

    for (system, strategy, model), grp in df.groupby(
        ["system", "strategy", "model"]
    ):
        strategy  = grp["strategy"].iloc[0]
        placement = grp["placement"].iloc[0]
        gpu_model = grp["gpu_model"].iloc[0]

        color  = clust_color[system]
        ls     = place_ls.get(placement, "-")
        marker = base_markers.get(strategy, "o")
        label  = _make_label(strategy, model, placement, system, gpu_model)

        grp = _dedup_gpus(grp, label, metric)

        ax.errorbar(
            grp["gpus"], grp[f"throughput_{metric}"],
            yerr=grp["throughput_std"],
            label=label, color=color, marker=marker, linestyle=ls,
            linewidth=1, markersize=5, capsize=2,
        )

        annotate_points(
            ax, grp["gpus"], grp[f"throughput_{metric}"],
            label=label, color=color, plot_type="scaling",
            label_filters=label_filters, label_gpus=label_gpus,
            label_every_nth=label_every_nth, series_index=series_idx,
            fontsize=label_fontsize,
        )

        if show_ideal:
            base_row = grp.iloc[0]
            gpus_range = np.array(sorted(df["gpus"].unique()))
            gpus_range = gpus_range[gpus_range >= base_row["gpus"]]
            ideal = base_row[f"throughput_{metric}"] * (gpus_range / base_row["gpus"])
            ax.plot(gpus_range, ideal, color=color, linestyle=":", linewidth=1, alpha=0.4)

        if plot_efficiency:
            g0  = grp.iloc[0]["gpus"]
            T0  = grp.iloc[0][f"throughput_{metric}"]
            eff = (grp[f"throughput_{metric}"] / T0) * (g0 / grp["gpus"])
            min_eff = min(min_eff, eff.min())
            ax_eff.plot(grp["gpus"], eff, label=label,
                        color=color, marker=marker, linestyle=ls,
                        linewidth=1, markersize=5)
            annotate_points(
                ax_eff, grp["gpus"], eff.values,
                label=label, color=color, plot_type="efficiency",
                label_filters=label_filters, label_gpus=label_gpus,
                label_every_nth=label_every_nth, series_index=series_idx,
                fontsize=label_fontsize,
            )

        series_idx += 1

    if show_ideal:
        ax.plot([], [], color="gray", linestyle=":", linewidth=1,
                alpha=0.7, label="Ideal scaling")

    all_gpus = sorted(df["gpus"].unique())
    for _ax in [ax]:
        _ax.set_xscale("log", base=2)
        _ax.set_yscale("log", base=2)
        _ax.set_xticks(all_gpus)
        _ax.set_xticklabels([str(g) for g in all_gpus])
        _ax.set_xlabel("Number of GPUs", fontsize=12)
        _ax.set_ylabel("Throughput (samples/s)", fontsize=12)
        _ax.set_title(title, fontsize=14, fontweight="bold")
        _ax.legend(fontsize=9, loc="center left",
                   bbox_to_anchor=(1.01, 0.5), borderaxespad=0)
        _ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 0.78, 1])
    _save_or_show(fig, output_file, "Scaling plot")

    if plot_efficiency:
        ax_eff.axhline(1.0, linestyle=":", linewidth=1, alpha=0.7,
                       color="gray", label="Ideal efficiency")
        ax_eff.set_xscale("log", base=2)
        ax_eff.set_xticks(all_gpus)
        ax_eff.set_xticklabels([str(g) for g in all_gpus])
        ax_eff.set_yticks(
            list(np.linspace(min_eff, 1.0, 10)),
            labels=[f"{int(e*100)}%" for e in np.linspace(min_eff, 1.0, 10)],
        )
        ax_eff.set_xlabel("Number of GPUs", fontsize=12)
        ax_eff.set_ylabel("Parallel Efficiency", fontsize=12)
        ax_eff.set_title("Scaling Efficiency", fontsize=14, fontweight="bold")
        ax_eff.legend(fontsize=8, loc="center left",
                      bbox_to_anchor=(1.01, 0.5), borderaxespad=0)
        ax_eff.grid(True, alpha=0.3)
        fig_eff.tight_layout(rect=[0, 0, 0.78, 1])

        eff_file = output_file.replace(".png", "_efficiency.png") if output_file else None
        _save_or_show(fig_eff, eff_file, "Efficiency plot")
        return (fig, ax), (fig_eff, ax_eff)

    return fig, ax, (None, None)


# ============================================================================
#  Comm-pct LaTeX table
# ============================================================================

def generate_comm_pct_table(
    summary: pd.DataFrame,
    output_file: Optional[str] = None,
    gpus: Optional[int] = None,
) -> str:
    df = summary.copy()
    if gpus is not None:
        df = df[df["gpus"] == gpus]

    df["row_key"] = df["strategy"] + " / " + df["model"]

    systems  = sorted(df["system"].unique())
    row_keys = sorted(df["row_key"].unique())

    def _cell(sub: pd.DataFrame) -> str:
        per_placement = sub.groupby("placement")["comm_pct"].mean().dropna()
        if per_placement.empty:
            return "---"
        lo, hi = per_placement.min(), per_placement.max()
        if len(per_placement) == 1 or abs(hi - lo) < 0.05:
            return f"{lo:.1f}\\%"
        return f"{lo:.1f}--{hi:.1f}\\%"

    cells: Dict[Tuple[str, str], str] = {}
    for (row_key, system), grp in df.groupby(["row_key", "system"]):
        cells[(row_key, system)] = _cell(grp)

    def _esc(s: str) -> str:
        return s.replace("_", r"\_")

    col_spec    = "ll" + "c" * len(systems)
    header_cols = " & ".join(
        [r"\textbf{Strategy / Model}"] +
        [r"\textbf{" + _esc(c) + "}" for c in systems]
    )
    rows_tex = [
        _esc(rk) + " & " + " & ".join(cells.get((rk, c), "---") for c in systems) + r" \\"
        for rk in row_keys
    ]
    gpu_note = f"GPU count: {gpus}" if gpus is not None else "all GPU counts"
    caption  = (
        f"Communication time (\\%) by strategy, model and system ({gpu_note}). "
        f"Cells show the min--max range across placements."
    )
    lines = (
        [r"\begin{table}[ht]", r"  \centering",
         r"  \caption{" + caption + "}", r"  \label{tab:comm_pct}",
         r"  \begin{tabular}{" + col_spec + "}", r"    \toprule",
         f"    {header_cols} \\\\", r"    \midrule"]
        + [f"    {r}" for r in rows_tex]
        + [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    )
    tex = "\n".join(lines) + "\n"

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text(tex)
        print(f"  Comm-pct table saved to: {output_file}")

    return tex


# ============================================================================
#  Per-system orchestration
# ============================================================================

def process_system(
    system_name: str,
    backup_dir: str,
    skip_first: int,
    output_dir: str,
    prefix: str = "",
    no_ideal: bool = False,
    aggregate_placements_flag: bool = False,
    only_all: bool = False,
    table_gpus: Optional[int] = None,
    strategyegies_filter: Optional[List[str]] = None,
    label_filters: Optional[List[str]] = None,
    label_gpus: Optional[List[int]] = None,
    label_every_nth: int = 1,
    label_fontsize: int = 8,
):
    pfx = f"{prefix}_" if prefix else ""
    output_dir = Path(output_dir)
    per_system_dir   = output_dir / "per_system"
    cross_system_dir = output_dir / "cross_system"

    print(f"\n{'='*60}")
    print(f"  System     : {system_name}")
    print(f"  Backup dir : {backup_dir}")
    print(f"  Skip first : {skip_first} iterations")
    print(f"{'='*60}\n")

    print("Loading baseline records ...")
    records = _build_baseline_records(backup_dir, system_name, skip_first)
    print(f"  {len(records)} baseline jobs loaded\n")
    if not records:
        print("  Nothing to plot.")
        return

    summary = build_summary(records)
    print(f"Summary ({len(summary)} jobs):")
    print(summary[["system", "gpu_model", "strategy", "placement",
                   "gpus", "throughput_mean", "comm_pct", "compute_pct"]
                  ].to_string(index=False))
    print()

    systems       = sorted(summary["system"].unique())
    strategyegies = sorted(summary["strategy"].unique())
    if strategyegies_filter:
        strategyegies = [s for s in strategyegies if s in strategyegies_filter]
    all_strategies = sorted(summary["strategy"].unique())
    all_models     = sorted(summary["model"].unique())

    # Shared kwargs for point labels
    lbl_kw = dict(
        label_filters=label_filters,
        label_gpus=label_gpus,
        label_every_nth=label_every_nth,
        label_fontsize=label_fontsize,
    )

    # 1) Per-system, per-base-strategy, per-model: compare placements
    if not only_all:
        print("[Per-system placement comparison]")
        for system in systems:
            for strategy in strategyegies:
                for model in all_models:
                    strats_here = summary[
                        (summary["system"]   == system) &
                        (summary["strategy"] == strategy) &
                        (summary["model"]    == model)
                    ]["strategy"].unique().tolist()
                    if not strats_here:
                        continue

                    tag = f"{pfx}{system_name}_{strategy}_{model}_on_{system}"
                    print(f"  {tag}  placements={strats_here}")

                    plot_scaling(
                        summary, strategies=strats_here, systems=[system],
                        output_file=str(per_system_dir / f"{tag}_scaling.png"),
                        title=f"Scaling — {strategy} / {model} placements on {system}",
                        show_ideal=not no_ideal,
                        **lbl_kw,
                    )

        # 2) Cross-system per strategy+placement+model
        if len(systems) > 1:
            print("\n[Cross-system comparison per strategy+placement+model]")
            for strategy in all_strategies:
                base = summary[summary["strategy"] == strategy]["strategy"].iloc[0]
                if strategyegies_filter and base not in strategyegies_filter:
                    continue
                for model in all_models:
                    clust_here = summary[
                        (summary["strategy"] == strategy) &
                        (summary["model"]    == model)
                    ]["system"].unique().tolist()
                    if not clust_here:
                        continue
                    print(f"  {strategy} / {model}  systems={clust_here}")
                    plot_scaling(
                        summary, strategies=[strategy], systems=clust_here,
                        output_file=str(cross_system_dir / f"{pfx}{system_name}_{strategy}_{model}_xsystem_scaling.png"),
                        title=f"Scaling — {strategy} / {model} across systems",
                        show_ideal=not no_ideal,
                        **lbl_kw,
                    )

    if not only_all:
        # 3) Cross-system overview per base-strategy, per-model
        print("\n[Cross-system + cross-placement overview per base strategy / model]")
        for strategy in strategyegies:
            for model in all_models:
                sub = summary[
                    (summary["strategy"] == strategy) &
                    (summary["model"]    == model)
                ]
                if sub.empty:
                    continue
                plot_summary = aggregate_placements(sub) if aggregate_placements_flag else sub
                agg_label    = " (placements averaged)" if aggregate_placements_flag else ""
                strats_here  = plot_summary["strategy"].unique().tolist()
                mode_tag     = "aggr" if aggregate_placements_flag else "all"
                print(f"  {strategy} / {model}  [{mode_tag}] strategies={strats_here}")

                plot_scaling(
                    plot_summary, strategies=strats_here,
                    output_file=str(output_dir / f"{pfx}{system_name}_{strategy}_{model}_{mode_tag}_scaling.png"),
                    title=f"Scaling — {strategy} / {model} (all systems{agg_label})",
                    show_ideal=not no_ideal,
                    plot_efficiency=True,
                    **lbl_kw,
                )

    # 4) Comm-pct LaTeX table
    print("\n[Comm-pct LaTeX table]")
    generate_comm_pct_table(
        summary,
        output_file=str(output_dir / f"{pfx}{system_name}_comm_pct_table.tex"),
        gpus=table_gpus,
    )

    return summary


# ============================================================================
#  Global plots  (all systems combined)
# ============================================================================

def plot_global(
    combined: pd.DataFrame,
    output_dir: Path,
    pfx: str = "",
    no_ideal: bool = False,
    aggregate_placements_flag: bool = False,
    label_filters: Optional[List[str]] = None,
    label_gpus: Optional[List[int]] = None,
    label_every_nth: int = 1,
    label_fontsize: int = 8,
):
    """Single-figure global overview across all systems."""
    print("\n[Global overview — all systems combined]")

    plot_summary = aggregate_placements(combined) if aggregate_placements_flag else combined
    agg_label    = " (placements averaged)" if aggregate_placements_flag else ""
    mode_tag     = "aggr" if aggregate_placements_flag else "all"
    all_strats   = plot_summary["strategy"].unique().tolist()
    systems      = sorted(combined["system"].unique())
    sys_label    = "+".join(systems)

    print(f"  [{mode_tag}] systems={systems}  strategies={sorted(all_strats)}")

    lbl_kw = dict(
        label_filters=label_filters,
        label_gpus=label_gpus,
        label_every_nth=label_every_nth,
        label_fontsize=label_fontsize,
    )

    plot_scaling(
        plot_summary, strategies=all_strats,
        output_file=str(output_dir / f"{pfx}global_{sys_label}_{mode_tag}_scaling.png"),
        title=f"Scaling — all strategies / all models ({sys_label}{agg_label})",
        show_ideal=not no_ideal,
        plot_efficiency=True,
        **lbl_kw,
    )


def plot_global_faceted(
    combined: pd.DataFrame,
    output_dir: Path,
    metric: str,
    pfx: str = "",
    no_ideal: bool = False,
    aggregate_placements_flag: bool = False,
    n_rows: int = 1,
    plot_type: str = "scaling",   # "scaling" | "efficiency"
    figsize_per_cell: Tuple[int, int] = (8, 6),
    label_filters: Optional[List[str]] = None,
    label_gpus: Optional[List[int]] = None,
    label_every_nth: int = 1,
    label_fontsize: int = 12,
):
    """
    One subplot per system, laid out in a grid of *n_rows* rows.

    Parameters
    ----------
    combined : pd.DataFrame
        Concatenated summary from all systems.
    output_dir : Path
        Directory where the figure is saved.
    metric : str
        Throughput column suffix (e.g. "mean", "median").
    pfx : str
        Filename prefix.
    no_ideal : bool
        Suppress ideal-scaling reference lines.
    aggregate_placements_flag : bool
        Average across placements before plotting each panel.
    n_rows : int
        Number of rows in the subplot grid.
    plot_type : str
        "scaling" or "efficiency".
    figsize_per_cell : (width, height)
        Size of each individual subplot cell in inches.
    label_filters : list[str] | None
        Substrings / regexes that select which series get point labels.
        None → label all series.  Pass an empty list [] to suppress all labels.
    label_gpus : list[int] | None
        Annotate only these GPU counts.  None → respect *label_every_nth*.
    label_every_nth : int
        Annotate every n-th point along each series (1 = all points).
    label_fontsize : int
        Font size used for point annotations.
    """
    import math
    from matplotlib.ticker import FuncFormatter

    sys_order = SYSTEM_ORDER
    systems = [s for s in sys_order if s in combined["system"].unique()]
    n_sys   = len(systems)
    if n_sys == 0:
        print("  [SKIP] No systems in combined summary.")
        return

    n_rows  = max(1, min(n_rows, n_sys))
    n_cols  = math.ceil(n_sys / n_rows)
    mode_tag = "aggr" if aggregate_placements_flag else "all"

    fig_w = n_cols * figsize_per_cell[0]
    fig_h = n_rows * figsize_per_cell[1]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    axes_flat = [axes[r][c] for r in range(n_rows) for c in range(n_cols)]
    for ax in axes_flat[n_sys:]:
        ax.set_visible(False)

    global_handles = []
    global_labels  = []

    def format_throughput(throughput: float, *_):
        if throughput < 100:
            return str(int(throughput))
        if throughput < 1e3:
            return f"{(throughput / 1e3):.1f}K"
        if throughput < 1e6:
            return f"{int(throughput / 1e3)}K"
        if throughput < 1e9:
            return f"{int(throughput / 1e6)}M"
        return f"{int(throughput / 1e9)}B"

    def format_gpus(gpus: int, *_):
        if gpus == 224:
            return ''
        if gpus > 512:
            return f'{int(gpus / 1e3)}K'
        return str(gpus)

    def format_efficiency(e: float, *_):
        return f'{int(e * 100.0)}'

    for idx, system_name in enumerate(systems):
        ax  = axes_flat[idx]
        sub = combined[combined["system"] == system_name].copy()

        plot_summary = aggregate_placements(sub) if aggregate_placements_flag else sub

        gpus_local    = sorted(plot_summary["gpus"].unique())
        place_ls      = _placement_linestyles(plot_summary["placement"].unique())
        model_ls      = _model_linestyles(plot_summary["model"].unique())
        place_markers = _placement_markers(plot_summary["placement"].unique())
        base_color    = _strategy_colors(plot_summary["strategy"].unique())

        min_eff    = 1.0
        
        # ── draw once so transforms/log-scale ticks are initialized ──────────
        # This is the single draw() call for the entire subplot. It is O(1)
        # per subplot, not O(n_points), so performance stays acceptable.
        fig.set_layout_engine("none")          # prevents tight_layout from firing early
        renderer = fig.canvas.get_renderer() if hasattr(fig.canvas, "get_renderer") \
            else plt.matplotlib.backends.backend_agg.FigureCanvasAgg(fig).get_renderer()
        
        handler = OverlapHandler(padding_px=1.0)
        series_idx = 0
        
        for (strategy, system, model, placement), grp in plot_summary.groupby(
            ["strategy", "system", "model", "placement"]
        ):
            color  = base_color[strategy]
            ls     = model_ls[model]
            marker = place_markers.get(placement, "o")
            label  = _make_label(strategy, model, placement, system, include_system=False)

            grp = _dedup_gpus(grp, label, metric)

            if plot_type == "scaling":
                ax.errorbar(
                    grp["gpus"], grp[f"throughput_{metric}"],
                    yerr=grp["throughput_std"],
                    label=label, color=color, marker=marker, linestyle=ls,
                    linewidth=2, markersize=10, capsize=4,
                )
                if not (system == 'jupiter' and ( \
                    (ensure_strategy(strategy) == Strategy.DP and ensure_placement(placement) != Placement.INTRA_L1_RANDOM) \
                    or (ensure_strategy(strategy) == Strategy.DP_PP_TP and ensure_placement(placement) != Placement.INTER_GROUP_RANDOM) \
                    or (ensure_strategy(strategy) == Strategy.DP_PP_EXPERT and ensure_placement(placement) != Placement.INTRA_GROUP_RANDOM) \
                    )):
                    annotate_points(
                        ax, grp["gpus"], grp[f"throughput_{metric}"],
                        label=label, color=color, plot_type="scaling", strategy=ensure_strategy(strategy),
                        label_filters=label_filters, label_gpus=label_gpus,
                        label_every_nth=label_every_nth, series_index=series_idx,
                        fontsize=label_fontsize, overlap_handler=handler, renderer=renderer,
                    )
                if not no_ideal:
                    base_row   = grp.iloc[0]
                    gpus_range = np.array([g for g in grp["gpus"].unique()
                                           if g >= base_row["gpus"]])
                    ideal = base_row[f"throughput_{metric}"] * (gpus_range / base_row["gpus"])
                    ax.plot(gpus_range, ideal, color=color,
                            linestyle=":", linewidth=2, alpha=0.35)

            else:  # efficiency
                g0  = grp.iloc[0]["gpus"]
                T0  = grp.iloc[0][f"throughput_{metric}"]

                if (
                    system_name == 'lumi'
                    and strategy in [Strategy.DP, Strategy.DP_PP, Strategy.FSDP]
                    # and (grp["gpus"] >= 8).any()
                ):
                    row_8 = grp[grp["gpus"] == 8]
                    if not row_8.empty:
                        print(f'LUMI efficiency base {grp.iloc[0][f"throughput_{metric}"]} ---> {row_8.iloc[0][f"throughput_{metric}"]}')
                        g0 = row_8.iloc[0]["gpus"]
                        T0 = row_8.iloc[0][f"throughput_{metric}"]
                        
                eff = (grp[f"throughput_{metric}"] / T0) * (g0 / grp["gpus"])
                min_eff = min(min_eff, float(eff.min()))
                ax.plot(grp["gpus"], eff, label=label,
                        color=color, marker=marker, linestyle=ls,
                        linewidth=2, markersize=10)
                # annotate_points(
                #     ax, grp["gpus"], eff.values,
                #     label=label, color=color, plot_type="efficiency",
                #     label_filters=label_filters, label_gpus=label_gpus,
                #     label_every_nth=label_every_nth, series_index=series_idx,
                #     fontsize=label_fontsize, overlap_handler=handler, renderer=renderer,
                # )

            series_idx += 1

        # Axis formatting
        if plot_type == "scaling":
            ax.set_xscale("log", base=2)
            ax.set_yscale("log", base=2)
            if idx % n_cols == 0:
                ax.set_ylabel("Throughput (samples/s)", fontsize=22)
            if not no_ideal:
                ax.plot([], [], color="gray", linestyle=":", linewidth=2,
                        alpha=0.9, label="Ideal")
            ax.yaxis.set_major_formatter(FuncFormatter(format_throughput))
            ax.yaxis.set_tick_params(labelsize=20)
        else:  # efficiency
            ax.axhline(1.0, linestyle=":", linewidth=2, alpha=0.9,
                       color="gray", label="Ideal")
            ax.set_xscale("log", base=2)
            if idx % n_cols == 0:
                ax.set_ylabel("Parallel Efficiency (%)", fontsize=22)
            ax.yaxis.set_major_formatter(FuncFormatter(format_efficiency))
            ax.yaxis.set_tick_params(labelsize=20)

        ax.set_xticks(gpus_local)
        ax.xaxis.set_major_formatter(FuncFormatter(format_gpus))
        ax.xaxis.set_tick_params(
            labelsize=19 if system_name != 'lumi' else 16,
        )
        ax.set_title(SYSTEM_NAMES_MAP.get(system_name, system_name),
                     fontsize=24, fontweight="bold")
        if plot_type != "scaling":
            ax.set_xlabel("GPUs", fontsize=20)
        ax.grid(True, alpha=0.45)

        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            if l not in global_labels:
                global_handles.append(h)
                global_labels.append(l)

    # One global legend at the bottom
    if global_labels and plot_type == "efficiency":
        legend_properties = {'weight': 'bold', 'size': 14}
        sorted_pairs = sorted(zip(global_handles, global_labels),
                              key=lambda x: x[1].split('-'))
        global_handles, global_labels = zip(*sorted_pairs)
        fig.legend(
            list(global_handles),
            list(global_labels),
            loc="lower center",
            ncol=min(9, max(1, len(global_labels))),
            fontsize=20,
            frameon=False,
            prop=legend_properties,
        )

    fig.tight_layout(rect=[0, 0.18, 1, 1.0])

    out = str(output_dir / f"{pfx}global_{mode_tag}_{metric}_{plot_type}_faceted.png")
    _save_or_show(fig, out, f"Faceted {plot_type} plot")


# ============================================================================
#  CLI
# ============================================================================

def main():
    parser = ArgumentParser(
        description="Plot DLNetBench baseline scaling results from raw stdout/yaml files."
    )
    parser.add_argument(
        "--systems", nargs="+", default=list(SYSTEMS.keys()),
        help=f"Systems to process (default: {list(SYSTEMS.keys())})",
    )
    parser.add_argument(
        "--backup-dirs", nargs="+", default=None,
        help=(
            "Raw data directories, one per system (in the same order as --systems). "
            "Overrides the built-in SYSTEMS paths."
        ),
    )
    parser.add_argument("--output-dir", default="plots/baselines",
                        help="Output directory for plots")
    parser.add_argument("--prefix", default="",
                        help="Optional filename prefix")
    parser.add_argument("--skip-first", type=int, default=1,
                        help="Warm-up iterations to skip per rank (default: 1)")
    parser.add_argument("--no-ideal", action="store_true",
                        help="Suppress ideal-scaling lines")
    parser.add_argument("--strategies", nargs="*",
                        help="Filter to specific base strategies")
    parser.add_argument("--aggregate-placements", action="store_true",
                        help="Average across placements before plotting")
    parser.add_argument("--only-all", action="store_true",
                        help="Produce only cross-system overview plots")
    parser.add_argument("--table-gpus", type=int, default=None,
                        help="GPU count to filter for the comm-pct table")
    parser.add_argument(
        "--grid-rows", type=int, default=1,
        help=(
            "Number of rows in the faceted global plot (one panel per system). "
            "n_cols is derived as ceil(n_systems / grid-rows). (default: 1)"
        ),
    )
    parser.add_argument(
        "--cell-size", type=int, nargs=2, default=[8, 6],
        metavar=("W", "H"),
        help="Width and height in inches of each panel cell in the faceted plot (default: 8 6)",
    )

    # ---- Point-label arguments ----
    label_group = parser.add_argument_group(
        "Point labels",
        "Control which data-point labels are drawn next to plotted markers.",
    )
    label_group.add_argument(
        "--label-filter", nargs="*", dest="label_filter",
        metavar="PATTERN",
        help=(
            "Show labels only for series whose name matches at least one "
            "PATTERN (case-insensitive substring or Python regex).  "
            "Omit the flag entirely to label ALL series; pass the flag with "
            "no arguments (--label-filter) to suppress ALL labels."
        ),
    )
    label_group.add_argument(
        "--label-gpu", nargs="+", type=int, dest="label_gpu",
        metavar="N",
        help=(
            "Annotate only the listed GPU counts (e.g. --label-gpu 128 256). "
            "When omitted, respects --label-every-nth instead."
        ),
    )
    label_group.add_argument(
        "--label-every-nth", type=int, default=1, dest="label_every_nth",
        metavar="N",
        help=(
            "Annotate every N-th point along each series "
            "(1 = every point [default], 2 = every other, …). "
            "Ignored when --label-gpu is given."
        ),
    )
    label_group.add_argument(
        "--label-fontsize", type=int, default=8, dest="label_fontsize",
        metavar="PT",
        help="Font size for point annotations (default: 8).",
    )

    args = parser.parse_args()

    # Interpret --label-filter:
    #   not provided at all  → None  (label everything)
    #   --label-filter       → []    (label nothing)
    #   --label-filter A B   → ["A","B"]
    label_filters: Optional[List[str]] = args.label_filter  # None or list

    output_dir  = Path(args.output_dir)
    pfx         = f"{args.prefix}_" if args.prefix else ""
    backup_dirs = args.backup_dirs or [SYSTEMS.get(s) for s in args.systems]

    # Shared label kwargs threaded through every plot call
    lbl_kw = dict(
        label_filters   = label_filters,
        label_gpus      = args.label_gpu,
        label_every_nth = args.label_every_nth,
        label_fontsize  = args.label_fontsize,
    )

    all_summaries: List[pd.DataFrame] = []

    for system_name, backup_dir in zip(args.systems, backup_dirs):
        if backup_dir is None:
            print(f"ERROR: no backup dir known for system '{system_name}'.")
            continue
        if not os.path.isdir(backup_dir):
            print(f"WARNING: backup dir not found for {system_name}: {backup_dir}")
            continue

        summary = process_system(
            system_name               = system_name,
            backup_dir                = backup_dir,
            skip_first                = args.skip_first,
            output_dir                = args.output_dir,
            prefix                    = args.prefix,
            no_ideal                  = args.no_ideal,
            aggregate_placements_flag = args.aggregate_placements,
            only_all                  = args.only_all,
            table_gpus                = args.table_gpus,
            strategyegies_filter      = args.strategies,
            **lbl_kw,
        )
        if summary is not None and not summary.empty:
            all_summaries.append(summary)

    if len(all_summaries) == 0:
        print("\nNo data loaded; skipping global plots.")
    else:
        combined = pd.concat(all_summaries, ignore_index=True)

        plot_global(combined, output_dir=output_dir, pfx=pfx,
                    no_ideal=args.no_ideal,
                    aggregate_placements_flag=args.aggregate_placements,
                    **lbl_kw)

        for plot_type in ("scaling", "efficiency"):
            plot_global_faceted(
                combined,
                output_dir                = output_dir,
                pfx                       = pfx,
                no_ideal                  = args.no_ideal,
                aggregate_placements_flag = args.aggregate_placements,
                n_rows                    = args.grid_rows,
                plot_type                 = plot_type,
                figsize_per_cell          = tuple(args.cell_size),
                **lbl_kw,
            )

    print(f"\nDone.")


if __name__ == "__main__":
    main()