"""
Multi-Training Congestion: Experiment Design Framework
======================================================
Python implementation of the formalization by Thomas Pasquali.

Structure:
  - Section 1:  System model (GPU count, strategies, feasibility sets)
  - Section 2:  Baseline set B
  - Section 3:  GPU allocation patterns  (families A / B / C / D / E)
  - Section 4:  Strategy assignment & entropy-stratified sampling
  - Section 5:  Final experiment set  E
  - Section 6:  Network topology model & placement-class sampling
  - Section 7:  TopologyOracle  – thin wrapper around the external topology program
  - Section 8:  Extended experiment set  E_hier
  - Section 9:  Pretty-printing / summary utilities
  - Section 10: JSON serialization

Configure everything in the CONFIG block at the top of this file.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import subprocess
import json
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional
from datetime import datetime, timezone


# ===========================================================================
# CONFIG  –  edit this block to customise the experiment design
# ===========================================================================

# ── Hardware ────────────────────────────────────────────────────────────────
G: int = 72                # Total number of GPUs in the cluster

# ── Strategies ──────────────────────────────────────────────────────────────
# Each entry: (name, feasible_gpu_counts)
STRATEGY_DEFS: list[tuple[str, list[int]]] = [
    ("DP",           [2, 4, 8, 16]),
    ("FSDP",         [4, 8, 16, 32]),
    ("DP+PP",        [4, 8, 16, 32, 64]),
    ("DP+PP+Expert", [64, 128, 192, 256, 320, 384, 448, 512]),
    ("DP+PP+TP",     [320, 640, 960]),
]

# ── Pattern generation ───────────────────────────────────────────────────────

# Minimum slot size (GPUs); slots smaller than this are dropped
G_MIN: int = 2

# Maximum number of concurrent jobs in a single configuration
K_MAX: int = math.floor(G / G_MIN)

# Geometric pattern: decay factor β  (Section 3B)
GEOMETRIC_BETA: float = 0.5

# Hierarchical dominant-minority pattern (Section 3C):
#   List of (tier_fractions, n_tiny) tuples.
#   Each entry generates one pattern family.
#   tier_fractions: [α_1, …, α_d]  (fractions of the *residual* budget)
#   n_tiny: number of equal-size tiny slots that consume the remaining budget
HIERARCHICAL_PATTERNS: list[tuple[list[float], int]] = [
    ([0.50, 0.25], 4),          # one dominant + one medium + 4 tiny
    ([0.05, 0.21], 6),          # 5% large, ~20% medium, ~75% tiny (paper example)
    ([0.89],       4),          # one very dominant + 4 tiny  (S4-style: 64/72)
]

# ── Power-law patterns (Family D) ────────────────────────────────────────────
# Tail exponents α to sweep.  α close to 1 → heavy tail (very unequal slots);
# large α → light tail (slots become more equal, approaches family A).
POWERLAW_ALPHAS: list[float] = [1.2]

# Utilization range ρ ∈ (0, 1] applied to ALL pattern families (A–E).
# A uniform grid of UTIL_STEPS values is generated in [UTIL_MIN, UTIL_MAX]
# (both endpoints inclusive).  Each family generates one variant per grid
# point.  Set UTIL_STEPS = 1 to use only UTIL_MAX (single point).
UTIL_MIN:   float = 0.8
UTIL_MAX:   float = 1.0
UTIL_STEPS: int   = 1

# ── Stochastic tier-sampled patterns (Family E) ───────────────────────────────
# Each tier has a weight (probability of being chosen for the next job),
# a list of allowed GPU sizes, and optional sub-weights for specific sizes
# within the tier.  Sub-weights are specified as {gpu_size: probability};
# unlisted sizes share the remaining probability equally.
#
# Tier weights must sum to 1.0.
# Sub-weights within a tier must sum to ≤ 1.0.
STOCHASTIC_TIER_CONFIG: dict[str, dict] = {
    "small": {
        "tier_weight": 0.75,
        "sizes": [2],
        "sub_weights": {},              # only one size, so no sub-weight needed
    },
    "medium": {
        "tier_weight": 0.20,
        "sizes": [4, 8, 16],
        "sub_weights": {8: 0.50},       # 50% are 8 GPUs; 4 and 16 share the rest equally
    },
    "large": {
        "tier_weight": 0.05,
        "sizes": [32, 64],
        "sub_weights": {64: 0.20},      # 20% are 64 GPUs, 80% are 32 GPUs
    },
}

# Number of distinct stochastic patterns to generate for family E.
# Each draw is an independent realisation of the tier-sampling process.
N_STOCHASTIC_PATTERNS: int = 5

# ── Entropy-stratified sampling ──────────────────────────────────────────────
# Bin boundaries for mixture entropy H ∈ [0, log m]
ENTROPY_DELTA_1: float = 0.3     # low  / medium boundary
ENTROPY_DELTA_2: float = 0.7     # medium / high  boundary  (as fraction of log m)

# Number of labellings to sample per entropy bin, per pattern
N_SAMPLES_PER_BIN: int = 4

# Maximum |Φ(P)| for which full enumeration is used instead of rejection
# sampling.  Enumeration is exact but O(|Φ(P)|); rejection sampling is
# O(n_per_bin × k) but may miss sparse bins.  The default (50 000) keeps
# memory well under 10 MB for any realistic strategy set.
# Set to 0 to force rejection sampling always; set to math.inf to force
# enumeration always (dangerous for large k).
ENUM_THRESHOLD: int = 50_000

# Random seed for reproducibility (set to None for non-deterministic)
RANDOM_SEED: Optional[int] = 42

# ── Topology (Section 6) ─────────────────────────────────────────────────────
# Set USE_TOPOLOGY = False to skip hierarchical placement entirely (flat model)
USE_TOPOLOGY: bool = False

# Path / command of the external topology program (Section 7)
TOPOLOGY_PROGRAM: str = "topology_oracle"   # must be on PATH or an absolute path

# Number of within-class variance shuffles for high-interest configs (Section 8.3)
N_VARIANCE_SHUFFLES: int = 3

# ── JSON output ──────────────────────────────────────────────────────────────
# Default output path for the serialized experiment set.
# Pass --output-json <path> on the CLI to override.
DEFAULT_JSON_OUTPUT: str = "experiments.json"


# ===========================================================================
# SECTION 1 – Data structures
# ===========================================================================

@dataclass(frozen=True)
class Strategy:
    """A distributed training strategy together with its feasible GPU counts."""
    name: str
    feasible: frozenset[int]

    def supports(self, g: int) -> bool:
        return g in self.feasible


@dataclass(frozen=True)
class SingleRun:
    """r = (S_i, g)  –  one job with a chosen strategy and GPU count."""
    strategy: Strategy
    gpus: int

    def __str__(self) -> str:
        return f"({self.strategy.name}:{self.gpus})"


@dataclass
class Config:
    """
    A concurrent configuration C = {r_1, …, r_k}.
    Satisfies: sum(r.gpus for r in runs) <= G
    """
    runs: list[SingleRun]
    cfg: argparse.Namespace

    @property
    def total_gpus(self) -> int:
        return sum(r.gpus for r in self.runs)

    @property
    def utilization(self) -> float:
        return self.total_gpus / self.cfg.G

    @property
    def k(self) -> int:
        return len(self.runs)

    def __str__(self) -> str:
        slots = ", ".join(str(r) for r in self.runs)
        return f"[{slots:<40}]\n  util={int(self.utilization*100):3<}%"


# ===========================================================================
# SECTION 2 – Baseline set  B
# ===========================================================================

def build_strategies(defs: list[tuple[str, list[int]]]) -> list[Strategy]:
    return [Strategy(name, frozenset(gpus)) for name, gpus in defs]


def build_baseline_set(strategies: list[Strategy], g_total: int) -> list[SingleRun]:
    """
    B = { (S_i, g) | S_i ∈ S, g ∈ G_i, g <= G }
    Every feasible single run that could appear as a slot.
    """
    baseline: list[SingleRun] = []
    for s in strategies:
        for g in sorted(s.feasible):
            if g < g_total:
                baseline.append(SingleRun(s, g))
    return baseline


# ===========================================================================
# SECTION 3 – GPU Allocation Patterns
# ===========================================================================

AllocationPattern = tuple[int, ...]   # ordered vector of GPU counts (slots only)


@dataclass(frozen=True)
class TaggedPattern:
    """An allocation pattern together with the family that generated it."""
    slots: AllocationPattern
    family: str   # one of: "A", "B", "C", "D", "E"

    # Delegate common tuple-like queries so call-sites stay concise
    def __len__(self) -> int:
        return len(self.slots)

    def __iter__(self):
        return iter(self.slots)

    def __getitem__(self, idx):
        return self.slots[idx]


def compute_feasible_gpu_counts(
    strategies: list[Strategy],
    g_total: int,
) -> frozenset[int]:
    """
    Return the set of GPU counts that are (a) supported by at least one
    strategy and (b) strictly less than g_total (so a slot can coexist with
    at least one other slot).

    This is the universe of valid slot sizes for any allocation pattern.
    Every pattern family must restrict its slot sizes to this set.
    """
    counts: set[int] = set()
    for s in strategies:
        for g in s.feasible:
            if g < g_total:
                counts.add(g)
    return frozenset(counts)


def _snap_down(raw: int, feasible: frozenset[int]) -> Optional[int]:
    """
    Return the largest value in `feasible` that is ≤ `raw`.
    Returns None if no such value exists (raw is smaller than every feasible count).
    """
    candidates = [g for g in feasible if g <= raw]
    return max(candidates) if candidates else None


def pattern_A_equal_splits(
    g_total: int,
    k_max: int,
    g_min: int,
    feasible_gpu_counts: frozenset[int],
    utilization: float = 1.0,
) -> list[TaggedPattern]:
    """
    Family A: equal-split patterns  P = (g, …, g)  with  k*g ≤ floor(ρ·G).

    Only emits patterns whose slot size g is in feasible_gpu_counts.
    """
    budget = int(math.floor(utilization * g_total))
    patterns: list[TaggedPattern] = []
    for k in range(2, k_max + 1):
        if budget % k == 0:
            g = budget // k
            if g >= g_min and g in feasible_gpu_counts:
                patterns.append(TaggedPattern(slots=tuple([g] * k), family="A"))
    return patterns


def pattern_B_geometric(
    g_total: int,
    k_max: int,
    g_min: int,
    beta: float,
    feasible_gpu_counts: frozenset[int],
    utilization: float = 1.0,
) -> list[TaggedPattern]:
    """
    Family B: geometric decay patterns.

    Start with g_1 = floor(ρ·G) // 2, then g_{j+1} = floor(β·g_j).
    Slots are accumulated until the budget floor(ρ·G) is exhausted or
    k_max is reached.  Each slot size is snapped down to the nearest
    feasible GPU count.
    """
    budget = int(math.floor(utilization * g_total))
    patterns: list[TaggedPattern] = []
    g1 = budget // 2
    while g1 >= g_min:
        slots: list[int] = []
        remaining = budget
        g_curr = g1
        while len(slots) < k_max:
            if g_curr < g_min:
                break
            snapped = _snap_down(g_curr, feasible_gpu_counts)
            if snapped is None or snapped < g_min or snapped > remaining:
                break
            slots.append(snapped)
            remaining -= snapped
            g_curr = max(g_min, int(math.floor(beta * g_curr)))
        if len(slots) >= 2:
            patterns.append(TaggedPattern(slots=tuple(slots), family="B"))
        g1 = g1 // 2
    return patterns


def pattern_C_hierarchical(
    g_total: int,
    k_max: int,
    g_min: int,
    tier_fractions: list[float],
    n_tiny: int,
    feasible_gpu_counts: frozenset[int],
    utilization: float = 1.0,
) -> Optional[TaggedPattern]:
    """
    Family C: hierarchical dominant-minority patterns.

    Tier fractions are applied to floor(ρ·G) as the starting residual.
    Each computed tier size and tiny slot size is snapped down to the nearest
    feasible GPU count.  Returns None if any slot cannot be snapped or falls
    below g_min.
    """
    budget = int(math.floor(utilization * g_total))
    slots: list[int] = []
    residual = budget

    for alpha in tier_fractions:
        raw = int(math.floor(alpha * residual))
        snapped = _snap_down(raw, feasible_gpu_counts)
        if snapped is None or snapped < g_min:
            return None
        slots.append(snapped)
        residual -= snapped

    if n_tiny > 0 and residual > 0:
        raw_tiny = int(math.floor(residual / n_tiny))
        snapped_tiny = _snap_down(raw_tiny, feasible_gpu_counts)
        if snapped_tiny is None or snapped_tiny < g_min:
            return None
        slots.extend([snapped_tiny] * n_tiny)

    if len(slots) < 2 or len(slots) > k_max:
        return None

    if sum(slots) > g_total:
        return None

    return TaggedPattern(slots=tuple(slots), family="C")


def pattern_D_powerlaw(
    g_total: int,
    k: int,
    g_min: int,
    alpha: float,
    utilization: float,
    feasible_gpu_counts: frozenset[int],
) -> Optional[TaggedPattern]:
    """
    Family D: power-law (Zipf) allocation pattern.

    Slot sizes are proportional to rank^{-alpha}, rescaled to sum to
    floor(utilization * G), then each slot is snapped down to the nearest
    feasible GPU count (≥ g_min).  Discards the pattern if any slot cannot
    be snapped or if the total exceeds g_total.

    Construction:
      1. Raw weights:    g_hat[j] = j^{-alpha}   for j = 1 … k
      2. Rescale:        g[j] = floor(g_hat[j] / sum(g_hat) * floor(rho * G))
      3. Snap down:      g[j] = largest feasible count ≤ g[j], ≥ g_min
      4. Feasibility:    discard if any slot unsnappable or sum > g_total
    """
    if alpha <= 1.0:
        raise ValueError(f"alpha must be > 1 for a normalizable power law, got {alpha}")

    budget = int(math.floor(utilization * g_total))
    raw = [j ** (-alpha) for j in range(1, k + 1)]
    total_raw = sum(raw)

    slots: list[int] = []
    for w in raw:
        raw_size = int(math.floor(w / total_raw * budget))
        snapped = _snap_down(raw_size, feasible_gpu_counts)
        if snapped is None or snapped < g_min:
            return None
        slots.append(snapped)

    if sum(slots) > g_total or len(slots) < 2:
        return None

    return TaggedPattern(slots=tuple(slots), family="D")


# ---------------------------------------------------------------------------
# Family E – Stochastic tier-sampled patterns
# ---------------------------------------------------------------------------

def _tier_size_weights(
    tier_name: str,
    tier_cfg: dict,
    feasible_gpu_counts: frozenset[int],
) -> tuple[list[int], list[float]]:
    """
    Return (sizes, weights) for the intra-tier GPU-size distribution,
    restricted to sizes that appear in feasible_gpu_counts.

    Sizes absent from feasible_gpu_counts are silently dropped before
    computing weights; their probability mass is redistributed to the
    remaining sizes proportionally (pinned weights scale together; free
    sizes share the residual equally).

    Raises ValueError if no sizes remain after filtering.
    """
    all_sizes: list[int] = tier_cfg["sizes"]
    sub_weights_map: dict[int, float] = tier_cfg.get("sub_weights", {})

    # Keep only feasible sizes
    sizes = [s for s in all_sizes if s in feasible_gpu_counts]
    if not sizes:
        raise ValueError(
            f"Tier '{tier_name}' has no sizes in feasible_gpu_counts "
            f"(sizes={all_sizes}, feasible={sorted(feasible_gpu_counts)})."
        )

    # Recompute weights over the surviving sizes
    pinned = {s: sub_weights_map[s] for s in sizes if s in sub_weights_map}
    free = [s for s in sizes if s not in pinned]

    total_pinned = sum(pinned.values())
    if total_pinned > 1.0 + 1e-9:
        raise ValueError(
            f"Sub-weights for tier '{tier_name}' exceed 1.0 after filtering "
            f"(got {total_pinned:.6f})."
        )

    rem_prob = (1.0 - total_pinned) / len(free) if free else 0.0

    weights: list[float] = []
    for s in sizes:
        weights.append(pinned[s] if s in pinned else rem_prob)

    # Normalise to absorb any floating-point drift
    total_w = sum(weights)
    if total_w <= 0:
        raise ValueError(f"All weights are zero for tier '{tier_name}'.")
    weights = [w / total_w for w in weights]

    return sizes, weights


def pattern_E_stochastic_tier(
    g_total: int,
    k_max: int,
    g_min: int,
    tier_config: dict[str, dict],
    feasible_gpu_counts: frozenset[int],
    rng: random.Random,
    utilization: float = 1.0,
) -> Optional[TaggedPattern]:
    """
    Family E: stochastic tier-sampled allocation pattern.

    Slots are drawn until the budget floor(ρ·G) is exhausted or k_max is
    reached.  Each tier's size pool is pre-filtered to feasible_gpu_counts
    before any sampling occurs, so every drawn slot size is guaranteed to be
    supported by at least one strategy.

    Generation procedure:
      1. Validate tier weights sum to 1.0.
      2. For each tier, drop sizes not in feasible_gpu_counts and
         renormalise intra-tier weights.  Tiers with no remaining sizes
         are dropped entirely and the tier-level weights are renormalised.
      3. Repeat until remaining budget < g_min or k_max slots filled:
           a. If no feasible size fits the remaining budget, stop.
           b. Draw tier t ~ Categorical(tier_weights).
           c. Draw size g ~ Categorical(intra-tier weights for t).
           d. If g ≤ remaining budget, append g; else skip and retry.
      4. Sort descending and return; return None if fewer than 2 slots.
    """
    tier_names = list(tier_config.keys())

    total_tw = sum(tier_config[t]["tier_weight"] for t in tier_names)
    if abs(total_tw - 1.0) > 1e-9:
        raise ValueError(
            f"Tier weights in STOCHASTIC_TIER_CONFIG must sum to 1.0 "
            f"(got {total_tw:.6f})."
        )

    # Pre-compute per-tier (sizes, weights), dropping infeasible sizes.
    # Also drop entire tiers whose size pool becomes empty after filtering.
    tier_size_info: dict[str, tuple[list[int], list[float]]] = {}
    surviving_tiers: list[str] = []
    surviving_tw: list[float] = []

    for t in tier_names:
        try:
            sizes, weights = _tier_size_weights(t, tier_config[t], feasible_gpu_counts)
            tier_size_info[t] = (sizes, weights)
            surviving_tiers.append(t)
            surviving_tw.append(tier_config[t]["tier_weight"])
        except ValueError:
            # Tier has no feasible sizes; skip it entirely
            pass

    if not surviving_tiers:
        return None  # no tier can contribute any slot

    # Renormalise tier-level weights over surviving tiers
    total_stw = sum(surviving_tw)
    surviving_tw = [w / total_stw for w in surviving_tw]

    budget = int(math.floor(utilization * g_total))
    slots: list[int] = []
    remaining = budget

    while remaining >= g_min and len(slots) < k_max:
        # Check if any surviving tier can produce a size that fits
        any_fits = any(
            s <= remaining
            for t in surviving_tiers
            for s in tier_size_info[t][0]
        )
        if not any_fits:
            break

        tier = rng.choices(surviving_tiers, weights=surviving_tw, k=1)[0]
        sizes, weights = tier_size_info[tier]
        g = rng.choices(sizes, weights=weights, k=1)[0]

        if g <= remaining:
            slots.append(g)
            remaining -= g
        # else: size doesn't fit; skip this draw and try again

    if len(slots) < 2:
        return None

    return TaggedPattern(slots=tuple(sorted(slots, reverse=True)), family="E")


def pattern_E_stochastic_tier_batch(
    g_total: int,
    k_max: int,
    g_min: int,
    tier_config: dict[str, dict],
    feasible_gpu_counts: frozenset[int],
    n_patterns: int,
    rng: random.Random,
    utilization: float = 1.0,
) -> list[TaggedPattern]:
    """
    Generate n_patterns distinct realisations of the family-E stochastic process.
    Duplicate patterns (identical sorted slot tuples) are discarded; the generator
    retries until n_patterns distinct patterns are collected or an attempt budget
    is exhausted (10 × n_patterns attempts).
    """
    seen: set[AllocationPattern] = set()
    patterns: list[TaggedPattern] = []
    budget = 10 * n_patterns

    for _ in range(budget):
        if len(patterns) >= n_patterns:
            break
        p = pattern_E_stochastic_tier(
            g_total, k_max, g_min, tier_config, feasible_gpu_counts, rng,
            utilization=utilization,
        )
        if p is not None and p.slots not in seen:
            seen.add(p.slots)
            patterns.append(p)

    return patterns


def build_pattern_set(
    g_total: int,
    k_max: int,
    g_min: int,
    beta: float,
    hierarchical_defs: list[tuple[list[float], int]],
    powerlaw_alphas: list[float],
    utilizations: list[float],
    util_min: float,
    util_max: float,
    stochastic_tier_config: dict[str, dict],
    n_stochastic_patterns: int,
    feasible_gpu_counts: frozenset[int],
    rng: random.Random,
    generate_equal_splits: bool,
    generate_geometric: bool,
    generate_hierarchical: bool,
    generate_powerlaw: bool,
    generate_stochastic: bool,
) -> list[TaggedPattern]:
    """
    Build the canonical pattern set P = families A ∪ B ∪ C ∪ D ∪ E (deduplicated).

    The `utilizations` grid is swept across all families:
      A – one equal-split sweep per ρ value
      B – one geometric-decay sweep per ρ value
      C – one hierarchical pattern per (def, ρ) pair
      D – one power-law pattern per (α, k, ρ) triple
      E – one independent stochastic batch per ρ value

    After generation, patterns whose actual utilization sum(slots)/G falls
    outside [util_min, util_max] are silently dropped.  This filters out
    patterns where post-snap rounding pushed the real utilization outside the
    configured range, regardless of the target ρ used during generation.

    Deduplication is on slot content only (identical slot tuples from different
    families or utilization levels keep the first occurrence, preserving generation
    order A→B→C→D→E and ascending ρ within each family).
    Every slot size is guaranteed to be in feasible_gpu_counts.
    """
    seen: set[AllocationPattern] = set()
    patterns: list[TaggedPattern] = []

    def add(tp: TaggedPattern) -> None:
        actual_util = sum(tp.slots) / g_total
        if (
            tp.slots not in seen
            and sum(tp.slots) <= g_total
            and util_min <= actual_util <= util_max
        ):
            seen.add(tp.slots)
            patterns.append(tp)

    for rho in utilizations:
        if generate_equal_splits:
            for tp in pattern_A_equal_splits(
                g_total, k_max, g_min, feasible_gpu_counts, utilization=rho
            ):
                add(tp)

        if generate_geometric:
            for tp in pattern_B_geometric(
                g_total, k_max, g_min, beta, feasible_gpu_counts, utilization=rho
            ):
                add(tp)

        if generate_hierarchical:
            for tier_fracs, n_tiny in hierarchical_defs:
                tp = pattern_C_hierarchical(
                    g_total, k_max, g_min, tier_fracs, n_tiny,
                    feasible_gpu_counts, utilization=rho,
                )
                if tp is not None:
                    add(tp)

        if generate_powerlaw:
            for alpha in powerlaw_alphas:
                for k in range(2, k_max + 1):
                    tp = pattern_D_powerlaw(
                        g_total, k, g_min, alpha, rho, feasible_gpu_counts
                    )
                    if tp is not None:
                        add(tp)

        if generate_stochastic:
            for tp in pattern_E_stochastic_tier_batch(
                g_total, k_max, g_min, stochastic_tier_config,
                feasible_gpu_counts, n_stochastic_patterns, rng,
                utilization=rho,
            ):
                add(tp)

    return patterns


# ===========================================================================
# SECTION 4 – Strategy Assignment & Entropy-Stratified Sampling
# ===========================================================================

def feasible_strategies_for_slot(strategies: list[Strategy], g: int) -> list[Strategy]:
    """All strategies that support GPU count g."""
    return [s for s in strategies if s.supports(g)]


def all_feasible_labellings(
    pattern: TaggedPattern,
    strategies: list[Strategy],
) -> list[list[Strategy]]:
    """
    Enumerate Φ(P): all feasible strategy labellings for the given pattern.
    WARNING: exponential in k – use only for small patterns or sampling.
    """
    per_slot = [feasible_strategies_for_slot(strategies, g) for g in pattern.slots]
    if any(len(opts) == 0 for opts in per_slot):
        return []   # pattern has an infeasible slot
    return [list(combo) for combo in itertools.product(*per_slot)]


def mixture_entropy(labelling: list[Strategy], pattern: AllocationPattern) -> float:
    """
    Compute H(C) = -Σ p̃_i log p̃_i where p̃_i = p_i / ρ(C).
    """
    rho = sum(pattern) / G
    if rho == 0:
        return 0.0
    gpu_per_strategy: dict[str, int] = {}
    for strat, g in zip(labelling, pattern):
        gpu_per_strategy[strat.name] = gpu_per_strategy.get(strat.name, 0) + g
    entropy = 0.0
    for g_s in gpu_per_strategy.values():
        p_tilde = (g_s / G) / rho
        if p_tilde > 0:
            entropy -= p_tilde * math.log(p_tilde)
    return entropy


def entropy_bin(
    h: float,
    m_feasible: int,
    k: int,
    delta1_frac: float,
    delta2_frac: float,
) -> str:
    """Return 'low', 'medium', or 'high' entropy bin label.

    h_max = ln(min(k, m_feasible)) where:
      k           = number of slots in the pattern (caps distinct strategies
                    that can co-appear in one labelling)
      m_feasible  = number of distinct strategies feasible for this pattern
                    (strategies not feasible for any slot can never appear)
    """
    effective_m = min(k, m_feasible)
    h_max = math.log(effective_m) if effective_m > 1 else 1.0
    d1 = delta1_frac * h_max
    d2 = delta2_frac * h_max
    if h < d1:
        return "low"
    elif h < d2:
        return "medium"
    else:
        return "high"


def _labelling_space_size(per_slot: list[list[Strategy]], threshold: int) -> int:
    """
    Compute |Φ(P)| = ∏ |feasible(slot_j)| without materialising the product.
    Returns immediately once the running product exceeds `threshold`.
    """
    size = 1
    for opts in per_slot:
        size *= len(opts)
        if size > threshold:
            return size
    return size


def _random_labelling(
    per_slot: list[list[Strategy]],
    rng: random.Random,
) -> list[Strategy]:
    """Draw one labelling by independently sampling each slot."""
    return [rng.choice(opts) for opts in per_slot]


def _sample_exact(
    per_slot: list[list[Strategy]],
    pattern: AllocationPattern,
    n_per_bin: int,
    m_feasible: int,
    k: int,
    delta1_frac: float,
    delta2_frac: float,
    rng: random.Random,
) -> list[tuple[list[Strategy], str]]:
    """
    Exact path: enumerate all of Φ(P), bin every labelling by entropy, then
    draw up to n_per_bin samples uniformly at random from each bin.
    """
    BIN_NAMES = ("low", "medium", "high")
    bins: dict[str, list[list[Strategy]]] = {b: [] for b in BIN_NAMES}

    for combo in itertools.product(*per_slot):
        lab = list(combo)
        h = mixture_entropy(lab, pattern)
        b = entropy_bin(h, m_feasible, k, delta1_frac, delta2_frac)
        bins[b].append(lab)

    result: list[tuple[list[Strategy], str]] = []
    for bin_name in BIN_NAMES:
        chosen = rng.sample(bins[bin_name], min(n_per_bin, len(bins[bin_name])))
        for lab in chosen:
            result.append((lab, bin_name))
    return result


def _sample_rejection(
    per_slot: list[list[Strategy]],
    pattern: AllocationPattern,
    n_per_bin: int,
    m_feasible: int,
    k: int,
    delta1_frac: float,
    delta2_frac: float,
    rng: random.Random,
) -> list[tuple[list[Strategy], str]]:
    """
    Rejection-sampling path: draw labellings one at a time, route each into its
    entropy bin, stop when all bins are full or the attempt budget is exhausted.
    """
    BIN_NAMES = ("low", "medium", "high")
    ATTEMPTS_PER_SAMPLE = 200

    bins: dict[str, list[list[Strategy]]] = {b: [] for b in BIN_NAMES}
    seen: set[tuple[str, ...]] = set()
    total_budget = n_per_bin * ATTEMPTS_PER_SAMPLE * len(BIN_NAMES)

    for _ in range(total_budget):
        if all(len(bins[b]) >= n_per_bin for b in BIN_NAMES):
            break

        lab = _random_labelling(per_slot, rng)
        key = tuple(s.name for s in lab)
        if key in seen:
            continue

        h = mixture_entropy(lab, pattern)
        b = entropy_bin(h, m_feasible, k, delta1_frac, delta2_frac)

        if len(bins[b]) < n_per_bin:
            bins[b].append(lab)
            seen.add(key)

    result: list[tuple[list[Strategy], str]] = []
    for bin_name in BIN_NAMES:
        for lab in bins[bin_name]:
            result.append((lab, bin_name))
    return result


def sample_labellings_stratified(
    pattern: TaggedPattern,
    strategies: list[Strategy],
    n_per_bin: int,
    delta1_frac: float,
    delta2_frac: float,
    enum_threshold: int,
    rng: random.Random,
) -> list[tuple[list[Strategy], str]]:
    """
    Φ_sample(P): return up to n_per_bin labellings per entropy bin.
    Selects exact enumeration or rejection sampling based on |Φ(P)|.
    """
    per_slot = [feasible_strategies_for_slot(strategies, g) for g in pattern.slots]
    if any(len(opts) == 0 for opts in per_slot):
        return []

    # m_feasible: strategies reachable by at least one slot in this pattern.
    # k: number of slots — caps how many distinct strategies can co-appear.
    # Together these give h_max = ln(min(k, m_feasible)).
    m_feasible = len({s.name for opts in per_slot for s in opts})
    k = len(pattern.slots)
    space = _labelling_space_size(per_slot, enum_threshold)

    if space <= enum_threshold:
        return _sample_exact(per_slot, pattern.slots, n_per_bin, m_feasible, k,
                             delta1_frac, delta2_frac, rng)
    else:
        return _sample_rejection(per_slot, pattern.slots, n_per_bin, m_feasible, k,
                                 delta1_frac, delta2_frac, rng)


# ===========================================================================
# SECTION 5 – Final Experiment Set  E  (flat / uniform topology)
# ===========================================================================

@dataclass
class Experiment:
    """A single element of E: pattern + labelling → Config."""
    pattern: TaggedPattern
    labelling: list[Strategy]
    entropy_bin: str
    config: Config
    placement: Optional["PlacementClassVector"] = None   # set in Section 8

    def __str__(self) -> str:
        pattern_str = str(self.pattern.slots)
        config_str = str(self.config)
        placement_str = (
            f"  placement={self.placement}" if self.placement else ""
        )
        return f"  pattern={pattern_str:<20}\n  H-bin={self.entropy_bin:<6}{placement_str}\n  config={config_str:<40}\n"


def build_experiment_set(
    cfg: argparse.Namespace,
    patterns: list[TaggedPattern],
    strategies: list[Strategy],
    n_per_bin: int,
    delta1_frac: float,
    delta2_frac: float,
    enum_threshold: int,
    rng: random.Random,
) -> list[Experiment]:
    """
    E = { C(P, φ) | P ∈ P, φ ∈ Φ_sample(P) }
    """
    experiments: list[Experiment] = []
    for pattern in patterns:
        sampled = sample_labellings_stratified(
            pattern, strategies, n_per_bin, delta1_frac, delta2_frac,
            enum_threshold, rng
        )
        for labelling, bin_label in sampled:
            runs = [SingleRun(s, g) for s, g in zip(labelling, pattern.slots)]
            config = Config(runs, cfg)
            experiments.append(
                Experiment(
                    pattern=pattern,
                    labelling=labelling,
                    entropy_bin=bin_label,
                    config=config,
                )
            )
    return experiments


# ===========================================================================
# SECTION 6 – Network Topology Model
# ===========================================================================

class PlacementClass:
    """The four placement classes from Section 6.3."""
    INTRA_NODE  = "intra-node"
    INTRA_L1    = "intra-L1"
    INTRA_GROUP = "intra-group"
    MULTI_GROUP = "multi-group"
    ALL = [INTRA_NODE, INTRA_L1, INTRA_GROUP, MULTI_GROUP]


PlacementClassVector = tuple[str, ...]   # one entry per job in the config


# ===========================================================================
# SECTION 7 – TopologyOracle  (wrapper around the external topology program)
# ===========================================================================

class TopologyOracle:
    """
    Thin wrapper around the external topology analysis program.

    If the program is not available, a stub fallback is used so the rest of the
    framework can still run (all jobs default to 'intra-node').
    """

    def __init__(self, program: str = TOPOLOGY_PROGRAM) -> None:
        self.program = program
        self._available = self._check_available()

    def _check_available(self) -> bool:
        try:
            result = subprocess.run(
                [self.program, "--ping"],
                capture_output=True, timeout=2
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _query(self, payload: dict) -> dict:
        if not self._available:
            return self._stub_response(payload)
        try:
            proc = subprocess.run(
                [self.program],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=10,
            )
            return json.loads(proc.stdout)
        except Exception as exc:
            print(f"[TopologyOracle] Warning: query failed ({exc}), using stub.")
            return self._stub_response(payload)

    def _stub_response(self, payload: dict) -> dict:
        cmd = payload.get("command", "")
        if cmd == "describe_topology":
            return {"G": G, "q1": 8, "q2": 64, "q3": G}
        if cmd == "feasible_placement_classes":
            g = payload.get("gpu_count", 1)
            free = payload.get("free_capacity", {})
            classes = []
            if g <= free.get("intra_node", 0):
                classes.append(PlacementClass.INTRA_NODE)
            if g <= free.get("intra_l1", 0):
                classes.append(PlacementClass.INTRA_L1)
            classes.append(PlacementClass.INTRA_GROUP)
            classes.append(PlacementClass.MULTI_GROUP)
            return {"classes": classes[:1]}
        if cmd == "find_placement":
            jobs = payload.get("jobs", [])
            return {
                "feasible": True,
                "assignments": [
                    {"job_id": j["job_id"],
                     "nodes": [f"node-{j['job_id']}-{n}"
                               for n in range(max(1, j["gpu_count"] // 8))]}
                    for j in jobs
                ],
            }
        if cmd == "shuffle_within_class":
            jobs = payload.get("jobs", [])
            n = payload.get("n_shuffles", 1)
            single = [
                {"job_id": j["job_id"],
                 "nodes": [f"node-{j['job_id']}-shuffle-{n}"
                           for _ in range(max(1, j["gpu_count"] // 8))]}
                for j in jobs
            ]
            return {"shuffles": [single] * n}
        return {}

    def describe_topology(self) -> dict:
        return self._query({"command": "describe_topology"})

    def feasible_placement_classes(
        self,
        gpu_count: int,
        free_capacity: dict[str, int],
    ) -> list[str]:
        payload = {
            "command": "feasible_placement_classes",
            "gpu_count": gpu_count,
            "free_capacity": free_capacity,
        }
        resp = self._query(payload)
        return resp.get("classes", [PlacementClass.INTRA_GROUP])

    def find_placement(
        self,
        jobs: list[dict],
    ) -> dict:
        return self._query({"command": "find_placement", "jobs": jobs})

    def shuffle_within_class(
        self,
        jobs: list[dict],
        n_shuffles: int = 3,
    ) -> list[list[dict]]:
        resp = self._query({
            "command": "shuffle_within_class",
            "jobs": jobs,
            "n_shuffles": n_shuffles,
        })
        return resp.get("shuffles", [])


# ===========================================================================
# SECTION 8 – Extended Experiment Set  E_hier
# ===========================================================================

@dataclass
class HierarchicalExperiment:
    """
    X = (P, φ, ψ)  –  one element of E_hier.
    """
    base: Experiment
    placement_class_vector: PlacementClassVector
    node_assignment: Optional[dict] = None
    is_variance_shuffle: bool = False
    shuffle_index: Optional[int] = None

    def __str__(self) -> str:
        suffix = ""
        if self.is_variance_shuffle:
            suffix = f"  [variance shuffle #{self.shuffle_index}]"
        return (
            f"{self.base}"
            f"  κ={list(self.placement_class_vector)}{suffix}\n"
        )


def _free_capacity_from_topology(topo: dict, allocated: int) -> dict[str, int]:
    q1 = topo.get("q1", 8)
    q2 = topo.get("q2", 64)
    q3 = topo.get("q3", G)
    remaining = G - allocated
    return {
        "intra_node":  min(remaining, q1),
        "intra_l1":    min(remaining, q2),
        "intra_group": min(remaining, q3),
    }


def build_placement_class_vectors(
    config: Config,
    oracle: TopologyOracle,
) -> list[PlacementClassVector]:
    topo = oracle.describe_topology()
    allocated_so_far = 0

    per_job_classes: list[list[str]] = []
    for run in config.runs:
        free = _free_capacity_from_topology(topo, allocated_so_far)
        classes = oracle.feasible_placement_classes(run.gpus, free)
        per_job_classes.append(classes)
        allocated_so_far += run.gpus

    vectors: list[PlacementClassVector] = [
        tuple(combo) for combo in itertools.product(*per_job_classes)
    ]
    return vectors


def build_hierarchical_experiment_set(
    flat_experiments: list[Experiment],
    oracle: TopologyOracle,
    n_variance_shuffles: int,
    high_interest_predicate: Optional[Callable[[Experiment], bool]] = None,
) -> list[HierarchicalExperiment]:
    if high_interest_predicate is None:
        high_interest_predicate = lambda exp: exp.config.k >= 2

    hier_experiments: list[HierarchicalExperiment] = []

    for exp in flat_experiments:
        kappa_vectors = build_placement_class_vectors(exp.config, oracle)

        for kappa in kappa_vectors:
            jobs_payload = [
                {"job_id": j, "gpu_count": run.gpus, "placement_class": kappa[j]}
                for j, run in enumerate(exp.config.runs)
            ]
            placement = oracle.find_placement(jobs_payload)

            he = HierarchicalExperiment(
                base=exp,
                placement_class_vector=kappa,
                node_assignment=placement if placement.get("feasible") else None,
            )
            hier_experiments.append(he)

            if high_interest_predicate(exp) and placement.get("feasible"):
                shuffles = oracle.shuffle_within_class(
                    jobs_payload, n_shuffles=n_variance_shuffles
                )
                for idx, shuffle_assign in enumerate(shuffles):
                    he_var = HierarchicalExperiment(
                        base=exp,
                        placement_class_vector=kappa,
                        node_assignment={"feasible": True, "assignments": shuffle_assign},
                        is_variance_shuffle=True,
                        shuffle_index=idx,
                    )
                    hier_experiments.append(he_var)

    return hier_experiments


# ===========================================================================
# SECTION 9 – Utilities: pretty-printing & summary
# ===========================================================================

PRINTS_SEP_WIDTH=140

def print_baseline_set(cfg: argparse.Namespace, baseline: list[SingleRun]) -> None:
    print(f"\n\033[34m{'='*PRINTS_SEP_WIDTH}")
    print(f"BASELINE SET  |B| = {len(baseline)}")
    print(f"{'='*PRINTS_SEP_WIDTH}\033[0m")
    by_strategy: dict[str, list[int]] = {}
    for run in baseline:
        by_strategy.setdefault(run.strategy.name, []).append(run.gpus)
    for name, gpus in by_strategy.items():
        print(f"  {name:20s}  gpus = {sorted(gpus)}")


def print_patterns(cfg: argparse.Namespace, patterns: list[TaggedPattern]) -> None:
    print(f"\n\033[34m{'='*PRINTS_SEP_WIDTH}")
    print(f"PATTERN SET  |P| = {len(patterns)}")
    print(f"{'='*PRINTS_SEP_WIDTH}\033[0m")
    for tp in patterns:
        s = sum(tp.slots)
        print(f"  [{tp.family}] {str(tp.slots):<80}  totGPUs={s:<4}  util={int(s/cfg.G*100):<3}%  k={len(tp.slots)}")


def print_experiment_set(
    cfg: argparse.Namespace,
    experiments: list[Experiment],
    title: str = "EXPERIMENT SET",
) -> None:
    print(f"\n\033[34m{'='*PRINTS_SEP_WIDTH}")
    print(f"{title}  |E| = {len(experiments)}")
    print(f"{'='*PRINTS_SEP_WIDTH}\033[0m")
    by_bin: dict[str, list[Experiment]] = {"low": [], "medium": [], "high": []}
    for exp in experiments:
        by_bin[exp.entropy_bin].append(exp)
    for bin_name in ("low", "medium", "high"):
        exps = by_bin[bin_name]
        if exps:
            print(f"\n  \033[33m── H-bin: {bin_name} ({len(exps)} experiments) ──\033[0m\n")
            for exp in exps:
                print(exp)


def print_hierarchical_experiment_set(
    cfg: argparse.Namespace,
    hier_experiments: list[HierarchicalExperiment],
) -> None:
    core = [e for e in hier_experiments if not e.is_variance_shuffle]
    var  = [e for e in hier_experiments if e.is_variance_shuffle]
    print(f"\n\033[34m{'='*PRINTS_SEP_WIDTH}")
    print(f"HIERARCHICAL EXPERIMENT SET  E_hier")
    print(f"  Core experiments:           {len(core)}")
    print(f"  Variance-shuffle runs:      {len(var)}")
    print(f"  Total:                      {len(hier_experiments)}")
    print(f"{'='*PRINTS_SEP_WIDTH}\033[0m")
    for e in core:
        print(e)


def print_summary(
    g_total: int,
    strategies: list[Strategy],
    baseline: list[SingleRun],
    patterns: list[TaggedPattern],
    experiments: list[Experiment],
    hier_experiments: Optional[list[HierarchicalExperiment]] = None,
) -> None:
    print(f"\n\033[34m{'='*PRINTS_SEP_WIDTH}")
    print("SUMMARY")
    print(f"{'='*PRINTS_SEP_WIDTH}\033[0m")
    print(f"  G (total GPUs)        : {g_total}")
    print(f"  Strategies  |S|       : {len(strategies)}")
    print(f"  Baseline set |B|      : {len(baseline)}")
    print(f"  Pattern set |P|       : {len(patterns)}")
    print(f"  Flat experiments |E|  : {len(experiments)}")
    if hier_experiments is not None:
        core = sum(1 for e in hier_experiments if not e.is_variance_shuffle)
        var  = sum(1 for e in hier_experiments if e.is_variance_shuffle)
        print(f"  Hier experiments      : {core} core + {var} variance shuffles")
    print(f"  O(log G) bound        : log₂({g_total}) ≈ {math.log2(g_total):.1f}")


# ===========================================================================
# SECTION 10 – JSON Serialization
# ===========================================================================

def _strategy_to_dict(s: Strategy) -> dict:
    return {"name": s.name, "feasible_gpu_counts": sorted(s.feasible)}


def _single_run_to_dict(r: SingleRun) -> dict:
    return {"strategy": r.strategy.name, "gpus": r.gpus}


def _config_to_dict(c: Config) -> dict:
    return {
        "runs": [_single_run_to_dict(r) for r in c.runs],
        "total_gpus": c.total_gpus,
        "utilization": round(c.utilization, 6),
        "k": c.k,
    }


def _experiment_to_dict(exp: Experiment) -> dict:
    return {
        "pattern": list(exp.pattern.slots),
        "pattern_family": exp.pattern.family,
        "pattern_sum": sum(exp.pattern.slots),
        "utilization": round(sum(exp.pattern.slots) / G, 6),
        "k": len(exp.pattern.slots),
        "entropy_bin": exp.entropy_bin,
        "labelling": [s.name for s in exp.labelling],
        "config": _config_to_dict(exp.config),
        "placement_class_vector": list(exp.placement) if exp.placement else None,
    }


def _hier_experiment_to_dict(he: HierarchicalExperiment) -> dict:
    d = _experiment_to_dict(he.base)
    d["placement_class_vector"] = list(he.placement_class_vector)
    d["node_assignment"] = he.node_assignment
    d["is_variance_shuffle"] = he.is_variance_shuffle
    d["shuffle_index"] = he.shuffle_index
    return d


def build_json_output(
    cfg: argparse.Namespace,
    strategies: list[Strategy],
    baseline: list[SingleRun],
    patterns: list[TaggedPattern],
    experiments: list[Experiment],
    hier_experiments: Optional[list[HierarchicalExperiment]] = None,
) -> dict:
    """
    Assemble the complete experiment design into a single JSON-serializable dict.

    Top-level keys
    --------------
    meta          : run metadata (timestamp, seed, G, …)
    parameters    : all tunable constants used during generation
    strategies    : the strategy set S with feasible GPU counts
    baseline_set  : B – all feasible isolated runs
    pattern_set   : P – all allocation patterns (with family tag)
    experiments   : E – flat experiment set (pattern + labelling + config)
    hierarchical_experiments : E_hier (only present when topology is enabled)
    summary       : scalar counts mirroring the printed summary
    """

    # ── meta ────────────────────────────────────────────────────────────────
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "G": cfg.G,
        "seed": cfg.seed,
        "use_topology": cfg.use_topology,
        "topology_program": cfg.topology_program if cfg.use_topology else None,
    }

    # ── parameters ──────────────────────────────────────────────────────────
    parameters = {
        "G": cfg.G,
        "g_min": cfg.g_min,
        "k_max": cfg.k_max,
        "geometric_beta": cfg.beta,
        "hierarchical_patterns": [
            {"tier_fractions": tf, "n_tiny": nt}
            for tf, nt in HIERARCHICAL_PATTERNS
        ],
        "powerlaw_alphas": cfg.powerlaw_alphas,
        "util_min": cfg.util_min,
        "util_max": cfg.util_max,
        "util_steps": cfg.util_steps,
        "utilizations": _utilization_grid(
            cfg.util_min, cfg.util_max, cfg.util_steps
        ),
        "stochastic_tier_config": STOCHASTIC_TIER_CONFIG,
        "n_stochastic_patterns": cfg.n_stochastic_patterns,
        "entropy_delta_1": cfg.entropy_delta_1,
        "entropy_delta_2": cfg.entropy_delta_2,
        "n_samples_per_bin": cfg.n_samples_per_bin,
        "enum_threshold": cfg.enum_threshold,
        "n_variance_shuffles": cfg.n_variance_shuffles if cfg.use_topology else None,
    }

    # ── strategies ──────────────────────────────────────────────────────────
    strategies_list = [_strategy_to_dict(s) for s in strategies]

    # ── baseline set ────────────────────────────────────────────────────────
    baseline_list = [_single_run_to_dict(r) for r in baseline]

    # ── pattern set ──────────────────────────────────────────────────────────
    pattern_list = []
    for tp in patterns:
        pattern_list.append({
            "slots": list(tp.slots),
            "k": len(tp.slots),
            "total_gpus": sum(tp.slots),
            "utilization": round(sum(tp.slots) / cfg.G, 6),
            "family": tp.family,
        })

    # ── flat experiments ─────────────────────────────────────────────────────
    experiments_list = [_experiment_to_dict(exp) for exp in experiments]

    # ── hierarchical experiments ─────────────────────────────────────────────
    hier_list = None
    if hier_experiments is not None:
        hier_list = [_hier_experiment_to_dict(he) for he in hier_experiments]

    # ── summary ──────────────────────────────────────────────────────────────
    summary: dict = {
        "n_strategies": len(strategies),
        "n_baseline_runs": len(baseline),
        "n_patterns": len(patterns),
        "n_flat_experiments": len(experiments),
        "log2_G": round(math.log2(cfg.G), 3),
    }
    if hier_experiments is not None:
        n_core = sum(1 for e in hier_experiments if not e.is_variance_shuffle)
        n_var  = sum(1 for e in hier_experiments if e.is_variance_shuffle)
        summary["n_hier_core"] = n_core
        summary["n_hier_variance_shuffles"] = n_var
        summary["n_hier_total"] = len(hier_experiments)

    doc = {
        "meta": meta,
        "parameters": parameters,
        "strategies": strategies_list,
        "baseline_set": baseline_list,
        "pattern_set": pattern_list,
        "experiments": experiments_list,
        "summary": summary,
    }
    if hier_list is not None:
        doc["hierarchical_experiments"] = hier_list

    return doc


def serialize_to_json(doc: dict, path: str) -> None:
    """Write the experiment design document to a JSON file."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    print(f"\n\033[32m[JSON] Experiment design serialized → {path}  "
          f"({len(doc['experiments'])} experiments)\033[0m")


# ===========================================================================
# MAIN  –  wire everything together
# ===========================================================================

def main(cfg: argparse.Namespace) -> None:
    rng = random.Random(cfg.seed)

    # 1. Strategies  (not CLI-configurable; edit STRATEGY_DEFS in the file)
    strategies = build_strategies(STRATEGY_DEFS)

    # 2. Baseline set
    baseline = build_baseline_set(strategies, cfg.G)
    print_baseline_set(cfg, baseline)

    # 2b. Feasible GPU counts – the universe of valid slot sizes
    feasible_gpu_counts = compute_feasible_gpu_counts(strategies, cfg.G)

    # 3. Pattern set
    utilizations = _utilization_grid(
        cfg.util_min, cfg.util_max, cfg.util_steps
    )
    patterns = build_pattern_set(
        g_total=cfg.G,
        k_max=cfg.k_max,
        g_min=cfg.g_min,
        beta=cfg.beta,
        hierarchical_defs=HIERARCHICAL_PATTERNS,
        powerlaw_alphas=cfg.powerlaw_alphas,
        utilizations=utilizations,
        util_min=cfg.util_min,
        util_max=cfg.util_max,
        stochastic_tier_config=STOCHASTIC_TIER_CONFIG,
        n_stochastic_patterns=cfg.n_stochastic_patterns,
        feasible_gpu_counts=feasible_gpu_counts,
        rng=rng,
        generate_equal_splits=not cfg.use_topology,
        generate_geometric=False,
        generate_hierarchical=False,
        generate_powerlaw=cfg.use_topology,
        generate_stochastic=True,
    )
    print_patterns(cfg, patterns)

    # 4–5. Flat experiment set E
    experiments = build_experiment_set(
        cfg=cfg,
        patterns=patterns,
        strategies=strategies,
        n_per_bin=cfg.n_samples_per_bin,
        delta1_frac=cfg.entropy_delta_1,
        delta2_frac=cfg.entropy_delta_2,
        enum_threshold=cfg.enum_threshold,
        rng=rng,
    )
    print_experiment_set(cfg, experiments)

    # 6–8. Hierarchical experiment set E_hier (if enabled)
    hier_experiments: Optional[list[HierarchicalExperiment]] = None
    if cfg.use_topology:
        oracle = TopologyOracle(program=cfg.topology_program)
        if not oracle._available:
            print(
                "\n[TopologyOracle] External program not found – using stub fallback.\n"
                f"  Program path: '{cfg.topology_program}'\n"
                "  To use your real oracle, pass --topology-program <path>."
            )
        hier_experiments = build_hierarchical_experiment_set(
            flat_experiments=experiments,
            oracle=oracle,
            n_variance_shuffles=cfg.n_variance_shuffles,
        )
        print_hierarchical_experiment_set(cfg, hier_experiments)

    # Summary
    print_summary(cfg.G, strategies, baseline, patterns, experiments, hier_experiments)

    # 10. JSON serialization
    doc = build_json_output(
        cfg=cfg,
        strategies=strategies,
        baseline=baseline,
        patterns=patterns,
        experiments=experiments,
        hier_experiments=hier_experiments,
    )
    serialize_to_json(doc, cfg.output_json)


# ===========================================================================
# CLI  –  argument parsing
# ===========================================================================

def _positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return n


def _unit_float(value: str) -> float:
    f = float(value)
    if not (0.0 < f < 1.0):
        raise argparse.ArgumentTypeError(f"must be in (0, 1), got {value}")
    return f


def _fraction_float(value: str) -> float:
    f = float(value)
    if not (0.0 <= f <= 1.0):
        raise argparse.ArgumentTypeError(f"must be in [0, 1], got {value}")
    return f


def _utilization_grid(
    util_min: float, util_max: float, steps: int
) -> list[float]:
    """
    Build a uniform grid of `steps` utilization values in [util_min, util_max].

    - steps == 1  →  [util_max]  (single point; util_min is ignored)
    - steps >= 2  →  linspace(util_min, util_max, steps), both endpoints included
    """
    if steps < 1:
        raise ValueError(f"--util-steps must be ≥ 1, got {steps}")
    if not (0.0 < util_min <= util_max <= 1.0):
        raise ValueError(
            f"Utilization range must satisfy 0 < util_min ≤ util_max ≤ 1, "
            f"got [{util_min}, {util_max}]"
        )
    if steps == 1:
        return [util_max]
    return [
        util_min + (util_max - util_min) * i / (steps - 1)
        for i in range(steps)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="experiment_design.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Multi-Training Congestion – Experiment Design Generator
=======================================================
Generates structured experiment sets for studying interference between
concurrent distributed training jobs, following the formalization in:
  "Multi-Training Congestion: Experiment Design Formalization" (Pasquali).

STRATEGY_DEFS, HIERARCHICAL_PATTERNS, and STOCHASTIC_TIER_CONFIG can only
be changed by editing the CONFIG block at the top of this file.
""",
        epilog="""
examples:
  # Minimal: only G is required
  python experiment_design.py --G 8

  # DGX H100 node, no topology analysis, fixed seed
  python experiment_design.py --G 8 --no-topology --seed 0

  # GB200 NVL72 cluster, more samples per entropy bin
  python experiment_design.py --G 72 --n-samples-per-bin 3

  # Custom decay, wider entropy bins, real oracle
  python experiment_design.py --G 72 --beta 0.33 \\
      --entropy-delta-1 0.25 --entropy-delta-2 0.75 \\
      --topology-program /usr/local/bin/topology_oracle

  # Larger cluster with more concurrent jobs allowed
  python experiment_design.py --G 512 --k-max 8 --g-min 4

  # Custom output path for the serialized experiment set
  python experiment_design.py --G 72 --output-json my_experiments.json
""",
    )

    # ── Required ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--G", "-G",
        required=True,
        type=_positive_int,
        metavar="N",
        help="Total number of GPUs in the cluster (e.g. 8, 72, 512). REQUIRED.",
    )

    # ── Pattern generation ────────────────────────────────────────────────────
    pattern_group = parser.add_argument_group(
        "pattern generation",
        "Controls GPU allocation pattern families (Sections 3A/B/C/D/E).",
    )
    pattern_group.add_argument(
        "--k-max",
        type=_positive_int,
        default=K_MAX,
        metavar="K",
        help=f"Maximum number of concurrent jobs per configuration (default: {K_MAX}).",
    )
    pattern_group.add_argument(
        "--g-min",
        type=_positive_int,
        default=G_MIN,
        metavar="G",
        help=f"Minimum GPU slot size; smaller slots are dropped (default: {G_MIN}).",
    )
    pattern_group.add_argument(
        "--beta",
        type=_unit_float,
        default=GEOMETRIC_BETA,
        metavar="β",
        help=(
            f"Geometric decay factor β ∈ (0,1) for family-B patterns (default: {GEOMETRIC_BETA}). "
            "Each successive slot size is floor(β × previous)."
        ),
    )
    pattern_group.add_argument(
        "--powerlaw-alphas",
        type=float,
        nargs="+",
        default=POWERLAW_ALPHAS,
        metavar="α",
        help=(
            f"Tail exponent(s) α > 1 for family-D power-law patterns "
            f"(default: {POWERLAW_ALPHAS}). "
        ),
    )
    pattern_group.add_argument(
        "--util-min",
        type=_fraction_float,
        default=UTIL_MIN,
        metavar="ρ_min",
        help=(
            f"Minimum utilization ρ ∈ (0,1] for pattern generation and filtering "
            f"(default: {UTIL_MIN}). Patterns whose actual sum(slots)/G falls below "
            "this threshold are dropped. Also controls the lower bound of the "
            "generation sweep grid when --util-steps > 1."
        ),
    )
    pattern_group.add_argument(
        "--util-max",
        type=_fraction_float,
        default=UTIL_MAX,
        metavar="ρ_max",
        help=(
            f"Maximum utilization ρ ∈ (0,1] for the shared sweep grid "
            f"(default: {UTIL_MAX})."
        ),
    )
    pattern_group.add_argument(
        "--util-steps",
        type=_positive_int,
        default=UTIL_STEPS,
        metavar="N",
        help=(
            f"Number of utilization steps in [ρ_min, ρ_max] shared across all "
            f"pattern families (default: {UTIL_STEPS}). "
            "1 → only ρ_max; 2 → {ρ_min, ρ_max}; N ≥ 2 → uniform linspace."
        ),
    )
    pattern_group.add_argument(
        "--n-stochastic-patterns",
        type=_positive_int,
        default=N_STOCHASTIC_PATTERNS,
        metavar="N",
        help=(
            f"Number of distinct family-E stochastic patterns to generate "
            f"(default: {N_STOCHASTIC_PATTERNS}). "
            "Each draw is an independent realisation of the tier-sampling process. "
            "Duplicate patterns are discarded and retried."
        ),
    )

    # ── Entropy-stratified sampling ───────────────────────────────────────────
    entropy_group = parser.add_argument_group(
        "entropy-stratified sampling",
        "Controls how strategy labellings are sampled per pattern (Section 4).",
    )
    entropy_group.add_argument(
        "--n-samples-per-bin",
        type=_positive_int,
        default=N_SAMPLES_PER_BIN,
        metavar="N",
        help=f"Labellings to sample per entropy bin per pattern (default: {N_SAMPLES_PER_BIN}).",
    )
    entropy_group.add_argument(
        "--entropy-delta-1",
        type=_fraction_float,
        default=ENTROPY_DELTA_1,
        metavar="δ1",
        help=f"Low/medium entropy bin boundary as fraction of log(m) (default: {ENTROPY_DELTA_1}).",
    )
    entropy_group.add_argument(
        "--entropy-delta-2",
        type=_fraction_float,
        default=ENTROPY_DELTA_2,
        metavar="δ2",
        help=f"Medium/high entropy bin boundary as fraction of log(m) (default: {ENTROPY_DELTA_2}).",
    )
    entropy_group.add_argument(
        "--enum-threshold",
        type=int,
        default=ENUM_THRESHOLD,
        metavar="N",
        help=(
            f"Maximum |Φ(P)| for exact enumeration (default: {ENUM_THRESHOLD:,}). "
            "Set to 0 to always use rejection sampling."
        ),
    )

    # ── Reproducibility ───────────────────────────────────────────────────────
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        metavar="S",
        help=f"Random seed for reproducible sampling (default: {RANDOM_SEED}).",
    )

    # ── Topology ──────────────────────────────────────────────────────────────
    topo_group = parser.add_argument_group(
        "topology / placement",
        "Controls hierarchical placement-class sampling (Sections 6–8).",
    )
    topo_exclusive = topo_group.add_mutually_exclusive_group()
    topo_exclusive.add_argument(
        "--use-topology",
        dest="use_topology",
        action="store_true",
        default=USE_TOPOLOGY,
        help="Enable hierarchical placement analysis.",
    )
    topo_exclusive.add_argument(
        "--no-topology",
        dest="use_topology",
        action="store_false",
        help="Skip placement analysis; produce the flat experiment set E only.",
    )
    topo_group.add_argument(
        "--topology-program",
        default=TOPOLOGY_PROGRAM,
        metavar="PATH",
        help=f"Path to the external topology oracle program (default: '{TOPOLOGY_PROGRAM}').",
    )
    topo_group.add_argument(
        "--n-variance-shuffles",
        type=_positive_int,
        default=N_VARIANCE_SHUFFLES,
        metavar="N",
        help=f"Within-class variance shuffles for high-interest configs (default: {N_VARIANCE_SHUFFLES}).",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output-json",
        default=DEFAULT_JSON_OUTPUT,
        metavar="PATH",
        help=f"Path for the serialized JSON experiment design (default: '{DEFAULT_JSON_OUTPUT}').",
    )

    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.entropy_delta_1 >= args.entropy_delta_2:
        parser.error(
            f"--entropy-delta-1 ({args.entropy_delta_1}) must be strictly less than "
            f"--entropy-delta-2 ({args.entropy_delta_2})."
        )
    if args.g_min > args.G:
        parser.error(f"--g-min ({args.g_min}) cannot exceed --G ({args.G}).")
    if args.enum_threshold < 0:
        parser.error(f"--enum-threshold ({args.enum_threshold}) must be >= 0.")
    for a in args.powerlaw_alphas:
        if a <= 1.0:
            parser.error(
                f"--powerlaw-alphas: each α must be > 1 for a normalizable power law, got {a}."
            )
    if args.util_min > args.util_max:
        parser.error(
            f"--util-min ({args.util_min}) must be ≤ "
            f"--util-max ({args.util_max})."
        )
    if args.util_min <= 0.0:
        parser.error(
            f"--util-min ({args.util_min}) must be > 0."
        )


if __name__ == "__main__":
    _parser = build_parser()
    _args = _parser.parse_args()
    _validate_args(_args, _parser)
    main(_args)