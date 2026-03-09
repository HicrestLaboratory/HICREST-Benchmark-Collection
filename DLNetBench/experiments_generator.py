"""
Multi-Training Congestion: Experiment Design Framework
======================================================
Python implementation of the formalization by Thomas Pasquali.

Structure:
  - Section 1:  System model (GPU count, strategies, feasibility sets)
  - Section 2:  Baseline set B  (extended with topology placements when enabled)
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
STRATEGY_DEFS: list[tuple[str, list[int]]] = [
    ("DP",           [2, 4, 8, 16]),
    ("FSDP",         [4, 8, 16, 32]),
    ("DP+PP",        [4, 8, 16, 32, 64]),
    ("DP+PP+Expert", [64, 128, 192, 256, 320, 384, 448, 512]),
    ("DP+PP+TP",     [320, 640, 960]),
]

# ── Pattern generation ───────────────────────────────────────────────────────
G_MIN: int = 2
K_MAX: int = math.floor(G / G_MIN)
GEOMETRIC_BETA: float = 0.5

HIERARCHICAL_PATTERNS: list[tuple[list[float], int]] = [
    ([0.50, 0.25], 4),
    ([0.05, 0.21], 6),
    ([0.89],       4),
]

POWERLAW_ALPHAS: list[float] = [1.2]

UTIL_MIN:   float = 0.8
UTIL_MAX:   float = 1.0
UTIL_STEPS: int   = 1

STOCHASTIC_TIER_CONFIG: dict[str, dict] = {
    "small": {
        "tier_weight": 0.75,
        "sizes": [2, 4],
        "sub_weights": {},
    },
    "medium": {
        "tier_weight": 0.20,
        "sizes": [8, 16],
        "sub_weights": {},
    },
    "large": {
        "tier_weight": 0.05,
        "sizes": [32, 64], # FIXME
        "sub_weights": {},
    },
}

N_STOCHASTIC_PATTERNS: int = 5

# ── Entropy-stratified sampling ──────────────────────────────────────────────
ENTROPY_DELTA_1: float = 0.3
ENTROPY_DELTA_2: float = 0.7
N_SAMPLES_PER_BIN: int = 4
ENUM_THRESHOLD: int = 50_000
RANDOM_SEED: Optional[int] = 42

# ── Experiment list size bounds ───────────────────────────────────────────────
# MIN_EXPERIMENTS: lower bound on the output list.  An error is raised if the
#   generator produces fewer experiments than this value.  Set to 0 to disable.
# MAX_EXPERIMENTS: upper bound.  When the generated set exceeds this value,
#   experiments are trimmed with a round-robin strategy that preserves variety
#   across pattern families, entropy bins, and (with topology) placement bins.
#   Set to None to disable.
MIN_EXPERIMENTS: int = 0
MAX_EXPERIMENTS: Optional[int] = None

# ── Topology (Section 6) ─────────────────────────────────────────────────────
USE_TOPOLOGY: bool = False

# Path / command of the external topology program.
# The oracle must support exactly two commands:
#   find_placement        – assign jobs to nodes given target placement classes
#   shuffle_within_class  – return N random same-class alternative node assignments
# Topology structure is passed statically (TOPO_Q1/Q2/Q3); no topology-describe
# query is ever issued to the external program.
TOPOLOGY_PROGRAM: str = "topology_oracle"

# Static topology parameters – describe the cluster hierarchy without any
# external query.
TOPO_Q1: int = 4    # GPUs per intra-node NVLink domain
TOPO_Q2: int = 64   # GPUs per L1-switch / rack domain
TOPO_Q3: int = G    # GPUs per group (set equal to G for single-group clusters)

# Whether to include intra-node as a valid placement class.
# Excluded by default: in multi-job interference studies jobs rarely share a node,
# so the intra-node class adds no practical coverage.
INCLUDE_INTRA_NODE: bool = False

# Number of additional node-selection replicates per baseline run for
# intra-group and inter-group placement classes.
# For each (S_i, g) in B the generator produces:
#   1                      x intra-L1    (one canonical representative)
#   N_BASELINE_TOPO_REPS   x intra-group (different physical node selections)
#   N_BASELINE_TOPO_REPS   x inter-group (different physical node selections)
N_BASELINE_TOPO_REPS: int = 2

# Number of alternative node assignments to generate per (experiment, placement
# class vector) pair.  These are recorded as (vector, seed) pairs only – actual
# nodelists are resolved later by expand_experiments.py.
N_PLACEMENT_SEEDS_PER_VECTOR: int = 1

# Placement-vector scoring and bin boundaries.
# Each placement class receives a numeric score:
#   intra-node  -> 0  (excluded by default)
#   intra-L1    -> 1
#   intra-group -> 2
#   inter-group -> 3
# A placement-class vector κ = (κ_1, …, κ_k) is scored by
#   mean_score(κ) = (sum_j score(κ_j)) / k  ∈ [1, 3]
# and partitioned into three locality bins:
#   low    [1.00, PLACEMENT_BIN_LO_HI)  – mostly intra-L1
#   medium [PLACEMENT_BIN_LO_HI, PLACEMENT_BIN_MED_HI)
#   high   [PLACEMENT_BIN_MED_HI, 3.00] – mostly inter-group
PLACEMENT_BIN_LO_HI:  float = 1.67
PLACEMENT_BIN_MED_HI: float = 2.33

# Placement-class vectors to sample per locality bin per experiment.
N_PLACEMENT_SAMPLES_PER_BIN: int = 2

# ── JSON output ──────────────────────────────────────────────────────────────
DEFAULT_JSON_OUTPUT: str = "experiments.json"


# ===========================================================================
# SECTION 1 – Data structures
# ===========================================================================

@dataclass(frozen=True)
class Strategy:
    name: str
    feasible: frozenset[int]

    def supports(self, g: int) -> bool:
        return g in self.feasible


@dataclass(frozen=True)
class SingleRun:
    strategy: Strategy
    gpus: int

    def __str__(self) -> str:
        return f"({self.strategy.name}:{self.gpus})"


@dataclass
class Config:
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
        return f"[{slots}]\n  util={int(self.utilization*100):3}%"


# ===========================================================================
# SECTION 2 – Baseline set  B
# ===========================================================================

def build_strategies(defs: list[tuple[str, list[int]]]) -> list[Strategy]:
    return [Strategy(name, frozenset(gpus)) for name, gpus in defs]


def build_baseline_set(strategies: list[Strategy], g_total: int) -> list[SingleRun]:
    baseline: list[SingleRun] = []
    for s in strategies:
        for g in sorted(s.feasible):
            if g < g_total:
                baseline.append(SingleRun(s, g))
    return baseline


@dataclass
class BaselineWithPlacement:
    """
    A baseline run annotated with a topology placement class and a seed.

    When USE_TOPOLOGY is enabled every (S_i, g) in B is replicated across:
      1                    x intra-L1    (canonical, seed=0)
      N_BASELINE_TOPO_REPS x intra-group (seeds 1..N)
      N_BASELINE_TOPO_REPS x inter-group (seeds 1..N)
    intra-node is added only if INCLUDE_INTRA_NODE is True.

    The 'seed' field is forwarded to the topology oracle's shuffle_within_class
    command when nodelists are generated later, guaranteeing reproducibility
    without storing actual nodelists at design time.
    """
    run: SingleRun
    placement_class: str
    replicate_index: int   # 0 = canonical; 1..N = variance replicates
    seed: int              # reproducibility seed for later nodelist generation

    def __str__(self) -> str:
        return (
            f"  {str(self.run):<30}  class={self.placement_class:<14}  "
            f"rep={self.replicate_index}  seed={self.seed}"
        )


def build_baseline_set_with_topology(
    strategies: list[Strategy],
    g_total: int,
    oracle: "TopologyOracle",
    n_baseline_topo_reps: int,
    include_intra_node: bool,
    base_seed: int = 0,
) -> list[BaselineWithPlacement]:
    """
    Build the topology-extended baseline set.

    No nodelists are stored; each entry carries a 'seed' for later
    reproducible nodelist generation via the oracle.  The oracle is
    instantiated here only to read topology_params; no query is issued.
    """
    flat_baseline = build_baseline_set(strategies, g_total)
    result: list[BaselineWithPlacement] = []
    seed_ctr = base_seed

    for run in flat_baseline:
        classes_in_scope = _placement_classes_in_scope(
            run.gpus, g_total, oracle, include_intra_node
        )
        for pc in classes_in_scope:
            # canonical representative (rep 0)
            result.append(BaselineWithPlacement(
                run=run,
                placement_class=pc,
                replicate_index=0,
                seed=seed_ctr,
            ))
            seed_ctr += 1

            # variance replicates only for intra-group and inter-group
            n_reps = (
                n_baseline_topo_reps
                if pc in (PlacementClass.INTRA_GROUP, PlacementClass.INTER_GROUP)
                else 0
            )
            for rep in range(1, n_reps + 1):
                result.append(BaselineWithPlacement(
                    run=run,
                    placement_class=pc,
                    replicate_index=rep,
                    seed=seed_ctr,
                ))
                seed_ctr += 1

    return result


# ===========================================================================
# SECTION 3 – GPU Allocation Patterns
# ===========================================================================

AllocationPattern = tuple[int, ...]


@dataclass(frozen=True)
class TaggedPattern:
    slots: AllocationPattern
    family: str

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
    counts: set[int] = set()
    for s in strategies:
        for g in s.feasible:
            if g < g_total:
                counts.add(g)
    return frozenset(counts)


def _snap_down(raw: int, feasible: frozenset[int]) -> Optional[int]:
    candidates = [g for g in feasible if g <= raw]
    return max(candidates) if candidates else None


def pattern_A_equal_splits(
    g_total: int, k_max: int, g_min: int,
    feasible_gpu_counts: frozenset[int], utilization: float = 1.0,
) -> list[TaggedPattern]:
    budget = int(math.floor(utilization * g_total))
    patterns: list[TaggedPattern] = []
    for k in range(2, k_max + 1):
        if budget % k == 0:
            g = budget // k
            if g >= g_min and g in feasible_gpu_counts:
                patterns.append(TaggedPattern(slots=tuple([g] * k), family="A"))
    return patterns


def pattern_B_geometric(
    g_total: int, k_max: int, g_min: int, beta: float,
    feasible_gpu_counts: frozenset[int], utilization: float = 1.0,
) -> list[TaggedPattern]:
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
    g_total: int, k_max: int, g_min: int,
    tier_fractions: list[float], n_tiny: int,
    feasible_gpu_counts: frozenset[int], utilization: float = 1.0,
) -> Optional[TaggedPattern]:
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
    if len(slots) < 2 or len(slots) > k_max or sum(slots) > g_total:
        return None
    return TaggedPattern(slots=tuple(slots), family="C")


def pattern_D_powerlaw(
    g_total: int, k: int, g_min: int, alpha: float,
    utilization: float, feasible_gpu_counts: frozenset[int],
) -> Optional[TaggedPattern]:
    if alpha <= 1.0:
        raise ValueError(f"alpha must be > 1, got {alpha}")
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


def _tier_size_weights(
    tier_name: str, tier_cfg: dict, feasible_gpu_counts: frozenset[int],
) -> tuple[list[int], list[float]]:
    all_sizes: list[int] = tier_cfg["sizes"]
    sub_weights_map: dict[int, float] = tier_cfg.get("sub_weights", {})
    sizes = [s for s in all_sizes if s in feasible_gpu_counts]
    if not sizes:
        raise ValueError(f"Tier '{tier_name}' has no feasible sizes.")
    pinned = {s: sub_weights_map[s] for s in sizes if s in sub_weights_map}
    free = [s for s in sizes if s not in pinned]
    total_pinned = sum(pinned.values())
    if total_pinned > 1.0 + 1e-9:
        raise ValueError(f"Sub-weights for tier '{tier_name}' exceed 1.0.")
    rem_prob = (1.0 - total_pinned) / len(free) if free else 0.0
    weights = [pinned[s] if s in pinned else rem_prob for s in sizes]
    total_w = sum(weights)
    if total_w <= 0:
        raise ValueError(f"All weights zero for tier '{tier_name}'.")
    return sizes, [w / total_w for w in weights]


def pattern_E_stochastic_tier(
    g_total: int, k_max: int, g_min: int,
    tier_config: dict[str, dict], feasible_gpu_counts: frozenset[int],
    rng: random.Random, utilization: float = 1.0,
) -> Optional[TaggedPattern]:
    tier_names = list(tier_config.keys())
    total_tw = sum(tier_config[t]["tier_weight"] for t in tier_names)
    if abs(total_tw - 1.0) > 1e-9:
        raise ValueError("Tier weights must sum to 1.0.")

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
            pass
    if not surviving_tiers:
        return None
    total_stw = sum(surviving_tw)
    surviving_tw = [w / total_stw for w in surviving_tw]

    budget = int(math.floor(utilization * g_total))
    slots: list[int] = []
    remaining = budget
    while remaining >= g_min and len(slots) < k_max:
        any_fits = any(s <= remaining for t in surviving_tiers for s in tier_size_info[t][0])
        if not any_fits:
            break
        tier = rng.choices(surviving_tiers, weights=surviving_tw, k=1)[0]
        sizes, weights = tier_size_info[tier]
        g = rng.choices(sizes, weights=weights, k=1)[0]
        if g <= remaining:
            slots.append(g)
            remaining -= g
    if len(slots) < 2:
        return None
    return TaggedPattern(slots=tuple(sorted(slots, reverse=True)), family="E")


def pattern_E_stochastic_tier_batch(
    g_total: int, k_max: int, g_min: int,
    tier_config: dict[str, dict], feasible_gpu_counts: frozenset[int],
    n_patterns: int, rng: random.Random, utilization: float = 1.0,
) -> list[TaggedPattern]:
    seen: set[AllocationPattern] = set()
    patterns: list[TaggedPattern] = []
    for _ in range(10 * n_patterns):
        if len(patterns) >= n_patterns:
            break
        p = pattern_E_stochastic_tier(
            g_total, k_max, g_min, tier_config, feasible_gpu_counts, rng, utilization=utilization,
        )
        if p is not None and p.slots not in seen:
            seen.add(p.slots)
            patterns.append(p)
    return patterns


def build_pattern_set(
    g_total: int, k_max: int, g_min: int, beta: float,
    hierarchical_defs: list[tuple[list[float], int]],
    powerlaw_alphas: list[float], utilizations: list[float],
    util_min: float, util_max: float,
    stochastic_tier_config: dict[str, dict], n_stochastic_patterns: int,
    feasible_gpu_counts: frozenset[int], rng: random.Random,
    generate_equal_splits: bool, generate_geometric: bool,
    generate_hierarchical: bool, generate_powerlaw: bool,
    generate_stochastic: bool,
) -> list[TaggedPattern]:
    seen: set[AllocationPattern] = set()
    patterns: list[TaggedPattern] = []

    def add(tp: TaggedPattern) -> None:
        actual_util = sum(tp.slots) / g_total
        if (tp.slots not in seen and sum(tp.slots) <= g_total
                and util_min <= actual_util <= util_max):
            seen.add(tp.slots)
            patterns.append(tp)

    for rho in utilizations:
        if generate_equal_splits:
            for tp in pattern_A_equal_splits(g_total, k_max, g_min, feasible_gpu_counts, rho):
                add(tp)
        if generate_geometric:
            for tp in pattern_B_geometric(g_total, k_max, g_min, beta, feasible_gpu_counts, rho):
                add(tp)
        if generate_hierarchical:
            for tf, nt in hierarchical_defs:
                tp = pattern_C_hierarchical(g_total, k_max, g_min, tf, nt, feasible_gpu_counts, rho)
                if tp is not None:
                    add(tp)
        if generate_powerlaw:
            for alpha in powerlaw_alphas:
                for k in range(2, k_max + 1):
                    tp = pattern_D_powerlaw(g_total, k, g_min, alpha, rho, feasible_gpu_counts)
                    if tp is not None:
                        add(tp)
        if generate_stochastic:
            for tp in pattern_E_stochastic_tier_batch(
                g_total, k_max, g_min, stochastic_tier_config,
                feasible_gpu_counts, n_stochastic_patterns, rng, utilization=rho,
            ):
                add(tp)
    return patterns


# ===========================================================================
# SECTION 4 – Strategy Assignment & Entropy-Stratified Sampling
# ===========================================================================

def feasible_strategies_for_slot(strategies: list[Strategy], g: int) -> list[Strategy]:
    return [s for s in strategies if s.supports(g)]


def mixture_entropy(labelling: list[Strategy], pattern: AllocationPattern) -> float:
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


def entropy_bin(h: float, m_feasible: int, k: int,
                delta1_frac: float, delta2_frac: float) -> str:
    effective_m = min(k, m_feasible)
    h_max = math.log(effective_m) if effective_m > 1 else 1.0
    d1, d2 = delta1_frac * h_max, delta2_frac * h_max
    if h < d1:
        return "low"
    elif h < d2:
        return "medium"
    return "high"


def _labelling_space_size(per_slot: list[list[Strategy]], threshold: int) -> int:
    size = 1
    for opts in per_slot:
        size *= len(opts)
        if size > threshold:
            return size
    return size


def _random_labelling(per_slot: list[list[Strategy]], rng: random.Random) -> list[Strategy]:
    return [rng.choice(opts) for opts in per_slot]


def _sample_exact(per_slot, pattern, n_per_bin, m_feasible, k, d1, d2, rng):
    BIN_NAMES = ("low", "medium", "high")
    bins: dict[str, list] = {b: [] for b in BIN_NAMES}
    for combo in itertools.product(*per_slot):
        lab = list(combo)
        b = entropy_bin(mixture_entropy(lab, pattern), m_feasible, k, d1, d2)
        bins[b].append(lab)
    result = []
    for bn in BIN_NAMES:
        for lab in rng.sample(bins[bn], min(n_per_bin, len(bins[bn]))):
            result.append((lab, bn))
    return result


def _sample_rejection(per_slot, pattern, n_per_bin, m_feasible, k, d1, d2, rng):
    BIN_NAMES = ("low", "medium", "high")
    bins: dict[str, list] = {b: [] for b in BIN_NAMES}
    seen: set[tuple[str, ...]] = set()
    budget = n_per_bin * 200 * len(BIN_NAMES)
    for _ in range(budget):
        if all(len(bins[b]) >= n_per_bin for b in BIN_NAMES):
            break
        lab = _random_labelling(per_slot, rng)
        key = tuple(s.name for s in lab)
        if key in seen:
            continue
        b = entropy_bin(mixture_entropy(lab, pattern), m_feasible, k, d1, d2)
        if len(bins[b]) < n_per_bin:
            bins[b].append(lab)
            seen.add(key)
    result = []
    for bn in BIN_NAMES:
        for lab in bins[bn]:
            result.append((lab, bn))
    return result


def sample_labellings_stratified(
    pattern: TaggedPattern, strategies: list[Strategy],
    n_per_bin: int, delta1_frac: float, delta2_frac: float,
    enum_threshold: int, rng: random.Random,
) -> list[tuple[list[Strategy], str]]:
    per_slot = [feasible_strategies_for_slot(strategies, g) for g in pattern.slots]
    if any(len(opts) == 0 for opts in per_slot):
        return []
    m_feasible = len({s.name for opts in per_slot for s in opts})
    k = len(pattern.slots)
    space = _labelling_space_size(per_slot, enum_threshold)
    if space <= enum_threshold:
        return _sample_exact(per_slot, pattern.slots, n_per_bin, m_feasible, k,
                             delta1_frac, delta2_frac, rng)
    return _sample_rejection(per_slot, pattern.slots, n_per_bin, m_feasible, k,
                             delta1_frac, delta2_frac, rng)


# ===========================================================================
# SECTION 5 – Final Experiment Set  E
# ===========================================================================

@dataclass
class Experiment:
    pattern: TaggedPattern
    labelling: list[Strategy]
    entropy_bin: str
    config: Config

    def __str__(self) -> str:
        return (
            f"  pattern={str(self.pattern.slots)}\n"
            f"  H-bin={self.entropy_bin}\n"
            f"  config={str(self.config)}\n"
        )


def build_experiment_set(
    cfg: argparse.Namespace, patterns: list[TaggedPattern],
    strategies: list[Strategy], n_per_bin: int,
    delta1_frac: float, delta2_frac: float,
    enum_threshold: int, rng: random.Random,
) -> list[Experiment]:
    experiments: list[Experiment] = []
    for pattern in patterns:
        sampled = sample_labellings_stratified(
            pattern, strategies, n_per_bin, delta1_frac, delta2_frac,
            enum_threshold, rng
        )
        for labelling, bin_label in sampled:
            runs = [SingleRun(s, g) for s, g in zip(labelling, pattern.slots)]
            experiments.append(Experiment(
                pattern=pattern, labelling=labelling,
                entropy_bin=bin_label, config=Config(runs, cfg),
            ))
    return experiments


# ===========================================================================
# SECTION 6 – Network Topology Model & Placement Classification
# ===========================================================================

class PlacementClass:
    """
    Placement classes and their numeric scores for vector scoring.

      intra-node  -> score 0  (excluded by default; set INCLUDE_INTRA_NODE=True to enable)
      intra-L1    -> score 1  (all GPUs within one L1-switch / rack domain)
      intra-group -> score 2  (spans multiple L1 domains within one group)
      inter-group -> score 3  (spans multiple group domains)

    A placement-class vector κ = (κ_1, …, κ_k) is scored by
      mean_score(κ) = (sum_j score(κ_j)) / k
    which lies in [1, 3] when intra-node is excluded.
    The score partitions κ into low / medium / high locality bins.
    """
    INTRA_NODE  = "intra-node"
    INTRA_L1    = "intra-L1"
    INTRA_GROUP = "intra-group"
    INTER_GROUP = "inter-group"

    # Short names used in experiment output JSONs consumed by expand_experiments.py
    SHORT_NAME: dict[str, str] = {
        "intra-node":  "intra-node",  # excluded by default
        "intra-L1":    "intra-L1",
        "intra-group": "intra-group",
        "inter-group": "inter-group",
    }

    SCORE: dict[str, float] = {
        "intra-node":  0.0,
        "intra-L1":    1.0,
        "intra-group": 2.0,
        "inter-group": 3.0,
    }


PlacementClassVector = tuple[str, ...]


def placement_vector_score(kappa: PlacementClassVector) -> float:
    """
    mean_score(κ) = (Σ_j score(κ_j)) / k.

    With scores intra-L1→1, intra-group→2, inter-group→3 the result lies
    in [1, 3] for vectors that exclude intra-node.

    Example: κ = (intra-L1 ×2, intra-group ×4, inter-group ×10)
      → mean_score = (2·1 + 4·2 + 10·3) / 16 = 2.5  →  high bin.
    """
    if not kappa:
        return 0.0
    return sum(PlacementClass.SCORE.get(c, 0.0) for c in kappa) / len(kappa)


def placement_bin(score: float, lo_hi: float, med_hi: float) -> str:
    """
    Assign a mean placement score to a locality bin.
      low    [1.00, lo_hi)  – mostly intra-L1
      medium [lo_hi, med_hi)
      high   [med_hi, 3.00] – mostly inter-group
    """
    if score < lo_hi:
        return "low"
    elif score < med_hi:
        return "medium"
    return "high"


def _placement_classes_in_scope(
    gpu_count: int, g_total: int,
    oracle: "TopologyOracle", include_intra_node: bool,
) -> list[str]:
    """Feasible placement classes for a job of gpu_count GPUs."""
    q1 = oracle.topology_params["q1"]
    q2 = oracle.topology_params["q2"]
    q3 = oracle.topology_params["q3"]
    feasible: list[str] = []
    if include_intra_node and gpu_count <= q1:
        feasible.append(PlacementClass.INTRA_NODE)
    if gpu_count <= q2:
        feasible.append(PlacementClass.INTRA_L1)
    if gpu_count <= q3:
        feasible.append(PlacementClass.INTRA_GROUP)
    feasible.append(PlacementClass.INTER_GROUP)
    return feasible


# ===========================================================================
# SECTION 7 – TopologyOracle
# ===========================================================================

class TopologyOracle:
    """
    Thin wrapper around the external topology analysis program.

    Minimal external interface (two commands only)
    ----------------------------------------------
    The oracle communicates over stdin/stdout with newline-delimited JSON.

    1. find_placement
       Input:  { "command": "find_placement",
                 "jobs": [ { "job_id": <int>,
                             "gpu_count": <int>,
                             "placement_class": <str> }, … ] }
       Output: { "feasible": <bool>,
                 "assignments": [ { "job_id": <int>,
                                    "nodes": [ <str>, … ] }, … ] }

    2. shuffle_within_class
       Input:  { "command": "shuffle_within_class",
                 "jobs": [ { "job_id": <int>,
                             "gpu_count": <int>,
                             "placement_class": <str> }, … ],
                 "n_shuffles": <int> }
       Output: { "shuffles": [ [ { "job_id": <int>,
                                   "nodes": [ <str>, … ] }, … ], … ] }

    The topology structure (q1, q2, q3) is passed to the constructor and
    described statically – no topology-describe command is ever issued to
    the external program, keeping the required interface minimal.
    """

    def __init__(
        self,
        program: str = TOPOLOGY_PROGRAM,
        q1: int = TOPO_Q1,
        q2: int = TOPO_Q2,
        q3: int = TOPO_Q3,
    ) -> None:
        self.program = program
        self.topology_params = {"q1": q1, "q2": q2, "q3": q3}
        self._available = self._check_available()

    def _check_available(self) -> bool:
        try:
            result = subprocess.run([self.program, "--ping"], capture_output=True, timeout=2)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _query(self, payload: dict) -> dict:
        if not self._available:
            return self._stub_response(payload)
        try:
            proc = subprocess.run(
                [self.program], input=json.dumps(payload),
                capture_output=True, text=True, timeout=10,
            )
            return json.loads(proc.stdout)
        except Exception as exc:
            print(f"[TopologyOracle] Warning: query failed ({exc}), using stub.")
            return self._stub_response(payload)

    def _stub_response(self, payload: dict) -> dict:
        cmd = payload.get("command", "")
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
                 "nodes": [f"node-{j['job_id']}-shuf-{n}"
                           for _ in range(max(1, j["gpu_count"] // 8))]}
                for j in jobs
            ]
            return {"shuffles": [single] * n}
        return {}

    def find_placement(self, jobs: list[dict]) -> dict:
        return self._query({"command": "find_placement", "jobs": jobs})

    def shuffle_within_class(self, jobs: list[dict], n_shuffles: int = 3) -> list[list[dict]]:
        resp = self._query({"command": "shuffle_within_class",
                            "jobs": jobs, "n_shuffles": n_shuffles})
        return resp.get("shuffles", [])


# ===========================================================================
# SECTION 8 – Extended Experiment Set  E_hier
# ===========================================================================

@dataclass
class HierarchicalExperiment:
    """
    X = (P, φ, κ, seed) – one element of E_hier.

    Nodelists are NOT stored at design time.  Only the placement-class vector κ
    and a reproducibility seed are recorded.  The concrete node assignments are
    resolved later by expand_experiments.py using the topology oracle.

    placement_class_vector: one PlacementClass entry per job in the config.
    placement_seed: integer seed forwarded to the oracle's shuffle_within_class
        command to produce a deterministic nodelist for this (experiment, κ) pair.
    """
    base: Experiment
    placement_class_vector: PlacementClassVector
    placement_bin_label: str
    placement_score: float
    placement_seed: int

    def __str__(self) -> str:
        short = [PlacementClass.SHORT_NAME.get(c, c) for c in self.placement_class_vector]
        return (
            f"{self.base}"
            f"  κ={short}"
            f"  P-bin={self.placement_bin_label}"
            f"  score={self.placement_score:.2f}"
            f"  seed={self.placement_seed}\n"
        )


def _all_placement_class_vectors(
    config: Config, oracle: "TopologyOracle",
    include_intra_node: bool, g_total: int,
) -> list[PlacementClassVector]:
    per_job_classes = [
        _placement_classes_in_scope(run.gpus, g_total, oracle, include_intra_node)
        for run in config.runs
    ]
    return [tuple(combo) for combo in itertools.product(*per_job_classes)]

def _sample_placement_vectors(
    vectors: list[PlacementClassVector],
    n_per_bin: int, lo_hi: float, med_hi: float,
    rng: random.Random,
) -> list[tuple[PlacementClassVector, str]]:
    """
    Stratified sampling of placement-class vectors by locality bin.

    Each vector κ is scored by mean_score(κ) = (Σ score(κ_j)) / k and placed
    into a low / medium / high bin.  Up to n_per_bin vectors are drawn
    uniformly at random from each non-empty bin, ensuring variety across the
    full locality spectrum.
    """
    BIN_NAMES = ("low", "medium", "high")
    bins: dict[str, list[PlacementClassVector]] = {b: [] for b in BIN_NAMES}
    for kappa in vectors:
        score = placement_vector_score(kappa)
        b = placement_bin(score, lo_hi, med_hi)
        bins[b].append(kappa)
    result: list[tuple[PlacementClassVector, str]] = []
    for bn in BIN_NAMES:
        for kappa in rng.sample(bins[bn], min(n_per_bin, len(bins[bn]))):
            result.append((kappa, bn))
    return result


def build_hierarchical_experiment_set(
    flat_experiments: list[Experiment],
    oracle: "TopologyOracle",
    n_placement_samples_per_bin: int,
    placement_bin_lo_hi: float,
    placement_bin_med_hi: float,
    include_intra_node: bool,
    g_total: int,
    rng: random.Random,
    base_seed: int = 10_000,
    n_seeds_per_vector: int = 1,
) -> list[HierarchicalExperiment]:
    """
    Build E_hier: for every flat experiment, sample placement-class vectors
    from each locality bin, then assign deterministic seeds.

    No oracle calls are made here.  Concrete node assignments are deferred to
    expand_experiments.py, which uses (placement_class_vector, placement_seed)
    to request nodelists from the topology oracle at experiment launch time.

    n_seeds_per_vector: how many independent seeds (and therefore independent
        nodelist realisations) to record per (experiment, κ) pair.
    """
    hier_experiments: list[HierarchicalExperiment] = []
    seed_ctr = base_seed

    for exp in flat_experiments:
        all_vectors = _all_placement_class_vectors(
            exp.config, oracle, include_intra_node, g_total
        )
        sampled = _sample_placement_vectors(
            all_vectors, n_placement_samples_per_bin,
            placement_bin_lo_hi, placement_bin_med_hi, rng,
        )

        for kappa, p_bin in sampled:
            score = placement_vector_score(kappa)
            for _ in range(n_seeds_per_vector):
                hier_experiments.append(HierarchicalExperiment(
                    base=exp,
                    placement_class_vector=kappa,
                    placement_bin_label=p_bin,
                    placement_score=round(score, 4),
                    placement_seed=seed_ctr,
                ))
                seed_ctr += 1

    return hier_experiments


# ===========================================================================
# SECTION 9 – Size-bounded trimming
# ===========================================================================

def _diversity_key_flat(exp: "Experiment") -> tuple:
    """Diversity key for flat experiments: (pattern_family, entropy_bin)."""
    return (exp.pattern.family, exp.entropy_bin)


def _diversity_key_hier(he: "HierarchicalExperiment") -> tuple:
    """Diversity key for hierarchical experiments: (family, entropy_bin, placement_bin)."""
    return (he.base.pattern.family, he.base.entropy_bin, he.placement_bin_label)


def _trim_to_max(items: list, max_count: int, key_fn, rng: random.Random) -> list:
    """
    Trim *items* to at most *max_count* while maximising variety.

    Buckets items by diversity key, shuffles within each bucket, then
    round-robins across buckets so that every (pattern_family × entropy_bin ×
    placement_bin) combination is represented before any single combination is
    over-represented.
    """
    if len(items) <= max_count:
        return items
    buckets: dict[tuple, list] = {}
    for item in items:
        k = key_fn(item)
        buckets.setdefault(k, []).append(item)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = sorted(buckets.keys())
    result: list = []
    indices = {k: 0 for k in keys}
    while len(result) < max_count:
        progressed = False
        for k in keys:
            if len(result) >= max_count:
                break
            if indices[k] < len(buckets[k]):
                result.append(buckets[k][indices[k]])
                indices[k] += 1
                progressed = True
        if not progressed:
            break
    return result


def apply_size_bounds(
    experiments: list["Experiment"],
    hier_experiments: Optional[list["HierarchicalExperiment"]],
    min_count: int,
    max_count: Optional[int],
    rng: random.Random,
) -> tuple[list["Experiment"], Optional[list["HierarchicalExperiment"]]]:
    """
    Enforce --min-experiments / --max-experiments.

    When topology is enabled, bounds apply to |E_hier|.  After trimming E_hier,
    the flat set E is filtered to only patterns still referenced, keeping the
    JSON consistent.  Without topology, bounds apply directly to |E|.
    """
    primary = hier_experiments if hier_experiments is not None else experiments
    primary_name = "hierarchical" if hier_experiments is not None else "flat"

    if min_count > 0 and len(primary) < min_count:
        raise ValueError(
            f"Generated {primary_name} experiment set has {len(primary)} entries, "
            f"below --min-experiments={min_count}.  "
            "Raise --n-samples-per-bin / --n-stochastic-patterns, or lower the bound."
        )

    if max_count is not None and len(primary) > max_count:
        if hier_experiments is not None:
            hier_experiments = _trim_to_max(
                hier_experiments, max_count, _diversity_key_hier, rng
            )
            used_ids = {id(he.base) for he in hier_experiments}
            experiments = [e for e in experiments if id(e) in used_ids]
        else:
            experiments = _trim_to_max(
                experiments, max_count, _diversity_key_flat, rng
            )

    return experiments, hier_experiments


# ===========================================================================
# SECTION 10 – Utilities
# ===========================================================================

PRINTS_SEP_WIDTH = 140


def print_baseline_set(cfg: argparse.Namespace, baseline: list[SingleRun]) -> None:
    print(f"\n\033[34m{'='*PRINTS_SEP_WIDTH}")
    print(f"BASELINE SET  |B| = {len(baseline)}")
    print(f"{'='*PRINTS_SEP_WIDTH}\033[0m")
    by_strategy: dict[str, list[int]] = {}
    for run in baseline:
        by_strategy.setdefault(run.strategy.name, []).append(run.gpus)
    for name, gpus in by_strategy.items():
        print(f"  {name:20s}  gpus = {sorted(gpus)}")


def print_baseline_set_topology(
    cfg: argparse.Namespace, baseline_topo: list[BaselineWithPlacement],
) -> None:
    print(f"\n\033[34m{'='*PRINTS_SEP_WIDTH}")
    print(f"TOPOLOGY BASELINE SET  |B_topo| = {len(baseline_topo)}")
    print(f"{'='*PRINTS_SEP_WIDTH}\033[0m")
    for bwp in baseline_topo:
        print(bwp)


def print_patterns(cfg: argparse.Namespace, patterns: list[TaggedPattern]) -> None:
    print(f"\n\033[34m{'='*PRINTS_SEP_WIDTH}")
    print(f"PATTERN SET  |P| = {len(patterns)}")
    print(f"{'='*PRINTS_SEP_WIDTH}\033[0m")
    for tp in patterns:
        s = sum(tp.slots)
        print(f"  [{tp.family}] {str(tp.slots):<80}  totGPUs={s:<4}  util={int(s/cfg.G*100):<3}%  k={len(tp.slots)}")


def print_experiment_set(
    cfg: argparse.Namespace, experiments: list[Experiment],
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
    cfg: argparse.Namespace, hier_experiments: list[HierarchicalExperiment],
) -> None:
    print(f"\n\033[34m{'='*PRINTS_SEP_WIDTH}")
    print(f"HIERARCHICAL EXPERIMENT SET  |E_hier| = {len(hier_experiments)}")
    print(f"{'='*PRINTS_SEP_WIDTH}\033[0m")
    by_pbin: dict[str, list[HierarchicalExperiment]] = {"low": [], "medium": [], "high": []}
    for e in hier_experiments:
        by_pbin[e.placement_bin_label].append(e)
    for pbin in ("low", "medium", "high"):
        if by_pbin[pbin]:
            print(f"\n  \033[33m── P-bin: {pbin} ({len(by_pbin[pbin])} experiments) ──\033[0m\n")
            for e in by_pbin[pbin]:
                print(e)


def print_summary(
    g_total: int, strategies: list[Strategy], baseline: list[SingleRun],
    patterns: list[TaggedPattern], experiments: list[Experiment],
    hier_experiments: Optional[list[HierarchicalExperiment]] = None,
    baseline_topo: Optional[list[BaselineWithPlacement]] = None,
    min_experiments: int = 0,
    max_experiments: Optional[int] = None,
) -> None:
    print(f"\n\033[34m{'='*PRINTS_SEP_WIDTH}")
    print("SUMMARY")
    print(f"{'='*PRINTS_SEP_WIDTH}\033[0m")
    print(f"  G (total GPUs)         : {g_total}")
    print(f"  Strategies  |S|        : {len(strategies)}")
    print(f"  Baseline set |B|       : {len(baseline)}")
    if baseline_topo is not None:
        print(f"  Topology baseline |B_t|: {len(baseline_topo)}")
    print(f"  Pattern set |P|        : {len(patterns)}")
    print(f"  Flat experiments |E|   : {len(experiments)}")
    if hier_experiments is not None:
        print(f"  Hier experiments |E_h| : {len(hier_experiments)}")
    bounds = (
        f"min={min_experiments}" if min_experiments else "no min"
    ) + " / " + (
        f"max={max_experiments}" if max_experiments is not None else "no max"
    )
    print(f"  Size bounds            : {bounds}")
    print(f"  O(log G) bound         : log₂({g_total}) ≈ {math.log2(g_total):.1f}")


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
    }


def _hier_experiment_to_dict(he: HierarchicalExperiment) -> dict:
    d = _experiment_to_dict(he.base)
    # Emit short names (class1/class2/class3) for downstream consumers
    d["placement_class_vector"] = [
        PlacementClass.SHORT_NAME.get(c, c) for c in he.placement_class_vector
    ]
    d["placement_bin"] = he.placement_bin_label
    d["placement_score"] = he.placement_score
    d["placement_seed"] = he.placement_seed
    return d


def _baseline_topo_to_dict(bwp: BaselineWithPlacement) -> dict:
    return {
        "run": _single_run_to_dict(bwp.run),
        "placement_class": PlacementClass.SHORT_NAME.get(bwp.placement_class, bwp.placement_class),
        "replicate_index": bwp.replicate_index,
        "seed": bwp.seed,
    }


def build_json_output(
    cfg: argparse.Namespace, strategies: list[Strategy],
    baseline: list[SingleRun], patterns: list[TaggedPattern],
    experiments: list[Experiment],
    hier_experiments: Optional[list[HierarchicalExperiment]] = None,
    baseline_topo: Optional[list[BaselineWithPlacement]] = None,
) -> dict:
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "G": cfg.G,
        "seed": cfg.seed,
        "use_topology": cfg.use_topology,
        "topology_program": cfg.topology_program if cfg.use_topology else None,
    }
    parameters = {
        "G": cfg.G,
        "g_min": cfg.g_min,
        "k_max": cfg.k_max,
        "geometric_beta": cfg.beta,
        "hierarchical_patterns": [
            {"tier_fractions": tf, "n_tiny": nt} for tf, nt in HIERARCHICAL_PATTERNS
        ],
        "powerlaw_alphas": cfg.powerlaw_alphas,
        "util_min": cfg.util_min,
        "util_max": cfg.util_max,
        "util_steps": cfg.util_steps,
        "utilizations": _utilization_grid(cfg.util_min, cfg.util_max, cfg.util_steps),
        "stochastic_tier_config": STOCHASTIC_TIER_CONFIG,
        "n_stochastic_patterns": cfg.n_stochastic_patterns,
        "entropy_delta_1": cfg.entropy_delta_1,
        "entropy_delta_2": cfg.entropy_delta_2,
        "n_samples_per_bin": cfg.n_samples_per_bin,
        "enum_threshold": cfg.enum_threshold,
        "min_experiments": cfg.min_experiments,
        "max_experiments": cfg.max_experiments,
        "include_intra_node": cfg.include_intra_node,
        "topo_q1": cfg.topo_q1,
        "topo_q2": cfg.topo_q2,
        "topo_q3": cfg.topo_q3,
        "n_baseline_topo_reps": cfg.n_baseline_topo_reps if cfg.use_topology else None,
        "n_placement_seeds_per_vector": cfg.n_placement_seeds_per_vector if cfg.use_topology else None,
        "placement_bin_lo_hi": cfg.placement_bin_lo_hi,
        "placement_bin_med_hi": cfg.placement_bin_med_hi,
        "n_placement_samples_per_bin": cfg.n_placement_samples_per_bin,
    }
    doc = {
        "meta": meta,
        "parameters": parameters,
        "strategies": [_strategy_to_dict(s) for s in strategies],
        "baseline_set": [_single_run_to_dict(r) for r in baseline],
        "pattern_set": [
            {"slots": list(tp.slots), "k": len(tp.slots),
             "total_gpus": sum(tp.slots),
             "utilization": round(sum(tp.slots) / cfg.G, 6),
             "family": tp.family}
            for tp in patterns
        ],
        "experiments": [_experiment_to_dict(e) for e in experiments],
        "summary": {
            "n_strategies": len(strategies),
            "n_baseline_runs": len(baseline),
            "n_patterns": len(patterns),
            "n_flat_experiments": len(experiments),
            "log2_G": round(math.log2(cfg.G), 3),
        },
    }
    if baseline_topo is not None:
        doc["baseline_set_topology"] = [_baseline_topo_to_dict(b) for b in baseline_topo]
        doc["summary"]["n_baseline_topo_runs"] = len(baseline_topo)
    if hier_experiments is not None:
        doc["hierarchical_experiments"] = [_hier_experiment_to_dict(e) for e in hier_experiments]
        doc["summary"]["n_hier_experiments"] = len(hier_experiments)
    return doc


def serialize_to_json(doc: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    n = len(doc.get("hierarchical_experiments", doc.get("experiments", [])))
    print(f"\n\033[32m[JSON] Serialized → {path}  ({n} experiments)\033[0m")


# ===========================================================================
# MAIN
# ===========================================================================

def main(cfg: argparse.Namespace) -> None:
    rng = random.Random(cfg.seed)

    strategies = build_strategies(STRATEGY_DEFS)
    baseline = build_baseline_set(strategies, cfg.G)
    print_baseline_set(cfg, baseline)

    feasible_gpu_counts = compute_feasible_gpu_counts(strategies, cfg.G)
    utilizations = _utilization_grid(cfg.util_min, cfg.util_max, cfg.util_steps)

    patterns = build_pattern_set(
        g_total=cfg.G, k_max=cfg.k_max, g_min=cfg.g_min, beta=cfg.beta,
        hierarchical_defs=HIERARCHICAL_PATTERNS, powerlaw_alphas=cfg.powerlaw_alphas,
        utilizations=utilizations, util_min=cfg.util_min, util_max=cfg.util_max,
        stochastic_tier_config=STOCHASTIC_TIER_CONFIG,
        n_stochastic_patterns=cfg.n_stochastic_patterns,
        feasible_gpu_counts=feasible_gpu_counts, rng=rng,
        generate_equal_splits=not cfg.use_topology,
        generate_geometric=False,
        generate_hierarchical=False,
        generate_powerlaw=cfg.use_topology,
        generate_stochastic=True,
    )
    print_patterns(cfg, patterns)

    experiments = build_experiment_set(
        cfg=cfg, patterns=patterns, strategies=strategies,
        n_per_bin=cfg.n_samples_per_bin,
        delta1_frac=cfg.entropy_delta_1, delta2_frac=cfg.entropy_delta_2,
        enum_threshold=cfg.enum_threshold, rng=rng,
    )
    print_experiment_set(cfg, experiments)

    hier_experiments: Optional[list[HierarchicalExperiment]] = None
    baseline_topo: Optional[list[BaselineWithPlacement]] = None

    if cfg.use_topology:
        oracle = TopologyOracle(
            program=cfg.topology_program,
            q1=cfg.topo_q1, q2=cfg.topo_q2, q3=cfg.topo_q3,
        )
        if not oracle._available:
            print(f"\n[TopologyOracle] External program not found – oracle available for later use.\n"
                  f"  Program: '{cfg.topology_program}'")

        baseline_topo = build_baseline_set_with_topology(
            strategies=strategies, g_total=cfg.G, oracle=oracle,
            n_baseline_topo_reps=cfg.n_baseline_topo_reps,
            include_intra_node=cfg.include_intra_node,
        )
        print_baseline_set_topology(cfg, baseline_topo)

        hier_experiments = build_hierarchical_experiment_set(
            flat_experiments=experiments, oracle=oracle,
            n_placement_samples_per_bin=cfg.n_placement_samples_per_bin,
            placement_bin_lo_hi=cfg.placement_bin_lo_hi,
            placement_bin_med_hi=cfg.placement_bin_med_hi,
            include_intra_node=cfg.include_intra_node,
            g_total=cfg.G, rng=rng,
            n_seeds_per_vector=cfg.n_placement_seeds_per_vector,
        )
        print_hierarchical_experiment_set(cfg, hier_experiments)

    # ── Size bounds: trim and/or validate ────────────────────────────────────
    experiments, hier_experiments = apply_size_bounds(
        experiments=experiments,
        hier_experiments=hier_experiments,
        min_count=cfg.min_experiments,
        max_count=cfg.max_experiments,
        rng=rng,
    )
    if cfg.max_experiments is not None:
        active = hier_experiments if hier_experiments is not None else experiments
        print(f"\n[bounds] After trimming: {len(active)} experiments "
              f"(max={cfg.max_experiments})")

    print_summary(cfg.G, strategies, baseline, patterns, experiments,
                  hier_experiments, baseline_topo,
                  cfg.min_experiments, cfg.max_experiments)

    doc = build_json_output(
        cfg=cfg, strategies=strategies, baseline=baseline,
        patterns=patterns, experiments=experiments,
        hier_experiments=hier_experiments, baseline_topo=baseline_topo,
    )
    serialize_to_json(doc, cfg.output_json)


# ===========================================================================
# CLI
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


def _utilization_grid(util_min: float, util_max: float, steps: int) -> list[float]:
    if steps < 1:
        raise ValueError(f"--util-steps must be >= 1, got {steps}")
    if not (0.0 < util_min <= util_max <= 1.0):
        raise ValueError(f"Need 0 < util_min <= util_max <= 1, got [{util_min}, {util_max}]")
    if steps == 1:
        return [util_max]
    return [util_min + (util_max - util_min) * i / (steps - 1) for i in range(steps)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="experiment_design.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Multi-Training Congestion – Experiment Design Generator
=======================================================
Generates structured experiment sets for studying interference between
concurrent distributed training jobs.
""",
        epilog="""
examples:
  python experiment_design.py --G 8
  python experiment_design.py --G 72 --no-topology --seed 0
  python experiment_design.py --G 72 --use-topology --n-baseline-topo-reps 3
  python experiment_design.py --G 72 --min-experiments 20 --max-experiments 60
  python experiment_design.py --G 72 --output-json my_experiments.json
""",
    )

    parser.add_argument("--G", "-G", required=True, type=_positive_int, metavar="N",
                        help="Total number of GPUs. REQUIRED.")

    pg = parser.add_argument_group("pattern generation")
    pg.add_argument("--k-max", type=_positive_int, default=K_MAX, metavar="K",
                    help=f"Max concurrent jobs (default: {K_MAX}).")
    pg.add_argument("--g-min", type=_positive_int, default=G_MIN, metavar="G",
                    help=f"Min GPU slot size (default: {G_MIN}).")
    pg.add_argument("--beta", type=_unit_float, default=GEOMETRIC_BETA, metavar="β",
                    help=f"Geometric decay factor (default: {GEOMETRIC_BETA}).")
    pg.add_argument("--powerlaw-alphas", type=float, nargs="+", default=POWERLAW_ALPHAS,
                    metavar="α", help=f"Power-law tail exponents (default: {POWERLAW_ALPHAS}).")
    pg.add_argument("--util-min", type=_fraction_float, default=UTIL_MIN, metavar="ρ",
                    help=f"Min utilization (default: {UTIL_MIN}).")
    pg.add_argument("--util-max", type=_fraction_float, default=UTIL_MAX, metavar="ρ",
                    help=f"Max utilization (default: {UTIL_MAX}).")
    pg.add_argument("--util-steps", type=_positive_int, default=UTIL_STEPS, metavar="N",
                    help=f"Utilization grid steps (default: {UTIL_STEPS}).")
    pg.add_argument("--n-stochastic-patterns", type=_positive_int,
                    default=N_STOCHASTIC_PATTERNS, metavar="N",
                    help=f"Family-E patterns to generate (default: {N_STOCHASTIC_PATTERNS}).")

    eg = parser.add_argument_group("entropy-stratified sampling")
    eg.add_argument("--n-samples-per-bin", type=_positive_int, default=N_SAMPLES_PER_BIN,
                    metavar="N", help=f"Labellings per entropy bin (default: {N_SAMPLES_PER_BIN}).")
    eg.add_argument("--entropy-delta-1", type=_fraction_float, default=ENTROPY_DELTA_1,
                    metavar="δ1", help=f"Low/medium entropy boundary (default: {ENTROPY_DELTA_1}).")
    eg.add_argument("--entropy-delta-2", type=_fraction_float, default=ENTROPY_DELTA_2,
                    metavar="δ2", help=f"Medium/high entropy boundary (default: {ENTROPY_DELTA_2}).")
    eg.add_argument("--enum-threshold", type=int, default=ENUM_THRESHOLD, metavar="N",
                    help=f"Max |Φ(P)| for exact enumeration (default: {ENUM_THRESHOLD:,}).")

    # ── size bounds ──────────────────────────────────────────────────────────
    sg = parser.add_argument_group("experiment list size bounds")
    sg.add_argument("--min-experiments", type=int, default=MIN_EXPERIMENTS, metavar="N",
                    help=(
                        f"Minimum required experiments (default: {MIN_EXPERIMENTS}, 0=disabled). "
                        "Raises an error if the generated set is smaller."
                    ))
    sg.add_argument("--max-experiments", type=int, default=MAX_EXPERIMENTS, metavar="N",
                    help=(
                        "Maximum experiments to keep (default: unlimited). "
                        "Excess entries are trimmed by round-robin across pattern families, "
                        "entropy bins, and placement bins to maximise variety."
                    ))

    parser.add_argument("--seed", type=int, default=RANDOM_SEED, metavar="S",
                        help=f"Random seed (default: {RANDOM_SEED}).")

    tg = parser.add_argument_group("topology / placement")
    tex = tg.add_mutually_exclusive_group()
    tex.add_argument("--use-topology", dest="use_topology", action="store_true",
                     default=USE_TOPOLOGY, help="Enable hierarchical placement analysis.")
    tex.add_argument("--no-topology", dest="use_topology", action="store_false",
                     help="Skip placement analysis (flat model only).")
    tg.add_argument("--topology-program", default=TOPOLOGY_PROGRAM, metavar="PATH",
                    help=f"Path to topology oracle (default: '{TOPOLOGY_PROGRAM}'). "
                         "Required commands: find_placement, shuffle_within_class.")
    tg.add_argument("--topo-q1", type=_positive_int, default=TOPO_Q1, metavar="N",
                    help=f"GPUs per intra-node NVLink domain (default: {TOPO_Q1}).")
    tg.add_argument("--topo-q2", type=_positive_int, default=TOPO_Q2, metavar="N",
                    help=f"GPUs per L1-switch/rack domain (default: {TOPO_Q2}).")
    tg.add_argument("--topo-q3", type=_positive_int, default=TOPO_Q3, metavar="N",
                    help=f"GPUs per group (default: {TOPO_Q3}).")
    tg.add_argument("--include-intra-node", dest="include_intra_node",
                    action="store_true", default=INCLUDE_INTRA_NODE,
                    help="Include intra-node placement class (excluded by default).")
    tg.add_argument("--n-baseline-topo-reps", type=_positive_int,
                    default=N_BASELINE_TOPO_REPS, metavar="N",
                    help=(
                        f"Replicates per baseline for intra-group and inter-group classes "
                        f"(default: {N_BASELINE_TOPO_REPS}). "
                        "Each (S_i,g) in B: 1x intra-L1 + N x intra-group + N x inter-group."
                    ))
    tg.add_argument("--n-placement-seeds-per-vector", type=_positive_int,
                    default=N_PLACEMENT_SEEDS_PER_VECTOR, metavar="N",
                    help=(
                        f"Independent nodelist seeds per (experiment, placement vector) pair "
                        f"(default: {N_PLACEMENT_SEEDS_PER_VECTOR}). "
                        "Each seed produces a distinct nodelist at expand time."
                    ))
    tg.add_argument("--n-placement-samples-per-bin", type=_positive_int,
                    default=N_PLACEMENT_SAMPLES_PER_BIN, metavar="N",
                    help=f"Placement vectors per locality bin per experiment (default: {N_PLACEMENT_SAMPLES_PER_BIN}).")
    tg.add_argument("--placement-bin-lo-hi", type=float, default=PLACEMENT_BIN_LO_HI,
                    metavar="F", help=f"Low/medium placement score boundary (default: {PLACEMENT_BIN_LO_HI}).")
    tg.add_argument("--placement-bin-med-hi", type=float, default=PLACEMENT_BIN_MED_HI,
                    metavar="F", help=f"Medium/high placement score boundary (default: {PLACEMENT_BIN_MED_HI}).")

    parser.add_argument("--output-json", default=DEFAULT_JSON_OUTPUT, metavar="PATH",
                        help=f"Output JSON path (default: '{DEFAULT_JSON_OUTPUT}').")

    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.entropy_delta_1 >= args.entropy_delta_2:
        parser.error("--entropy-delta-1 must be < --entropy-delta-2.")
    if args.g_min > args.G:
        parser.error(f"--g-min ({args.g_min}) cannot exceed --G ({args.G}).")
    if args.enum_threshold < 0:
        parser.error("--enum-threshold must be >= 0.")
    for a in args.powerlaw_alphas:
        if a <= 1.0:
            parser.error(f"--powerlaw-alphas: each α must be > 1, got {a}.")
    if args.util_min > args.util_max:
        parser.error("--util-min must be <= --util-max.")
    if args.util_min <= 0.0:
        parser.error("--util-min must be > 0.")
    if args.placement_bin_lo_hi >= args.placement_bin_med_hi:
        parser.error("--placement-bin-lo-hi must be < --placement-bin-med-hi.")
    if args.min_experiments < 0:
        parser.error("--min-experiments must be >= 0.")
    if args.max_experiments is not None:
        if args.max_experiments < 1:
            parser.error("--max-experiments must be >= 1.")
        if args.min_experiments > args.max_experiments:
            parser.error("--min-experiments must be <= --max-experiments.")


if __name__ == "__main__":
    _parser = build_parser()
    _args = _parser.parse_args()
    _validate_args(_args, _parser)
    main(_args)