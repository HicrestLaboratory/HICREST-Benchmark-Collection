"""
Multi-Training Congestion: Experiment Design Framework
======================================================
Python implementation of the formalization by Thomas Pasquali.

Structure:
  - Section 1:  System model (GPU count, strategies, feasibility sets)
  - Section 2:  Baseline set B
  - Section 3:  GPU allocation patterns  (families A / B / C)
  - Section 4:  Strategy assignment & entropy-stratified sampling
  - Section 5:  Final experiment set  E
  - Section 6:  Network topology model & placement-class sampling
  - Section 7:  TopologyOracle  – thin wrapper around the external topology program
  - Section 8:  Extended experiment set  E_hier
  - Section 9:  Pretty-printing / summary utilities

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
from xmlrpc.client import FastMarshaller


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

# ── Entropy-stratified sampling ──────────────────────────────────────────────
# Bin boundaries for mixture entropy H ∈ [0, log m]
ENTROPY_DELTA_1: float = 0.3     # low  / medium boundary
ENTROPY_DELTA_2: float = 0.7     # medium / high  boundary  (as fraction of log m)

# Number of labellings to sample per entropy bin, per pattern
N_SAMPLES_PER_BIN: int = 2

# Random seed for reproducibility (set to None for non-deterministic)
RANDOM_SEED: Optional[int] = 42

# ── Topology (Section 6) ─────────────────────────────────────────────────────
# Set USE_TOPOLOGY = False to skip hierarchical placement entirely (flat model)
USE_TOPOLOGY: bool = False

# Path / command of the external topology program (Section 7)
TOPOLOGY_PROGRAM: str = "topology_oracle"   # must be on PATH or an absolute path

# Number of within-class variance shuffles for high-interest configs (Section 8.3)
N_VARIANCE_SHUFFLES: int = 3

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

AllocationPattern = tuple[int, ...]   # ordered vector of GPU counts


def pattern_A_equal_splits(g_total: int, k_max: int, g_min: int) -> list[AllocationPattern]:
    """
    Family A: equal-split patterns  P = (g, …, g)  with  k*g = G.
    """
    patterns: list[AllocationPattern] = []
    for k in range(2, k_max + 1):
        if g_total % k == 0:
            g = g_total // k
            if g >= g_min:
                patterns.append(tuple([g] * k))
    return patterns


def pattern_B_geometric(
    g_total: int,
    k_max: int,
    g_min: int,
    beta: float,
) -> list[AllocationPattern]:
    """
    Family B: geometric decay patterns.
    Start with g_1 = G // 2, then g_{j+1} = floor(β * g_j).
    Stop when sum >= G or g_j < g_min.
    """
    patterns: list[AllocationPattern] = []
    g1 = g_total // 2
    while g1 >= g_min:
        slots: list[int] = []
        remaining = g_total
        g_curr = g1
        while g_curr >= g_min and remaining >= g_curr and len(slots) < k_max:
            slots.append(g_curr)
            remaining -= g_curr
            g_curr = max(g_min, int(math.floor(beta * g_curr)))
        if len(slots) >= 2:
            patterns.append(tuple(slots))
        g1 = g1 // 2
    return patterns


def pattern_C_hierarchical(
    g_total: int,
    k_max: int,
    g_min: int,
    tier_fractions: list[float],
    n_tiny: int,
) -> Optional[AllocationPattern]:
    """
    Family C: hierarchical dominant-minority patterns.
    tier_fractions = [α_1, …, α_d]: each tier takes α_ℓ of the residual budget.
    n_tiny: number of equal tiny slots that consume the remaining budget.
    Returns None if the pattern is degenerate (e.g. tiny slots too small).
    """
    slots: list[int] = []
    residual = g_total

    for alpha in tier_fractions:
        g_tier = int(math.floor(alpha * residual))
        if g_tier < g_min:
            return None
        slots.append(g_tier)
        residual -= g_tier

    if n_tiny > 0 and residual > 0:
        tiny_size = int(math.floor(residual / n_tiny))
        if tiny_size < g_min:
            return None
        slots.extend([tiny_size] * n_tiny)

    if len(slots) < 2 or len(slots) > k_max:
        return None

    return tuple(slots)


def build_pattern_set(
    g_total: int,
    k_max: int,
    g_min: int,
    beta: float,
    hierarchical_defs: list[tuple[list[float], int]],
    generate_equal_splits: bool,
    generate_geometric: bool,
    generate_hierarchical: bool,
) -> list[AllocationPattern]:
    """
    Build the canonical pattern set P = families A ∪ B ∪ C (deduplicated).
    """
    seen: set[AllocationPattern] = set()
    patterns: list[AllocationPattern] = []

    def add(p: AllocationPattern) -> None:
        if p not in seen and sum(p) <= g_total:
            seen.add(p)
            patterns.append(p)

    if generate_equal_splits:
        for p in pattern_A_equal_splits(g_total, k_max, g_min):
            add(p)

    if generate_geometric:
        for p in pattern_B_geometric(g_total, k_max, g_min, beta):
            add(p)

    if generate_hierarchical:
        for tier_fracs, n_tiny in hierarchical_defs:
            p = pattern_C_hierarchical(g_total, k_max, g_min, tier_fracs, n_tiny)
            if p is not None:
                add(p)

    return patterns


# ===========================================================================
# SECTION 4 – Strategy Assignment & Entropy-Stratified Sampling
# ===========================================================================

def feasible_strategies_for_slot(strategies: list[Strategy], g: int) -> list[Strategy]:
    """All strategies that support GPU count g."""
    return [s for s in strategies if s.supports(g)]


def all_feasible_labellings(
    pattern: AllocationPattern,
    strategies: list[Strategy],
) -> list[list[Strategy]]:
    """
    Enumerate Φ(P): all feasible strategy labellings for the given pattern.
    WARNING: exponential in k – use only for small patterns or sampling.
    """
    per_slot = [feasible_strategies_for_slot(strategies, g) for g in pattern]
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
    m: int,
    delta1_frac: float,
    delta2_frac: float,
) -> str:
    """Return 'low', 'medium', or 'high' entropy bin label."""
    h_max = math.log(m) if m > 1 else 1.0
    d1 = delta1_frac * h_max
    d2 = delta2_frac * h_max
    if h < d1:
        return "low"
    elif h < d2:
        return "medium"
    else:
        return "high"


def _random_labelling(
    per_slot: list[list[Strategy]],
    rng: random.Random,
) -> list[Strategy]:
    """Draw one labelling by independently sampling each slot."""
    return [rng.choice(opts) for opts in per_slot]


def sample_labellings_stratified(
    pattern: AllocationPattern,
    strategies: list[Strategy],
    n_per_bin: int,
    delta1_frac: float,
    delta2_frac: float,
    rng: random.Random,
) -> list[tuple[list[Strategy], str]]:
    """
    Φ_sample(P): sample n_per_bin labellings per entropy bin.
    Returns list of (labelling, bin_label).

    Replaces the previous enumerate-all approach with lazy rejection sampling:
    labellings are drawn one at a time and routed into the appropriate entropy
    bin until every bin is full or a budget of attempts is exhausted.  This is
    O(n_per_bin × k) per pattern regardless of |Φ(P)|, so it scales to large k.

    The attempt budget is:
        MAX_ATTEMPTS = n_per_bin × ATTEMPTS_PER_SAMPLE × n_bins
    which is a small constant relative to the work done per experiment.

    Deduplication: a labelling (as a tuple of strategy names) is rejected if it
    has already been accepted into any bin, so all returned labellings are
    distinct.
    """
    BIN_NAMES = ("low", "medium", "high")
    ATTEMPTS_PER_SAMPLE = 200   # max draws per missing sample before giving up

    per_slot = [feasible_strategies_for_slot(strategies, g) for g in pattern]
    if any(len(opts) == 0 for opts in per_slot):
        return []   # pattern has a slot no strategy can fill

    m = len(strategies)
    bins: dict[str, list[list[Strategy]]] = {b: [] for b in BIN_NAMES}
    seen: set[tuple[str, ...]] = set()          # deduplication across all bins

    total_budget = n_per_bin * ATTEMPTS_PER_SAMPLE * len(BIN_NAMES)

    for _ in range(total_budget):
        # Stop early when every bin has reached its quota
        if all(len(bins[b]) >= n_per_bin for b in BIN_NAMES):
            break

        lab = _random_labelling(per_slot, rng)
        key = tuple(s.name for s in lab)
        if key in seen:
            continue

        h = mixture_entropy(lab, pattern)
        b = entropy_bin(h, m, delta1_frac, delta2_frac)

        if len(bins[b]) < n_per_bin:
            bins[b].append(lab)
            seen.add(key)

    result: list[tuple[list[Strategy], str]] = []
    for bin_name in BIN_NAMES:
        for lab in bins[bin_name]:
            result.append((lab, bin_name))
    return result


# ===========================================================================
# SECTION 5 – Final Experiment Set  E  (flat / uniform topology)
# ===========================================================================

@dataclass
class Experiment:
    """A single element of E: pattern + labelling → Config."""
    pattern: AllocationPattern
    labelling: list[Strategy]
    entropy_bin: str
    config: Config
    placement: Optional["PlacementClassVector"] = None   # set in Section 8

    def __str__(self) -> str:
        pattern_str = str(self.pattern)
        config_str = str(self.config)
        placement_str = (
            f"  placement={self.placement}" if self.placement else ""
        )
        return f"  pattern={pattern_str:<20}\n  H-bin={self.entropy_bin:<6}{placement_str}\n  config={config_str:<40}\n"


def build_experiment_set(
    cfg: argparse.Namespace,
    patterns: list[AllocationPattern],
    strategies: list[Strategy],
    n_per_bin: int,
    delta1_frac: float,
    delta2_frac: float,
    rng: random.Random,
) -> list[Experiment]:
    """
    E = { C(P, φ) | P ∈ P, φ ∈ Φ_sample(P) }
    """
    experiments: list[Experiment] = []
    for pattern in patterns:
        sampled = sample_labellings_stratified(
            pattern, strategies, n_per_bin, delta1_frac, delta2_frac, rng
        )
        for labelling, bin_label in sampled:
            runs = [SingleRun(s, g) for s, g in zip(labelling, pattern)]
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

    The external program is assumed to be a CLI tool that accepts JSON on stdin
    and returns JSON on stdout.  All methods here translate Python calls into
    JSON queries and parse the responses.

    If the program is not available, a stub fallback is used so the rest of the
    framework can still run (all jobs default to 'intra-node').

    ── Protocol ──────────────────────────────────────────────────────────────
    Every query is a JSON object with a "command" field.  Supported commands:

      describe_topology
        → Returns the cluster topology parameters:
          { "G": int, "q1": int, "q2": int, "q3": int,
            "b1": float, "b2": float, "b3": float }

      feasible_placement_classes
        Input:  { "command": "feasible_placement_classes", "gpu_count": int,
                  "free_capacity": {"intra_node": int, "intra_l1": int,
                                    "intra_group": int} }
        Output: { "classes": [str, …] }   # subset of PlacementClass.ALL

      find_placement
        Input:  { "command": "find_placement",
                  "jobs": [{"job_id": int, "gpu_count": int,
                             "placement_class": str}, …] }
        Output: { "assignments": [{"job_id": int, "nodes": [str, …]}, …],
                  "feasible": bool }

      shuffle_within_class
        Input:  { "command": "shuffle_within_class",
                  "jobs": [{"job_id": int, "gpu_count": int,
                             "placement_class": str}, …],
                  "n_shuffles": int }
        Output: { "shuffles": [ [{"job_id": int, "nodes": [str, …]}, …], … ] }
    """

    def __init__(self, program: str = TOPOLOGY_PROGRAM) -> None:
        self.program = program
        self._available = self._check_available()

    # ── internal helpers ──────────────────────────────────────────────────

    def _check_available(self) -> bool:
        """Return True if the external program can be invoked."""
        try:
            result = subprocess.run(
                [self.program, "--ping"],
                capture_output=True, timeout=2
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _query(self, payload: dict) -> dict:
        """
        Send a JSON query to the external program and return the parsed response.
        Falls back to stub responses when the program is not available.
        """
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
        """
        Fallback stubs used when the external program is unavailable.
        All jobs are treated as intra-node; a single representative placement
        is returned.
        """
        cmd = payload.get("command", "")
        if cmd == "describe_topology":
            return {"G": G, "q1": 8, "q2": 64, "q3": G,
                    "b1": 600.0, "b2": 200.0, "b3": 100.0}
        if cmd == "feasible_placement_classes":
            g = payload.get("gpu_count", 1)
            free = payload.get("free_capacity", {})
            # Conservative stub: return the most local class that fits
            classes = []
            if g <= free.get("intra_node", 0):
                classes.append(PlacementClass.INTRA_NODE)
            if g <= free.get("intra_l1", 0):
                classes.append(PlacementClass.INTRA_L1)
            classes.append(PlacementClass.INTRA_GROUP)
            classes.append(PlacementClass.MULTI_GROUP)
            return {"classes": classes[:1]}   # return only the best class
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

    # ── public API ────────────────────────────────────────────────────────

    def describe_topology(self) -> dict:
        """
        Returns cluster topology parameters:
          G, q1 (GPUs per NVLink domain), q2 (GPUs per L1-switch domain),
          q3 (GPUs per group), b1/b2/b3 (bandwidths in GB/s).
        """
        return self._query({"command": "describe_topology"})

    def feasible_placement_classes(
        self,
        gpu_count: int,
        free_capacity: dict[str, int],
    ) -> list[str]:
        """
        Ask the oracle which placement classes are achievable for a job
        requesting `gpu_count` GPUs given the current free capacity at each
        topology level.

        free_capacity keys: "intra_node", "intra_l1", "intra_group"
        Returns: subset of PlacementClass.ALL
        """
        payload = {
            "command": "feasible_placement_classes",
            "gpu_count": gpu_count,
            "free_capacity": free_capacity,
        }
        resp = self._query(payload)
        return resp.get("classes", [PlacementClass.INTRA_GROUP])

    def find_placement(
        self,
        jobs: list[dict],   # [{"job_id": int, "gpu_count": int, "placement_class": str}]
    ) -> dict:
        """
        Ask the oracle for a concrete GPU assignment realising the given
        placement-class vector.

        Returns {"feasible": bool, "assignments": [{"job_id", "nodes"}, …]}
        """
        return self._query({"command": "find_placement", "jobs": jobs})

    def shuffle_within_class(
        self,
        jobs: list[dict],
        n_shuffles: int = 3,
    ) -> list[list[dict]]:
        """
        Generate `n_shuffles` random node-list permutations of a placement,
        keeping each job within its assigned placement class.

        Returns a list of n_shuffles assignments (each like find_placement output).
        """
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
    base: Experiment                           # carries pattern, labelling, config
    placement_class_vector: PlacementClassVector
    node_assignment: Optional[dict] = None    # concrete output from oracle
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
    """
    Simple heuristic: compute remaining free GPUs at each topology level
    based on how many have already been allocated in this configuration.
    In a real deployment the oracle itself tracks live cluster state.
    """
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
    """
    For a given Config, enumerate all feasible placement-class vectors
    κ = (κ_1, …, κ_k) by querying the oracle per job.

    Returns one entry per feasible combination (Cartesian product of per-job
    feasible classes).  Since each κ_j has at most 4 values, the result is O(1).
    """
    topo = oracle.describe_topology()
    allocated_so_far = 0

    per_job_classes: list[list[str]] = []
    for run in config.runs:
        free = _free_capacity_from_topology(topo, allocated_so_far)
        classes = oracle.feasible_placement_classes(run.gpus, free)
        per_job_classes.append(classes)
        allocated_so_far += run.gpus   # greedy sequential allocation estimate

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
    """
    E_hier = { (C(P,φ), ψ) | P ∈ P, φ ∈ Φ_sample(P), ψ ∈ Ψ_sample(C) }

    For each flat experiment:
      1. Enumerate feasible placement-class vectors (via oracle).
      2. For each vector, find one concrete placement (oracle.find_placement).
      3. Optionally add within-class variance shuffles for high-interest configs.

    high_interest_predicate: callable(Experiment) → bool.  If None, variance
      shuffles are added for all experiments with k >= 2 jobs.
    """
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

            # Stage 2: within-class variance shuffles for high-interest configs
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

def print_baseline_set(cfg: argparse.Namespace, baseline: list[SingleRun]) -> None:
    print(f"\n\033[34m{'='*70}")
    print(f"BASELINE SET  |B| = {len(baseline)}")
    print(f"{'='*70}\033[0m")
    by_strategy: dict[str, list[int]] = {}
    for run in baseline:
        by_strategy.setdefault(run.strategy.name, []).append(run.gpus)
    for name, gpus in by_strategy.items():
        print(f"  {name:20s}  gpus = {sorted(gpus)}")


def print_patterns(cfg: argparse.Namespace, patterns: list[AllocationPattern]) -> None:
    print(f"\n\033[34m{'='*70}")
    print(f"PATTERN SET  |P| = {len(patterns)}")
    print(f"{'='*70}\033[0m")
    for p in patterns:
        print(f"  {str(p):<30}  totGPUs={sum(p):<4}  util={int(sum(p)/cfg.G*100):<3}%  k={len(p)}")


def print_experiment_set(
    cfg: argparse.Namespace,
    experiments: list[Experiment],
    title: str = "EXPERIMENT SET",
) -> None:
    print(f"\n\033[34m{'='*70}")
    print(f"{title}  |E| = {len(experiments)}")
    print(f"{'='*70}\033[0m")
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
    print(f"\n\033[34m{'='*70}")
    print(f"HIERARCHICAL EXPERIMENT SET  E_hier")
    print(f"  Core experiments:           {len(core)}")
    print(f"  Variance-shuffle runs:      {len(var)}")
    print(f"  Total:                      {len(hier_experiments)}")
    print(f"{'='*70}\033[0m")
    for e in core:
        print(e)


def print_summary(
    g_total: int,
    strategies: list[Strategy],
    baseline: list[SingleRun],
    patterns: list[AllocationPattern],
    experiments: list[Experiment],
    hier_experiments: Optional[list[HierarchicalExperiment]] = None,
) -> None:
    print(f"\n\033[34m{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\033[0m")
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
# MAIN  –  wire everything together
# ===========================================================================

def main(cfg: argparse.Namespace) -> None:
    rng = random.Random(cfg.seed)

    # 1. Strategies  (not CLI-configurable; edit STRATEGY_DEFS in the file)
    strategies = build_strategies(STRATEGY_DEFS)

    # 2. Baseline set
    baseline = build_baseline_set(strategies, cfg.G)
    print_baseline_set(cfg, baseline)

    # 3. Pattern set
    patterns = build_pattern_set(
        g_total=cfg.G,
        k_max=cfg.k_max,
        g_min=cfg.g_min,
        beta=cfg.beta,
        hierarchical_defs=HIERARCHICAL_PATTERNS,
        generate_equal_splits=not cfg.use_topology,
        generate_geometric=False,
        generate_hierarchical=True,
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


# ===========================================================================
# CLI  –  argument parsing
# ===========================================================================

def _positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return n


def _unit_float(value: str) -> float:
    """A float strictly in (0, 1)."""
    f = float(value)
    if not (0.0 < f < 1.0):
        raise argparse.ArgumentTypeError(f"must be in (0, 1), got {value}")
    return f


def _fraction_float(value: str) -> float:
    """A float in [0, 1]."""
    f = float(value)
    if not (0.0 <= f <= 1.0):
        raise argparse.ArgumentTypeError(f"must be in [0, 1], got {value}")
    return f


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

STRATEGY_DEFS and HIERARCHICAL_PATTERNS can only be changed by editing the
CONFIG block at the top of this file.
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
""",
    )

    # ── Required ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--G",
        "-G",
        required=True,
        type=_positive_int,
        metavar="N",
        help="Total number of GPUs in the cluster (e.g. 8, 72, 512). REQUIRED.",
    )

    # ── Pattern generation ────────────────────────────────────────────────────
    pattern_group = parser.add_argument_group(
        "pattern generation",
        "Controls GPU allocation pattern families (Sections 3A/B/C).",
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
        help=(
            f"Labellings to sample per entropy bin per pattern (default: {N_SAMPLES_PER_BIN}). "
            "Three bins (low/medium/high) means up to 3×N experiments per pattern."
        ),
    )
    entropy_group.add_argument(
        "--entropy-delta-1",
        type=_fraction_float,
        default=ENTROPY_DELTA_1,
        metavar="δ1",
        help=(
            f"Low/medium entropy bin boundary as a fraction of log(m) (default: {ENTROPY_DELTA_1}). "
            "Must be < --entropy-delta-2."
        ),
    )
    entropy_group.add_argument(
        "--entropy-delta-2",
        type=_fraction_float,
        default=ENTROPY_DELTA_2,
        metavar="δ2",
        help=(
            f"Medium/high entropy bin boundary as a fraction of log(m) (default: {ENTROPY_DELTA_2}). "
            "Must be > --entropy-delta-1."
        ),
    )

    # ── Reproducibility ───────────────────────────────────────────────────────
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        metavar="S",
        help=(
            f"Random seed for reproducible sampling (default: {RANDOM_SEED}). "
            "Pass --seed '' or omit for non-deterministic runs."
        ),
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
        help="Enable hierarchical placement analysis (default: enabled).",
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
        help=(
            f"Path or command name of the external topology oracle program "
            f"(default: '{TOPOLOGY_PROGRAM}'). Ignored when --no-topology is set."
        ),
    )
    topo_group.add_argument(
        "--n-variance-shuffles",
        type=_positive_int,
        default=N_VARIANCE_SHUFFLES,
        metavar="N",
        help=(
            f"Within-class variance shuffles for high-interest configs (default: {N_VARIANCE_SHUFFLES}). "
            "Set to 0 to disable variance estimation entirely."
        ),
    )

    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Cross-argument validation that argparse cannot express declaratively."""
    if args.entropy_delta_1 >= args.entropy_delta_2:
        parser.error(
            f"--entropy-delta-1 ({args.entropy_delta_1}) must be strictly less than "
            f"--entropy-delta-2 ({args.entropy_delta_2})."
        )
    if args.g_min > args.G:
        parser.error(
            f"--g-min ({args.g_min}) cannot exceed --G ({args.G})."
        )


if __name__ == "__main__":
    _parser = build_parser()
    _args = _parser.parse_args()
    _validate_args(_args, _parser)
    main(_args)