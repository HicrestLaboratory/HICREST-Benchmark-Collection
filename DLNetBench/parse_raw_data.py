#!/usr/bin/env python3
"""
parse_results.py — Data parsing and slowdown computation for DLNetBench.

For each (strategy, nodes, placement) combination this module:
  1. Parses per-rank data from raw ccutils stdout files.
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

Public API
----------
  parse_baselines(backup_dir, skip_first, system_name)  -> dict
  parse_concurrent(backup_dir, skip_first, system_name) -> list[dict]
  compute_slowdowns(baselines, concurrent)               -> dict
  min_throughput_across_ranks(ranks, skip_first, adaptive_skip) -> float | None
"""

import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import yaml

try:
    from command_map import get_model_from_command
except ImportError:
    def get_model_from_command(_cmd):
        return None

# ============================================================================
#  Placement mapping
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

# ============================================================================
#  Raw stdout parsing
# ============================================================================

# Regex for ccutils stdout format:
#   [[Rank N]]
#   { ... JSON ... }
#   [[END Rank N]]
_RANK_RE = re.compile(r"\[\[Rank (\d+)\]\]\n(.+?)\n\[\[END Rank \d+\]\]")

# Keys extracted from each rank's JSON block.
# "throughputs" is the primary metric; all others are carried along so
# callers can access the full per-rank record without re-parsing the file.
_RANK_KEYS = [
    "throughputs",       # list[float] — primary metric
    "losses",            # list[float] — training loss curve
    "iteration_times",   # list[float] — wall-clock time per iteration (s)
    "memory_allocated",  # float       — peak GPU memory allocated (bytes or GiB)
    "memory_reserved",   # float       — peak GPU memory reserved
    "model_name",        # str         — model identifier
    "gpu_model",         # str         — GPU hardware identifier
    "n_params",          # int         — model parameter count
    "batch_size",        # int         — local batch size used
    "seq_len",           # int         — sequence length
    "world_size",        # int         — total number of ranks
]


def parse_stdout_throughputs(filepath):
    """
    Parse a ccutils stdout file and extract all available per-rank metrics.

    For each rank block the full JSON object is read and all keys listed in
    ``_RANK_KEYS`` are captured if present.  Unknown keys are also preserved
    under a ``"extra"`` sub-dict so no information is silently discarded.

    Parameters
    ----------
    filepath : str
        Path to the stdout file.

    Returns
    -------
    dict
        ``{rank_id (int): {key: value, ..., "extra": {unknown_key: value}}}``

        The ``"throughputs"`` key (list of floats) is always present (possibly
        empty) so downstream code can rely on it without a guard.
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

        record = {"throughputs": []}   # guarantee key presence
        extra = {}

        for key, value in data.items():
            if key in _RANK_KEYS:
                record[key] = value
            else:
                extra[key] = value

        # Ensure all known keys are present (None when absent in the JSON)
        for key in _RANK_KEYS:
            record.setdefault(key, None if key != "throughputs" else [])

        if extra:
            record["extra"] = extra

        if record["throughputs"]:       # skip ranks with no throughput data
            ranks[rank_id] = record

    return ranks


def parse_model_name_from_stdout(filepath):
    """Return model_name from stdout JSON metadata in the run file."""
    with open(filepath, "r", errors="ignore") as f:
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
        ``{rank_id: rank_record}`` as returned by ``parse_stdout_throughputs``.
        Each record must contain a ``"throughputs"`` list; plain
        ``{rank_id: [float]}`` dicts are also accepted for backwards
        compatibility.
    skip_first : int
        Default iterations to discard (overridden to 3 when >= 6 are available
        and *adaptive_skip* is True).
    adaptive_skip : bool
        If True (default), use skip=3 for ranks with >= 6 iterations (tuned
        for baselines whose first iterations are warm-up).  Set to False for
        concurrent runs where throughput variability is the signal, not
        warm-up noise.

    Returns
    -------
    float or None
        Minimum per-rank median throughput, or None if no usable data.
    """
    medians = []
    for rank_record in ranks_throughputs.values():
        # Accept both the new dict-of-dicts format and legacy list format.
        tp = (rank_record["throughputs"]
              if isinstance(rank_record, dict)
              else rank_record)

        # Adaptive skip: DP baselines typically have 6 iterations where the
        # first 3 are warm-up; shorter runs (< 6 iters) use skip_first.
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
        ``{(strategy, nodes, placement_class, model_name): min_throughput}``
        per baseline.
    """
    baselines = {}
    if system_name == "nvl72":
        base = os.path.join(backup_dir, "SbatchMan", "experiments", "nvl72")
        meta_pattern = os.path.join(base, "*", "baseline_*", "*", "metadata.yaml")
    else:
        base = os.path.join(backup_dir, "SbatchMan", "experiments", system_name)
        meta_pattern = os.path.join(
            base, f"{system_name}_*", "*", "20*", "metadata.yaml"
        )

    for meta_path in glob.glob(meta_pattern):
        with open(meta_path) as f:
            meta = yaml.safe_load(f)

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
    system_name : str
        System name used for optional GPU-model filtering (e.g. "nvl72").

    Returns
    -------
    list of dict
        Each dict has keys: strategy, gpus, nodes, placement, model_name,
        app_id, rep, throughput.
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
            with open(filepath, "r", errors="ignore") as f:
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
                "strategy":   strategy,
                "gpus":       int(gpus),
                "nodes":      int(nodes),
                "placement":  placement,
                "model_name": model_name,
                "app_id":     app_id,
                "rep":        int(rep),
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

    Matching uses (strategy, nodes, model_name) and maps the concurrent
    placement name to the baseline's placement class via PLACEMENT_MAP.

    Parameters
    ----------
    baselines : dict
        ``{(strategy, nodes, placement, model_name): throughput}`` from
        ``parse_baselines``.
    concurrent : list of dict
        Concurrent run records from ``parse_concurrent``.

    Returns
    -------
    dict
        ``{(strategy, nodes, placement, model_name): [ratio, ...]}``

        ratio = baseline_throughput / concurrent_throughput.
        Values > 1.0 indicate congestion-induced slowdown.
    """
    slowdowns = defaultdict(list)
    unmatched = set()
    skipped_na = 0

    for run in concurrent:
        placement = run["placement"]

        baseline_placement = PLACEMENT_MAP.get(placement)
        if baseline_placement is None:
            skipped_na += 1
            continue

        model_name = run.get("model_name", "unknown")
        bkey = (run["strategy"], run["nodes"], baseline_placement, model_name)
        t0 = baselines.get(bkey)
        if t0 is None or t0 == 0:
            unmatched.add(bkey)
            continue

        sigma = t0 / run["throughput"]
        cat = (run["strategy"], run["nodes"], placement, model_name)
        slowdowns[cat].append(sigma)

    if skipped_na:
        print(
            f"  Skipped {skipped_na} concurrent runs with unmapped placement "
            f"(e.g. 'na')"
        )
    if unmatched:
        print(f"  WARNING: {len(unmatched)} combos had no matching baseline:")
        for u in sorted(unmatched, key=str):
            print(f"    {u}")

    return slowdowns