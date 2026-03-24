"""
expand_experiments.py
=====================
Parses the master JSON produced by experiment_design.py and writes:

  1. A folder of per-experiment JSON files   (experiment_NNNN.json)
  2. A global JSON configuration file        (global_config.json)

Each experiment file contains a single top-level key "pattern_N" with the
structure expected by the runtime scheduler.

Interconnect types
------------------
The interconnect type is inferred from the master JSON:

  uniform        key "experiments"              → default placement: device
  hierarchical   key "hierarchical_experiments" → default placement: runtime

Placement modes
---------------
Exactly one placement value is written to the top-level "placement" key of
every experiment file.  Allowed values differ by interconnect:

  uniform      → device | random | linear | hardcoded
  hierarchical → runtime | random | linear | hardcoded

  device   [uniform default]
    Top-level placement = "device".
    Job entries carry NO placement_class key.

  random
    Top-level placement = "random".
    Job entries carry NO placement_class key.

  linear
    Top-level placement = "linear".
    Job entries carry NO placement_class key.

  runtime  [hierarchical default]
    Top-level placement = "runtime".
    Job entries carry placement_class from the design's placement_class_vector.

  hardcoded
    Requires --system NAME.  Queries the placement oracle once per experiment.
    Feasible   → top-level placement = "hardcoded";
                 job entries carry an explicit "nodelist" (no placement_class).
    Infeasible → top-level placement = "hardcoded_infeasible";
                 job entries carry placement_class = "hardcoded_infeasible"
                 and an "infeasible_reason" field.
    A placement summary is printed at the end.

    Oracle interface (stdin/stdout JSON):
      Input:
        {"command": "find_placement",
         "system":  <str>,
         "reserved_nodes": [<str>, ...],
         "jobs": [{"job_id": <int>, "node_count": <int>,
                   "placement_class": <str>}, ...]}
      Output (feasible):
        {"feasible": true,
         "assignments": [{"job_id": <int>, "nodelist": [<str>, ...]}, ...]}
      Output (infeasible):
        {"feasible": false, "reason": <str>}

    When the oracle program is not reachable a stub is used (all infeasible).

Small vs. large job classification
------------------------------------
Default (no flags):  ALL jobs are classified as small.

--small-job-threshold N
  Jobs with node_count >= N → large; all others → small.

--split-large-small
  Exactly ONE job is large – the one requiring the most nodes
  (ties: first occurrence wins).  All others are small.
  --small-job-threshold has no effect when this flag is active.

Global config templates
-----------------------
Two built-in templates are provided (uniform / hierarchical).
Edit GLOBAL_TEMPLATE_UNIFORM and GLOBAL_TEMPLATE_HIERARCHICAL near the top of
this file to permanently customise the output shape.

Use --global-template PATH to supply an external JSON file that is deep-merged
over the built-in template (user keys win).  Useful for overriding sbatch
directives, timeouts, etc. without touching this script.

App entry builders
------------------
The functions _build_uniform_apps() and _build_hierarchical_apps() define the
per-experiment "apps" sub-structure inside the global config.  Edit them here
if the runtime scheduler's expected schema changes.

GPU / node accounting
----------------------
  node_count   = ceil(gpus / --gpus-per-node)
  n_total_gpus = sum of raw GPU counts from the design (exact, no rounding)

Usage examples
--------------
  # uniform, device placement (default for uniform)
  python expand_experiments.py experiments.json \\
      --gpus-per-node 8 --n-total-nodes 1 --output-dir ./exp_files

  # hierarchical, runtime placement (default for hierarchical)
  python expand_experiments.py experiments.json \\
      --gpus-per-node 4 --n-total-nodes 16 --output-dir ./exp_files

  # explicit random placement
  python expand_experiments.py experiments.json \\
      --gpus-per-node 4 --n-total-nodes 16 \\
      --placement-mode random --output-dir ./exp_files

  # hardcoded via oracle
  python expand_experiments.py experiments.json \\
      --gpus-per-node 4 --n-total-nodes 16 \\
      --placement-mode hardcoded --system mycluster \\
      --oracle-program ./placement_oracle \\
      --reserved-nodes node0005,node0042 \\
      --output-dir ./exp_files

  # force exactly one large job per experiment
  python expand_experiments.py experiments.json \\
      --gpus-per-node 4 --n-total-nodes 16 \\
      --split-large-small --output-dir ./exp_files

  # merge a custom global-config template
  python expand_experiments.py experiments.json \\
      --gpus-per-node 4 --n-total-nodes 16 \\
      --global-template my_overrides.json --output-dir ./exp_files

  # suppress design traceability metadata
  python expand_experiments.py experiments.json \\
      --gpus-per-node 4 --n-total-nodes 16 \\
      --no-design-meta --output-dir ./exp_files
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pprint import pprint
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from command_map import get_command, _STRATEGIES_NUM_RUNS, _STRATEGIES_NUM_RUNS_B200

sys.path.append(str(Path(__file__).parent.parent / "common"))
from utils.slurm import expand_slurm_nodelist
from JobPlacer.cli_wrapper import JobPlacer, JobRequest, PlacementResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR: str     = "experiments"
DEFAULT_ORACLE_PROGRAM: str = "placement_oracle"
GLOBAL_CONFIG_FILENAME: str = "global_config.json"

# Valid placement modes per interconnect type
VALID_PLACEMENTS: dict = {
    "uniform":      {"device", "random", "linear"},
    "hierarchical": {"runtime", "random", "linear", "hardcoded"},
}

# In seconds
RUNTIME_ESTIMATES = {
    # Leonardo
    'leonardo__DP__8__A100':                5.935904,
    'leonardo__DP__16__A100':               7.114242,
    
    'leonardo__FSDP__16__A100':             90.286853,
    'leonardo__FSDP__32__A100':             89.194575,
    
    'leonardo__DP+PP__16__A100':            14.222422,
    'leonardo__DP+PP__32__A100':            14.404939,
    'leonardo__DP+PP__64__A100':            21.96247,
    
    'leonardo__DP+PP+TP__224__A100':        34.089267,
    'leonardo__DP+PP+TP__256__A100':        47.625327,
    'leonardo__DP+PP+TP__512__A100':        37.044405,
    
    'leonardo__DP+PP+Expert__512__A100':    214.022749,
    'leonardo__DP+PP+Expert__1024__A100':   221.582458,
    
    # Alps (daint)
    # 'alps__DP__8__H200':                    15.139591,
    # 'alps__DP__16__H200':                   17.387984,
    
    # 'alps__FSDP__16__H200':                 34.149995,
    # 'alps__FSDP__32__H200':                 35.57791,
    
    # 'alps__DP+PP__16__H200':                23.598698,
    # 'alps__DP+PP__32__H200':                70.0, # FIXME random estimate
    # 'alps__DP+PP__64__H200':                70.0,
    
    # 'alps__DP+PP+TP__224__H200':            72.094135,
    # 'alps__DP+PP+TP__256__H200':            72.666106,
}

MIN_CONCURRENT_RUNTIME = 105 # FIXME

# ---------------------------------------------------------------------------
# Placement Oracle  (hardcoded mode only)
# ---------------------------------------------------------------------------

class PlacementOracle:
    """
    Thin wrapper around the external placement oracle program.

    Queried once per experiment in hardcoded mode.  If unreachable, a stub is
    used that always returns infeasible so every experiment file is still written.
    """

    def __init__(
        self,
        program: str,
        system: str,
        reserved_nodes: Optional[list],
        use_placer_files: bool
    ) -> None:
        topology_file=None
        topology_toml_file=None
        sinfo_file=None
        if use_placer_files:
            topology_file=f'../common/JobPlacer/{system}_topo.txt'
            sinfo_file=f'../common/JobPlacer/{system}_sinfo.txt'
        if system.lower() == 'alps':
            topology_toml_file=f'../common/JobPlacer/systems/{system.upper()}.toml'
            
        self.oracle = JobPlacer(
            system=system,
            topology_file=topology_file,
            topology_toml_file=topology_toml_file,
            sinfo_file=sinfo_file,
            nodelist=reserved_nodes,
            verbose=False,
        )
        self.system = system
        self.reserved_nodes: list = reserved_nodes or []
        self._available = self._probe()
        if not self._available:
            print(
                f"[PlacementOracle] '{program}' not found / not responding. "
                "Using stub (all placements infeasible).",
                file=sys.stderr,
            )

    def _probe(self) -> bool:
        try:
            r = subprocess.run([self.oracle._binary, "--help"],
                               capture_output=True, timeout=2)
            return r.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def find_placement(self, jobs: list, seed=None, timeout=5.0, svg_out=None) -> PlacementResult:
        """Query the oracle for one experiment's node assignments."""        
        oracle_jobs = {}
        for j in jobs:
            oracle_jobs[j["job_name"]] = JobRequest(
                num_nodes=j['node_count'],
                placement_class=str(j['placement_class']).lower()
            )
        
        tot_nodes_asked = sum([int(j["node_count"]) for j in jobs])
        print(f'[oracle] Finding placement for {",".join([j["job_name"] for j in jobs])}. Total required nodes: {tot_nodes_asked}')
        print(f'[oracle] Pattern: [{",".join([str(j["node_count"]) for j in jobs])}]')
        res = self.oracle.place(
            oracle_jobs,
            seed=seed,
            timeout=timeout,
            extra_args=['--out-svg', str(svg_out)] if svg_out else None,
        )
        print(f'[oracle] {"OK" if res.ok else "FAILED"}')
        if res.placements:
            used_nodes = set()
            # pprint(res.placements)
            tot_placed = sum([len(p) for p in res.placements.values()])
            print((f"[oracle] Tot placed nodes: {tot_placed}"))
            assert tot_placed == tot_nodes_asked
            for j in jobs:
                job_nodes = set(res.placements[j["job_name"]])
                prev_used_nodes_len = len(used_nodes)
                assert len(res.placements[j["job_name"]]) == len(job_nodes)
                # print(f'{used_nodes=}')
                # print(f'{job_nodes=}')
                used_nodes.update(job_nodes)
                # print(prev_used_nodes_len + len(job_nodes), len(used_nodes))
                assert prev_used_nodes_len + len(job_nodes) == len(used_nodes)
                # print(j['node_count'], len(res.placements[j["job_name"]]))
                if j['node_count'] != len(res.placements[j["job_name"]]):
                    print(j)
                    print(res.placements[j["job_name"]])
                    exit(15)
                
        print()
        
        return res

# ---------------------------------------------------------------------------
# Placement summary  (hardcoded mode only)
# ---------------------------------------------------------------------------

@dataclass
class PlacementSummary:
    """Accumulates per-experiment oracle statistics."""
    total: int = 0
    feasible: int = 0
    infeasible: int = 0
    infeasible_reasons: Counter = field(default_factory=Counter)
    by_class: dict = field(default_factory=dict)   # dominant class → [ok, nok]

    def record(self, result: PlacementResult, placement_classes: list) -> None:
        self.total += 1
        dominant = (Counter(placement_classes).most_common(1)[0][0]
                    if placement_classes else "unknown")
        self.by_class.setdefault(dominant, [0, 0])
        if result.ok:
            self.feasible += 1
            self.by_class[dominant][0] += 1
        else:
            self.infeasible += 1
            self.by_class[dominant][1] += 1
            self.infeasible_reasons[result.reason or "unknown reason"] += 1

    def print_report(self) -> None:
        W = 72
        print(f"\n\033[34m{'='*W}")
        print("PLACEMENT ORACLE SUMMARY")
        print(f"{'='*W}\033[0m")
        print(f"  Total experiments queried : {self.total}")
        print(f"  Feasible placements       : \033[32m{self.feasible}\033[0m")
        print(f"  Infeasible placements     : \033[31m{self.infeasible}\033[0m")
        if self.total:
            print(f"  Feasibility rate          : "
                  f"{100.0 * self.feasible / self.total:.1f}%")
        if self.by_class:
            print("\n  By dominant placement class:")
            for pc in sorted(self.by_class):
                ok, nok = self.by_class[pc]
                bar = "█" * ok + "░" * nok
                print(f"    {pc:<22}  {bar:<20}  "
                      f"feasible={ok}  infeasible={nok}  ({ok+nok})")
        if self.infeasible_reasons:
            print("\n  Infeasibility reasons:")
            for reason, n in self.infeasible_reasons.most_common():
                print(f"    [{n:3}]  {reason}")
        print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict: override values win; nested dicts are merged."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _extract_records(doc: dict) -> tuple[list, list, str]:
    """
    Return (records, interconnect_type).
    interconnect_type is "uniform" or "hierarchical".
    """
    if "hierarchical_experiments" in doc:
        return doc["hierarchical_experiments"], doc["baseline_set_topology"], "hierarchical"
    
    return doc["experiments"], doc["baseline_set"], "uniform"


def _default_placement(interconnect: str) -> str:
    return "device" if interconnect == "uniform" else "runtime"


def _gpus_to_nodes(gpus: int, gpus_per_node: int) -> int:
    return math.ceil(gpus / gpus_per_node) if gpus_per_node is not None else 1


def get_job_name(strategy, gpus, nodes, placement, idx) -> str:
    return f"{strategy}_g{gpus}_n{nodes}_{placement}_{idx}"

def get_concurrent_run_descriptor(pattern_id, entropy_bin: str, placement_score: float) -> str:
    if not placement_score:
        return f"patt{pattern_id}_ent{entropy_bin.capitalize()}_placNA"
    return f"patt{pattern_id}_ent{entropy_bin.capitalize()}_plac{int(placement_score*100.0)}"


def _classify_jobs(
    runs: list,
    small_threshold: Optional[int],
    force_single_large: bool,
) -> list:
    """
    Return [(run_dict, is_large: bool), ...].

    force_single_large=True  → exactly one job (largest by gpus) is large.
    small_threshold is None  → all jobs are small.
    small_threshold is int   → jobs with gpu_count > threshold are large.
    """
    gpu_counts = [r["gpus"] for r in runs]
    if force_single_large:
        large_idx = gpu_counts.index(max(gpu_counts))
        return [(run, i == large_idx) for i, run in enumerate(runs)]
    if small_threshold is None:
        return [(run, False) for run in runs]
    return [(run, gpu_counts[i] >= small_threshold)
            for i, run in enumerate(runs)]


# ---------------------------------------------------------------------------
# Per-experiment JSON builder
# ---------------------------------------------------------------------------

def build_experiment_json(
    rec: dict,
    gpu_model: str,
    gpus_per_node: int,
    comm_lib: str,
    placement_mode: str,           # device|random|linear|runtime|hardcoded
    oracle: Optional[PlacementOracle],
    placement_summary: Optional[PlacementSummary],
    small_threshold: Optional[int],
    force_single_large: bool,
    include_design_meta: bool,
    idx: int,
    out_dir: Path,
) -> dict:
    """
    Build and return the dict for a single experiment JSON file.

    Job-entry placement rules
    -------------------------
    device / random / linear  → no placement_class key on job entries
    runtime                   → placement_class from placement_class_vector
    hardcoded (feasible)      → explicit nodelist on job entries
    hardcoded (infeasible)    → placement_class = "hardcoded_infeasible"
                                + infeasible_reason
    """
    runs = rec["config"]["runs"]                  # [{strategy, gpus}, ...]
    pcv  = rec.get("placement_class_vector", [])  # per-job class (hierarchical)
    seed = rec.get('placement_seed')

    classified = _classify_jobs(runs, small_threshold,
                                force_single_large)

    # ── Oracle query (hardcoded mode) ────────────────────────────────────────
    oracle_result: Optional[PlacementResult] = None
    if placement_mode == "hardcoded":
        assert oracle is not None, "Oracle must be set for hardcoded mode"
        payload = [
            {
                "job_name":        get_job_name(run['strategy'], run['gpus'], _gpus_to_nodes(run["gpus"], gpus_per_node), pcv[i], i),
                "node_count":      _gpus_to_nodes(run["gpus"], gpus_per_node),
                "placement_class": pcv[i] if i < len(pcv) else "random",
            }
            for i, (run, _) in enumerate(classified)
        ]
        svg_out = None
        # svg_out = '_'.join([f'{r["strategy"]}-{int(r["gpus"]/4)}' for r, _ in classified])
        # can be commented
        svg_out = str(idx).zfill(2)
        svg_out = Path(out_dir / f'_topos/topo_{svg_out}_{get_concurrent_run_descriptor(rec.get("pattern_id"), rec.get("entropy_bin"), rec.get("placement_score"))}.svg')
        svg_out.parent.mkdir(exist_ok=True, parents=True)
        oracle_result = oracle.find_placement(payload, seed=seed, svg_out=svg_out)
        # print(oracle_result)
        # print()
        # print()
        if placement_summary is not None:
            classes = [pcv[i] if i < len(pcv) else "random"
                       for i in range(len(classified))]
            placement_summary.record(oracle_result, classes)

    # ── Build job entries ────────────────────────────────────────────────────
    small_jobs: dict = {}
    large_jobs: dict = {}
    sc = lc = 1

    for i, (run, is_large) in enumerate(classified):
        gpus  = run["gpus"]
        nodes = _gpus_to_nodes(gpus, gpus_per_node)

        # Base fields — always present
        entry: dict = {
            "command":          get_command(run["strategy"], gpus, comm_lib, gpu_model=gpu_model, use_dgx=False),
            "nodes":            nodes,
            "gpus":             gpus,
        }
        
        placement = 'na'

        # Placement-mode-specific fields
        if placement_mode in ("device", "random", "linear"):
            # No extra key: placement is expressed at the top level only
            pass

        elif placement_mode == "runtime":
            # placement_class comes from the design vector
            entry["placement_class"] = pcv[i] if i < len(pcv) else "random"
            placement = entry["placement_class"]

        elif placement_mode == "hardcoded":
            job_name = get_job_name(run['strategy'], gpus, nodes, pcv[i], i)
            if oracle_result is not None and oracle_result.ok and oracle_result.placements is not None:
                nodelist = oracle_result.placements.get(job_name)
                if nodelist:
                    entry["nodelist"] = nodelist
                    entry["placement_class"] = pcv[i]
                else:
                    # Oracle said feasible but returned no list for this job
                    entry["placement_class"] = "hardcoded_partial"
            else:
                entry["placement_class"] = "hardcoded_infeasible"
                if oracle_result and oracle_result.reason:
                    entry["infeasible_reason"] = oracle_result.reason
                    
            placement = entry["placement_class"]

        job_name = get_job_name(run['strategy'], gpus, nodes, placement, i)
        if is_large:
            large_jobs[job_name] = entry; lc += 1
        else:
            small_jobs[job_name] = entry; sc += 1

    # ── Top-level placement ──────────────────────────────────────────────────
    if placement_mode == "hardcoded":
        top_placement = (
            "hardcoded"
            if oracle_result is not None and oracle_result.ok
            else "hardcoded_infeasible"
        )
    else:
        # device | random | linear | runtime — written verbatim
        top_placement = placement_mode

    # ── n_total_gpus: sum of raw GPU counts (exact, no rounding) ─────────────
    total_gpus = sum(run["gpus"] for run, _ in classified)

    inner: dict = {
        "placement":      top_placement,
        "placement_seed": seed,
        "gpus_per_node":  gpus_per_node if gpus_per_node else total_gpus,
        "n_total_gpus":   total_gpus,
        "n_small_jobs":   len(small_jobs),
        "n_large_jobs":   len(large_jobs),
        "n_total_jobs":   len(classified),
    }
    if small_jobs:
        inner["small_jobs"] = small_jobs
    if large_jobs:
        inner["large_jobs"] = large_jobs
    if include_design_meta:
        inner["_meta"] = {
            "pattern":          rec.get("pattern"),
            "pattern_family":   rec.get("pattern_family"),
            "pattern_id":       rec.get("pattern_id"),
            "entropy_bin":      rec.get("entropy_bin"),
            "placement_bin":    rec.get("placement_bin"),
            "placement_vector": pcv,
            "placement_score":  rec.get("placement_score"),
            "placement_seed":   seed,
        }

    return inner

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    with open(args.input_json, encoding="utf-8") as fh:
        doc = json.load(fh)

    records, baselines, interconnect = _extract_records(doc)
    # print(interconnect)
    # print(records)
    # exit()
    if not records:
        print("No experiments found in input JSON.", file=sys.stderr)
        sys.exit(1)

    # Estimate total runtime
    runtimes = defaultdict(dict)
    for meta, runtime in RUNTIME_ESTIMATES.items():
        system, strategy, gpus, compute = meta.split('__')
        runtimes[f'{system}__{compute}'][f'{strategy}__{int(gpus)}'] = runtime
    
    for sys_compute, est_runtimes in runtimes.items():
        estimates = estimate_experiment_times(records, baselines, est_runtimes, args.gpu_model)
        print(f"\n\033[36m{'='*72}")
        print(f"RUNTIME ESTIMATION for {sys_compute}")
        print(f"{'='*72}\033[0m")
        print(f"  Total Baseline (Sequential), #runs={len(estimates['baseline_times']):<3}  : {estimates['baseline_mins']:.2f} minutes")
        print(f"  Total Concurrent Execution,  #runs={len(estimates['concurrent_times']):<3}  : {estimates['concurrent_mins']:.2f} minutes")
        print(f"{'-'*72}")
        print(f"  Baseline times   (seconds): {estimates['baseline_times']}")
        print(f"  Concurrent times (seconds): {estimates['concurrent_times'][0]} and {len(estimates['concurrent_times'])-1} more")
        print(f"{'-'*72}")
        print()

    # ── Resolve placement mode ───────────────────────────────────────────────
    if args.placement_mode is None:
        placement_mode = _default_placement(interconnect)
        print(f"[expand] Interconnect: {interconnect!r}  →  "
              f"default placement: {placement_mode!r}")
    else:
        placement_mode = args.placement_mode

    # Validate placement against interconnect type
    allowed = VALID_PLACEMENTS[interconnect]
    if placement_mode not in allowed:
        print(
            f"Error: placement {placement_mode!r} is not valid for "
            f"{interconnect!r} interconnect.  Allowed: {sorted(allowed)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Warn if runtime requested but placement_class_vector is missing
    if placement_mode == "runtime":
        missing = sum(1 for r in records if "placement_class_vector" not in r)
        if missing:
            print(
                f"Error: {missing} record(s) lack placement_class_vector; "
                "those jobs will fall back to 'random'.",
                file=sys.stderr,
            )
            exit(1)

    # ── Oracle setup (hardcoded mode only) ───────────────────────────────────
    oracle: Optional[PlacementOracle] = None
    summary: Optional[PlacementSummary] = None
    if placement_mode == "hardcoded":
        reserved: list = []
        if args.reserved_nodes:
            reserved = expand_slurm_nodelist(args.reserved_nodes)
            preview = ", ".join(reserved[:10]) + ("…" if len(reserved) > 10 else "")
            print(f"[oracle] Reserved nodes ({len(reserved)}): {preview}")
        oracle  = PlacementOracle(
            args.oracle_program,
            args.system,
            reserved,
            args.use_placer_files
        )
        summary = PlacementSummary()
        print(f"[oracle] System: {args.system!r}  "
              f"Program: {args.oracle_program!r}  "
              f"Available: {oracle._available}")

    # ── Output directory ─────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_digits   = len(str(len(records)))
    exp_files: list = []   # bare filenames, for the global config
    written:   list = []   # full paths, for the final summary

    # ── Write per-experiment files ───────────────────────────────────────────
    for idx, rec in enumerate(records, start=1):
        exp_doc = build_experiment_json(
            rec=rec,
            gpu_model=args.gpu_model,
            comm_lib=args.comm_lib,
            gpus_per_node=args.gpus_per_node,
            placement_mode=placement_mode,
            oracle=oracle,
            placement_summary=summary,
            small_threshold=args.small_job_threshold,
            force_single_large=args.split_large_small,
            include_design_meta=args.include_design_meta,
            idx=idx,
            out_dir=out_dir,
        )
        fname = f"exp_{str(idx).zfill(n_digits)}_{get_concurrent_run_descriptor(rec.get("pattern_id"), rec.get("entropy_bin"), rec.get("placement_score"))}.json"
        fpath = out_dir / fname
        with open(fpath, "w", encoding="utf-8") as fh:
            json.dump(exp_doc, fh, indent=2, ensure_ascii=False)
        exp_files.append(fname)
        written.append(fpath)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n\033[32m[expand] Wrote {len(written)} experiment files "
          f"→ {out_dir}/\033[0m")
    if written:
        print(f"  First : {written[0].name}")
        print(f"  Last  : {written[-1].name}")
    # print(f"\033[32m[expand] Wrote global config → {global_path}\033[0m")
    print(f"  Interconnect  : {interconnect}")
    print(f"  Placement     : {placement_mode}")

    if args.split_large_small:
        print("  Classification: --split-large-small  "
              "(1 large = largest job, rest small)")
    elif args.small_job_threshold is None:
        print("  Classification: all jobs small  (no threshold set)")
    else:
        print(f"  Classification: threshold = {args.small_job_threshold} gpus ({_gpus_to_nodes(args.small_job_threshold, args.gpus_per_node)} nodes)")

    if summary is not None:
        summary.print_report()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="expand_experiments.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    p.add_argument(
        "input_json", metavar="EXPERIMENTS_JSON",
        help="Master JSON produced by experiment_design.py.",
    )

    p.add_argument(
        "--comm-lib", type=str, required=True, metavar="COMM_LIB", 
        help="Communication library to use", choices=["nccl", "rccl", "oneccl"]
    )
    
    p.add_argument(
        "--gpu-model", type=str, required=True, metavar="GPU_MODEL", 
        help="The GPU model to emulate compute time (sleep)", choices=["B200", "H200", "A100"]
    )

    # Required
    p.add_argument(
        "--gpus-per-node", type=int, required=False, metavar="N", default=None,
        help="GPUs per physical node on the target system.  REQUIRED.",
    )

    # Placement mode
    pm = p.add_argument_group("placement mode")
    pm.add_argument(
        "--placement-mode",
        choices=["device", "random", "linear", "runtime", "hardcoded"],
        default=None,
        metavar="MODE",
        help=(
            "Placement strategy written to every experiment file.  "
            "Default: 'device' for uniform interconnects, 'runtime' for hierarchical.  "
            "Allowed values per interconnect — "
            "uniform: device, random, linear, hardcoded; "
            "hierarchical: runtime, random, linear, hardcoded.  "
            "device/random/linear: job entries carry NO placement_class.  "
            "runtime: job entries carry placement_class from the design vector.  "
            "hardcoded: job entries carry explicit nodelists from the oracle."
        ),
    )

    # Oracle options
    og = p.add_argument_group("oracle options  (--placement-mode hardcoded)")
    og.add_argument(
        "--system", metavar="NAME", default=None,
        help="Target system name passed to the oracle.  "
             "REQUIRED with --placement-mode hardcoded.",
    )
    og.add_argument(
        "--oracle-program", metavar="PATH", default=DEFAULT_ORACLE_PROGRAM,
        help=f"Oracle executable (default: '{DEFAULT_ORACLE_PROGRAM}').  "
             "Must read JSON from stdin and write JSON to stdout.  "
             "If unreachable a stub is used (all infeasible).",
    )
    og.add_argument(
        "--reserved-nodes", metavar="N1,N2,…",
        help="Comma-separated nodes to exclude; forwarded to the oracle.",
    )

    # Job classification
    cg = p.add_argument_group("job classification")
    cg.add_argument(
        "--split-large-small", action="store_true", default=False,
        help=(
            "Classify exactly ONE job as large (the one with the most nodes; "
            "ties broken by first occurrence).  All others are small.  "
            "Overrides --small-job-threshold."
        ),
    )
    cg.add_argument(
        "--small-job-threshold", type=int, default=None, metavar="N",
        help=(
            "Jobs with node_count > N → large; rest → small.  "
            "Default: unset (all jobs are small).  "
            "No effect when --split-large-small is active."
        ),
    )

    # Output
    p.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR, metavar="DIR",
        help=f"Output folder (default: '{DEFAULT_OUTPUT_DIR}').  "
             f"Created if absent.  Experiment files and '{GLOBAL_CONFIG_FILENAME}' "
             "are both written here.",
    )
    p.add_argument(
        "--results-dir", default="results", metavar="DIR",
        help="Results directory embedded in app args in the global config "
             "(default: 'results').",
    )
    p.add_argument(
        "--no-design-meta", dest="include_design_meta",
        action="store_false", default=True,
        help=(
            "Omit the '_design_meta' traceability block from output files "
            "(pattern, family, entropy_bin, placement_bin, placement_score, "
            "placement_seed).  Included by default."
        ),
    )
    p.add_argument(
        "--use-placer-files", action="store_true", default=False,
        help=(
            "If true, passes to JobPlacer the txt/toml topo and sinfo files"
        ),
    )

    return p


def _validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not os.path.isfile(args.input_json):
        parser.error(f"Input file not found: '{args.input_json}'.")
    if args.small_job_threshold is not None and args.small_job_threshold < 1:
        parser.error("--small-job-threshold must be ≥ 1.")
    if args.placement_mode == "hardcoded" and not args.system:
        parser.error("--placement-mode hardcoded requires --system NAME.")
    if args.placement_mode != "device" and (args.gpus_per_node is None or args.gpus_per_node < 1):
        parser.error("--gpus-per-node must be ≥ 1.")

def get_total_runs(strategy: str, gpu_model: str) -> int:
    """Calculate total iterations (min_runs + max_runs) using command_map dicts."""
    
    # Select the correct dictionary based on the GPU model
    target_dict = _STRATEGIES_NUM_RUNS_B200 if gpu_model == "B200" else _STRATEGIES_NUM_RUNS
    
    # Raise an exception if the strategy isn't found
    if strategy not in target_dict:
        raise ValueError(f"Unknown strategy '{strategy}' for GPU model '{gpu_model}'.")
        
    runs_tuple = target_dict[strategy]
    
    # Total runs = min_runs + max_runs
    return runs_tuple[0] + runs_tuple[1]

def estimate_experiment_times(records: list, baselines: list, profile_data: dict, gpu_model: str) -> dict:
    """
    Estimates the runtime of the baseline (sequential) vs concurrent placements.
    profile_data: dict mapping strategy names to their single-iteration time in seconds.
    """
    baseline_secs = []
    total_baseline_secs = 0.0
    concurrent_secs = []
    total_concurrent_secs = 0.0
    
    for bas in baselines:
        if 'run' in bas:
            bas = bas['run']
        time = profile_data[f'{bas["strategy"]}__{bas["gpus"]}']
        baseline_secs.append(time)
        total_baseline_secs += time
    
    for rec in records:
        runs = rec["config"]["runs"]
        job_times = []
        
        for run in runs:
            strat = run["strategy"]
            gpus = run["gpus"]
            
            time_per_iter = profile_data[f'{strat}__{gpus}']
            job_times.append(time_per_iter)
        
        if job_times:
            concurrent_secs.append(job_times)
            total_concurrent_secs += max(MIN_CONCURRENT_RUNTIME, max(job_times))
            
    return {
        "concurrent_times": concurrent_secs,
        "baseline_times": baseline_secs,
        "concurrent_mins": total_concurrent_secs / 60.0,
        "baseline_mins": total_baseline_secs / 60.0
    }

if __name__ == "__main__":
    _p = build_parser()
    _a = _p.parse_args()
    _validate(_a, _p)
    main(_a)
