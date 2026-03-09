"""
expand_experiments.py
=====================
Parses the master JSON produced by experiment_design.py and writes one
experiment file per generated experiment into an output folder.

Each output file is named  experiment_NNNN.json  and contains a single
top-level key  "pattern_N"  with the structure expected by the runtime
scheduler.

Placement modes
---------------
Three modes are supported; only one may be active per run:

  (a) random  [default]
      Top-level placement_class = "random".
      Each job entry: {"strategy", "nodes", "gpus", "placement_class": "random"}

  (b) class-based  [--placement-mode class]
      Top-level placement_class = the dominant class (or "mixed").
      Each job entry: {"strategy", "nodes", "gpus", "placement_class": "classN"}
      The per-job class comes from the design's placement_class_vector.

  (c) hardcoded nodelists  [--placement-mode hardcoded]
      Requires --system NAME.
      Queries an external placement oracle once per experiment.
      Feasible: top-level placement_class = "hardcoded";
        job entry: {"strategy", "nodes", "gpus", "nodelist": [...]}
      Infeasible: top-level placement_class = "hardcoded_infeasible";
        job entry: {"strategy", "nodes", "gpus",
                    "placement_class": "hardcoded_infeasible",
                    "infeasible_reason": "..."}
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

--small-job-threshold N :
  jobs with node_count >  N  → large; all others → small.

--split-large-small :
  Exactly ONE job is large – the one requiring the most nodes
  (ties: first occurrence wins).  All others are small.
  --small-job-threshold has no effect when this flag is active.

Job entry structure (all modes)
---------------------------------
  random:
    {"strategy": "DP",  "nodes": 1, "gpus": 4,
     "placement_class": "random"}

  class:
    {"strategy": "DP",  "nodes": 4, "gpus": 16,
     "placement_class": "class2"}

  hardcoded (feasible):
    {"strategy": "DP",  "nodes": 4, "gpus": 16,
     "nodelist": ["node0001", ..., "node0004"]}

  hardcoded (infeasible):
    {"strategy": "DP",  "nodes": 4, "gpus": 16,
     "placement_class": "hardcoded_infeasible",
     "infeasible_reason": "<reason>"}

GPU / node accounting
----------------------
  node_count  = ceil(gpus / --gpus-per-node)
  n_total_gpus = sum of raw gpu counts from the design
                 (no rounding — design GPU counts are exact multiples)

Usage examples
--------------
  # random placement (default)
  python expand_experiments.py experiments.json \\
      --gpus-per-node 4 --n-total-nodes 92 --output-dir ./exp_files

  # class-based (topology design JSON)
  python expand_experiments.py experiments.json \\
      --gpus-per-node 4 --n-total-nodes 92 \\
      --placement-mode class --output-dir ./exp_files

  # hardcoded via oracle
  python expand_experiments.py experiments.json \\
      --gpus-per-node 4 --n-total-nodes 92 \\
      --placement-mode hardcoded --system mycluster \\
      --oracle-program ./placement_oracle \\
      --reserved-nodes node0005,node0042 \\
      --output-dir ./exp_files

  # force exactly one large job per experiment
  python expand_experiments.py experiments.json \\
      --gpus-per-node 4 --n-total-nodes 92 \\
      --split-large-small --output-dir ./exp_files

  # suppress design traceability metadata
  python expand_experiments.py experiments.json \\
      --gpus-per-node 4 --n-total-nodes 92 \\
      --no-design-meta --output-dir ./exp_files
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR: str = "experiments"
DEFAULT_ORACLE_PROGRAM: str = "placement_oracle"


# ---------------------------------------------------------------------------
# Placement Oracle  (mode c only)
# ---------------------------------------------------------------------------

@dataclass
class OracleResult:
    """Result of a single find_placement call."""
    feasible: bool
    assignments: dict = field(default_factory=dict)  # job_id → nodelist
    reason: str = ""


class PlacementOracle:
    """
    Thin wrapper around the external placement oracle program.

    Queried once per experiment in hardcoded mode.  The oracle receives the
    system name, reserved nodes, and a list of {job_id, node_count,
    placement_class} dicts, and returns either a per-job nodelist or an
    infeasibility signal.

    If the program is not reachable a stub is used that always returns
    infeasible, so every experiment file is still written.
    """

    def __init__(
        self,
        program: str = DEFAULT_ORACLE_PROGRAM,
        system: str = "",
        reserved_nodes: Optional[list] = None,
    ) -> None:
        self.program = program
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
            r = subprocess.run([self.program, "--ping"],
                               capture_output=True, timeout=2)
            return r.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def find_placement(self, jobs: list) -> OracleResult:
        """Query the oracle for one experiment's node assignments."""
        payload = {
            "command":        "find_placement",
            "system":         self.system,
            "reserved_nodes": self.reserved_nodes,
            "jobs":           jobs,
        }
        if not self._available:
            return OracleResult(feasible=False, reason="oracle unavailable (stub)")
        try:
            proc = subprocess.run(
                [self.program],
                input=json.dumps(payload),
                capture_output=True, text=True, timeout=30,
            )
            resp = json.loads(proc.stdout)
        except subprocess.TimeoutExpired:
            return OracleResult(feasible=False, reason="oracle timeout")
        except Exception as exc:
            return OracleResult(feasible=False, reason=f"oracle error: {exc}")

        if not resp.get("feasible", False):
            return OracleResult(
                feasible=False,
                reason=resp.get("reason", "oracle reported infeasible"),
            )
        assignments = {
            e["job_id"]: e["nodelist"]
            for e in resp.get("assignments", [])
        }
        return OracleResult(feasible=True, assignments=assignments)


# ---------------------------------------------------------------------------
# Placement summary  (mode c only)
# ---------------------------------------------------------------------------

@dataclass
class PlacementSummary:
    """Accumulates per-experiment oracle statistics."""
    total: int = 0
    feasible: int = 0
    infeasible: int = 0
    infeasible_reasons: Counter = field(default_factory=Counter)
    # dominant placement class → [feasible_count, infeasible_count]
    by_class: dict = field(default_factory=dict)

    def record(self, result: OracleResult, placement_classes: list) -> None:
        self.total += 1
        dominant = (Counter(placement_classes).most_common(1)[0][0]
                    if placement_classes else "unknown")
        self.by_class.setdefault(dominant, [0, 0])
        if result.feasible:
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
                print(f"    {pc:<14}  {bar}  "
                      f"feasible={ok}  infeasible={nok}  ({ok+nok})")
        if self.infeasible_reasons:
            print("\n  Infeasibility reasons:")
            for reason, n in self.infeasible_reasons.most_common():
                print(f"    [{n:3}]  {reason}")
        print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_records(doc: dict) -> list:
    """Return experiment list; prefer hierarchical_experiments (topo mode)."""
    if "hierarchical_experiments" in doc:
        return doc["hierarchical_experiments"]
    return doc["experiments"]


def _gpus_to_nodes(gpus: int, gpus_per_node: int) -> int:
    return math.ceil(gpus / gpus_per_node)


def _classify_jobs(
    runs: list,
    gpus_per_node: int,
    small_threshold: Optional[int],
    force_single_large: bool,
) -> list:
    """
    Return [(run_dict, is_large: bool), ...].

    force_single_large=True  → exactly one job (largest by nodes) is large.
    small_threshold is None  → all jobs are small.
    small_threshold is int   → jobs with node_count > threshold are large.
    """
    node_counts = [_gpus_to_nodes(r["gpus"], gpus_per_node) for r in runs]
    if force_single_large:
        large_idx = node_counts.index(max(node_counts))
        return [(run, i == large_idx) for i, run in enumerate(runs)]
    if small_threshold is None:
        return [(run, False) for run in runs]
    return [(run, node_counts[i] > small_threshold)
            for i, run in enumerate(runs)]


# ---------------------------------------------------------------------------
# Per-experiment JSON builder
# ---------------------------------------------------------------------------

def build_experiment_json(
    rec: dict,
    exp_index: int,
    gpus_per_node: int,
    n_total_nodes: int,
    placement_mode: str,          # "random" | "class" | "hardcoded"
    oracle: Optional[PlacementOracle],
    placement_summary: Optional[PlacementSummary],
    small_threshold: Optional[int],
    force_single_large: bool,
    include_design_meta: bool,
) -> dict:
    runs = rec["config"]["runs"]                 # [{strategy, gpus}, ...]
    pcv  = rec.get("placement_class_vector", []) # per-job class (topo mode)

    classified = _classify_jobs(runs, gpus_per_node, small_threshold,
                                force_single_large)

    # ── Oracle query (hardcoded mode) ────────────────────────────────────────
    oracle_result: Optional[OracleResult] = None
    if placement_mode == "hardcoded":
        assert oracle is not None
        payload = [
            {
                "job_id":          i,
                "node_count":      _gpus_to_nodes(run["gpus"], gpus_per_node),
                "placement_class": pcv[i] if i < len(pcv) else "random",
            }
            for i, (run, _) in enumerate(classified)
        ]
        oracle_result = oracle.find_placement(payload)
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

        # Base fields always present
        entry: dict = {
            "strategy": run["strategy"],
            "nodes":    nodes,
            "gpus":     gpus,
        }

        if placement_mode == "random":
            entry["placement_class"] = "random"

        elif placement_mode == "class":
            # placement_class at job level; no seed embedded here
            entry["placement_class"] = pcv[i] if i < len(pcv) else "random"

        elif placement_mode == "hardcoded":
            if oracle_result is not None and oracle_result.feasible:
                nodelist = oracle_result.assignments.get(i)
                if nodelist:
                    entry["nodelist"] = nodelist
                else:
                    # Oracle said feasible but gave no list for this job
                    entry["placement_class"] = "hardcoded_partial"
            else:
                entry["placement_class"] = "hardcoded_infeasible"
                if oracle_result and oracle_result.reason:
                    entry["infeasible_reason"] = oracle_result.reason

        if is_large:
            large_jobs[f"job_{lc}"] = entry; lc += 1
        else:
            small_jobs[f"job_{sc}"] = entry; sc += 1

    # ── Top-level placement_class ────────────────────────────────────────────
    if placement_mode == "random":
        top_pc = "random"
    elif placement_mode == "class":
        if pcv:
            cnt = Counter(pcv)
            top_pc = cnt.most_common(1)[0][0] if len(cnt) == 1 else "mixed"
        else:
            top_pc = "random"
    else:
        top_pc = ("hardcoded"
                  if oracle_result is not None and oracle_result.feasible
                  else "hardcoded_infeasible")

    # ── n_total_gpus: sum raw GPU counts (exact, no rounding) ────────────────
    total_gpus = sum(run["gpus"] for run, _ in classified)

    inner: dict = {
        "placement_class": top_pc,
        "gpus_per_node":   gpus_per_node,
        "n_total_nodes":   n_total_nodes,
        "n_total_gpus":    total_gpus,
        "n_total_jobs":    len(classified),
        "n_small_jobs":    len(small_jobs),
        "n_large_jobs":    len(large_jobs),
    }
    if small_jobs:
        inner["small_jobs"] = small_jobs
    if large_jobs:
        inner["large_jobs"] = large_jobs
    if include_design_meta:
        inner["_design_meta"] = {
            "pattern":         rec.get("pattern"),
            "pattern_family":  rec.get("pattern_family"),
            "entropy_bin":     rec.get("entropy_bin"),
            "placement_bin":   rec.get("placement_bin"),
            "placement_score": rec.get("placement_score"),
            "placement_seed":  rec.get("placement_seed"),
        }

    return {f"pattern_{exp_index}": inner}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    with open(args.input_json, encoding="utf-8") as fh:
        doc = json.load(fh)

    records = _extract_records(doc)
    if not records:
        print("No experiments found in input JSON.", file=sys.stderr)
        sys.exit(1)

    # Topology / class-mode consistency
    has_topology = "hierarchical_experiments" in doc
    if args.placement_mode == "class" and not has_topology:
        print(
            "Warning: --placement-mode class requested but the master JSON "
            "has no placement_class_vector fields.  Falling back to 'random'.",
            file=sys.stderr,
        )
        args.placement_mode = "random"

    # Oracle setup (hardcoded mode only)
    oracle: Optional[PlacementOracle] = None
    summary: Optional[PlacementSummary] = None
    if args.placement_mode == "hardcoded":
        reserved: list = []
        if args.reserved_nodes:
            reserved = [n.strip() for n in args.reserved_nodes.split(",")
                        if n.strip()]
            preview = ", ".join(reserved[:10]) + ("…" if len(reserved) > 10 else "")
            print(f"[oracle] Reserved nodes ({len(reserved)}): {preview}")
        oracle  = PlacementOracle(args.oracle_program, args.system, reserved)
        summary = PlacementSummary()
        print(f"[oracle] System: '{args.system}'  "
              f"Program: '{args.oracle_program}'  "
              f"Available: {oracle._available}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_digits = len(str(len(records)))
    written: list = []

    for idx, rec in enumerate(records, start=1):
        exp_doc = build_experiment_json(
            rec=rec,
            exp_index=idx,
            gpus_per_node=args.gpus_per_node,
            n_total_nodes=args.n_total_nodes,
            placement_mode=args.placement_mode,
            oracle=oracle,
            placement_summary=summary,
            small_threshold=args.small_job_threshold,
            force_single_large=args.split_large_small,
            include_design_meta=args.include_design_meta,
        )
        fname = out_dir / f"experiment_{str(idx).zfill(n_digits)}.json"
        with open(fname, "w", encoding="utf-8") as fh:
            json.dump(exp_doc, fh, indent=2, ensure_ascii=False)
        written.append(fname)

    print(f"\n\033[32m[expand] Wrote {len(written)} experiment files "
          f"→ {out_dir}/\033[0m")
    if written:
        print(f"  First : {written[0].name}")
        print(f"  Last  : {written[-1].name}")

    if args.split_large_small:
        print("  Classification : --split-large-small  "
              "(1 large = largest job, rest small)")
    elif args.small_job_threshold is None:
        print("  Classification : all jobs small  (no threshold set)")
    else:
        print(f"  Classification : threshold = {args.small_job_threshold} nodes")

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

    p.add_argument("input_json", metavar="EXPERIMENTS_JSON",
                   help="Master JSON produced by experiment_design.py.")

    # required
    p.add_argument("--gpus-per-node", type=int, required=True, metavar="N",
                   help="GPUs per physical node on the target system.  REQUIRED.")
    p.add_argument("--n-total-nodes", type=int, required=True, metavar="N",
                   help="Total node count of the target cluster.  REQUIRED.")

    # placement mode
    pm = p.add_argument_group("placement mode")
    pm.add_argument(
        "--placement-mode", choices=["random", "class", "hardcoded"],
        default="class",
        help=("'random' (default): placement_class='random' for every job.  "
              "'class': per-job class from the design's placement_class_vector.  "
              "'hardcoded': query the placement oracle for concrete nodelists."),
    )

    # oracle options
    og = p.add_argument_group(
        "oracle options  (--placement-mode hardcoded)")
    og.add_argument("--system", metavar="NAME", default=None,
                    help="Target system name passed to the oracle.  "
                         "REQUIRED with --placement-mode hardcoded.")
    og.add_argument("--oracle-program", metavar="PATH",
                    default=DEFAULT_ORACLE_PROGRAM,
                    help=f"Oracle executable (default: '{DEFAULT_ORACLE_PROGRAM}').  "
                         "Must read JSON from stdin and write JSON to stdout.  "
                         "If unreachable, a stub is used (all infeasible).")
    og.add_argument("--reserved-nodes", metavar="N1,N2,…",
                    help="Comma-separated nodes to exclude; forwarded to oracle.")

    # classification
    cg = p.add_argument_group("job classification")
    cg.add_argument(
        "--split-large-small", action="store_true", default=False,
        help=("Classify exactly ONE job as large (the one with the most nodes; "
              "ties broken by first occurrence).  All others are small.  "
              "Overrides --small-job-threshold."),
    )
    cg.add_argument(
        "--small-job-threshold", type=int, default=None, metavar="N",
        help=("Jobs with node_count > N → large; rest → small.  "
              "Default: unset (all jobs are small).  "
              "No effect when --split-large-small is active."),
    )

    # output
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, metavar="DIR",
                   help=f"Output folder (default: '{DEFAULT_OUTPUT_DIR}').  "
                        "Created if absent.")
    p.add_argument(
        "--no-design-meta", dest="include_design_meta",
        action="store_false", default=True,
        help=("Omit the '_design_meta' traceability block from output files "
              "(pattern, family, entropy_bin, placement_bin, placement_score, "
              "placement_seed).  Included by default."),
    )

    return p


def _validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not os.path.isfile(args.input_json):
        parser.error(f"Input file not found: '{args.input_json}'.")
    if args.gpus_per_node < 1:
        parser.error("--gpus-per-node must be ≥ 1.")
    if args.n_total_nodes < 1:
        parser.error("--n-total-nodes must be ≥ 1.")
    if args.small_job_threshold is not None and args.small_job_threshold < 1:
        parser.error("--small-job-threshold must be ≥ 1.")
    if args.placement_mode == "hardcoded" and not args.system:
        parser.error("--placement-mode hardcoded requires --system NAME.")


if __name__ == "__main__":
    _p = build_parser()
    _a = _p.parse_args()
    _validate(_a, _p)
    main(_a)