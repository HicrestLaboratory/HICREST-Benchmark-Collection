"""
launch_baseline_singlenode.py
==============================
Launches single-node baseline experiments using sbatchman.

Reads the baseline_set from experiments.json, filters runs that fit on a
single node (gpus <= gpus_per_node), and launches them sequentially via
sbatchman.launch_job — each job waits for the previous one to complete.

Usage
-----
  python launch_baseline_singlenode.py experiments.json \\
      --config-name my_config \\
      --comm-lib nccl \\
      --gpus-per-node 72

  # dry run (does not actually submit jobs)
  python launch_baseline_singlenode.py experiments.json \\
      --config-name my_config \\
      --comm-lib nccl \\
      --gpus-per-node 72 \\
      --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path

import sbatchman as sbm


sys.path.append(str(Path(__file__).parent))
from command_map import get_command, FEASIBLE_GPU_COUNTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_baseline_runs(input_json: str, gpus_per_node: int) -> list[dict]:
    with open(input_json, encoding="utf-8") as fh:
        doc = json.load(fh)

    if "baseline_set" not in doc:
        print("Error: 'baseline_set' key not found in input JSON.", file=sys.stderr)
        sys.exit(1)

    all_runs = doc["baseline_set"]
    single_node = [r for r in all_runs if r["gpus"] <= gpus_per_node]
    skipped     = len(all_runs) - len(single_node)

    print(f"[baseline] Total baseline runs      : {len(all_runs)}")
    print(f"[baseline] Single-node (≤ {gpus_per_node} GPUs)  : {len(single_node)}")
    if skipped:
        print(f"[baseline] Skipped (multi-node)     : {skipped}")

    if not single_node:
        print("Error: no single-node baseline runs after filtering.", file=sys.stderr)
        sys.exit(1)

    return single_node


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    runs = _load_baseline_runs(args.input_json, args.gpus_per_node)

    print(f"\n[baseline] Config name : baseline_gpus{num_gpus}")
    print(f"[baseline] Comm lib    : {args.comm_lib}")
    print(f"[baseline] Dry run     : {args.dry_run}")
    print(f"[baseline] Launching {len(runs)} jobs sequentially...\n")

    previous_job_id: int | None = None

    for i, run in enumerate(runs, start=1):
        strategy = run["strategy"]
        num_gpus = run["gpus"]
        nodes = num_gpus / args.gpus_per_node

        command = get_command(strategy, num_gpus, args.comm_lib)

        command = f"srun -N {nodes} {command}"

        print(f"[{i:02d}/{len(runs):02d}] strategy={strategy}  gpus={num_gpus}")
        print(f"        command: {command}")
        if previous_job_id is not None:
            print(f"        waiting for job_id={previous_job_id}")

        job = sbm.launch_job(
            config_name      = f"baseline_gpus{num_gpus}",
            command          = command,
            tag              = f"baseline_{strategy}_{num_gpus}gpus",
            previous_job_id  = previous_job_id,
            dry_run          = args.dry_run,
        )

        previous_job_id = job.job_id
        print(f"        → job_id={job.job_id}\n")

    print(f"\n\033[32m[baseline] Done. {len(runs)} jobs launched.\033[0m")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="baseline_single_node.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument(
        "input_json", metavar="EXPERIMENTS_JSON",
        help="Master JSON produced by experiment_design.py. Must contain 'baseline_set'.",
    )
    p.add_argument(
        "--comm-lib", required=True, metavar="LIB",
        help="Communication library (e.g. 'nccl', 'mpi'). Injected into the command path.",
    )
    p.add_argument(
        "--gpus-per-node", type=int, default=72, metavar="N",
        help="GPUs per physical node. Runs with gpus > N are skipped (default: 72).",
    )
    p.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Pass dry_run=True to sbatchman — jobs are not actually submitted.",
    )
    return p


def _validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not os.path.isfile(args.input_json):
        parser.error(f"Input file not found: '{args.input_json}'.")
    if args.gpus_per_node < 1:
        parser.error("--gpus-per-node must be >= 1.")


if __name__ == "__main__":
    _p = build_parser()
    _a = _p.parse_args()
    _validate(_a, _p)
    main(_a)
