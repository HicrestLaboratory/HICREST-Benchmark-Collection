"""
run_baselines_no_placement.py
==============================
Launches single-node baseline experiments using sbatchman.

Reads the baseline_set from experiments.json, filters runs that fit on a
single node (gpus <= gpus_per_node), and launches them sequentially via
sbatchman.launch_job — each job waits for the previous one to complete.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import sbatchman as sbm


sys.path.append(str(Path(__file__).parent))
from command_map import EXTRA_SRUN_FLAGS, get_command
from collections import Counter

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
    single_node_count = len([r for r in all_runs if r["gpus"] <= gpus_per_node])

    print(f"[baseline] Total baseline runs     :{len(all_runs)}")
    print(f"[baseline] Single-node (≤ {gpus_per_node} GPUs)  : {single_node_count}")
    gpu_counts = Counter(r["gpus"] for r in all_runs)
    print("[baseline] Histogram of runs by GPU count:")
    for gpus, count in sorted(gpu_counts.items()):
        print(f"    {gpus:>3} GPUs : {count} runs")

    return all_runs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace, config_prefix:str) -> None:
    runs = _load_baseline_runs(args.input_json, args.gpus_per_node)

    print(f"[baseline] Comm lib    : {args.comm_lib}")
    print(f"[baseline] Dry run     : {args.dry_run}")
    print(f"[baseline] Launching {len(runs)} jobs sequentially...\n")

    previous_job_id: int | None = None

    for i, run in enumerate(runs, start=1):
        strategy = run["strategy"]
        num_gpus = int(run["gpus"])
        nodes = int(num_gpus / args.gpus_per_node) if num_gpus > args.gpus_per_node else 1
        
        if args.max_n_nodes and nodes > args.max_n_nodes:
            print(f'Skipping job {strategy} @ {nodes} nodes. Limit is {args.max_n_nodes}.')
            continue

        command = get_command(strategy, num_gpus, args.comm_lib, args.gpu_model, num_warmup_override=0, use_dgx=(args.dgx == "DGX_A100"))

        if args.use_mpirun:
            command = f"mpirun -np {num_gpus} {command}"
        else:
            command = f"srun -N{nodes} -n{num_gpus} --ntasks-per-node={args.gpus_per_node} --cpus-per-task={args.cpus_per_task} {' '.join(EXTRA_SRUN_FLAGS.get(args.system, []))} {command}"

        print(f"[{i:02d}/{len(runs):02d}] strategy={strategy}  gpus={num_gpus}")
        print(f"        command: {command}")
        if previous_job_id is not None:
            print(f"        waiting for job_id={previous_job_id}")

        config_name = f"{nodes}_{config_prefix}" if config_prefix == "nodes" else f"{num_gpus}_{config_prefix}"

        try:
            job = sbm.launch_job(
                config_name      = config_name,
                preprocess       = 'echo "Allocated nodes: $SLURM_JOB_NODELIST"',
                command          = command,
                tag              = f"baseline_{strategy}_{num_gpus}gpus_{nodes}nodes_comm-{args.comm_lib}_gpu-{args.gpu_model}",
                previous_job_id  = previous_job_id,
                dry_run          = args.dry_run,
                variables        = {'strategy': strategy, 'gpus': num_gpus, 'nodes': nodes, 'comm_lib': args.comm_lib, 'gpu_model': args.gpu_model, 'placement': 'na'},
                ignore_archived  = True,
                ignore_commands_in_dup_check = True,
            )
            previous_job_id = job.job_id
            print(f"        → job_id={job.job_id}\n")
        except Exception as e:
            previous_job_id = None
            print(e)
        print()

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
    p.add_argument("--dgx", required=False, help="Use DGX-A100 node.", choices=["DGX_A100"], default=None)

    p.add_argument(
        "--system", required=True, metavar="SYSTEM", choices=['leonardo', 'jupiter', 'alps', 'nvl72', 'baldo'],
        help="The name of the system",
    )
    p.add_argument(
        "--comm-lib", required=True, metavar="LIB",
        help="Communication library to use", choices=["nccl", "rccl", "oneccl", "mpi_gpu_cuda"]
    )
    p.add_argument(
        "--gpu-model", type=str, required=True, metavar="GPU_MODEL", 
        help="The GPU model to emulate compute time (sleep)", choices=["GB300", "B200", "H100", "H200", "A100", "GH200"]
    )
    p.add_argument(
        "--gpus-per-node", type=int, required=True, metavar="N",
        help="GPUs per physical node.",
    )
    p.add_argument(
        "--cpus-per-task", type=int, required=True, metavar="C",
        help="CPUs per task",
    )
    p.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Pass dry_run=True to sbatchman — jobs are not actually submitted.",
    )
    p.add_argument(
        "--config-prefix", type=str, default="nodes", metavar="PREFIX",
        help="Prefix for config names (default: 'nodes'). If set to 'gpus', config names will be based on GPU count instead of node count.",
    )
    p.add_argument(
        "--use-mpirun", action="store_true", default=False,
        help="If set, mpirun will be used instead of srun.",
    )
    p.add_argument(
        "--max-n-nodes", type=int, default=None,
        help="If set, it will only launch jobs that need at most --max-n-nodes nodes.",
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
    main(_a, config_prefix=_a.config_prefix)
