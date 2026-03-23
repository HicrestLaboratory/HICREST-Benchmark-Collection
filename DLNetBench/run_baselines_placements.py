"""
run_baselines_with_placement.py
================================
Launches the topology baseline set from experiments.json with hardcoded
(oracle) placement, sequentially (each job waits for the previous one).

The oracle program path is read from meta.topology_program in the JSON itself.

Supported systems
-----------------
  leonardo  -- CINECA Leonardo (boost_usr_prod, A100)
  alps      -- CSCS Alps        (dummy values, update as needed)
  jupyter   -- Jupyter cluster  (dummy values, update as needed)

Adding a new system: add one entry to SYSTEM_CONFIGS.

Usage
-----
  python run_baselines_with_placement.py experiments.json \\
      --system leonardo \\
      --comm-lib nccl \\
      --gpu-model A100 \\
      --gpus-per-node 4

  # dry run
  python run_baselines_with_placement.py experiments.json \\
      --system leonardo --comm-lib nccl --gpu-model A100 \\
      --gpus-per-node 4 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import sbatchman as sbm

sys.path.append(str(Path(__file__).parent))
from command_map import EXTRA_SRUN_FLAGS, get_command
from expand_experiments import PlacementOracle

sys.path.append(str(Path(__file__).parent.parent / "common"))
from utils.slurm import expand_slurm_nodelist


# ---------------------------------------------------------------------------
# Per-system sbatch configs
# ---------------------------------------------------------------------------

SYSTEM_CONFIGS: dict[str, dict] = {
    "leonardo": {
        "cluster_name":  "leonardo",
        "partition":     "boost_usr_prod",
        "account":       "IscrC_OMG-25",
        "cpus_per_task": 32,
        "time":          "00:30:00",
        "gpus":          0,
        "qos":           "normal",
        "modules": [
            "gcc/12.2.0",
            "openmpi/4.1.6--gcc--12.2.0-cuda-12.2",
        ],
        "env": [
            "OMP_PROC_BIND=true",
            "OMP_NUM_THREADS=32",
            "NCCL_IB_SL=1",
            "UCX_IB_SL=1",
        ],
    },
    # TODO: fill in real values
    "alps": {
        "cluster_name":  "alps",
        "partition":     "normal",
        "account":       "your_alps_account",
        "cpus_per_task": 64,
        "time":          "00:30:00",
        "gpus":          0,
        "qos":           "default",
        "modules":       ["cray-mpich/8.1.28"],
        "env":           ["OMP_NUM_THREADS=64"],
    },
    # TODO: fill in real values
    "jupyter": {
        "cluster_name":  "jupyter",
        "partition":     "gpu",
        "account":       "your_jupyter_account",
        "cpus_per_task": 16,
        "time":          "00:30:00",
        "gpus":          0,
        "qos":           "normal",
        "modules":       ["openmpi/4.1.0"],
        "env":           ["OMP_NUM_THREADS=16"],
    },
}

ORACLE_TIMEOUT_S: float = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_doc(input_json: str) -> dict:
    with open(input_json, encoding="utf-8") as fh:
        doc = json.load(fh)
    if "baseline_set_topology" not in doc:
        print("Error: 'baseline_set_topology' not found in input JSON.\n"
              "Re-generate experiments.json with --use-topology.",
              file=sys.stderr)
        sys.exit(1)
    return doc


def _config_name(system: str, strategy: str, nodes: int,
                 placement_class_name: str, replicate_index: int) -> str:
    safe_strat = strategy.replace("+", "_")
    safe_class = placement_class_name.lower()
    return f"{system}_{safe_strat}_{nodes}nodes_{safe_class}_rep{replicate_index}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    sys_cfg  = SYSTEM_CONFIGS[args.system]
    doc      = _load_doc(args.input_json)
    baselines = doc["baseline_set_topology"]

    # oracle program comes from the JSON that generated this experiment set
    oracle_program = doc["meta"]["topology_program"]

    print(f"\n[baseline] Input         : {args.input_json}")
    print(f"[baseline] System          : {args.system}")
    print(f"[baseline] GPUs per node   : {args.gpus_per_node}")
    print(f"[baseline] CPUs per task   : {args.cpus_per_task}")
    print(f"[baseline] Comm lib        : {args.comm_lib}")
    print(f"[baseline] GPU model       : {args.gpu_model}")
    print(f"[baseline] Oracle program  : {oracle_program}")
    print(f"[baseline] Dry run         : {args.dry_run}")
    print(f"[baseline] Total entries   : {len(baselines)}")

    gpu_hist = Counter(e["run"]["gpus"] for e in baselines)
    print("[baseline] GPU count histogram:")
    for gpus, count in sorted(gpu_hist.items()):
        nodes = max(1, gpus // args.gpus_per_node)
        print(f"    {gpus:>5} GPUs  ({nodes:>3} nodes) : {count} run(s)")

    # -- Reserved nodes -------------------------------------------------------
    nodelist: list[str] = []
    if args.nodelist:
        nodelist = expand_slurm_nodelist(args.nodelist)
        preview = ", ".join(nodelist[:8]) + ("..." if len(nodelist) > 8 else "")
        print(f"[baseline] Available nodes ({len(nodelist)}): {preview}")

    # -- Oracle ---------------------------------------------------------------
    oracle = PlacementOracle(
        program=oracle_program,
        system=args.system,
        reserved_nodes=nodelist,
        use_placer_files=args.use_placer_files,
    )

    # -- Sequential launch loop -----------------------------------------------
    print(f"\n[baseline] Launching {len(baselines)} jobs sequentially...\n")

    previous_job_id: Optional[int] = None
    n_ok = 0
    n_skipped = 0

    for job_idx, entry in enumerate(baselines, start=1):
        strategy        = entry["run"]["strategy"]
        num_gpus        = int(entry["run"]["gpus"])
        nodes           = max(1, num_gpus // args.gpus_per_node)
        placement_class      = entry["placement_class"]       # e.g. "intra-group"       -- oracle format
        placement_class_name = entry["placement_class_name"]  # e.g. "INTRA_GROUP_RANDOM" -- used for config/tags
        seed                 = entry["seed"]
        replicate_index      = entry["replicate_index"]

        print(f"[{job_idx:03d}/{len(baselines):03d}] "
              f"strategy={strategy:<14}  gpus={num_gpus:<5}  nodes={nodes:<4}  "
              f"class={placement_class}  rep={replicate_index}  seed={seed}")

        # -- Oracle: resolve concrete nodelist --------------------------------
        job_name = (f"{strategy}_g{num_gpus}_n{nodes}"
                    f"_{placement_class_name}_rep{replicate_index}")
        payload = [{"job_name":        job_name,
                    "node_count":      nodes,
                    "placement_class": placement_class}]  # oracle expects label with dashes

        svg_out = Path("svgs") / f"topo_{job_name}.svg"
        svg_out.parent.mkdir(exist_ok=True, parents=True)
        oracle_result = oracle.find_placement(
            payload, seed=seed, timeout=ORACLE_TIMEOUT_S, svg_out=svg_out
        )

        if not oracle_result.ok:
            print(f"  [SKIP] Oracle failed: {oracle_result.reason}\n")
            n_skipped += 1
            continue

        nodelist: list[str] = oracle_result.placements.get(job_name, [])
        if len(nodelist) != nodes:
            print(f"  [SKIP] Oracle returned {len(nodelist)} nodes, "
                  f"expected {nodes}.\n")
            n_skipped += 1
            continue

        # -- Build command ----------------------------------------------------
        command = get_command(
            strategy, num_gpus, args.comm_lib,
            gpu_model=args.gpu_model,
            num_warmup_override=0,
        )
        command = f"srun -N{nodes} -n{num_gpus} --ntasks-per-node={args.gpus_per_node} --cpus-per-task={args.cpus_per_task} {' '.join(EXTRA_SRUN_FLAGS.get(args.system, []))} {command}"

        print(f"  nodelist : {nodelist}")
        print(f"  command  : {command}")
        if previous_job_id is not None:
            print(f"  depends  : job_id={previous_job_id}")

        # -- Create per-job sbatch config -------------------------------------
        config_name = _config_name(
            args.system, strategy, nodes, placement_class_name, replicate_index,
        )
        sbm.create_slurm_config(
            name=config_name,
            nodes=str(nodes),
            ntasks=str(nodes),
            nodelist=nodelist,
            overwrite=True,
            **sys_cfg,
        )

        # -- Submit -----------------------------------------------------------
        try:
            job = sbm.launch_job(
                config_name     = config_name,
                preprocess      = 'echo "Allocated nodes: $SLURM_JOB_NODELIST"',
                command         = command,
                tag             = (
                    f"baseline_{strategy}_{num_gpus}gpus_{nodes}nodes"
                    f"_comm-{args.comm_lib}_gpu-{args.gpu_model}"
                    f"_class-{placement_class_name}_rep{replicate_index}"
                ),
                previous_job_id = previous_job_id,
                dry_run         = args.dry_run,
                variables       = {
                    "strategy":        strategy,
                    "gpus":            num_gpus,
                    "nodes":           nodes,
                    "comm_lib":        args.comm_lib,
                    "gpu_model":       args.gpu_model,
                    "placement_class": placement_class_name,
                    "replicate_index": replicate_index,
                    "seed":            seed,
                    "system":          args.system,
                },
            )
            previous_job_id = job.job_id
            print(f"  -> job_id={job.job_id}\n")
            n_ok += 1
        except Exception as exc:
            print(f"  [ERROR] Submission failed: {exc}\n")
            previous_job_id = None

    # -- Summary --------------------------------------------------------------
    print(f"\n[baseline] Done.")
    print(f"  Submitted : {n_ok}")
    print(f"  Skipped   : {n_skipped}")
    print(f"  Total     : {len(baselines)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_baselines_with_placement.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument(
        "input_json", metavar="EXPERIMENTS_JSON",
        help="experiments.json produced by experiment_design.py "
             "(must contain 'baseline_set_topology').",
    )
    p.add_argument(
        "--system", required=True, choices=sorted(SYSTEM_CONFIGS),
        help="Target HPC system.",
    )
    p.add_argument(
        "--comm-lib", required=True, choices=["nccl", "rccl", "oneccl"],
        help="Communication library.",
    )
    p.add_argument(
        "--gpu-model", required=True, choices=["B200", "H200", "A100"],
        help="GPU model (selects compute-time emulation profile).",
    )
    p.add_argument(
        "--gpus-per-node", type=int, required=True, metavar="N",
        help="GPUs per physical node (derives node count from gpu count).",
    )
    p.add_argument(
        "--cpus-per-task", type=int, required=True, metavar="C",
        help="CPUs per task",
    )
    p.add_argument(
        "--nodelist", metavar="NODELIST", required=True,
        help="SLURM nodelist expression of nodes available for placement (e.g. 'node[001-340]'). Required when running outside a SLURM allocation.",
    )
    p.add_argument(
        "--use-placer-files", action="store_true", default=False,
        help="Use pre-computed topology/sinfo files instead of live scontrol/sinfo.",
    )
    p.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Print jobs without actually submitting them.",
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