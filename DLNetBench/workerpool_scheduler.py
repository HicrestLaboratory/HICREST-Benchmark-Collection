#!/usr/bin/env python3
"""
SLURM srun Job Scheduler
=========================
Must be launched INSIDE an existing salloc session.

Maintains N concurrent srun jobs split as:
  - 80% microjobs  : 2 nodes, 4 tasks/node, 1 cpu/task
  - 20% medium jobs: 4 or 8 nodes (chosen randomly), 4 tasks/node, 1 cpu/task

The main loop polls all running processes. When one finishes, its nodelist is
reused to immediately launch a replacement of the same type.

Each job writes its stdout to its own file inside workerpool_out/.
Scheduler debug logs go to stdout by default, or to a file if --output-log is given.

Usage:
    python slurm_scheduler.py <N> --nodelist [node01,node02,...] [--output-dir DIR] [--srun-extra "..."] [--output-log FILE]

Examples:
    python slurm_scheduler.py 10 --nodelist [node01,node02,node03,node04,node05,node06,node07,node08]
    python slurm_scheduler.py 20 --nodelist [node01,node02,node03,node04] --output-dir my_out --output-log scheduler.log
"""

import argparse
import math
import os
import random
import shlex
import subprocess
import sys
import time
import json
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURATION — edit application paths to match your environment
# ---------------------------------------------------------------------------

APPS: dict = {
    "FSDP": "$HOME/CRAB/benchmarks/blink/bin/a2a_comm_only -msgsize 512 -iter 1000",
    "DP+PP": "$HOME/CRAB/benchmarks/blink/bin/a2a_comm_only -msgsize 512 -iter 1000",
    "DP": "$HOME/CRAB/benchmarks/blink/bin/a2a_comm_only -msgsize 512 -iter 1000",
    "DP+PP+TP": "$HOME/CRAB/benchmarks/blink/bin/a2a_comm_only -msgsize 262144 -iter 10000"
}


# Node layout
MICROJOB_NODE_COUNT = 2       # microjobs always use exactly 2 nodes
MEDIUM_NODE_CHOICES = [4, 8]  # medium jobs randomly use 4 or 8 nodes
TASKS_PER_NODE      = 4
CPUS_PER_TASK       = 1

DEFAULT_OUTPUT_DIR  = "workerpool_out"
SCHEDULER_LOG_NAME  = "scheduler.log"
POLL_INTERVAL       = 2       # seconds between polls

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str, log_path: Path | None = None) -> None:
    """Print msg to stdout. If log_path is given, also append to that file."""
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    if log_path is not None:
        with open(log_path, "a") as f:
            f.write(line + "\n")

def compute_split(n: int) -> tuple[int, int]:
    """Return (n_micro, n_medium) summing to n with an ~80/20 split."""
    n_medium = max(1, math.floor(n * 0.20))
    n_micro  = n - n_medium
    return n_micro, n_medium


def pick_nodes(all_nodes: list[str], job_type: str) -> list[str]:
    """Sample nodes without replacement for one job."""
    if job_type == "micro":
        count = MICROJOB_NODE_COUNT
    else:
        available = [c for c in MEDIUM_NODE_CHOICES if c <= len(all_nodes)]
        count = random.choice(available if available else [min(MEDIUM_NODE_CHOICES)])
        count = min(count, len(all_nodes))
    return random.sample(all_nodes, count)

def expand_app(app_str: str) -> list[str]:
    """
    Expand environment variables (e.g. $HOME) and split the app string into
    a proper argv list so srun receives the executable and its flags separately.
    """
    expanded = os.path.expandvars(os.path.expanduser(app_str))
    return shlex.split(expanded)


def job_output_path(out_dir: Path, uid: str) -> Path:
    """Return the per-job output file path."""
    return out_dir / f"{uid}.out"

# ---------------------------------------------------------------------------
# Job launch
# ---------------------------------------------------------------------------

def launch(job_type: str, nodes: list[str], extra_flags: list[str],
           out_dir: Path, log_path: Path | None, task_id: int) -> subprocess.Popen:
    """Launch one srun job and return its Popen handle."""
    uid      = f"{job_type}_{task_id}"
    app      = APPS[job_type] #!RANDOM.CHOICE(list(APPS.keys()))] se voglio farlo randomicos
    app_argv = expand_app(app)
    nodelist = ",".join(nodes)
    job_out  = job_output_path(out_dir, uid)

    cmd = [
        "srun",
        "--export=ALL",
        f"--nodelist={nodelist}",
        f"--cpu-bind=socket",
        f"--ntasks-per-node={TASKS_PER_NODE}",
        f"--cpus-per-task={CPUS_PER_TASK}",
        f"--job-name={uid}",
        *extra_flags,
        *app_argv,
    ]

    # ------- LOGGING & METADATA -------
    header = (
        f"TASK     : {uid}\n"
        f"TYPE     : {job_type}\n"
        f"NODES    : {nodelist}\n"
        f"APP      : {app}\n"
        f"CMD      : {' '.join(cmd)}\n"
        f"STARTED  : {ts()}\n"
        f"{'=' * 72}\n"
    )
    with open(job_out, "w") as f:
        f.write(header)

    log(f"START [{uid}]  nodes={nodelist}  app={app}  out={job_out.name}", log_path)
    # ------- LOGGING & METADATA -------

    proc = subprocess.Popen(
        cmd,             
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    # Attach metadata directly to the Popen object for convenience
    proc.uid      = uid       # type: ignore[attr-defined]
    proc.job_type = job_type  # type: ignore[attr-defined]
    proc.nodes    = nodes     # type: ignore[attr-defined]
    proc.app      = app       # type: ignore[attr-defined]
    proc.job_out  = job_out   # type: ignore[attr-defined]
    return proc

# ---------------------------------------------------------------------------
# Output draining
# ---------------------------------------------------------------------------

def drain_output(proc: subprocess.Popen) -> None:
    """Read all remaining stdout from a finished process and append to its job file."""
    if proc.stdout:
        with open(proc.job_out, "a") as f:          # type: ignore[attr-defined]
            for line in proc.stdout:
                f.write(line)
    if proc.stderr:
        with open(proc.job_out, "a") as f:          # type: ignore[attr-defined]
            while True:
                line = proc.stderr.readline()
                if not line:
                    break
                f.write("[ERROR] " + line)

def drain_live(proc: subprocess.Popen) -> None:
    """Non-blocking drain of any currently available lines to the job file."""
    if proc.stdout:
        with open(proc.job_out, "a") as f:          # type: ignore[attr-defined]
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                f.write(line)

    if proc.stderr:
        with open(proc.job_out, "a") as f:          # type: ignore[attr-defined]
            while True:
                line = proc.stderr.readline()
                if not line:
                    break
                f.write("[ERROR] " + line)


# ---------------------------------------------------------------------------
# Main scheduler loop
# ---------------------------------------------------------------------------

def run_scheduler(jobs: dict, out_dir: Path,
                  extra_flags: list[str], walltime: int,
                  log_path: Path | None = None) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)

    # Write scheduler log header
    header = (
        f"SLURM srun Scheduler Log\n"
        f"Started  : {ts()}\n"
        f"Output   : {out_dir.resolve()}/\n"
        f"{'=' * 72}\n"
    )
    if log_path is not None:
        with open(log_path, "w") as f:
            f.write(header)
    else:
        print(header, flush=True)

    task_id = 0

    # Launch initial batch
    running: list[subprocess.Popen] = []
    larges: list[subprocess.Popen] = []
    for job_name, job in jobs["small"].items():
        proc  = launch(job["strategy"], job["nodelist"], extra_flags, out_dir, log_path, task_id)
        running.append(proc)
        task_id += 1

    for job_name, job in jobs["large"].items():
        proc  = launch(job["strategy"], job["nodelist"], extra_flags, out_dir, log_path, task_id)
        larges.append(proc)
        task_id += 1

    log(f"All {len(running)+len(larges)} jobs submitted. Monitoring for completions…", log_path)

    WALLTIME_SECONDS = walltime 
    deadline = time.time() + WALLTIME_SECONDS
    while running:

        #! --> Walltime Check
        if time.time() > deadline:
            log(f"WALLTIME of {WALLTIME_SECONDS}s exceeded. Terminating all jobs.", log_path)
            for r in running:
                r.terminate()
            for large in larges:
                large.terminate()
            return
        time.sleep(POLL_INTERVAL)

        #! --> Large Jobs Handler
        still_large = []
        for large in larges:
            ret = large.poll()
            if ret is None:
                # drain_live(large)
                still_large.append(large)
            else:
                drain_output(large)
                footer = (
                    f"{'=' * 72}\n"
                    f"FINISHED : {ts()}  exit_code={ret}\n"
                )
                with open(large.job_out, "a") as f:  # type: ignore[attr-defined]
                    f.write(footer)
                log(
                    f"FINISH [{large.uid}]  exit_code={ret}"       # type: ignore[attr-defined]
                    f"  nodes={','.join(large.nodes)}"             # type: ignore[attr-defined]
                    f"  out={large.job_out.name}",                 # type: ignore[attr-defined]
                    log_path,
                )

        larges = still_large
    
        #! --> Large Jobs Handler
        if not larges:
            for r in running:
                r.terminate()
            log("Terminated all remaining micro jobs — all large jobs finished.", log_path)
            return

        #! --> Small Jobs Handler
        still_running = []
        for proc in running:
            ret = proc.poll()
            if ret is None:
                # drain_live(proc)
                still_running.append(proc)
            else:
                # Finished — drain remaining output and write footer to job file
                drain_output(proc)
                footer = (
                    f"{'=' * 72}\n"
                    f"FINISHED : {ts()}  exit_code={ret}\n"
                )
                with open(proc.job_out, "a") as f:  # type: ignore[attr-defined]
                    f.write(footer)

                log(
                    f"FINISH [{proc.uid}]  exit_code={ret}"                 # type: ignore[attr-defined]
                    f"  nodes={','.join(proc.nodes)}"                        # type: ignore[attr-defined]
                    f"  out={proc.job_out.name}",                            # type: ignore[attr-defined]
                    log_path,
                )

                # Launch replacement of the same type reusing the same nodelist
                replacement = launch(
                    proc.job_type, proc.nodes,                   # type: ignore[attr-defined]
                    extra_flags, out_dir, log_path, task_id,
                )
                still_running.append(replacement)
                task_id += 1
        running = still_running


# ---------------------------------------------------------------------------
# NODELISTS PARSING (Now only random)
# ---------------------------------------------------------------------------

def load_node_list(path: str) -> list[str]:
    """Accept a plain text file (one node per line) or a JSON list."""
    with open(path) as f:
        content = f.read().strip()
    try:
        nodes = json.loads(content)
        if not isinstance(nodes, list):
            raise ValueError("JSON node file must contain a list.")
        return [str(n) for n in nodes]
    except json.JSONDecodeError:
        return [line.strip() for line in content.splitlines() if line.strip()]


def collect_jobs(pattern: dict) -> list[tuple[str, str, dict]]:
    """Return a flat list of (group, job_id, job_spec) tuples."""
    jobs = []
    for group in ("small_jobs", "large_jobs"):
        for job_id, spec in pattern.get(group, {}).items():
            jobs.append((group, job_id, spec))
    return jobs

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SLURM srun scheduler — maintains N concurrent jobs (80%% micro / 20%% medium).\n"
            f"  micro  : {MICROJOB_NODE_COUNT} nodes, {TASKS_PER_NODE} tasks/node, {CPUS_PER_TASK} cpu/task\n"
            f"  medium : {MEDIUM_NODE_CHOICES} nodes (random), {TASKS_PER_NODE} tasks/node, {CPUS_PER_TASK} cpu/task"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pattern", "-p", required=True,
        metavar="FILE",
        help="Path to the pattern JSON file. Uses built-in default if omitted.",
    )
    parser.add_argument("--nodelist", required=True, metavar="NODELIST",
                        help="Nodes in bracket format: name[node01,node02,...].")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, metavar="DIR",
                        help=f"Directory for all output files (default: {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--srun-extra", default="", metavar="FLAGS",
                        help='Extra srun flags for every launch, e.g. "--mem=4G".')
    parser.add_argument("--walltime", default=120, type=int, metavar="SECONDS",
                        help="Walltime limit for each job in seconds.")
    parser.add_argument("--output-log", default=None, metavar="FILE",
                        help=(
                            "File to redirect scheduler logs to. "
                            "If omitted, logs are printed to stdout (default)."
                        ))
    return parser.parse_args()


def assign_nodes(config_path: str, available_nodes: list) -> dict:
    with open(config_path) as f:          # ← read the file first
        config = json.load(f)             # ← use json.load(), not json.loads()
    available = available_nodes.copy()
    result = {
        "small": {},
        "large": {}
    }

    for pattern_name, pattern in config.items():

        pattern.get("n_total_small_nodes", -1)

        n_total_small = pattern.get("n_total_small_nodes", -1)
        n_total_large = pattern.get("n_total_large_nodes", -1)

        if n_total_small + n_total_large != len(available):
            raise ValueError(
                f"Total nodes in config ({n_total_small} + {n_total_large}) does not match the number of available nodes ({len(available)})."
            )

        group_small = available[:n_total_large]
        group_big = available[n_total_large:]

        if len(group_small) + len(group_big) != len(available):
            raise ValueError(
                f"Node split in config does not match the number of available nodes ({len(available)})."
            )
        

        jobs = pattern.get("large_jobs", {})
        for job_id, job_info in jobs.items():
            nodes_needed = job_info["nodes"]

            if len(group_big) < nodes_needed:
                raise ValueError(
                    f"needs {nodes_needed}, only {len(group_big)} left."
                )

            assigned = [group_big.pop(0) for _ in range(nodes_needed)]

            result["large"][f"large_jobs_{job_id}"] = {
                "strategy": job_info["strategy"],
                "nodelist": assigned
            }


        jobs = pattern.get("small_jobs", {})
        for job_id, job_info in jobs.items():
            nodes_needed = job_info["nodes"]

            if len(group_small) < nodes_needed:
                raise ValueError(
                    f"needs {nodes_needed}, only {len(group_small)} left."
                )

            assigned = [group_small.pop(0) for _ in range(nodes_needed)]

            result["small"][f"small_jobs_{job_id}"] = {
                "strategy": job_info["strategy"],
                "nodelist": assigned
            }

    return result

def main() -> None:
    args = parse_args()

    nodes = args.nodelist.strip().split(",")
    if not nodes:
        print(f"ERROR: No nodes parsed from '{args.nodelist}'", file=sys.stderr)
        sys.exit(1)

    jobs = assign_nodes(args.pattern, available_nodes=nodes)

    out_dir     = Path(args.output_dir)
    extra_flags = args.srun_extra.split() if args.srun_extra.strip() else []
    log_path    = Path(args.output_log) if args.output_log else None

    run_scheduler(jobs, out_dir, extra_flags, args.walltime, log_path)


if __name__ == "__main__":
    main()