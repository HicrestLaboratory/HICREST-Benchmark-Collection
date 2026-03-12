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

Each job writes its stdout to its own file inside workerpool_out_{pid}/.
Scheduler debug logs go to stdout by default, or to a file if --output-log is given.

Usage:
    python slurm_scheduler.py -p pattern.json --nodelist [node01,node02,...] [--output-dir DIR] [--debug]
"""

import argparse
import math
import os
import random
import shlex
import shutil
import signal
import subprocess
import sys
import time
import json
from datetime import datetime
from pathlib import Path

# Assuming parse_results is a custom module you have locally
from parse_results import *

from typing import Union, Optional

# ---------------------------------------------------------------------------
# CONFIGURATION — edit application paths to match your environment
# ---------------------------------------------------------------------------

# Node layout
MICROJOB_NODE_COUNT = 2       # microjobs always use exactly 2 nodes
MEDIUM_NODE_CHOICES = [4, 8]  # medium jobs randomly use 4 or 8 nodes
TASKS_PER_NODE      = 4
CPUS_PER_TASK       = 1

DEFAULT_OUTPUT_DIR  = "workerpool_out"
SCHEDULER_LOG_NAME  = "scheduler.log"
POLL_INTERVAL       = 2       # seconds between polls

# Global debug toggle
DEBUG_ENABLED       = False

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def debug_log(msg: str) -> None:
    """Print debug messages if the --debug flag is enabled."""
    if DEBUG_ENABLED:
        print(f"[DEBUG {ts()}] {msg}", flush=True)

def log(msg: str, log_path: Union[Path, None] = None, with_ts=True) -> None:
    """Print msg to stdout. If log_path is given, also append to that file."""
    line = f"[{ts()}] {msg}" if with_ts else msg
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

def job_output_paths(out_dir: Path, uid: str) -> tuple[Path, Path]:
    """Return the (stdout_path, stderr_path) for a given job uid."""
    return out_dir / f"{uid}.stdout", out_dir / f"{uid}.stderr"

# ---------------------------------------------------------------------------
# Job launch
# ---------------------------------------------------------------------------

def launch(command: str, strategy: str, nodes: list[str], extra_flags: list[str], 
           out_dir: Union[Path, None], log_path: Union[Path, None], task_id: int, launch_direct: bool, gpus: Union[int, None] = None) -> subprocess.Popen:
    """Launch one srun job and return its Popen handle."""
    uid      = f"{strategy}strategy_{gpus}gpus_{task_id}"
    nodelist = ",".join(nodes)
    job_stdout, job_stderr = job_output_paths(out_dir, uid) if out_dir else (None, None)

    cmd = []
    if launch_direct:
        cmd += ["mpirun", "-np", f"{gpus}", *extra_flags, *command.split(),'-d', nodelist]
    else:
        cmd += [
            "srun",
            "--export=ALL",
            f"--nodelist={nodelist}",
            f"--cpu-bind=socket",
            f"--ntasks-per-node={TASKS_PER_NODE}",
            f"--cpus-per-task={CPUS_PER_TASK}",
            f"--job-name={uid}",
            *extra_flags,
            *command.split()
        ]

    debug_log(f"Assembled launch command for {uid}: {' '.join(cmd)}")
    print(f"Launching job [{uid}] with command: {' '.join(cmd)}", flush=True)
    
    # ------- LOGGING & METADATA -------
    header = (
        f"{'=' * 72}\n"
        f"TASK       : {uid}\n"
        f"TYPE       : {strategy}\n"
        f"NODES/GPUs : {nodelist}\n"
        f"APP        : {command}\n"
        f"CMD        : {' '.join(cmd)}\n"
        f"STARTED    : {ts()}\n"
        f"STDOUT     : {job_stdout}\n"
        f"STDERR     : {job_stderr}\n"
        f"{'=' * 72}\n"
    )
    log(f"START [{uid}]  nodes={nodelist}  app={command}  stdout={job_stdout}  stderr={job_stderr}", log_path)
    # ------- LOGGING & METADATA -------

    stdout_fh = open(job_stdout, "w") if job_stdout else subprocess.PIPE
    stderr_fh = open(job_stderr, "w") if job_stderr else subprocess.PIPE

    proc = subprocess.Popen(
        cmd,
        stdout=stdout_fh,
        stderr=stderr_fh,
        text=True,
    )

    # Attach metadata directly to the Popen object for convenience
    proc.uid        = uid         # type: ignore[attr-defined]
    proc.header     = header      # type: ignore[attr-defined]
    proc.job_type   = strategy    # type: ignore[attr-defined]
    proc.nodes      = nodes       # type: ignore[attr-defined]
    proc.app        = command     # type: ignore[attr-defined]
    proc.job_stdout = job_stdout  # type: ignore[attr-defined]
    proc.job_stderr = job_stderr  # type: ignore[attr-defined]
    proc._stdout_fh = stdout_fh   # type: ignore[attr-defined]  keep handle to close later
    proc._stderr_fh = stderr_fh   # type: ignore[attr-defined]
    return proc

# ---------------------------------------------------------------------------
# Output draining
# ---------------------------------------------------------------------------

def drain_output(proc: subprocess.Popen, log_path: Union[Path, None] = None) -> None:
    """
    Close the live file handles, then print the contents of the job's stdout
    and stderr files to the workerpool stdout in a parsable block format.
    """
    uid = proc.uid  # type: ignore[attr-defined]
    debug_log(f"Draining output for job {uid}")

    # Close the file handles that were passed to Popen so all data is flushed
    for attr in ("_stdout_fh", "_stderr_fh"):
        fh = getattr(proc, attr, None)
        if fh and fh not in (subprocess.PIPE, subprocess.DEVNULL):
            try:
                fh.close()
            except Exception:
                pass

    for stream, path_attr in (("stdout", "job_stdout"), ("stderr", "job_stderr")):
        path: Union[Path, None] = getattr(proc, path_attr, None)
        if path is None:
            continue

        start_marker = f"<<<JOB_START uid={uid} stream={stream}>>>"
        end_marker   = f"<<<JOB_END   uid={uid} stream={stream}>>>"

        print(start_marker, flush=True)
        if log_path:
            with open(log_path, "a") as lf:
                lf.write(start_marker + "\n")

        try:
            content = path.read_text(errors="replace")
            if stream == 'stdout':
                print(stdout_to_csv(content))  # Assuming stdout_to_csv is from parse_results
            else:
                print(content, end="", flush=True)
            if log_path:
                with open(log_path, "a") as lf:
                    lf.write(content)
        except FileNotFoundError:
            msg = f"  <output file not found: {path}>\n"
            print(msg, end="", flush=True)
            if log_path:
                with open(log_path, "a") as lf:
                    lf.write(msg)

        print(end_marker, flush=True)
        if log_path:
            with open(log_path, "a") as lf:
                lf.write(end_marker + "\n")

def cleanup_output_dir(out_dir: Union[Path, None], log_path: Union[Path, None] = None) -> None:
    """Remove the per-run output directory once all output has been drained."""
    if out_dir is None:
        return
    try:
        debug_log(f"Attempting to remove output directory: {out_dir}")
        shutil.rmtree(out_dir)
        log(f"Removed output directory: {out_dir}", log_path)
    except Exception as exc:
        log(f"WARNING: could not remove output directory {out_dir}: {exc}", log_path)

# ---------------------------------------------------------------------------
# Scheduler Loop
# ---------------------------------------------------------------------------

def run_scheduler(jobs: dict, out_dir: Union[Path, None],
                  extra_flags: list[str], walltime: int, launch_direct: bool,
                  kill_signal: signal.Signals = signal.SIGTERM,
                  log_path: Union[Path, None] = None) -> None:

    debug_log(f"Starting scheduler with max walltime={walltime}s")

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Write scheduler log header
    header = (
        f"Started  : {ts()}\n"
        f"Output   : {out_dir.resolve() if out_dir else 'stdout'}\n\n"
    )
    log(header, log_path, with_ts=False)

    task_id = 0

    # Launch initial batch
    running: list[subprocess.Popen] = []
    larges: list[subprocess.Popen] = []
    
    # Track if we actually started with any large jobs
    had_large_jobs = len(jobs["large"]) > 0
    debug_log(f"Initial setup: had_large_jobs = {had_large_jobs}")

    debug_log("Launching initial batch of small jobs...")
    for job in jobs["small"].values():
        proc  = launch(job["command"], job["strategy"], job["nodelist"], extra_flags, out_dir, log_path, task_id, launch_direct, job.get("gpus"))
        running.append(proc)
        task_id += 1

    debug_log("Launching initial batch of large jobs...")
    for job in jobs["large"].values():
        proc  = launch(job["command"], job["strategy"], job["nodelist"], extra_flags, out_dir, log_path, task_id, launch_direct, job.get("gpus"))
        larges.append(proc)
        task_id += 1

    log(f"All {len(running)+len(larges)} jobs submitted. Monitoring for completions…", log_path)

    WALLTIME_SECONDS = walltime 
    deadline = time.time() + WALLTIME_SECONDS
    
    while running:

        #! --> Walltime Check
        if time.time() > deadline:
            debug_log("Walltime deadline reached!")
            log(f"WALLTIME of {WALLTIME_SECONDS}s exceeded. Terminating all jobs with {kill_signal.name}.", log_path)
            for r in running + larges:
                debug_log(f"Sending {kill_signal.name} to {r.uid}")
                r.send_signal(kill_signal)
                r.wait()
                drain_output(r, log_path)
            cleanup_output_dir(out_dir, log_path)
            return
            
        time.sleep(POLL_INTERVAL)
        debug_log(f"Polling jobs... Tracking {len(running)} small and {len(larges)} large jobs.")

        #! --> Large Jobs Handler
        still_large = []
        for large in larges:
            ret = large.poll()
            if ret is None:
                still_large.append(large)
            else:
                debug_log(f"Large job {large.uid} finished with code {ret}")
                drain_output(large, log_path)
                footer = (
                    f"{'=' * 72}\n"
                    f"FINISHED : {ts()}  exit_code={ret}\n"
                )
                log(footer, log_path, with_ts=False)
                log(
                    f"FINISH [{large.uid}]  exit_code={ret}"                        # type: ignore[attr-defined]
                    f"  nodes={','.join(large.nodes)}"                              # type: ignore[attr-defined]
                    f"  stdout={large.job_stdout}  stderr={large.job_stderr}",      # type: ignore[attr-defined]
                    log_path,
                )

        larges = still_large
    
        #! --> Early Exit Check: only trigger if we actually started with large jobs
        if had_large_jobs and not larges:
            debug_log("All large jobs completed. Initiating early exit for remaining small jobs.")
            for r in running:
                debug_log(f"Terminating small job {r.uid} due to early exit.")
                r.send_signal(kill_signal)
                r.wait()
                drain_output(r, log_path)
            log("Terminated all remaining micro jobs — all large jobs finished.", log_path)
            cleanup_output_dir(out_dir, log_path)
            return

        #! --> Small Jobs Handler
        still_running = []
        for proc in running:
            ret = proc.poll()
            if ret is None:
                still_running.append(proc)
            else:
                # Finished — drain remaining output and write footer to scheduler log
                debug_log(f"Small job {proc.uid} finished with code {ret}. Preparing replacement.")
                drain_output(proc, log_path)
                footer = (
                    f"{'=' * 72}\n"
                    f"FINISHED : {ts()}  exit_code={ret}\n"
                )
                log(footer, log_path, with_ts=False)

                log(
                    f"FINISH [{proc.uid}]  exit_code={ret}"                     # type: ignore[attr-defined]
                    f"  nodes={','.join(proc.nodes)}"                           # type: ignore[attr-defined]
                    f"  stdout={proc.job_stdout}  stderr={proc.job_stderr}",    # type: ignore[attr-defined]
                    log_path,
                )

                # Launch replacement of the same type reusing the same nodelist
                debug_log(f"Re-using nodes {proc.nodes} for replacement job.")
                replacement = launch(
                    proc.app, proc.job_type, proc.nodes,                        # type: ignore[attr-defined]
                    extra_flags, out_dir, log_path, task_id, launch_direct
                )
                still_running.append(replacement)
                task_id += 1
                
        running = still_running

    # Normal exit: while-loop exhausted because all small jobs finished naturally
    debug_log("All jobs finished naturally. Exiting scheduler loop.")
    cleanup_output_dir(out_dir, log_path)


# ---------------------------------------------------------------------------
# NODELISTS PARSING AND PLACEMENT
# ---------------------------------------------------------------------------

class PlacementOracle:
    """
    Template for your external placement oracle.
    """
    def __init__(self, system: str = ""):
        debug_log(f"Initializing PlacementOracle for system: '{system}'")
        self.system = system
        # Initialize your external binaries or checks here.
        self._available = True 

    def find_placement(self, jobs: list[dict], available_nodes: list[str]) -> dict[str, list[str]]:
        """
        Takes a list of job requests and available nodes.
        Returns a dictionary mapping job_ids to a list of assigned nodes.
        """
        debug_log(f"Oracle requested to place {len(jobs)} jobs across {len(available_nodes)} available nodes.")
        
        if not self._available:
            debug_log("Oracle marked as unavailable. Using sequential fallback.")
            print("Oracle unavailable, falling back...", file=sys.stderr)
            pass # Handle fallback

        # --- REPLACE WITH YOUR ACTUAL ORACLE SUBPROCESS LOGIC ---
        # For now, this is a stub that assigns nodes sequentially just so the script runs
        assignments = {}
        avail_copy = available_nodes.copy()
        
        for job in jobs:
            job_id = job["job_id"]
            req = job["req"]
            debug_log(f"Oracle processing job {job_id} (requires {req} nodes/gpus)")
            if len(avail_copy) < req:
                raise ValueError(f"Oracle: Not enough nodes for {job_id}")
            assignments[job_id] = [avail_copy.pop(0) for _ in range(req)]
            debug_log(f"Oracle assigned {assignments[job_id]} to {job_id}")
            
        return assignments

def load_node_list(path: str) -> list[str]:
    """Accept a plain text file (one node per line) or a JSON list."""
    debug_log(f"Loading node list from {path}")
    with open(path) as f:
        content = f.read().strip()
    try:
        nodes = json.loads(content)
        if not isinstance(nodes, list):
            raise ValueError("JSON node file must contain a list.")
        return [str(n) for n in nodes]
    except json.JSONDecodeError:
        return [line.strip() for line in content.splitlines() if line.strip()]

def assign_nodes(config_path: str, available_nodes: list, device: bool) -> dict:
    debug_log(f"Loading pattern config from: {config_path}")
    with open(config_path) as f:
        pattern = json.load(f)
        
    available = available_nodes.copy()
    placement_strategy = pattern.get("placement", "sequential").lower()
    debug_log(f"Detected placement strategy: {placement_strategy}")

    result = {
        "small": {},
        "large": {}
    }

    small_jobs = pattern.get("small_jobs", {})
    large_jobs = pattern.get("large_jobs", {})

    # 1. Derive required resources directly from the jobs (ignore metadata)
    total_needed = 0
    oracle_jobs_payload = []

    def parse_requirements(job_dict, group_prefix):
        nonlocal total_needed
        for job_id, job_info in job_dict.items():
            req = job_info.get("gpus" if device else "nodes", 1)
            total_needed += req
            oracle_jobs_payload.append({
                "job_id": f"{group_prefix}_{job_id}",
                "req": req,
                "strategy": job_info.get("strategy", "unknown")
            })

    parse_requirements(small_jobs, "small")
    parse_requirements(large_jobs, "large")

    debug_log(f"Total resources dynamically calculated as needed: {total_needed} (Available: {len(available)})")

    if total_needed > len(available):
        raise ValueError(
            f"Jobs require {total_needed} resources, but only {len(available)} are available."
        )

    # 2. Handle Placement Strategies
    runtime_assignments = {}
    
    if placement_strategy == "random":
        debug_log("Placement is random. Shuffling available nodes array.")
        random.shuffle(available)
        
    elif placement_strategy == "runtime":
        debug_log("Placement is runtime. Instantiating Oracle.")
        oracle = PlacementOracle(system="my_system")
        runtime_assignments = oracle.find_placement(oracle_jobs_payload, available)

    # 3. Assign nodes to the result dictionary
    def populate_results(job_dict, group_prefix, result_group):
        for job_id, job_info in job_dict.items():
            req = job_info.get("gpus" if device else "nodes", 1)
            full_job_id = f"{group_prefix}_{job_id}"

            if placement_strategy == "runtime":
                # Fetch assignment from Oracle's response
                assigned = runtime_assignments.get(full_job_id, [])
                if len(assigned) != req:
                    raise ValueError(f"Oracle failed to assign {req} nodes for {full_job_id}")
            else:
                # Pop sequentially (list is already shuffled if strategy was 'random')
                assigned = [available.pop(0) for _ in range(req)]

            debug_log(f"Job {full_job_id} successfully mapped to nodes/devices: {assigned}")

            result[result_group][full_job_id] = {
                "strategy": job_info["strategy"],
                "nodelist": assigned,
                "command": job_info["command"],
                "gpus": job_info.get("gpus")
            }

    populate_results(small_jobs, "small", "small")
    populate_results(large_jobs, "large", "large")

    return result

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
        help="Path to the pattern JSON file.",
    )
    parser.add_argument("--nodelist", required=False, default=None, metavar="NODELIST",
                        help="Nodes in bracket format: name[node01,node02,...].")
    parser.add_argument("--launch-direct", required=False,
                        help="If False launch using mpirun else use srun", action="store_true")
    parser.add_argument("--device-upperbound", required=False, default=None, metavar="DEVICE",
                        help="Device upperbound (8 --> device IDs considered 0..7)", type=int)
    parser.add_argument("--output-dir", default=None, metavar="DIR",
                        help=f"Directory for all output files.")
    parser.add_argument("--srun-extra", default="", metavar="FLAGS",
                        help='Extra srun flags for every launch, e.g. "--mem=4G".')
    parser.add_argument("--walltime", default=120, type=int, metavar="SECONDS",
                        help="Walltime limit for each job in seconds.")
    parser.add_argument("--output-log", default=None, metavar="FILE",
                        help=(
                            "File to redirect scheduler logs to. "
                            "If omitted, logs are printed to stdout (default)."
                        ))
    parser.add_argument("--kill-signal", default="SIGTERM", metavar="SIGNAL",
                        help=(
                            "Signal sent to force-terminate jobs (walltime exceeded or "
                            "early exit). Accepts signal names (e.g. SIGTERM, SIGINT, SIGKILL) "
                            "or integers. Default: SIGTERM."
                        ))
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose debug printing to trace script execution.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    # Toggle global debug state based on args
    global DEBUG_ENABLED
    if args.debug:
        DEBUG_ENABLED = True
        debug_log("Debug mode enabled.")
    
    if (not args.nodelist) and (not args.device_upperbound):
        print(f"--nodelist or --device-upperbound must be set!", file=sys.stderr)
        sys.exit(1)
    
    if args.nodelist and args.device_upperbound:
        print(f"You can't set both {args.nodelist} and {args.device_upperbound}")
        sys.exit(1)

    nodes = args.nodelist.strip().split(",") if args.nodelist else [str(i) for i in range(args.device_upperbound)]
    device = bool(args.device_upperbound)

    debug_log(f"Initial available nodes/devices list: {nodes}")

    jobs = assign_nodes(args.pattern, available_nodes=nodes, device=device)

    # Output directory is namespaced by workerpool PID to avoid collisions
    # across concurrent scheduler invocations.
    workerpool_pid = os.getpid()
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(f"workerpool_out_{workerpool_pid}")

    debug_log(f"Configured output directory: {out_dir}")

    extra_flags = args.srun_extra.split() if args.srun_extra.strip() else []
    log_path    = Path(args.output_log) if args.output_log else None

    # Resolve --kill-signal to a signal.Signals member (accepts names or integers)
    raw_sig = args.kill_signal.strip()
    try:
        kill_signal = signal.Signals[raw_sig] if raw_sig.isidentifier() else signal.Signals(int(raw_sig))
        debug_log(f"Resolved kill signal: {kill_signal.name}")
    except (KeyError, ValueError):
        print(f"ERROR: unknown signal '{raw_sig}'. Use a name like SIGTERM or an integer.", file=sys.stderr)
        sys.exit(1)

    run_scheduler(jobs, out_dir, extra_flags, args.walltime, args.launch_direct, kill_signal, log_path)

if __name__ == "__main__":
    main()