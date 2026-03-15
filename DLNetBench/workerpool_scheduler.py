#!/usr/bin/env python3
"""
Job Scheduler
=============
Launches and manages concurrent jobs from a pattern JSON file.

Two execution backends:
  - mpirun : used when placement="device" (jobs are mapped to specific GPU device IDs)
  - srun   : used for all other placement strategies (sequential, random, hardcoded, runtime)

Placement strategies:
  device    → mpirun; device IDs are integers (0..N-1), assigned linearly or from job spec
  hardcoded → srun;   each job in the JSON must carry a "nodelist" attribute
  linear    → srun;   nodes assigned in order from --nodelist
  random    → srun;   nodes shuffled, then assigned linearly
  runtime   → srun;   PlacementOracle assigns nodes

Behaviour:
  - At least two small jobs are always present.
  - Small jobs are continuously re-spawned until the scheduler exits.
  - If large jobs exist: scheduler exits when the last large job finishes (small jobs are killed).
  - If no large jobs: scheduler exits when --walltime seconds have elapsed (small jobs are killed).
  - Killing is graceful: the process-group of each mpirun/srun is signalled so every
    spawned child receives the signal and can clean up before the scheduler moves on.

Output:
  - Each job run writes stdout/stderr to separate files under --output-dir.
  - When a job finishes (naturally or by signal) its files are printed to stdout
    together with a metadata header describing what was run.

Usage:
    python slurm_scheduler.py -p pattern.json --nodelist node01,node02,... [options]
    python slurm_scheduler.py -p pattern.json --device-upperbound 8 [options]
"""

import argparse
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from parsers import stdout_to_csv

sys.path.append(str(Path(__file__).parent.parent / "common"))
from utils.slurm import expand_slurm_nodelist
from JobPlacer.cli_wrapper import JobPlacer, JobRequest, PlacementResult

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

TASKS_PER_NODE  = 4   # For almost any system
CPUS_PER_TASK   = 8   # For Leonardo
POLL_INTERVAL   = 1   # seconds between status polls

DEBUG_ENABLED   = False

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def debug(msg: str) -> None:
    if DEBUG_ENABLED:
        print(f"[DEBUG {ts()}] {msg}", flush=True)

def log(msg: str, log_path: Optional[Path] = None) -> None:
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    if log_path:
        with open(log_path, "a") as f:
            f.write(line + "\n")

# ---------------------------------------------------------------------------
# Process-group–aware launch
# ---------------------------------------------------------------------------

def _start_process(cmd: list[str], stdout_path: Path, stderr_path: Path) -> subprocess.Popen:
    """
    Launch cmd in its own process group so that sending a signal to the PGID
    propagates to every process spawned by mpirun/srun (workers, PMI daemons, …).
    """
    stdout_fh = open(stdout_path, "w")
    stderr_fh = open(stderr_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=stdout_fh,
        stderr=stderr_fh,
        text=True,
        start_new_session=True,   # creates a new process group (PGID == proc.pid)
    )
    proc._stdout_fh = stdout_fh   # type: ignore[attr-defined]
    proc._stderr_fh = stderr_fh   # type: ignore[attr-defined]
    return proc

def _wait_for_pgroup(pgid: int, timeout: float) -> bool:
    """
    Poll /proc until no process in *pgid* remains, or *timeout* seconds elapse.
    Returns True if the group is empty before the deadline, False otherwise.

    We cannot use waitpid(-pgid, ...) because the workers are not children of
    this process — they are grandchildren of mpirun/srun.  /proc is the only
    portable way to observe them on Linux.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            # os.killpg with signal 0 checks existence without sending a signal
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True   # group is gone
        except PermissionError:
            return True   # group exists but we can't signal it — treat as gone
        time.sleep(0.1)
    return False   # still running after timeout


def _graceful_kill(proc: subprocess.Popen, sig: signal.Signals, timeout: float = 30.0) -> None:
    """
    Send *sig* to the entire process group of *proc*, wait for every process in
    the group to exit (not just the mpirun/srun parent), then fall back to
    SIGKILL if they haven't all gone within *timeout* seconds.

    Why wait for the whole group and not just proc:
      mpirun/srun may exit (making proc.wait() return) while worker ranks are
      still running their signal handler and writing final output.  Reading the
      output files before those writes complete produces truncated or empty data.
    """
    pgid = os.getpgid(proc.pid)

    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass  # already gone

    # Wait for mpirun/srun itself first (updates proc.returncode)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass

    # Now wait for every worker in the process group to finish writing and exit
    if not _wait_for_pgroup(pgid, timeout=timeout):
        # Workers are still alive after the full timeout — escalate to SIGKILL
        log(f"WARNING: process group {pgid} did not exit within {timeout}s after "
            f"{sig.name}; sending SIGKILL.")
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _wait_for_pgroup(pgid, timeout=5.0)

    # Ensure proc.returncode is populated even if wait() timed out earlier
    proc.poll()

def _close_handles(proc: subprocess.Popen) -> None:
    for attr in ("_stdout_fh", "_stderr_fh"):
        fh = getattr(proc, attr, None)
        if fh:
            try:
                fh.close()
            except Exception:
                pass

# ---------------------------------------------------------------------------
# Job launch
# ---------------------------------------------------------------------------

def launch_job(
    job_name: str,
    repetition: int,
    command: str,
    strategy: str,
    assigned_resources: list,   # node names (srun) or device IDs as strings (mpirun)
    bind_to_device: bool,
    gpus_per_node: int,
    extra_srun_flags: list[str],
    out_dir: Path,
    log_path: Optional[Path],
) -> subprocess.Popen:
    """
    Build and launch one job. Returns the Popen handle with metadata attributes.

    Metadata attributes attached to the proc object:
      .uid          – unique run identifier  (job_name + repetition counter)
      .job_name     – original job key from the JSON
      .repetition   – how many times this job has been launched (0-indexed)
      .strategy     – strategy string from the JSON
      .resources    – list of assigned nodes or device IDs
      .bind_to_device – bool
      .app          – command string
      .stdout_path  – Path to stdout file
      .stderr_path  – Path to stderr file
      .start_ts     – launch timestamp string
    """
    uid = f"{job_name}_rep{repetition}"
    stdout_path = out_dir / f"{uid}.stdout"
    stderr_path = out_dir / f"{uid}.stderr"

    if bind_to_device:
        # resources are device IDs (integers stored as strings)
        device_ids   = ",".join(str(d) for d in assigned_resources)
        num_ranks    = len(assigned_resources)
        # cmd = [
        #     "mpirun",
        #     "-np", str(num_ranks),
        #     "--map-by", "slot",
        #     *extra_srun_flags,
        #     *command.split(),
        #     "-d", device_ids,
        # ]
        cmd = [
            "srun",
            "--ntasks=", str(num_ranks),
            "--cpus-per-task=1",
            "--gpus-per-task=1",
            f"--job-name={uid}",
            *extra_srun_flags,
            *command.split(),
            "-d", device_ids,
        ]
    else:
        # resources are node hostnames
        nodelist_str = ",".join(assigned_resources)
        num_nodes    = len(assigned_resources)
        cmd = [
            "srun",
            "--exact",
            "--export=ALL",
            f"--nodelist={nodelist_str}",
            f"--nodes={num_nodes}",
            f"--ntasks={num_nodes*TASKS_PER_NODE}",
            f"--ntasks-per-node={TASKS_PER_NODE}",
            # f"--gres=gpu:{TASKS_PER_NODE}",
            f"--cpus-per-task={CPUS_PER_TASK}",
            f"--job-name={uid}",
            # f"--gpu-bind=closest",
            *extra_srun_flags,
            *command.split(),
        ]

    start = ts()
    debug(f"Launching [{uid}]: {' '.join(cmd)}")
    log(f"START [{uid}]  strategy={strategy}  resources={assigned_resources}  cmd={' '.join(cmd)}", log_path)

    proc = _start_process(cmd, stdout_path, stderr_path)

    proc.uid         = uid           # type: ignore[attr-defined]
    proc.job_name    = job_name      # type: ignore[attr-defined]
    proc.repetition  = repetition    # type: ignore[attr-defined]
    proc.strategy    = strategy      # type: ignore[attr-defined]
    proc.resources   = assigned_resources  # type: ignore[attr-defined]
    proc.bind_to_device  = bind_to_device    # type: ignore[attr-defined]
    proc.app         = command       # type: ignore[attr-defined]
    proc.stdout_path = stdout_path   # type: ignore[attr-defined]
    proc.stderr_path = stderr_path   # type: ignore[attr-defined]
    proc.start_ts    = start         # type: ignore[attr-defined]
    proc.gpus_per_node = gpus_per_node  # type: ignore[attr-defined]
    return proc

# ---------------------------------------------------------------------------
# Output draining
# ---------------------------------------------------------------------------

METADATA_FIELDS = [
    "uid", "job_name", "repetition", "strategy",
    "resources", "bind_to_device", "app", "start_ts",
]

def drain_and_print(proc: subprocess.Popen, exit_code: Optional[int],
                    log_path: Optional[Path]) -> None:
    """
    Flush file handles, then emit a single self-contained block to stdout
    (and optionally log_path) with the following structure:

        ========================================================================
        JOB OUTPUT  uid=<uid>
        --- metadata ---
          key: value
          ...
        --- stdout ---
        <transformed stdout content>
        --- stderr ---
        <stderr content>
        ========================================================================
    """
    _close_handles(proc)

    uid = proc.uid  # type: ignore[attr-defined]
    end = ts()

    meta_lines = [f"  {k}: {getattr(proc, k, 'N/A')}" for k in METADATA_FIELDS]
    meta_lines.append(f"  exit_code: {exit_code}")
    meta_lines.append(f"  finished_at: {end}")
    meta_block = "\n".join(meta_lines)

    sep = "=" * 72

    def _read_stream(path_attr: str, transform) -> str:
        fpath: Optional[Path] = getattr(proc, path_attr, None)
        if fpath is None:
            return "  <no output file>\n"
        try:
            raw = fpath.read_text(errors="replace")
        except FileNotFoundError:
            return f"  <output file not found: {fpath}>\n"
        if transform:
            if exit_code != 0:
                try:
                    return transform(raw)
                except Exception as exc:
                    return (
                        f"  <stdout_to_csv failed with exit_code={exit_code}: {exc}>\n"
                        f"  Raw output follows:\n{raw}"
                    )
            return transform(raw)
        return raw

    stdout_content = _read_stream("stdout_path", stdout_to_csv)
    stderr_content = _read_stream("stderr_path", None)

    block = (
        f"\n{sep}\n"
        f"JOB OUTPUT  uid={uid}\n"
        f"--- metadata ---\n{meta_block}\n"
        f"--- stdout ---\n{stdout_content}\n"
        f"--- stderr ---\n{stderr_content}"
        f"{sep}\n"
    )

    print(block, flush=True)
    if log_path:
        with open(log_path, "a") as lf:
            lf.write(block)

# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

class PlacementOracle:
    """
    Thin wrapper around the external placement oracle program.

    Queried once per experiment in hardcoded mode.  If unreachable, a stub is
    used that always returns infeasible so every experiment file is still written.
    """

    def __init__(
        self,
        system: str,
        reserved_nodes: list,
    ) -> None:
        debug(f"PlacementOracle init for system='{system}'")
        self.program = '../common/JobPlacer/target/release/job_placer_placement_classes'
        self.oracle = JobPlacer(
            system=system,
            topology_file=f'../common/JobPlacer/{system}_topo.txt', # FIXME
            sinfo_file=f'../common/JobPlacer/{system}_sinfo.txt', # FIXME
            nodelist=reserved_nodes,
            binary=self.program
        )
        self.system = system
        self.reserved_nodes: list = reserved_nodes or []
        self._available = self._probe()
        if not self._available:
            print(
                f"[PlacementOracle] '{self.program}' not found / not responding. "
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

    def find_placement(self, jobs: list[dict]) -> PlacementResult:
        oracle_jobs = {}
        for j in jobs:
            oracle_jobs[j['job_id']] = JobRequest(
                num_nodes=j['req'],
                job_kind=j['job_id'],
                placement_class=str(j['placement_class']).lower()
            )
        res = self.oracle.place(
            oracle_jobs,
            seed=jobs[0]['seed'],
            timeout=5.0,
        )
        if not res.ok:
            print("Placement could not be satisfied", file=sys.stderr)
            print("Available nodes:", file=sys.stderr)
            print(self.reserved_nodes, file=sys.stderr)
            print("Request:", file=sys.stderr)
            print(oracle_jobs, file=sys.stderr)
        return res


def assign_resources(pattern: dict, available_resources: list, use_devices: bool, system: Union[str, None] = None) -> dict:
    """
    Read the pattern JSON, apply the placement strategy, and return:

    {
      "small": { "job_key": {"strategy": ..., "resources": [...], "command": ..., "gpus": ...}, ... },
      "large": { ... }
    }

    When use_devices=True the resource key means GPU count (integers → device IDs).
    When use_devices=False the resource key means node count (strings → hostnames).
    """
    placement = pattern.get("placement", "linear").lower()

    debug(f"Placement strategy: {placement}, use_devices={use_devices}")

    pool = available_resources.copy()

    # ---- collect all jobs and their resource requirements ----
    all_groups = [
        ("small", pattern.get("small_jobs", {})),
        ("large", pattern.get("large_jobs", {})),
    ]

    # For oracle / random we need the full list upfront
    oracle_payload = []
    for group_label, jobs_dict in all_groups:
        for job_key, job_info in jobs_dict.items():
            req = job_info.get("gpus" if use_devices else "nodes", 1)
            oracle_payload.append({
                "job_id": f"{group_label}/{job_key}",
                "req": req,
                "placement_class": job_info.get('placement_class'),
                "seed": job_info.get('seed', 0),
            })

    total_needed = sum(j["req"] for j in oracle_payload)
    if total_needed > len(pool):
        raise ValueError(
            f"Jobs require {total_needed} resources but only {len(pool)} are available."
        )

    # ---- resolve assignments ----
    if placement == "device":
        # same as linear but with integer IDs
        assignments = {}
        for item in oracle_payload:
            assignments[item["job_id"]] = [pool.pop(0) for _ in range(item["req"])]

    elif placement == "hardcoded":
        assignments = {}
        for group_label, jobs_dict in all_groups:
            for job_key, job_info in jobs_dict.items():
                full_id = f"{group_label}/{job_key}"
                if "nodelist" not in job_info:
                    raise ValueError(f"placement=hardcoded but job '{job_key}' has no 'nodelist'.")
                assignments[full_id] = job_info["nodelist"]

    elif placement == "random":
        random.shuffle(pool)
        assignments = {}
        for item in oracle_payload:
            assignments[item["job_id"]] = [pool.pop(0) for _ in range(item["req"])]

    elif placement == "linear":
        assignments = {}
        for item in oracle_payload:
            assignments[item["job_id"]] = [pool.pop(0) for _ in range(item["req"])]

    elif placement == "runtime":
        oracle = PlacementOracle(system=system, reserved_nodes=available_resources)
        assignments = oracle.find_placement(oracle_payload)
        if not assignments.ok:
            sys.exit(100)
        assignments = assignments.placements

    else:
        raise ValueError(f"Unknown placement strategy: '{placement}'")

    # ---- build result dict ----
    result = {"small": {}, "large": {}}
    for group_label, jobs_dict in all_groups:
        for job_key, job_info in jobs_dict.items():
            full_id = f"{group_label}/{job_key}"
            result[group_label][job_key] = {
                "strategy": job_info.get("strategy", ""),
                "resources": assignments[full_id],
                "command": job_info["command"],
                "gpus": job_info.get("gpus"),
            }
            debug(f"Assigned {full_id} → {assignments[full_id]}")

    return result

# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def run_scheduler(
    jobs: dict,
    out_dir: Path,
    extra_flags: list[str],
    walltime: int,
    bind_to_device: bool,
    gpus_per_node: int,
    kill_signal: signal.Signals,
    log_path: Optional[Path],
) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)

    # repetition counters per job name
    rep_counter: dict[str, int] = {}

    def _launch(job_key: str, group: str) -> subprocess.Popen:
        info = jobs[group][job_key]
        rep  = rep_counter.get(job_key, 0)
        rep_counter[job_key] = rep + 1
        return launch_job(
            job_name         = job_key,
            repetition       = rep,
            command          = info["command"],
            strategy         = info["strategy"],
            assigned_resources = info["resources"],
            bind_to_device   = bind_to_device,
            gpus_per_node    = gpus_per_node,
            extra_srun_flags = extra_flags,
            out_dir          = out_dir,
            log_path         = log_path,
        )

    have_large_jobs = bool(jobs["large"])

    # Launch everything
    small_procs: list[subprocess.Popen] = [_launch(k, "small") for k in jobs["small"]]
    large_procs: list[subprocess.Popen] = [_launch(k, "large") for k in jobs["large"]]

    log(f"Launched {len(small_procs)} small + {len(large_procs)} large jobs.", log_path)

    deadline = time.time() + walltime

    # ---- main poll loop ----
    while True:

        time.sleep(POLL_INTERVAL)

        # -- check walltime (only relevant when there are no large jobs) --
        if not have_large_jobs and time.time() > deadline:
            log(f"Walltime of {walltime}s reached. Terminating all jobs.", log_path)
            for p in small_procs:
                _graceful_kill(p, kill_signal)
                drain_and_print(p, p.returncode, log_path)
            break

        # -- poll large jobs --
        still_large = []
        for p in large_procs:
            rc = p.poll()
            if rc is None:
                still_large.append(p)
            else:
                log(f"FINISH (large) [{p.uid}]  exit_code={rc}", log_path)
                drain_and_print(p, rc, log_path)
        large_procs = still_large

        # -- early-exit when all large jobs are done --
        if have_large_jobs and not large_procs:
            log("All large jobs finished. Terminating remaining small jobs.", log_path)
            for p in small_procs:
                _graceful_kill(p, kill_signal)
                drain_and_print(p, p.returncode, log_path)
            break

        # -- poll small jobs and respawn finished ones --
        still_small = []
        for p in small_procs:
            rc = p.poll()
            if rc is None:
                still_small.append(p)
            else:
                log(f"FINISH (small) [{p.uid}]  exit_code={rc}", log_path)
                drain_and_print(p, rc, log_path)
                # Respawn: same job key, reusing the same resource assignment
                still_small.append(_launch(p.job_name, "small"))  # type: ignore[attr-defined]
        small_procs = still_small

    # cleanup
    try:
        shutil.rmtree(out_dir)
        log(f"Removed temporary output directory: {out_dir}", log_path)
    except Exception as exc:
        log(f"WARNING: could not remove {out_dir}: {exc}", log_path)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Job scheduler for congestion experiments (mpirun / srun backend).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-p", "--pattern", required=True, metavar="FILE",
                   help="Path to the pattern JSON file.")

    resource_group = p.add_mutually_exclusive_group(required=True)
    resource_group.add_argument("--nodelist", metavar="NODES",
                                help="Comma-separated list of node hostnames (for srun-based placements).")
    resource_group.add_argument("--device-upperbound", type=int, metavar="N",
                                help="Number of GPU devices available (device IDs will be 0..N-1).")

    p.add_argument("--output-dir", default=None, metavar="DIR",
                   help="Directory for per-job stdout/stderr files (cleaned up on exit).")
    p.add_argument("--srun-extra", default="", metavar="FLAGS",
                   help='Extra flags appended to every srun/mpirun invocation, e.g. "--mem=4G".')
    p.add_argument("--walltime", type=int, default=120, metavar="SECONDS",
                   help="Max run time in seconds (used when there are no large jobs). Default: 120.")
    p.add_argument("--output-log", default=None, metavar="FILE",
                   help="File to mirror scheduler log lines into (in addition to stdout).")
    p.add_argument("--kill-signal", default="SIGTERM", metavar="SIGNAL",
                   help="Signal used to terminate jobs gracefully. Default: SIGTERM.")
    p.add_argument("--system", type=str, default=None,
                   help="The system topology to query. Required if placement=runtime")
    p.add_argument("--debug", action="store_true",
                   help="Enable verbose debug output.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    global DEBUG_ENABLED
    DEBUG_ENABLED = args.debug

    with open(args.pattern) as f:
        pattern = json.load(f)

    placement = pattern.get("placement", "linear").lower()
    bind_to_device = (placement == "device")
    gpus_per_node = pattern.get("gpus_per_node", 1)
    
    if placement == 'runtime' and args.system is None:
        print("If placement == 'runtime', --system must be set")
        sys.exit(1)

    if bind_to_device:
        if args.nodelist:
            available = expand_slurm_nodelist(args.nodelist)
        else:
            available = [str(i) for i in range(args.device_upperbound)]
    else:
        if args.device_upperbound is not None:
            print("ERROR: --device-upperbound is only valid for placement=device.", file=sys.stderr)
            sys.exit(1)
        available = expand_slurm_nodelist(args.nodelist)

    debug(f"Available resources: {available}")

    jobs = assign_resources(pattern, available, use_devices=bind_to_device, system=args.system)

    pid = os.getpid()
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"workerpool_out_{pid}")

    extra_flags = args.srun_extra.split() if args.srun_extra.strip() else []
    log_path    = Path(args.output_log) if args.output_log else None

    raw_sig = args.kill_signal.strip()
    try:
        kill_signal = (
            signal.Signals[raw_sig] if raw_sig.isidentifier() else signal.Signals(int(raw_sig))
        )
    except (KeyError, ValueError):
        print(f"ERROR: unknown signal '{raw_sig}'.", file=sys.stderr)
        sys.exit(1)

    run_scheduler(
        jobs         = jobs,
        out_dir      = out_dir,
        extra_flags  = extra_flags,
        walltime     = args.walltime,
        bind_to_device   = bind_to_device,
        gpus_per_node= gpus_per_node,
        kill_signal  = kill_signal,
        log_path     = log_path,
    )


if __name__ == "__main__":
    main()