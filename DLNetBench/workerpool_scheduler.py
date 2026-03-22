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
  - With --no-compact: compact_all is skipped; output file paths are printed instead
    and output files are preserved even after the scheduler exits.

Performance notes (for ~500 concurrent processes):
  - drain_and_print runs in a ThreadPoolExecutor so compact_all (which may take
    seconds) never blocks the main poll loop or delays job respawning.
  - The poll loop itself is O(n) syscalls per POLL_INTERVAL tick; with 500 jobs
    and POLL_INTERVAL=0.3s this is well within budget on any modern kernel.
  - The scheduler process itself is single-threaded in the poll loop; the worker
    threads are only used for I/O-bound drain work.

Grace-Hopper / Grace-Blackwell affinity (--gh-affinity):
  On GH200/GB200 SoC nodes, each CPU die is NVLink-attached to one specific GPU.
  Crossing that boundary (i.e., a rank running on the wrong CPU socket for its GPU)
  causes all GPU↔CPU transfers to traverse the inter-socket fabric and can cut
  effective bandwidth by 30-50%.  When --gh-affinity is set the scheduler appends
  --gpu-bind=closest --cpu-bind=closest to every srun invocation.  These flags
  instruct SLURM's topology layer to pin each rank to the CPU cores that are
  NUMA-local to its assigned GPU, which is the correct behaviour on both Grace-Hopper
  and Grace-Blackwell without requiring a hard-coded core map.
  Requirements: SLURM ≥ 21.08 and that the node's NUMA/GPU topology is correctly
  reported to SLURM (it is on well-configured Leonardo-class / Alps-class systems).
  Do NOT combine with a manual --cpu-bind in --srun-extra; they will conflict.

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
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).parent.parent / "common"))
from command_map import EXTRA_SRUN_FLAGS
from utils.slurm import expand_slurm_nodelist
from JobPlacer.cli_wrapper import JobPlacer, JobRequest, PlacementResult

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

TASKS_PER_NODE  = 4   # For almost any system
CPUS_PER_TASK   = 8   # For Leonardo
POLL_INTERVAL   = 0.3 # seconds between status polls

# Thread pool used for drain_and_print so compact_all never blocks the poll loop.
# Size: enough to absorb bursts of simultaneous job completions without queueing.
DRAIN_WORKERS   = 16

DEBUG_ENABLED   = False

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def run_tag() -> str:
    """
    A short string that uniquely identifies this scheduler invocation.
    Incorporates the wall-clock time and the SLURM job ID when available,
    so output files from repeated runs on the same JSON are trivially
    distinguishable and sortable.

    Example:  20260317_190909_slurm42731
              20260317_190909_pid18234   (outside SLURM)
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slurm_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    suffix   = f"slurm{slurm_id}" if slurm_id else f"pid{os.getpid()}"
    return f"{stamp}_{suffix}"

# Computed once at startup so all files in one run share the same tag.
RUN_TAG: str = ""

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
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return True
        time.sleep(0.1)
    return False


def _graceful_kill(proc: subprocess.Popen, sig: signal.Signals, timeout: float = 30.0) -> None:
    """
    Send *sig* to the entire process group of *proc*, wait for every process in
    the group to exit, then fall back to SIGKILL if they haven't gone within
    *timeout* seconds.
    """
    pgid = os.getpgid(proc.pid)

    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass

    if not _wait_for_pgroup(pgid, timeout=timeout):
        log(f"WARNING: process group {pgid} did not exit within {timeout}s after "
            f"{sig.name}; sending SIGKILL.")
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _wait_for_pgroup(pgid, timeout=5.0)

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
    system: str,
    job_name: str,
    repetition: int,
    command: str,
    strategy: str,
    assigned_resources: list,
    bind_to_device: bool,
    gpus_per_node: int,
    extra_srun_flags: list[str],
    gh_affinity: bool,
    out_dir: Path,
    tasks_per_node: int,
    cpus_per_task: int,
    log_path: Optional[Path],
) -> subprocess.Popen:
    """
    Build and launch one job. Returns the Popen handle with metadata attributes.

    Output files are named:
        <job_name>_rep<repetition>_<RUN_TAG>.stdout / .stderr

    This naming scheme ensures files from different scheduler invocations on the
    same JSON are non-overlapping and can be correlated back to a specific SLURM
    job / wall-clock time without any external index.

    Metadata attributes attached to the proc object:
      .uid              – unique run identifier
      .job_name         – original job key from the JSON
      .repetition       – how many times this job has been launched (0-indexed)
      .strategy         – strategy string from the JSON
      .resources        – list of assigned nodes or device IDs
      .bind_to_device   – bool
      .app              – command string
      .stdout_path      – Path to stdout file
      .stderr_path      – Path to stderr file
      .start_ts         – launch timestamp string
    """
    uid = f"{job_name}_rep{repetition}_{RUN_TAG}"
    stdout_path = out_dir / f"{uid}.stdout"
    stderr_path = out_dir / f"{uid}.stderr"

    if bind_to_device:
        device_ids = ",".join(str(d) for d in assigned_resources)
        num_ranks  = len(assigned_resources)
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
            f"--ntasks={num_ranks}",
            f"--cpus-per-task={cpus_per_task}",
            "--gpus-per-task=1",
            f"--job-name={uid}",
            *extra_srun_flags,
            *command.split(),
            "-d", device_ids,
        ]
    else:
        nodelist_str = ",".join(assigned_resources)
        num_nodes    = len(assigned_resources)
        cmd = [
            "srun",
            "--exact",
            "--export=ALL",
            f"--nodelist={nodelist_str}",
            f"--nodes={num_nodes}",
            f"--ntasks={num_nodes * tasks_per_node}",
            f"--ntasks-per-node={tasks_per_node}",
            f"--cpus-per-task={cpus_per_task}",
            f"--job-name={uid}",
            *extra_srun_flags,
            *EXTRA_SRUN_FLAGS.get(system, [])
        ]
        # Grace-Hopper / Grace-Blackwell CPU-GPU affinity.
        # Injected just before the user command so it cannot be accidentally
        # overridden by extra_srun_flags (which precede it in the list above).
        # See module docstring for the rationale.
        if gh_affinity:
            cmd += ["--gpu-bind=closest", "--cpu-bind=closest"]
        cmd += command.split()

    start = ts()
    debug(f"Launching [{uid}]: {' '.join(cmd)}")
    log(f"START [{uid}]  strategy={strategy}  resources={assigned_resources}  cmd={' '.join(cmd)}", log_path)

    proc = _start_process(cmd, stdout_path, stderr_path)

    proc.uid             = uid                  # type: ignore[attr-defined]
    proc.job_name        = job_name             # type: ignore[attr-defined]
    proc.repetition      = repetition           # type: ignore[attr-defined]
    proc.strategy        = strategy             # type: ignore[attr-defined]
    proc.resources       = assigned_resources   # type: ignore[attr-defined]
    proc.bind_to_device  = bind_to_device       # type: ignore[attr-defined]
    proc.app             = command              # type: ignore[attr-defined]
    proc.stdout_path     = stdout_path          # type: ignore[attr-defined]
    proc.stderr_path     = stderr_path          # type: ignore[attr-defined]
    proc.start_ts        = start                # type: ignore[attr-defined]
    proc.gpus_per_node   = gpus_per_node        # type: ignore[attr-defined]
    return proc

# ---------------------------------------------------------------------------
# Output draining
# ---------------------------------------------------------------------------

METADATA_FIELDS = [
    "uid", "job_name", "repetition", "strategy",
    "resources", "bind_to_device", "app", "start_ts",
]

def drain_and_print(
    proc: subprocess.Popen,
    exit_code: Optional[int],
    log_path: Optional[Path],
    no_compact: bool,
    keep_files: bool,
) -> None:
    """
    Flush file handles, then emit a self-contained block to stdout.

    When no_compact=True:
      - compact_all is never called (avoids potentially multi-second stalls).
      - stdout/stderr file *paths* are printed instead of their contents.
      - Files are not deleted (keep_files is implicitly True in this mode).

    When no_compact=False (default):
      - compact_all is applied to stdout; raw stderr is printed.
      - Files are deleted afterwards unless keep_files=True.

    This function is designed to be called from a thread pool so that the
    potentially slow compact_all transform never blocks the main poll loop.
    """
    _close_handles(proc)

    uid      = proc.uid   # type: ignore[attr-defined]
    end      = ts()

    meta_lines = [f"  {k}: {getattr(proc, k, 'N/A')}" for k in METADATA_FIELDS]
    meta_lines.append(f"  exit_code: {exit_code}")
    meta_lines.append(f"  finished_at: {end}")
    meta_block = "\n".join(meta_lines)

    sep = "=" * 72

    if no_compact:
        stdout_section = (
            f"  stdout: {proc.stdout_path}\n"    # type: ignore[attr-defined]
            f"  stderr: {proc.stderr_path}\n"    # type: ignore[attr-defined]
        )
        stderr_section = ""
    else:
        def _read_and_transform(path: Path, transform) -> str:
            try:
                raw = path.read_text(errors="replace")
            except FileNotFoundError:
                return f"  <output file not found: {path}>\n"
            if transform:
                if exit_code != 0:
                    try:
                        return transform(raw)
                    except Exception as exc:
                        return (
                            f"  <transform failed with exit_code={exit_code}: {exc}>\n"
                            f"  Raw output follows:\n{raw}"
                        )
                return transform(raw)
            return raw

        from compact_csv import compact_all  # local import so --no-compact avoids the import entirely

        stdout_section = _read_and_transform(proc.stdout_path, compact_all)   # type: ignore[attr-defined]
        stderr_section = _read_and_transform(proc.stderr_path, None)          # type: ignore[attr-defined]

        if not keep_files:
            for p in (proc.stdout_path, proc.stderr_path):                    # type: ignore[attr-defined]
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

    block = (
        f"\n{sep}\n"
        f"JOB OUTPUT  uid={uid}\n"
        f"--- metadata ---\n{meta_block}\n"
        f"--- stdout ---\n{stdout_section}\n"
        f"--- stderr ---\n{stderr_section}"
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
    """

    def __init__(
        self,
        system: str,
        reserved_nodes: list,
        use_topo_files: bool = False,
    ) -> None:
        debug(f"PlacementOracle init for system='{system}', use_topo_files={use_topo_files}")
        topology_toml_file=None
        if system.lower() == 'alps':
            topology_toml_file=f'../common/JobPlacer/systems/{system.upper()}.toml'
            
        self.oracle = JobPlacer(
            system=system,
            topology_file=None,
            topology_toml_file=topology_toml_file,
            sinfo_file=None,
            nodelist=reserved_nodes,
            verbose=False,
        )
        self.program = self.oracle._binary
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


def assign_resources(
    pattern: dict,
    available_resources: list,
    use_devices: bool,
    system: str,
    use_topo_files: bool = False,
) -> dict:
    """
    Read the pattern JSON, apply the placement strategy, and return:

    {
      "small": { "job_key": {"strategy": ..., "resources": [...], "command": ..., "gpus": ...}, ... },
      "large": { ... }
    }
    """
    placement = pattern.get("placement", "linear").lower()

    debug(f"Placement strategy: {placement}, use_devices={use_devices}")

    pool = available_resources.copy()

    all_groups = [
        ("small", pattern.get("small_jobs", {})),
        ("large", pattern.get("large_jobs", {})),
    ]

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

    if placement == "device":
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
        oracle = PlacementOracle(
            system=system,
            reserved_nodes=available_resources,
            use_topo_files=use_topo_files,
        )
        assignments = oracle.find_placement(oracle_payload)
        if not assignments.ok:
            sys.exit(100)
        assignments = assignments.placements

    else:
        raise ValueError(f"Unknown placement strategy: '{placement}'")

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
    system: str,
    jobs: dict,
    out_dir: Path,
    extra_flags: list[str],
    walltime: int,
    bind_to_device: bool,
    gpus_per_node: int,
    kill_signal: signal.Signals,
    tasks_per_node: int,
    cpus_per_task: int,
    gh_affinity: bool,
    no_compact: bool,
    keep_files: bool,
    log_path: Optional[Path],
) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)

    rep_counter: dict[str, int] = {}

    # Drain futures: we track them so we can wait for all of them before exit.
    drain_futures: list[Future] = []
    executor = ThreadPoolExecutor(max_workers=DRAIN_WORKERS)

    def _submit_drain(proc: subprocess.Popen, exit_code: Optional[int]) -> None:
        """
        Submit drain_and_print to the thread pool.
        This returns immediately so the poll loop is never blocked by
        compact_all or file I/O.
        """
        fut = executor.submit(
            drain_and_print, proc, exit_code, log_path, no_compact, keep_files
        )
        drain_futures.append(fut)

    def _launch(job_key: str, group: str) -> subprocess.Popen:
        info = jobs[group][job_key]
        rep  = rep_counter.get(job_key, 0)
        rep_counter[job_key] = rep + 1
        return launch_job(
            system             = system,
            job_name           = job_key,
            repetition         = rep,
            command            = info["command"],
            strategy           = info["strategy"],
            assigned_resources = info["resources"],
            bind_to_device     = bind_to_device,
            gpus_per_node      = gpus_per_node,
            extra_srun_flags   = extra_flags,
            gh_affinity        = gh_affinity,
            out_dir            = out_dir,
            tasks_per_node     = tasks_per_node,
            cpus_per_task      = cpus_per_task,
            log_path           = log_path,
        )

    have_large_jobs = bool(jobs["large"])

    small_procs: list[subprocess.Popen] = [_launch(k, "small") for k in jobs["small"]]
    large_procs: list[subprocess.Popen] = [_launch(k, "large") for k in jobs["large"]]

    log(f"Launched {len(small_procs)} small + {len(large_procs)} large jobs.", log_path)

    deadline = time.time() + walltime

    # ---- main poll loop ----
    while True:

        time.sleep(POLL_INTERVAL)

        # -- check walltime --
        if not have_large_jobs and time.time() > deadline:
            log(f"Walltime of {walltime}s reached. Terminating all jobs.", log_path)
            for p in small_procs:
                _graceful_kill(p, kill_signal)
                _submit_drain(p, p.returncode)
            break

        # -- poll large jobs --
        still_large = []
        for p in large_procs:
            rc = p.poll()
            if rc is None:
                still_large.append(p)
            else:
                log(f"FINISH (large) [{p.uid}]  exit_code={rc}", log_path)
                _submit_drain(p, rc)
        large_procs = still_large

        # -- early-exit when all large jobs are done --
        if have_large_jobs and not large_procs:
            log("All large jobs finished. Terminating remaining small jobs.", log_path)
            for p in small_procs:
                _graceful_kill(p, kill_signal)
                _submit_drain(p, p.returncode)
            break

        # -- poll small jobs and respawn finished ones --
        # Drain is submitted to thread pool; respawn happens immediately after,
        # so compact_all on the old process never delays the new one launching.
        still_small = []
        for p in small_procs:
            rc = p.poll()
            if rc is None:
                still_small.append(p)
            else:
                log(f"FINISH (small) [{p.uid}]  exit_code={rc}", log_path)
                _submit_drain(p, rc)
                still_small.append(_launch(p.job_name, "small"))  # type: ignore[attr-defined]
        small_procs = still_small

    # Wait for all background drain tasks to complete before touching the directory.
    log("Waiting for drain tasks to complete…", log_path)
    executor.shutdown(wait=True)

    # Reap any drain exceptions so they surface rather than disappearing silently.
    for fut in drain_futures:
        exc = fut.exception()
        if exc:
            log(f"WARNING: drain task raised: {exc}", log_path)

    # Only remove the output directory when we are not preserving files.
    if no_compact or keep_files:
        log(f"Output files preserved in: {out_dir}", log_path)
    else:
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
                   help="Directory for per-job stdout/stderr files (cleaned up on exit "
                        "unless --no-compact or --keep-files is set).")
    p.add_argument("--srun-extra", default="", metavar="FLAGS",
                   help='Extra flags appended to every srun invocation, e.g. "--mem=4G".')
    p.add_argument("--walltime", type=int, default=120, metavar="SECONDS",
                   help="Max run time in seconds (used when there are no large jobs). Default: 120.")
    p.add_argument("--tasks-per-node", type=int, default=TASKS_PER_NODE,
                   help=f"Tasks per node. Default: {TASKS_PER_NODE}.")
    p.add_argument("--cpus-per-task", type=int, default=CPUS_PER_TASK,
                   help=f"CPUs per task. Default: {CPUS_PER_TASK}.")
    p.add_argument("--output-log", default=None, metavar="FILE",
                   help="File to mirror scheduler log lines into (in addition to stdout).")
    p.add_argument("--kill-signal", default="SIGUSR1", metavar="SIGNAL",
                   help="Signal used to terminate jobs gracefully. Default: SIGUSR1.")
    p.add_argument("--system", type=str, default='',
                   help="The system topology to query. Required if placement=runtime.")

    # --- new flags ---
    p.add_argument("--use-topo-files", action="store_true", default=False,
                   help="Pass topology_file and sinfo_file to JobPlacer "
                        "(requires the corresponding <system>_topo.txt / _sinfo.txt files). "
                        "By default these files are NOT passed.")
    p.add_argument("--gh-affinity", action="store_true", default=False,
                   help="Append --gpu-bind=closest --cpu-bind=closest to srun for "
                        "correct CPU-GPU NUMA affinity on Grace-Hopper / Grace-Blackwell nodes. "
                        "Has no effect when bind_to_device=True (device placement). "
                        "Do not combine with a manual --cpu-bind in --srun-extra.")
    p.add_argument("--no-compact", action="store_true", default=False,
                   help="Skip the compact_all transform on job stdout. "
                        "Prints output file paths instead of content, and preserves all files.")
    p.add_argument("--keep-files", action="store_true", default=False,
                   help="Keep per-job stdout/stderr files after the scheduler exits "
                        "(even when --no-compact is not set).")

    p.add_argument("--debug", action="store_true",
                   help="Enable verbose debug output.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    global DEBUG_ENABLED, RUN_TAG
    DEBUG_ENABLED = args.debug
    RUN_TAG       = run_tag()

    log(f"Run tag: {RUN_TAG}")

    with open(args.pattern) as f:
        pattern = json.load(f)

    placement = pattern.get("placement", "linear").lower()
    bind_to_device = (placement == "device")
    gpus_per_node  = pattern.get("gpus_per_node", 1)

    if placement == 'runtime' and not args.system:
        print("ERROR: If placement == 'runtime', --system must be set.", file=sys.stderr)
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

    jobs = assign_resources(
        pattern,
        available,
        use_devices=bind_to_device,
        system=args.system,
        use_topo_files=args.use_topo_files,
    )

    pid     = os.getpid()
    out_dir = Path(args.output_dir) if args.output_dir else Path("workerpool_out") / RUN_TAG

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
        system         = args.system,
        jobs           = jobs,
        out_dir        = out_dir,
        extra_flags    = extra_flags,
        walltime       = args.walltime,
        bind_to_device = bind_to_device,
        gpus_per_node  = gpus_per_node,
        kill_signal    = kill_signal,
        tasks_per_node = args.tasks_per_node,
        cpus_per_task  = args.cpus_per_task,
        gh_affinity    = args.gh_affinity,
        no_compact     = args.no_compact,
        keep_files     = args.keep_files,
        log_path       = log_path,
    )


if __name__ == "__main__":
    main()