import json
import sys
import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

sys.path.append(str(Path(__file__).parent.parent / "common" / "ccutils" / "parser"))
from ccutils_parser import MPIOutputParser

def stdout_to_csv(stdout_content):
    """
    Parse ccutils stdout and convert to CSV format string.
    
    For each run, returns the MAX runtime and MAX commtime across all ranks.
    
    Parameters:
    -----------
    stdout_content : str
        The complete stdout content containing ccutils output
    
    Returns:
    --------
    str : CSV-formatted string with header "runtime,commtime" and data rows
         Each row represents one run with max values across all ranks
    
    The function automatically detects the strategy and calculates communication time:
    - dp: commtime = barrier_time
    - fsdp: commtime = sum(allgather_fwd) + sum(allgather_bwd) + sum(reduce_scatter) + barrier
    - dp_pp: commtime = sum(pp_comm) + dp_comm
            (pp_comm is pure send/recv, merged recv+send for middle stages)
    - dp_pp_tp: commtime = sum(pp_comm) + sum(tp_comm) + dp_comm
            (pp_comm and tp_comm are pure communication)
    - dp_pp_ep: commtime = sum(pp_comm) + sum(ep_comm) + dp_ep_comm + dp_comm
            (pp_comm, ep_comm are pure communication; dp_ep_comm, dp_comm are all-reduces)
    """
    
    # Parse the stdout
    parser = MPIOutputParser()
    parser_output = parser.parse_string(stdout_content)
    
    # Auto-detect strategy (first non-empty section)
    if not parser_output:
        print(parser_output)
        raise ValueError("No ccutils sections found in stdout")
    
    strategy_name = list(parser_output.keys())[0]
    section = parser_output[strategy_name]
    
    print(f"Detected strategy: {strategy_name}")
    
    # Collect data per rank: {rank: [(runtime, commtime), ...]}
    rank_data = {}
    
    if strategy_name == "dp":
        # DP: runtime vs barrier_time
        rank_outputs = section.mpi_all_prints["ccutils_rank_json"].rank_outputs
        
        for rank, json_str in rank_outputs.items():
            parsed = json.loads(json_str)
            runtimes = parsed.get("runtimes", [])
            throughputs = parsed.get("throughputs", [])
            barrier_times = parsed.get("barrier_time", [])
            
            rank_results = []
            for runtime, barrier_time, throughput in zip(runtimes, barrier_times, throughputs):
                rank_results.append((runtime, barrier_time, throughput))
            
            rank_data[rank] = rank_results
    
    elif strategy_name == "fsdp":
        # FSDP: runtime vs (allgather_fwd + allgather_bwd + reduce_scatter + barrier)
        json_data = section.json_data
        num_units = json_data.get("num_units")
        rank_outputs = section.mpi_all_prints["ccutils_rank_json"].rank_outputs
        
        for rank, json_str in rank_outputs.items():
            parsed = json.loads(json_str)
            runtimes = parsed.get("runtime", [])
            throughputs = parsed.get("throughputs", [])
            barrier_times = parsed.get("barrier", [])
            allgather_times = parsed.get("allgather", [])
            allgather_wait_fwd_times = parsed.get("allgather_wait_fwd", [])
            allgather_wait_bwd_times = parsed.get("allgather_wait_bwd", [])
            reduce_scatter_times = parsed.get("reduce_scatter", [])
            
            num_runs = len(runtimes)
            
            rank_results = []
            for run_idx in range(num_runs):
                runtime = runtimes[run_idx]
                throughput = throughputs[run_idx]
                barrier = barrier_times[run_idx] if run_idx < len(barrier_times) else 0.0
                allgather = allgather_times[run_idx]
                
                # Sum allgather operations for this run
                ag_fwd_start = run_idx * (num_units - 1)
                ag_fwd_end = ag_fwd_start + (num_units - 1)
                ag_fwd_sum = sum(allgather_wait_fwd_times[ag_fwd_start:ag_fwd_end])
                
                ag_bwd_start = run_idx * (num_units - 1)
                ag_bwd_end = ag_bwd_start + (num_units - 1)
                ag_bwd_sum = sum(allgather_wait_bwd_times[ag_bwd_start:ag_bwd_end])
                
                # Sum reduce_scatter operations for this run
                rs_start = run_idx * num_units
                rs_end = rs_start + num_units
                rs_sum = sum(reduce_scatter_times[rs_start:rs_end])
                
                # Total comm time
                commtime = ag_fwd_sum + ag_bwd_sum + rs_sum + barrier + allgather
                
                rank_results.append((runtime, commtime, throughput))
            
            rank_data[rank] = rank_results
    
    elif strategy_name == "dp_pp":
        # DP+PP: pp_comm is pure communication (recv+send merged for middle stages)
        # commtime = sum(pp_comm) + dp_comm
        
        rank_outputs = section.mpi_all_prints["ccutils_rank_json"].rank_outputs
        
        for rank, json_str in rank_outputs.items():
            try:
                parsed = json.loads(json_str)
            except (json.JSONDecodeError, KeyError):
                continue
            
            runtimes = parsed.get("runtimes", [])
            throughputs = parsed.get("throughputs", [])
            pp_comm_times = parsed.get("pp_comm_time", [])
            dp_comm_times = parsed.get("dp_comm_time", [])
            
            num_runs = len(runtimes)
            if num_runs == 0:
                continue
            
            # Calculate ops per run
            ops_per_run = len(pp_comm_times) // num_runs if num_runs > 0 else 0
            
            rank_results = []
            for run_idx in range(num_runs):
                runtime = runtimes[run_idx]
                throughput = throughputs[run_idx]
                
                # Sum all pp_comm operations for this run (pure communication)
                start_idx = run_idx * ops_per_run
                end_idx = start_idx + ops_per_run
                pp_comm_sum = sum(pp_comm_times[start_idx:end_idx])
                
                # Add dp_comm time for this run
                dp_comm = dp_comm_times[run_idx] if run_idx < len(dp_comm_times) else 0
                
                commtime = pp_comm_sum + dp_comm
                
                rank_results.append((runtime, commtime, throughput))
            
            rank_data[rank] = rank_results
    
    elif strategy_name == "dp_pp_tp":
        # DP+PP+TP: pp_comm and tp_comm are both pure communication
        # commtime = sum(pp_comm) + sum(tp_comm) + dp_comm
        
        rank_outputs = section.mpi_all_prints["ccutils_rank_json"].rank_outputs
        
        for rank, json_str in rank_outputs.items():
            try:
                parsed = json.loads(json_str)
            except (json.JSONDecodeError, KeyError):
                continue
            
            runtimes = parsed.get("runtimes", [])
            throughputs = parsed.get("throughputs", [])
            pp_comm_times = parsed.get("pp_comm_time", [])
            tp_comm_times = parsed.get("tp_comm_time", [])
            dp_comm_times = parsed.get("dp_comm_time", [])
            
            num_runs = len(runtimes)
            if num_runs == 0:
                continue
            
            # Calculate ops per run
            pp_ops_per_run = len(pp_comm_times) // num_runs if num_runs > 0 else 0
            tp_ops_per_run = len(tp_comm_times) // num_runs if num_runs > 0 else 0
            
            rank_results = []
            for run_idx in range(num_runs):
                runtime = runtimes[run_idx]
                throughput = throughputs[run_idx]
                
                # Sum all pp_comm operations for this run (pure communication)
                pp_start = run_idx * pp_ops_per_run
                pp_end = pp_start + pp_ops_per_run
                pp_comm_sum = sum(pp_comm_times[pp_start:pp_end])
                
                # Sum all tp_comm operations for this run (pure communication)
                tp_start = run_idx * tp_ops_per_run
                tp_end = tp_start + tp_ops_per_run
                tp_comm_sum = sum(tp_comm_times[tp_start:tp_end])
                
                # Add dp_comm time for this run
                dp_comm = dp_comm_times[run_idx] if run_idx < len(dp_comm_times) else 0
                
                commtime = pp_comm_sum + tp_comm_sum + dp_comm
                
                rank_results.append((runtime, commtime, throughput))
            
            rank_data[rank] = rank_results
    
    elif strategy_name == "dp_pp_ep":
        # DP+PP+EP: pp_comm, ep_comm are pure communication, plus dp_ep_comm and dp_comm
        # commtime = sum(pp_comm) + sum(ep_comm) + dp_ep_comm + dp_comm
        
        rank_outputs = section.mpi_all_prints["ccutils_rank_json"].rank_outputs
        
        for rank, json_str in rank_outputs.items():
            try:
                parsed = json.loads(json_str)
            except (json.JSONDecodeError, KeyError):
                continue
            
            runtimes = parsed.get("runtimes", [])
            throughputs = parsed.get("throughputs", [])
            pp_comm_times = parsed.get("pp_comm_time", [])
            ep_comm_times = parsed.get("ep_comm_time", [])
            dp_ep_comm_times = parsed.get("dp_ep_comm_time", [])
            dp_comm_times = parsed.get("dp_comm_time", [])
            
            num_runs = len(runtimes)
            if num_runs == 0:
                continue
            
            # Calculate ops per run
            pp_ops_per_run = len(pp_comm_times) // num_runs if num_runs > 0 else 0
            ep_ops_per_run = len(ep_comm_times) // num_runs if num_runs > 0 else 0
            
            rank_results = []
            for run_idx in range(num_runs):
                runtime = runtimes[run_idx]
                throughput = throughputs[run_idx]
                
                # Sum all pp_comm operations for this run (pure communication)
                pp_start = run_idx * pp_ops_per_run
                pp_end = pp_start + pp_ops_per_run
                pp_comm_sum = sum(pp_comm_times[pp_start:pp_end])
                
                # Sum all ep_comm operations for this run (pure all-to-all)
                ep_start = run_idx * ep_ops_per_run
                ep_end = ep_start + ep_ops_per_run
                ep_comm_sum = sum(ep_comm_times[ep_start:ep_end])
                
                # Add dp_ep_comm time for this run (all-reduce within EP group)
                dp_ep_comm = dp_ep_comm_times[run_idx] if run_idx < len(dp_ep_comm_times) else 0
                
                # Add dp_comm time for this run (all-reduce for DP)
                dp_comm = dp_comm_times[run_idx] if run_idx < len(dp_comm_times) else 0
                
                commtime = pp_comm_sum + ep_comm_sum + dp_ep_comm + dp_comm
                
                rank_results.append((runtime, commtime, throughput))
            
            rank_data[rank] = rank_results
    
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    # Aggregate: for each run, take MAX across ranks
    if not rank_data:
        raise ValueError("No rank data collected")
    
    # Determine number of runs (assuming all ranks have same number of runs)
    num_runs = len(next(iter(rank_data.values())))
    
    results = []
    for run_idx in range(num_runs):
        # Collect all (runtime, commtime) for this run across ranks
        max_runtime = max(rank_data[rank][run_idx][0] for rank in rank_data)
        max_commtime = max(rank_data[rank][run_idx][1] for rank in rank_data)
        min_throughput = min(rank_data[rank][run_idx][2] for rank in rank_data)
        
        results.append((max_runtime, max_commtime, min_throughput))
    
    # Convert to CSV format string
    csv_lines = ["runtime,commtime,throughput"]
    for runtime, commtime, throughput in results:
        csv_lines.append(f"{runtime},{commtime},{throughput}")
    
    return "\n".join(csv_lines)

# ---------------------------------------------------------------------------
# Internal state machine types
# ---------------------------------------------------------------------------

SEP = "=" * 72
 
# Header line pattern:  "JOB OUTPUT  uid=job_1_rep0"
_HEADER_RE = re.compile(r"^JOB OUTPUT\s+uid=(\S+)$")
 
# Metadata key-value line:  "  key: value"
_META_RE = re.compile(r"^\s{2}(\w+):\s*(.*)")
 
@dataclass
class _Block:
    uid:    str
    meta:   dict = field(default_factory=dict)
    stdout_lines: list[str] = field(default_factory=list)
    stderr_lines: list[str] = field(default_factory=list)

# ---------------------------------------------------------------------------
# Type coercions for metadata fields
# ---------------------------------------------------------------------------

def _coerce(key: str, raw: str):
    """Best-effort type coercion for known metadata fields."""
    raw = raw.strip()
 
    if key == "repetition":
        try:
            return int(raw)
        except ValueError:
            return raw
 
    if key == "exit_code":
        if raw in ("None", "N/A", ""):
            return None
        try:
            return int(raw)
        except ValueError:
            return raw
 
    if key == "use_mpirun":
        return raw.lower() == "true"
 
    if key == "resources":
        # Stored as Python repr of a list, e.g. "['0', '1']" or "['node01']"
        try:
            val = ast.literal_eval(raw)
            return val if isinstance(val, list) else raw
        except Exception:
            return raw
 
    return raw  # str for everything else (uid, job_name, strategy, app, start_ts, finished_at)

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_scheduler_output(text: str) -> tuple[list[dict], list[str]]:
    """
    Parses the stdout produced by slurm_scheduler.py into a structured dict.
 
    Output format emitted by drain_and_print():
    
        ========================================================================
        JOB OUTPUT  uid=<uid>
        --- metadata ---
        uid: ...
        job_name: ...
        repetition: ...
        strategy: ...
        resources: ...
        use_mpirun: ...
        app: ...
        start_ts: ...
        exit_code: ...
        finished_at: ...
        --- stdout ---
        <transformed stdout content>
        --- stderr ---
        <stderr content>
        ========================================================================
    
    One block is emitted per job run, containing both streams.
    
    Return value of parse_scheduler_output()
    -----------------------------------------
    A list of JobRun dicts, each with the shape:
    
        {
            "uid":         str,                 # e.g. "job_1_rep2"
            "job_name":    str,                 # JSON key, e.g. "job_1"
            "repetition":  int,
            "strategy":    str,
            "resources":   list[str],           # nodes or device IDs
            "use_mpirun":  bool,
            "app":         str,                 # command string
            "start_ts":    str,
            "exit_code":   int | None,          # None if the field was missing
            "finished_at": str,
            "stdout":      str,                 # content of the stdout block
            "stderr":      str,                 # content of the stderr block
            "success":     bool,                # True iff exit_code == 0
        }
    
    Scheduler log lines (START / FINISH / walltime notices) that appear outside
    blocks are collected separately and returned as a list of raw strings.

    Parameters
    ----------
    text : str
        The complete stdout string to parse.

    Returns
    -------
    runs : list[dict]
        One dict per completed job run, with both stdout and stderr merged in.
        Sorted by (job_name, repetition).
    log_lines : list[str]
        Scheduler-level log lines that appeared outside any job block
        (START / FINISH notices, walltime messages, etc.).
    """
    lines = text.splitlines()
 
    blocks: list[_Block] = []
    log_lines: list[str] = []
 
    # ---- state machine ----
    # States: OUTSIDE, IN_HEADER, IN_META, IN_STDOUT, IN_STDERR
    current: Optional[_Block] = None
    state = "OUTSIDE"
 
    for line in lines:
        stripped = line.rstrip()
 
        if stripped == SEP:
            if state in ("IN_STDOUT", "IN_STDERR", "IN_META") and current is not None:
                # Closing separator — finalise current block
                blocks.append(current)
                current = None
                state = "OUTSIDE"
            elif state == "OUTSIDE":
                # Opening separator — next line should be the header
                state = "IN_HEADER"
            continue
 
        if state == "IN_HEADER":
            m = _HEADER_RE.match(stripped)
            if m:
                current = _Block(uid=m.group(1))
                state = "IN_META"
            else:
                # Not a recognised header — treat as log
                log_lines.append(SEP)
                log_lines.append(stripped)
                state = "OUTSIDE"
            continue
 
        if state == "IN_META":
            if stripped == "--- metadata ---":
                continue
            if stripped == "--- stdout ---":
                state = "IN_STDOUT"
                continue
            m = _META_RE.match(line)  # use original (indented) line
            if m and current is not None:
                key, raw_val = m.group(1), m.group(2)
                current.meta[key] = _coerce(key, raw_val)
            continue
 
        if state == "IN_STDOUT":
            if stripped == "--- stderr ---":
                state = "IN_STDERR"
                continue
            if current is not None:
                current.stdout_lines.append(line)
            continue
 
        if state == "IN_STDERR":
            if current is not None:
                current.stderr_lines.append(line)
            continue
 
        # OUTSIDE — collect as scheduler log lines (skip blank lines)
        if stripped:
            log_lines.append(stripped)
 
    # If the text ended mid-block (e.g. truncated), save whatever we have
    if current is not None:
        blocks.append(current)
 
    # ---- build runs dict directly from blocks (one block == one run) ----
    runs: dict[str, dict] = {}
 
    for block in blocks:
        uid = block.uid
        entry: dict = {
            "uid":         uid,
            "job_name":    block.meta.get("job_name", "N/A"),
            "repetition":  block.meta.get("repetition", None),
            "strategy":    block.meta.get("strategy", ""),
            "resources":   block.meta.get("resources", []),
            "use_mpirun":  block.meta.get("use_mpirun", False),
            "app":         block.meta.get("app", ""),
            "start_ts":    block.meta.get("start_ts", ""),
            "exit_code":   block.meta.get("exit_code", None),
            "finished_at": block.meta.get("finished_at", ""),
            "stdout":      "\n".join(block.stdout_lines),
            "stderr":      "\n".join(block.stderr_lines),
        }
        runs[uid] = entry
 
    # ---- derived field ----
    for entry in runs.values():
        entry["success"] = entry["exit_code"] == 0
 
    # ---- sort by (job_name, repetition) ----
    sorted_runs = sorted(
        runs.values(),
        key=lambda e: (e["job_name"], e["repetition"] if isinstance(e["repetition"], int) else -1),
    )
 
    return sorted_runs, log_lines
 
 
# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------
 
def parse_file(path: Union[str, Path]) -> tuple[list[dict], list[str]]:
    """Parse a file produced by redirecting the scheduler's stdout."""
    return parse_scheduler_output(Path(path).read_text(errors="replace"))
 
 
def failed_runs(runs: list[dict]) -> list[dict]:
    """Filter to runs that exited with a non-zero exit code."""
    return [r for r in runs if not r["success"]]
 
 
def runs_by_job(runs: list[dict]) -> dict[str, list[dict]]:
    """Group runs by job_name, preserving repetition order."""
    result: dict[str, list[dict]] = {}
    for r in runs:
        result.setdefault(r["job_name"], []).append(r)
    return result

# ---------------------------------------------------------------------------
# CLI — quick sanity check
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     import json
 
#     src = sys.argv[1] if len(sys.argv) > 1 else None
#     text = Path(src).read_text(errors="replace") if src else sys.stdin.read()
 
#     runs, log_lines = parse_scheduler_output(text)
 
#     print(f"Parsed {len(runs)} job run(s).\n")
 
#     for run in runs:
#         status = "OK" if run["success"] else f"FAILED (exit={run['exit_code']})"
#         print(
#             f"  {run['uid']:<30}  {status:<20}"
#             f"  stdout={len(run['stdout'])} chars"
#             f"  stderr={len(run['stderr'])} chars"
#         )
 
#     if log_lines:
#         print(f"\n{len(log_lines)} scheduler log line(s):")
#         for ll in log_lines[:10]:
#             print(f"  {ll}")
#         if len(log_lines) > 10:
#             print(f"  … ({len(log_lines) - 10} more)")

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python ccutils_to_csv_simple.py <stdout_file>")
        sys.exit(1)
    
    stdout_file = sys.argv[1]
    
    # Read stdout file
    with open(stdout_file, 'r') as f:
        stdout_content = f.read()
    
    # Convert to CSV format
    csv_output = stdout_to_csv(stdout_content)
    
    # Print to stdout
    print(csv_output)