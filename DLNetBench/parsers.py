import json
import sys
import ast
import re
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

sys.path.append(str(Path(__file__).parent.parent / "common" / "ccutils" / "parser"))
from ccutils_parser import MPIOutputParser


def stdout_file_to_csv_multi(stdout_path: Path, return_dataframes: bool = False):
    """Read a stdout file and parse it with stdout_to_csv_multi."""
    content = stdout_path.read_text(errors="replace")
    return stdout_to_csv_multi(content, return_dataframes)


def stdout_to_csv_multi(stdout_content, return_dataframes: bool = False) -> Tuple[Dict[str, Union[str, pd.DataFrame]], str]:
    """
    Parse ccutils stdout and convert to multiple CSV format strings (or DataFrames).
    
    Returns a dictionary of DataFrames for each metric category.
    ALL DataFrames now include run_id and rank columns - NO aggregation across ranks.
    
    Parameters:
    -----------
    stdout_content : str
        The complete stdout content containing ccutils output
    return_dataframes : bool, optional
        If True, return pandas DataFrames instead of CSV strings. Default is False.
    
    Returns:
    --------
    dict : Dictionary with keys as DataFrame names and values as either:
           - CSV-formatted strings (return_dataframes=False)
           - pandas DataFrames (return_dataframes=True)
    str  : Detected strategy name
    ...
    """
    
    # Parse the stdout
    parser = MPIOutputParser()
    parser_output = parser.parse_string(stdout_content)
    
    # Auto-detect strategy (first non-empty section)
    if not parser_output:
        raise ValueError("No ccutils sections found in stdout")
    
    strategy_name = list(parser_output.keys())[0]
    section = parser_output[strategy_name]

    result_dfs = {}

    def _finalise(name: str, columns: list[str], rows: list[tuple]):
        """Build either a CSV string or a DataFrame and store it in result_dfs."""
        if return_dataframes:
            result_dfs[name] = pd.DataFrame(rows, columns=columns)
        else:
            csv_lines = [",".join(columns)]
            for row in rows:
                csv_lines.append(",".join(str(v) for v in row))
            result_dfs[name] = "\n".join(csv_lines)
    
    if strategy_name == "dp":
        rank_outputs = section.mpi_all_prints["ccutils_rank_json"].rank_outputs
        
        results = []
        for rank, json_str in rank_outputs.items():
            parsed = json.loads(json_str)
            throughputs = parsed.get("throughputs", [])
            runtimes = parsed.get("runtimes", [])
            sync_times = parsed.get("barrier_time", [])
            comm_times = parsed.get("comm_time", [])
            # Bugfix
            if len(comm_times) > len(runtimes):
                comm_times = comm_times[1:]
            
            for run_idx, (throughput, runtime, sync_time, comm_time) in enumerate(
                zip(throughputs, runtimes, sync_times, comm_times)
            ):
                results.append((run_idx, rank, throughput, runtime, sync_time, comm_time))
        
        _finalise('main', ["run_id", "rank", "throughput", "runtime", "sync_time", "comm_time"], results)
    
    elif strategy_name == "fsdp":
        json_data = section.json_data
        num_units = json_data.get("num_units")
        rank_outputs = section.mpi_all_prints["ccutils_rank_json"].rank_outputs
        
        main_rows = []
        allgather_rows = []
        reduce_scatter_rows = []
        
        for rank, json_str in rank_outputs.items():
            parsed = json.loads(json_str)
            runtimes = parsed.get("runtime", [])
            throughputs = parsed.get("throughputs", [])
            barrier_times = parsed.get("barrier", [])
            allgather_times = parsed.get("allgather", [])
            allgather_comm_fwd_times = parsed.get("allgather_comm_fwd", [])
            allgather_wait_fwd_times = parsed.get("allgather_wait_fwd", [])
            allgather_wait_bwd_times = parsed.get("allgather_wait_bwd", [])
            reduce_scatter_times = parsed.get("reduce_scatter", [])
            
            num_runs = len(runtimes)
            sync_times = [0.0]*num_runs
            
            for run_idx in range(num_runs):
                fwd_start = run_idx * (num_units - 1)
                fwd_end = fwd_start + (num_units - 1)
                for unit_idx, (comm_time, wait_time) in enumerate(
                    zip(allgather_comm_fwd_times[fwd_start:fwd_end],
                        allgather_wait_fwd_times[fwd_start:fwd_end])
                ):
                    allgather_rows.append((run_idx, rank, unit_idx + 1, 'fwd', comm_time, wait_time))
                    sync_times[run_idx] += wait_time
            
            for run_idx in range(num_runs):
                bwd_start = run_idx * (num_units - 1)
                bwd_end = bwd_start + (num_units - 1)
                for unit_idx, wait_time in enumerate(allgather_wait_bwd_times[bwd_start:bwd_end]):
                    allgather_rows.append((run_idx, rank, unit_idx, 'bwd', 0.0, wait_time))
                    sync_times[run_idx] += wait_time
            
            for run_idx in range(num_runs):
                rs_start = run_idx * num_units
                rs_end = rs_start + num_units
                for unit_idx, rs_time in enumerate(reduce_scatter_times[rs_start:rs_end]):
                    reduce_scatter_rows.append((run_idx, rank, unit_idx, rs_time))
                    sync_times[run_idx] += rs_time
                    
            for run_idx in range(num_runs):
                try:
                    main_rows.append((
                        run_idx, rank, runtimes[run_idx],
                        sync_times[run_idx] + allgather_times[run_idx] + (barrier_times[run_idx] if run_idx < len(barrier_times) else 0.0),
                        allgather_times[run_idx] if run_idx < len(allgather_times) else 0.0,
                        barrier_times[run_idx] if run_idx < len(barrier_times) else 0.0,
                        throughputs[run_idx],
                    ))
                except Exception as e:
                    print(sync_times)
                    print()
                    print(allgather_times)
                    print()
                    print(barrier_times)
                    raise e
        
        _finalise('main',           ["run_id", "rank", "runtime", "sync_time", "allgather", "barrier", "throughput"],   main_rows)
        _finalise('allgather',      ["run_id", "rank", "unit_id", "phase", "comm_time", "wait_time"],                   allgather_rows)
        _finalise('reduce_scatter', ["run_id", "rank", "unit_id", "reduce_scatter_time"],                               reduce_scatter_rows)
    
    elif strategy_name == "dp_pp":
        rank_outputs = section.mpi_all_prints["ccutils_rank_json"].rank_outputs
        json_data = section.json_data
        num_microbatches = json_data.get("num_microbatches")
        
        main_results = []
        pp_comm_rows = []
        
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
            sync_times = [0.0]*num_runs
            if num_runs == 0:
                continue
            
            for run_idx in range(num_runs):
                start_idx = run_idx * num_microbatches
                for microbatch_idx, pp_time in enumerate(
                    pp_comm_times[start_idx : start_idx + num_microbatches]
                ):
                    pp_comm_rows.append((run_idx, rank, microbatch_idx, pp_time))
                    sync_times[run_idx] += pp_time
                    
            for run_idx in range(num_runs):
                main_results.append((
                    run_idx, rank, runtimes[run_idx],
                    dp_comm_times[run_idx] if run_idx < len(dp_comm_times) else 0,
                    sync_times[run_idx] + dp_comm_times[run_idx],
                    throughputs[run_idx],
                ))
        
        _finalise('main',    ["run_id", "rank", "runtime", "dp_comm_time", "sync_time", "throughput"],  main_results)
        _finalise('pp_comm', ["run_id", "rank", "microbatch_id", "pp_comm_time"],                       pp_comm_rows)
    
    elif strategy_name == "dp_pp_tp":
        rank_outputs = section.mpi_all_prints["ccutils_rank_json"].rank_outputs
        json_data = section.json_data
        num_microbatches = json_data.get("num_microbatches")
        layers_per_stage = json_data.get("layers_per_stage")
        
        main_rows = []
        pp_comm_rows = []
        tp_comm_rows = []
        
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
            sync_times = [0.0]*num_runs
            if num_runs == 0:
                continue
            
            for run_idx in range(num_runs):
                start_idx = run_idx * num_microbatches * 2
                for mb_idx in range(num_microbatches):
                    pp_comm_rows.append((run_idx, rank, mb_idx, 'fwd', pp_comm_times[start_idx + mb_idx]))
                    sync_times[run_idx] += pp_comm_times[start_idx + mb_idx]
                for mb_idx in range(num_microbatches):
                    pp_comm_rows.append((run_idx, rank, mb_idx, 'bwd', pp_comm_times[start_idx + num_microbatches + mb_idx]))
                    sync_times[run_idx] += pp_comm_times[start_idx + num_microbatches + mb_idx]
            
            tp_ops_per_mb = 2 * layers_per_stage
            for run_idx in range(num_runs):
                start_idx = run_idx * num_microbatches * 4 * layers_per_stage
                for mb_idx in range(num_microbatches):
                    fwd_start = start_idx + mb_idx * tp_ops_per_mb
                    for layer_idx in range(layers_per_stage):
                        tp_comm_rows.append((run_idx, rank, mb_idx, layer_idx, 'fwd', tp_comm_times[fwd_start + layer_idx * 2]))
                        tp_comm_rows.append((run_idx, rank, mb_idx, layer_idx, 'fwd', tp_comm_times[fwd_start + layer_idx * 2 + 1]))
                        sync_times[run_idx] += tp_comm_times[fwd_start + layer_idx * 2]
                        sync_times[run_idx] += tp_comm_times[fwd_start + layer_idx * 2 + 1]
                        
                bwd_base = start_idx + num_microbatches * tp_ops_per_mb
                for mb_idx in range(num_microbatches):
                    bwd_start = bwd_base + mb_idx * tp_ops_per_mb
                    for layer_idx in range(layers_per_stage):
                        tp_comm_rows.append((run_idx, rank, mb_idx, layer_idx, 'bwd', tp_comm_times[bwd_start + layer_idx * 2]))
                        tp_comm_rows.append((run_idx, rank, mb_idx, layer_idx, 'bwd', tp_comm_times[bwd_start + layer_idx * 2 + 1]))
                        sync_times[run_idx] += tp_comm_times[bwd_start + layer_idx * 2]
                        sync_times[run_idx] += tp_comm_times[bwd_start + layer_idx * 2 + 1]
            
            for run_idx in range(num_runs):
                main_rows.append((
                    run_idx, rank, runtimes[run_idx],
                    dp_comm_times[run_idx] if run_idx < len(dp_comm_times) else 0,
                    sync_times[run_idx] + dp_comm_times[run_idx],
                    throughputs[run_idx],
                ))
        
        _finalise('main',    ["run_id", "rank", "runtime", "dp_comm_time", "sync_time", "throughput"],     main_rows)
        _finalise('pp_comm', ["run_id", "rank", "microbatch_id", "phase", "pp_comm_time"],                 pp_comm_rows)
        _finalise('tp_comm', ["run_id", "rank", "microbatch_id", "layer_id", "phase", "tp_comm_time"],     tp_comm_rows)
    
    elif strategy_name == "dp_pp_ep":
        rank_outputs = section.mpi_all_prints["ccutils_rank_json"].rank_outputs
        json_data = section.json_data
        num_microbatches = json_data.get("num_microbatches")
        layers_per_stage = json_data.get("layers_per_stage", 1)
        
        main_rows = []
        pp_comm_rows = []
        ep_comm_rows = []
        
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
            sync_times = [0.0]*num_runs
            if num_runs == 0:
                continue
            
            for run_idx in range(num_runs):
                start_idx = run_idx * num_microbatches * 2
                for mb_idx in range(num_microbatches):
                    pp_comm_rows.append((run_idx, rank, mb_idx, 'fwd', pp_comm_times[start_idx + mb_idx]))
                    sync_times[run_idx] += pp_comm_times[start_idx + mb_idx]
                for mb_idx in range(num_microbatches):
                    pp_comm_rows.append((run_idx, rank, mb_idx, 'bwd', pp_comm_times[start_idx + num_microbatches + mb_idx]))
                    sync_times[run_idx] += pp_comm_times[start_idx + num_microbatches + mb_idx]
            
            ep_ops_per_mb = 2 * layers_per_stage
            for run_idx in range(num_runs):
                start_idx = run_idx * num_microbatches * ep_ops_per_mb * 2
                for mb_idx in range(num_microbatches):
                    fwd_start = start_idx + mb_idx * ep_ops_per_mb
                    for layer_idx in range(layers_per_stage):
                        ep_comm_rows.append((run_idx, rank, mb_idx, layer_idx, 'fwd', ep_comm_times[fwd_start + layer_idx * 2]))
                        ep_comm_rows.append((run_idx, rank, mb_idx, layer_idx, 'fwd', ep_comm_times[fwd_start + layer_idx * 2 + 1]))
                        sync_times[run_idx] += ep_comm_times[fwd_start + layer_idx * 2]
                        sync_times[run_idx] += ep_comm_times[fwd_start + layer_idx * 2 + 1]
                bwd_base = start_idx + num_microbatches * ep_ops_per_mb
                for mb_idx in range(num_microbatches):
                    bwd_start = bwd_base + mb_idx * ep_ops_per_mb
                    for layer_idx in range(layers_per_stage):
                        ep_comm_rows.append((run_idx, rank, mb_idx, layer_idx, 'bwd', ep_comm_times[bwd_start + layer_idx * 2]))
                        ep_comm_rows.append((run_idx, rank, mb_idx, layer_idx, 'bwd', ep_comm_times[bwd_start + layer_idx * 2 + 1]))
                        sync_times[run_idx] += ep_comm_times[bwd_start + layer_idx * 2]
                        sync_times[run_idx] += ep_comm_times[bwd_start + layer_idx * 2 + 1]
                        
            for run_idx in range(num_runs):
                main_rows.append((
                    run_idx, rank, runtimes[run_idx],
                    dp_ep_comm_times[run_idx] if run_idx < len(dp_ep_comm_times) else 0,
                    dp_comm_times[run_idx] if run_idx < len(dp_comm_times) else 0,
                    sync_times[run_idx] + dp_comm_times[run_idx] + dp_ep_comm_times[run_idx],
                    throughputs[run_idx],
                ))
        
        _finalise('main',    ["run_id", "rank", "runtime", "dp_ep_comm_time", "dp_comm_time", "sync_time", "throughput"], main_rows)
        _finalise('pp_comm', ["run_id", "rank", "microbatch_id", "phase", "pp_comm_time"],                                pp_comm_rows)
        _finalise('ep_comm', ["run_id", "rank", "microbatch_id", "layer_id", "phase", "ep_comm_time"],                    ep_comm_rows)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    return result_dfs, strategy_name

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
 
    if key == "bind_to_device":
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
        resources: ...
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
            "resources":   list[str],           # nodes or device IDs
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
            "resources":   block.meta.get("resources", []),
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
    from io import StringIO
    
    if len(sys.argv) != 2:
        print("Usage: python ccutils_to_csv_simple.py <stdout_file>")
        sys.exit(1)
    
    stdout_file = sys.argv[1]
    
    # Read stdout file
    with open(stdout_file, 'r') as f:
        stdout_content = f.read()
    
    # Convert to CSV format
    csv_output = stdout_to_csv_multi(stdout_content)
    
    # Print to stdout
    for df_name, csv_str in csv_output.items():
        print(f"=== {df_name} ===")
        df = pd.read_csv(StringIO(csv_str))
        print(df.head())  # Print first few lines of the DataFrame
