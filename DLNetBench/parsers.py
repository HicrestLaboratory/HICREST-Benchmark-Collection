import json
import sys
import ast
import re
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

sys.path.append(str(Path(__file__).parent.parent / "common" / "ccutils" / "parser"))
from ccutils_parser import MPIOutputParser

def stdout_to_csv_multi(stdout_content):
    """
    Parse ccutils stdout and convert to multiple CSV format strings.
    
    Returns a dictionary of DataFrames for each metric category.
    ALL DataFrames now include run_id and rank columns - NO aggregation across ranks.
    
    Parameters:
    -----------
    stdout_content : str
        The complete stdout content containing ccutils output
    
    Returns:
    --------
    dict : Dictionary with keys as DataFrame names and values as CSV-formatted strings
    
    For dp_pp strategy:
    - 'main': run_id, rank, runtime, dp_comm_time, throughput (per-rank, per-run)
    - 'pp_comm': run_id, rank, microbatch_id, pp_comm_time (detailed per microbatch)
    """
    
    # Parse the stdout
    parser = MPIOutputParser()
    parser_output = parser.parse_string(stdout_content)
    
    # Auto-detect strategy (first non-empty section)
    if not parser_output:
        raise ValueError("No ccutils sections found in stdout")
    
    strategy_name = list(parser_output.keys())[0]
    section = parser_output[strategy_name]
    
    print(f"Detected strategy: {strategy_name}")
    
    result_dfs = {}
    
    if strategy_name == "dp":
        # DP: runtime vs barrier_time
        # Keep per-rank, per-run data (no aggregation)
        rank_outputs = section.mpi_all_prints["ccutils_rank_json"].rank_outputs
        
        results = []
        for rank, json_str in rank_outputs.items():
            parsed = json.loads(json_str)
            runtimes = parsed.get("runtimes", [])
            throughputs = parsed.get("throughputs", [])
            barrier_times = parsed.get("barrier_time", [])
            comm_times = parsed.get("comm_time", [])
            
            for run_idx, (runtime, barrier_time, comm_time, throughput) in enumerate(zip(runtimes, barrier_times, comm_times, throughputs)):
                results.append((run_idx, rank, runtime, barrier_time, comm_time, throughput))
        
        # Convert to CSV
        csv_lines = ["run_id,rank,runtime,barrier_time,comm_time,throughput"]
        for run_id, rank, runtime, barrier_time, comm_time, throughput in results:
            csv_lines.append(f"{run_id},{rank},{runtime},{barrier_time},{comm_time},{throughput}")
        
        result_dfs['main'] = "\n".join(csv_lines)
    
    elif strategy_name == "fsdp":
        # FSDP: Split into main DF, allgather DF, and reduce_scatter DF
        json_data = section.json_data
        num_units = json_data.get("num_units")
        rank_outputs = section.mpi_all_prints["ccutils_rank_json"].rank_outputs
        
        # Main DF data: [(run_id, rank, runtime, allgather, barrier, throughput), ...]
        main_rows = []
        # Allgather DF data: [(run_id, rank, unit_id, phase, comm_time, wait_time), ...]
        allgather_rows = []
        # Reduce_scatter DF data: [(run_id, rank, unit_id, reduce_scatter_time), ...]
        reduce_scatter_rows = []
        
        for rank, json_str in rank_outputs.items():
            parsed = json.loads(json_str)
            runtimes = parsed.get("runtime", [])
            throughputs = parsed.get("throughputs", [])
            barrier_times = parsed.get("barrier", [])
            allgather_times = parsed.get("allgather", [])  # first blocking allgather
            allgather_comm_fwd_times = parsed.get("allgather_comm_fwd", [])
            allgather_wait_fwd_times = parsed.get("allgather_wait_fwd", [])
            allgather_wait_bwd_times = parsed.get("allgather_wait_bwd", [])
            reduce_scatter_times = parsed.get("reduce_scatter", [])
            
            num_runs = len(runtimes)
            
            # Main DF: one row per run per rank
            for run_idx in range(num_runs):
                runtime = runtimes[run_idx]
                throughput = throughputs[run_idx]
                barrier = barrier_times[run_idx] if run_idx < len(barrier_times) else 0.0
                allgather = allgather_times[run_idx] if run_idx < len(allgather_times) else 0.0
                main_rows.append((run_idx, rank, runtime, allgather, barrier, throughput))
            
            # Allgather DF: forward pass (num_units - 1 per run)
            for run_idx in range(num_runs):
                fwd_start = run_idx * (num_units - 1)
                fwd_end = fwd_start + (num_units - 1)
                
                for unit_idx, (comm_time, wait_time) in enumerate(
                    zip(allgather_comm_fwd_times[fwd_start:fwd_end],
                        allgather_wait_fwd_times[fwd_start:fwd_end])
                ):
                    allgather_rows.append((run_idx, rank, unit_idx + 1, 'fwd', comm_time, wait_time))
            
            # Allgather DF: backward pass (num_units - 1 per run, only wait time)
            for run_idx in range(num_runs):
                bwd_start = run_idx * (num_units - 1)
                bwd_end = bwd_start + (num_units - 1)
                
                for unit_idx, wait_time in enumerate(allgather_wait_bwd_times[bwd_start:bwd_end]):
                    # Note: backward has no comm_time (it's async, overlapped with compute)
                    allgather_rows.append((run_idx, rank, unit_idx, 'bwd', 0.0, wait_time))
            
            # Reduce_scatter DF: num_units per run
            for run_idx in range(num_runs):
                rs_start = run_idx * num_units
                rs_end = rs_start + num_units
                
                for unit_idx, rs_time in enumerate(reduce_scatter_times[rs_start:rs_end]):
                    reduce_scatter_rows.append((run_idx, rank, unit_idx, rs_time))
        
        # Convert to CSV
        csv_lines = ["run_id,rank,runtime,allgather,barrier,throughput"]
        for run_id, rank, runtime, allgather, barrier, throughput in main_rows:
            csv_lines.append(f"{run_id},{rank},{runtime},{allgather},{barrier},{throughput}")
        result_dfs['main'] = "\n".join(csv_lines)
        
        csv_lines = ["run_id,rank,unit_id,phase,comm_time,wait_time"]
        for run_id, rank, unit_id, phase, comm_time, wait_time in allgather_rows:
            csv_lines.append(f"{run_id},{rank},{unit_id},{phase},{comm_time},{wait_time}")
        result_dfs['allgather'] = "\n".join(csv_lines)
        
        csv_lines = ["run_id,rank,unit_id,reduce_scatter_time"]
        for run_id, rank, unit_id, rs_time in reduce_scatter_rows:
            csv_lines.append(f"{run_id},{rank},{unit_id},{rs_time}")
        result_dfs['reduce_scatter'] = "\n".join(csv_lines)
    
    elif strategy_name == "dp_pp":
        # DP+PP: Split into main DF and pp_comm DF
        # Keep per-rank, per-run data
        
        rank_outputs = section.mpi_all_prints["ccutils_rank_json"].rank_outputs
        json_data = section.json_data
        num_microbatches = json_data.get("num_microbatches")
        
        # Main DF data: [(run_id, rank, runtime, dp_comm_time, throughput), ...]
        main_results = []
        # PP Comm DF data: [(run_id, rank, microbatch_id, pp_comm_time), ...]
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
            if num_runs == 0:
                continue
            
            # Main DF: one row per run per rank
            for run_idx in range(num_runs):
                runtime = runtimes[run_idx]
                throughput = throughputs[run_idx]
                dp_comm = dp_comm_times[run_idx] if run_idx < len(dp_comm_times) else 0
                main_results.append((run_idx, rank, runtime, dp_comm, throughput))
            
            # PP Comm DF: one row per microbatch per run per rank
            for run_idx in range(num_runs):
                start_idx = run_idx * num_microbatches
                end_idx = start_idx + num_microbatches
                
                for microbatch_idx, pp_time in enumerate(pp_comm_times[start_idx:end_idx]):
                    pp_comm_rows.append((run_idx, rank, microbatch_idx, pp_time))
        
        # Convert main to CSV
        csv_lines = ["run_id,rank,runtime,dp_comm_time,throughput"]
        for run_id, rank, runtime, dp_comm, throughput in main_results:
            csv_lines.append(f"{run_id},{rank},{runtime},{dp_comm},{throughput}")
        result_dfs['main'] = "\n".join(csv_lines)
        
        # Convert pp_comm to CSV
        csv_lines = ["run_id,rank,microbatch_id,pp_comm_time"]
        for run_id, rank, mb_id, pp_time in pp_comm_rows:
            csv_lines.append(f"{run_id},{rank},{mb_id},{pp_time}")
        result_dfs['pp_comm'] = "\n".join(csv_lines)
    
    elif strategy_name == "dp_pp_tp":
        # DP+PP+TP: Split into main DF, pp_comm DF, and tp_comm DF
        
        rank_outputs = section.mpi_all_prints["ccutils_rank_json"].rank_outputs
        json_data = section.json_data
        num_microbatches = json_data.get("num_microbatches")
        layers_per_stage = json_data.get("layers_per_stage")
        
        # Main DF data: [(run_id, rank, runtime, dp_comm_time, throughput), ...]
        main_rows = []
        # PP Comm DF data: [(run_id, rank, microbatch_id, phase, pp_comm_time), ...]
        pp_comm_rows = []
        # TP Comm DF data: [(run_id, rank, microbatch_id, layer_id, phase, tp_comm_time), ...]
        tp_comm_rows = []
        
        for rank, json_str in rank_outputs.items():
            try:
                parsed = json.loads(json_str)
            except (json.JSONDecodeError, KeyError):
                continue
            
            stage_id = parsed.get("stage_id")
            runtimes = parsed.get("runtimes", [])
            throughputs = parsed.get("throughputs", [])
            pp_comm_times = parsed.get("pp_comm_time", [])
            tp_comm_times = parsed.get("tp_comm_time", [])
            dp_comm_times = parsed.get("dp_comm_time", [])
            
            num_runs = len(runtimes)
            if num_runs == 0:
                continue
            
            # Main DF: one row per run per rank
            for run_idx in range(num_runs):
                runtime = runtimes[run_idx]
                throughput = throughputs[run_idx]
                dp_comm = dp_comm_times[run_idx] if run_idx < len(dp_comm_times) else 0
                main_rows.append((run_idx, rank, runtime, dp_comm, throughput))
            
            # PP Comm DF: 2 entries per microbatch (fwd, bwd)
            # Already merged for middle stages in the C++ code
            for run_idx in range(num_runs):
                start_idx = run_idx * num_microbatches * 2
                end_idx = start_idx + num_microbatches * 2
                
                for mb_idx in range(num_microbatches):
                    # Forward phase
                    fwd_time = pp_comm_times[start_idx + mb_idx]
                    pp_comm_rows.append((run_idx, rank, mb_idx, 'fwd', fwd_time))
                
                for mb_idx in range(num_microbatches):
                    # Backward phase
                    bwd_time = pp_comm_times[start_idx + num_microbatches + mb_idx]
                    pp_comm_rows.append((run_idx, rank, mb_idx, 'bwd', bwd_time))
            
            # TP Comm DF: 4 * layers_per_stage allreduces per microbatch 
            # (2 per layer in fwd, 2 per layer in bwd)
            tp_ops_per_mb = 2 * layers_per_stage
            for run_idx in range(num_runs):
                start_idx = run_idx * num_microbatches * 4 * layers_per_stage
                
                # Forward pass: first half
                for mb_idx in range(num_microbatches):
                    fwd_start = start_idx + mb_idx * tp_ops_per_mb
                    for layer_idx in range(layers_per_stage):
                        # 2 allreduces per layer
                        tp_comm_rows.append((run_idx, rank, mb_idx, layer_idx, 'fwd', 
                                            tp_comm_times[fwd_start + layer_idx * 2]))
                        tp_comm_rows.append((run_idx, rank, mb_idx, layer_idx, 'fwd', 
                                            tp_comm_times[fwd_start + layer_idx * 2 + 1]))
                
                # Backward pass: second half
                bwd_base = start_idx + num_microbatches * tp_ops_per_mb
                for mb_idx in range(num_microbatches):
                    bwd_start = bwd_base + mb_idx * tp_ops_per_mb
                    for layer_idx in range(layers_per_stage):
                        # 2 allreduces per layer
                        tp_comm_rows.append((run_idx, rank, mb_idx, layer_idx, 'bwd', 
                                            tp_comm_times[bwd_start + layer_idx * 2]))
                        tp_comm_rows.append((run_idx, rank, mb_idx, layer_idx, 'bwd', 
                                            tp_comm_times[bwd_start + layer_idx * 2 + 1]))
        
        # Convert to CSV
        csv_lines = ["run_id,rank,runtime,dp_comm_time,throughput"]
        for run_id, rank, runtime, dp_comm, throughput in main_rows:
            csv_lines.append(f"{run_id},{rank},{runtime},{dp_comm},{throughput}")
        result_dfs['main'] = "\n".join(csv_lines)
        
        csv_lines = ["run_id,rank,microbatch_id,phase,pp_comm_time"]
        for run_id, rank, mb_id, phase, pp_time in pp_comm_rows:
            csv_lines.append(f"{run_id},{rank},{mb_id},{phase},{pp_time}")
        result_dfs['pp_comm'] = "\n".join(csv_lines)
        
        csv_lines = ["run_id,rank,microbatch_id,layer_id,phase,tp_comm_time"]
        for run_id, rank, mb_id, layer_id, phase, tp_time in tp_comm_rows:
            csv_lines.append(f"{run_id},{rank},{mb_id},{layer_id},{phase},{tp_time}")
        result_dfs['tp_comm'] = "\n".join(csv_lines)
    
    elif strategy_name == "dp_pp_ep":
        # DP+PP+EP: Split into main DF, pp_comm DF, and ep_comm DF
        
        rank_outputs = section.mpi_all_prints["ccutils_rank_json"].rank_outputs
        json_data = section.json_data
        num_microbatches = json_data.get("num_microbatches")
        layers_per_stage = json_data.get("layers_per_stage", 1)
        
        # Main DF data: [(run_id, rank, runtime, dp_ep_comm_time, dp_comm_time, throughput), ...]
        main_rows = []
        # PP Comm DF data: [(run_id, rank, microbatch_id, phase, pp_comm_time), ...]
        pp_comm_rows = []
        # EP Comm DF data: [(run_id, rank, microbatch_id, layer_id, phase, ep_comm_time), ...]
        ep_comm_rows = []
        
        for rank, json_str in rank_outputs.items():
            try:
                parsed = json.loads(json_str)
            except (json.JSONDecodeError, KeyError):
                continue
            
            stage_id = parsed.get("stage_id")
            runtimes = parsed.get("runtimes", [])
            throughputs = parsed.get("throughputs", [])
            pp_comm_times = parsed.get("pp_comm_time", [])
            ep_comm_times = parsed.get("ep_comm_time", [])
            dp_ep_comm_times = parsed.get("dp_ep_comm_time", [])
            dp_comm_times = parsed.get("dp_comm_time", [])
            
            num_runs = len(runtimes)
            if num_runs == 0:
                continue
            
            # Main DF: one row per run per rank
            for run_idx in range(num_runs):
                runtime = runtimes[run_idx]
                throughput = throughputs[run_idx]
                dp_ep_comm = dp_ep_comm_times[run_idx] if run_idx < len(dp_ep_comm_times) else 0
                dp_comm = dp_comm_times[run_idx] if run_idx < len(dp_comm_times) else 0
                main_rows.append((run_idx, rank, runtime, dp_ep_comm, dp_comm, throughput))
            
            # PP Comm DF: 2 entries per microbatch (fwd, bwd)
            # Already merged for middle stages in the C++ code
            for run_idx in range(num_runs):
                start_idx = run_idx * num_microbatches * 2
                end_idx = start_idx + num_microbatches * 2
                
                for mb_idx in range(num_microbatches):
                    # Forward phase
                    fwd_time = pp_comm_times[start_idx + mb_idx]
                    pp_comm_rows.append((run_idx, rank, mb_idx, 'fwd', fwd_time))
                
                for mb_idx in range(num_microbatches):
                    # Backward phase
                    bwd_time = pp_comm_times[start_idx + num_microbatches + mb_idx]
                    pp_comm_rows.append((run_idx, rank, mb_idx, 'bwd', bwd_time))
            
            # EP Comm DF: 2 * layers_per_stage alltoalls per microbatch per phase
            # (2 alltoalls per layer: to experts and from experts)
            ep_ops_per_mb = 2 * layers_per_stage
            for run_idx in range(num_runs):
                start_idx = run_idx * num_microbatches * ep_ops_per_mb * 2
                
                # Forward pass
                for mb_idx in range(num_microbatches):
                    fwd_start = start_idx + mb_idx * ep_ops_per_mb
                    for layer_idx in range(layers_per_stage):
                        # 2 alltoalls per layer (to experts, from experts)
                        ep_comm_rows.append((run_idx, rank, mb_idx, layer_idx, 'fwd', 
                                            ep_comm_times[fwd_start + layer_idx * 2]))
                        ep_comm_rows.append((run_idx, rank, mb_idx, layer_idx, 'fwd', 
                                            ep_comm_times[fwd_start + layer_idx * 2 + 1]))
                
                # Backward pass
                bwd_base = start_idx + num_microbatches * ep_ops_per_mb
                for mb_idx in range(num_microbatches):
                    bwd_start = bwd_base + mb_idx * ep_ops_per_mb
                    for layer_idx in range(layers_per_stage):
                        # 2 alltoalls per layer
                        ep_comm_rows.append((run_idx, rank, mb_idx, layer_idx, 'bwd', 
                                            ep_comm_times[bwd_start + layer_idx * 2]))
                        ep_comm_rows.append((run_idx, rank, mb_idx, layer_idx, 'bwd', 
                                            ep_comm_times[bwd_start + layer_idx * 2 + 1]))
        
        # Convert to CSV
        csv_lines = ["run_id,rank,runtime,dp_ep_comm_time,dp_comm_time,throughput"]
        for run_id, rank, runtime, dp_ep_comm, dp_comm, throughput in main_rows:
            csv_lines.append(f"{run_id},{rank},{runtime},{dp_ep_comm},{dp_comm},{throughput}")
        result_dfs['main'] = "\n".join(csv_lines)
        
        csv_lines = ["run_id,rank,microbatch_id,phase,pp_comm_time"]
        for run_id, rank, mb_id, phase, pp_time in pp_comm_rows:
            csv_lines.append(f"{run_id},{rank},{mb_id},{phase},{pp_time}")
        result_dfs['pp_comm'] = "\n".join(csv_lines)
        
        csv_lines = ["run_id,rank,microbatch_id,layer_id,phase,ep_comm_time"]
        for run_id, rank, mb_id, layer_id, phase, ep_time in ep_comm_rows:
            csv_lines.append(f"{run_id},{rank},{mb_id},{layer_id},{phase},{ep_time}")
        result_dfs['ep_comm'] = "\n".join(csv_lines)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    return result_dfs

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
        strategy: ...
        resources: ...
        bind_to_device: ...
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
            "bind_to_device":  bool,
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
            "bind_to_device":  block.meta.get("bind_to_device", False),
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
