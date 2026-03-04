#!/usr/bin/env python3
"""
Simple function to convert ccutils stdout to CSV format string.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "common" / "ccutils"))
from parser.ccutils_parser import MPIOutputParser


def stdout_to_csv(stdout_content):
    """
    Parse ccutils stdout and convert to CSV format string.
    
    Parameters:
    -----------
    stdout_content : str
        The complete stdout content containing ccutils output
    
    Returns:
    --------
    str : CSV-formatted string with header "runtime,commtime" and data rows
    
    The function automatically detects the strategy and calculates communication time:
    - dp: commtime = barrier_time
    - fsdp: commtime = sum(allgather_fwd) + sum(allgather_bwd) + sum(reduce_scatter) + barrier
    - dp_pp: commtime = sum(pp_comm) + dp_comm
            (pp_comm is pure send/recv, merged recv+send for middle stages)
    - dp_pp_tp: commtime = sum(pp_comm) + sum(tp_comm) + dp_comm
            (pp_comm and tp_comm are pure communication)
    """
    
    # Parse the stdout
    parser = MPIOutputParser()
    parser_output = parser.parse_string(stdout_content)
    
    # Auto-detect strategy (first non-empty section)
    if not parser_output:
        raise ValueError("No ccutils sections found in stdout")
    
    strategy_name = list(parser_output.keys())[0]
    section = parser_output[strategy_name]
    
    # Extract data based on strategy
    results = []
    
    if strategy_name == "dp":
        # DP: runtime vs barrier_time
        rank_outputs = section.mpi_all_prints["ccutils_rank_json"].rank_outputs
        
        for rank, json_str in rank_outputs.items():
            parsed = json.loads(json_str)
            runtimes = parsed.get("runtimes", [])
            barrier_times = parsed.get("barrier_time", [])
            
            for runtime, barrier_time in zip(runtimes, barrier_times):
                results.append((runtime, barrier_time))
    
    elif strategy_name == "fsdp":
        # FSDP: runtime vs (allgather_fwd + allgather_bwd + reduce_scatter + barrier)
        json_data = section.json_data
        num_units = json_data.get("num_units")
        rank_outputs = section.mpi_all_prints["ccutils_rank_json"].rank_outputs
        
        for rank, json_str in rank_outputs.items():
            parsed = json.loads(json_str)
            runtimes = parsed.get("runtime", [])
            barrier_times = parsed.get("barrier", [])
            all_gathers = parsed.get("allgather", [])
            allgather_wait_fwd_times = parsed.get("allgather_wait_fwd", [])
            allgather_wait_bwd_times = parsed.get("allgather_wait_bwd", [])
            reduce_scatter_times = parsed.get("reduce_scatter", [])
            
            num_runs = len(runtimes)
            
            for run_idx in range(num_runs):
                runtime = runtimes[run_idx]
                first_allgather = all_gathers[run_idx]
                barrier = barrier_times[run_idx] if run_idx < len(barrier_times) else 0
                
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
                commtime = ag_fwd_sum + ag_bwd_sum + rs_sum + barrier + first_allgather
                
                results.append((runtime, commtime))
    
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
            pp_comm_times = parsed.get("pp_comm_time", [])
            dp_comm_times = parsed.get("dp_comm_time", [])
            
            num_runs = len(runtimes)
            if num_runs == 0:
                continue
            
            # Calculate ops per run
            ops_per_run = len(pp_comm_times) // num_runs if num_runs > 0 else 0
            
            for run_idx in range(num_runs):
                runtime = runtimes[run_idx]
                
                # Sum all pp_comm operations for this run (pure communication)
                start_idx = run_idx * ops_per_run
                end_idx = start_idx + ops_per_run
                pp_comm_sum = sum(pp_comm_times[start_idx:end_idx])
                
                # Add dp_comm time for this run
                dp_comm = dp_comm_times[run_idx] if run_idx < len(dp_comm_times) else 0
                
                commtime = pp_comm_sum + dp_comm
                
                results.append((runtime, commtime))
    
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
            pp_comm_times = parsed.get("pp_comm_time", [])
            tp_comm_times = parsed.get("tp_comm_time", [])
            dp_comm_times = parsed.get("dp_comm_time", [])
            
            num_runs = len(runtimes)
            if num_runs == 0:
                continue
            
            # Calculate ops per run
            pp_ops_per_run = len(pp_comm_times) // num_runs if num_runs > 0 else 0
            tp_ops_per_run = len(tp_comm_times) // num_runs if num_runs > 0 else 0
            
            for run_idx in range(num_runs):
                runtime = runtimes[run_idx]
                
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
                
                results.append((runtime, commtime))
    
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
            
            for run_idx in range(num_runs):
                runtime = runtimes[run_idx]
                
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
                
                results.append((runtime, commtime))
    
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    # Convert to CSV format string
    csv_lines = ["runtime,commtime"]
    for runtime, commtime in results:
        csv_lines.append(f"{runtime},{commtime}")
    
    return "\n".join(csv_lines)


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python ccutils_to_csv.py <stdout_file>")
        sys.exit(1)
    
    stdout_file = sys.argv[1]
    
    # Read stdout file
    with open(stdout_file, 'r') as f:
        stdout_content = f.read()
    
    # Convert to CSV format
    csv_output = stdout_to_csv(stdout_content)
    
    # Print to stdout
    print(csv_output)