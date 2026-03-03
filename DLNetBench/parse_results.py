import json
import pandas as pd
import sys
import numpy as np
import os
from argparse import ArgumentParser
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "common" / "ccutils"))
from parser.ccutils_parser import *

import sbatchman as sbm


def extract_dp_metrics_df(dp_section, job_tag):
    """
    Extract DP metrics from a Section object and job variables into a Pandas DataFrame.
    Returns a DataFrame with one row per rank per run.
    """
    json_data = dp_section.json_data

    world_size = json_data.get("world_size")
    model_name = json_data.get("model_name")
    local_batch_size = json_data.get("local_batch_size")
    num_buckets = json_data.get("num_buckets")
    fwd_rt = json_data.get("fwd_rt_whole_model")
    bwd_rt = json_data.get("bwd_rt_per_bucket")
    msg_avg = json_data.get("msg_size_avg_bytes")
    backend = json_data.get("backend")
    device = json_data.get("device")
    total_model_size_params = json_data.get("total_model_size_params")

    rows = []
    rank_outputs = dp_section.mpi_all_prints["ccutils_rank_json"].rank_outputs

    for rank, json_str in rank_outputs.items():
        parsed = json.loads(json_str)
        runtimes = parsed["runtimes"]
        barrier_times = parsed["barrier_time"]
        hostname = parsed["hostname"]

        for run_idx, (rt, bt) in enumerate(zip(runtimes, barrier_times)):
            row = {
                "job_tag": job_tag,
                "world_size": world_size,
                "model_name": model_name,
                "local_batch_size": local_batch_size,
                "num_buckets": num_buckets,
                "fwd_rt_whole_model": fwd_rt,
                "bwd_rt_per_bucket": bwd_rt,
                "msg_size_avg_bytes": msg_avg,
                "backend": backend,
                "device": device,
                "total_model_size_params": total_model_size_params,
                "rank": rank,
                "hostname": hostname,
                "run": run_idx,
                "runtime": rt,
                "barrier_time": bt
            }
            rows.append(row)

    return pd.DataFrame(rows)

def extract_fsdp_metrics_df(fsdp_section, job_tag):
    """
    Extract FSDP metrics from a Section object and job variables into two Pandas DataFrames:
      - runtime_df: one row per rank per run
      - comm_df: one row per rank per run per unit with allgather and reduce_scatter times
    """
    global_data = fsdp_section.json_data

    # Global metrics
    world_size = global_data.get("world_size")
    sharding_factor = global_data.get("sharding_factor")
    num_replicas = global_data.get("num_replicas")
    model_size_bytes = global_data.get("model_size_bytes")
    local_batch_size = global_data.get("local_batch_size")
    num_units = global_data.get("num_units")
    fwd_time_per_unit_us = global_data.get("fwd_time_per_unit_us")
    bwd_time_per_unit_us = global_data.get("bwd_time_per_unit_us")
    allgather_msg_size_bytes = global_data.get("allgather_msg_size_bytes")
    reducescatter_msg_size_bytes = global_data.get("reducescatter_msg_size_bytes")
    allreduce_msg_size_bytes = global_data.get("allreduce_msg_size_bytes")
    backend = global_data.get("backend")
    device = global_data.get("device")

    runtime_rows = []
    comm_rows = []

    rank_outputs = fsdp_section.mpi_all_prints["ccutils_rank_json"].rank_outputs

    for rank, json_str in rank_outputs.items():
        parsed = json.loads(json_str)
        
        runtimes = parsed.get("runtime", [])
        barriers = parsed.get("barrier", [])
        hostname = parsed.get("hostname")
        allgather_times = parsed.get("allgather", [])
        allgather_wait_fwd_times = parsed.get("allgather_wait_fwd", [])
        allgather_wait_bwd_times = parsed.get("allgather_wait_bwd", [])
        reduce_scatter_times = parsed.get("reduce_scatter", [])
        
        num_runs = len(runtimes)

        # Build runtime DataFrame (one row per run)
        for run_idx in range(num_runs):
            runtime_rows.append({
                "job_tag": job_tag,
                "world_size": world_size,
                "sharding_factor": sharding_factor,
                "num_replicas": num_replicas,
                "model_size_bytes": model_size_bytes,
                "local_batch_size": local_batch_size,
                "num_units": num_units,
                "fwd_time_per_unit_us": fwd_time_per_unit_us,
                "bwd_time_per_unit_us": bwd_time_per_unit_us,
                "allgather_msg_size_bytes": allgather_msg_size_bytes,
                "reducescatter_msg_size_bytes": reducescatter_msg_size_bytes,
                "allreduce_msg_size_bytes": allreduce_msg_size_bytes,
                "backend": backend,
                "device": device,
                "rank": rank,
                "hostname": hostname,
                "run": run_idx,
                "runtime": runtimes[run_idx],
                "allgather": allgather_times[run_idx],
                "barrier": barriers[run_idx] if run_idx < len(barriers) else None
            })

        # Build communication DataFrame (one row per run per unit)
        for run_idx in range(num_runs):
            for unit_idx in range(num_units):
                # Calculate indices for each timing array
                # allgather_wait_fwd: (num_units - 1) per run
                # allgather_wait_bwd: (num_units - 1) per run
                # reduce_scatter: num_units per run
                
                fwd_idx = run_idx * (num_units - 1) + unit_idx
                bwd_idx = run_idx * (num_units - 1) + unit_idx
                rs_idx = run_idx * num_units + unit_idx

                comm_rows.append({
                    "job_tag": job_tag,
                    "world_size": world_size,
                    "sharding_factor": sharding_factor,
                    "num_replicas": num_replicas,
                    "model_size_bytes": model_size_bytes,
                    "local_batch_size": local_batch_size,
                    "num_units": num_units,
                    "fwd_time_per_unit_us": fwd_time_per_unit_us,
                    "bwd_time_per_unit_us": bwd_time_per_unit_us,
                    "allgather_msg_size_bytes": allgather_msg_size_bytes,
                    "reducescatter_msg_size_bytes": reducescatter_msg_size_bytes,
                    "allreduce_msg_size_bytes": allreduce_msg_size_bytes,
                    "backend": backend,
                    "device": device,
                    "rank": rank,
                    "hostname": hostname,
                    "run": run_idx,
                    "unit_idx": unit_idx,
                    "allgather_wait_fwd": allgather_wait_fwd_times[fwd_idx] if fwd_idx < len(allgather_wait_fwd_times) else None,
                    "allgather_wait_bwd": allgather_wait_bwd_times[bwd_idx] if bwd_idx < len(allgather_wait_bwd_times) else None,
                    "reduce_scatter": reduce_scatter_times[rs_idx] if rs_idx < len(reduce_scatter_times) else None
                })

    runtime_df = pd.DataFrame(runtime_rows)
    comm_df = pd.DataFrame(comm_rows)

    return runtime_df, comm_df

def extract_hybrid_2d_metrics_df(hybrid_2d_section, job_tag):
    """
    Extract metrics for hybrid DP+PP strategy into two DataFrames.
    
    Returns:
        tuple: (runtime_df, pp_comm_df)
        - runtime_df: One row per rank per run
        - pp_comm_df: One row per rank per run per pipeline communication operation
    """
    json_data = hybrid_2d_section.json_data
    
    # Extract global metrics
    model_name = json_data.get("model_name")
    num_stages = json_data.get("num_stages")
    num_microbatches = json_data.get("num_microbatches")
    samples_per_microbatch = json_data.get("samples_per_microbatch")
    local_batch_size = json_data.get("local_batch_size")
    global_batch_size = json_data.get("global_batch_size")
    world_size = json_data.get("world_size")
    dp_size = json_data.get("dp_size")
    fwd_rt_per_microbatch = json_data.get("fwd_rt_per_microbatch")
    bwd_rt_per_microbatch = json_data.get("bwd_rt_per_microbatch")
    total_model_size_params = json_data.get("total_model_size_params")
    pipe_msg_size_bytes = json_data.get("pipe_msg_size_bytes")
    dp_allreduce_size_bytes = json_data.get("dp_allreduce_size_bytes")
    device = json_data.get("device")
    backend = json_data.get("backend")
    
    global_metrics = {
        "job_tag": job_tag,
        "model_name": model_name,
        "num_stages": num_stages,
        "num_microbatches": num_microbatches,
        "samples_per_microbatch": samples_per_microbatch,
        "local_batch_size": local_batch_size,
        "global_batch_size": global_batch_size,
        "world_size": world_size,
        "dp_size": dp_size,
        "fwd_rt_per_microbatch": fwd_rt_per_microbatch,
        "bwd_rt_per_microbatch": bwd_rt_per_microbatch,
        "total_model_size_params": total_model_size_params,
        "pipe_msg_size_bytes": pipe_msg_size_bytes,
        "dp_allreduce_size_bytes": dp_allreduce_size_bytes,
        "device": device,
        "backend": backend
    }
    
    runtime_rows = []
    pp_comm_rows = []
    
    for rank, json_str in hybrid_2d_section.mpi_all_prints["ccutils_rank_json"].rank_outputs.items():
        try:
            parsed = json.loads(json_str)
        except (json.JSONDecodeError, KeyError):
            continue
        
        runtimes = parsed.get("runtimes", [])
        pp_comm_times = parsed.get("pp_comm_time", [])
        dp_comm_times = parsed.get("dp_comm_time", [])
        hostname = parsed.get("hostname")
        stage_id = parsed.get("stage_id")
        
        num_runs = len(runtimes)
        
        # Calculate pp_comm operations per run
        ops_per_run = len(pp_comm_times) // num_runs if num_runs > 0 else 0
        
        # Runtime DataFrame: one row per run
        for run_idx in range(num_runs):
            dp_comm_time = dp_comm_times[run_idx] if run_idx < len(dp_comm_times) else None
            
            runtime_row = {
                **global_metrics,
                "rank": rank,
                "hostname": hostname,
                "stage_id": stage_id,
                "run": run_idx,
                "runtime": runtimes[run_idx],
                "dp_comm_time": dp_comm_time
            }
            runtime_rows.append(runtime_row)
            
            # PP Communication DataFrame: one row per pp_comm operation
            start_idx = run_idx * ops_per_run
            end_idx = start_idx + ops_per_run
            
            for op_idx, pp_time in enumerate(pp_comm_times[start_idx:end_idx]):
                pp_comm_row = {
                    **global_metrics,
                    "rank": rank,
                    "hostname": hostname,
                    "stage_id": stage_id,
                    "run": run_idx,
                    "op_idx": op_idx,
                    "pp_comm_time": pp_time
                }
                pp_comm_rows.append(pp_comm_row)
    
    runtime_df = pd.DataFrame(runtime_rows)
    pp_comm_df = pd.DataFrame(pp_comm_rows)
    
    return (runtime_df, pp_comm_df)

def extract_hybrid_3d_metrics_df(hybrid_3d_section, job_tag):
    """
    Extract metrics for hybrid DP+PP+TP (3D parallelism) strategy into three DataFrames.
    
    Returns:
        tuple: (runtime_df, pp_comm_df, tp_comm_df)
        - runtime_df: One row per rank per run
        - pp_comm_df: One row per rank per run per pipeline communication operation
        - tp_comm_df: One row per rank per run per tensor parallel communication operation
    """
    json_data = hybrid_3d_section.json_data
    
    # Extract global metrics (including tensor parallelism metrics)
    model_name = json_data.get("model_name")
    num_stages = json_data.get("num_stages")
    num_microbatches = json_data.get("num_microbatches")
    samples_per_microbatch = json_data.get("samples_per_microbatch")
    local_batch_size = json_data.get("local_batch_size")
    global_batch_size = json_data.get("global_batch_size")
    world_size = json_data.get("world_size")
    dp_size = json_data.get("dp_size")
    num_tensor_shards = json_data.get("num_tensor_shards")
    fwd_rt_per_microbatch = json_data.get("fwd_rt_per_microbatch")
    bwd_rt_per_microbatch = json_data.get("bwd_rt_per_microbatch")
    total_model_size_params = json_data.get("total_model_size_params")
    pipe_msg_size_bytes = json_data.get("pipe_msg_size_bytes")
    dp_allreduce_size_bytes = json_data.get("dp_allreduce_size_bytes")
    tp_allreduce_size_bytes = json_data.get("tp_allreduce_size_bytes")
    device = json_data.get("device")
    backend = json_data.get("backend")
    
    global_metrics = {
        "job_tag": job_tag,
        "model_name": model_name,
        "num_stages": num_stages,
        "num_microbatches": num_microbatches,
        "samples_per_microbatch": samples_per_microbatch,
        "local_batch_size": local_batch_size,
        "global_batch_size": global_batch_size,
        "world_size": world_size,
        "dp_size": dp_size,
        "num_tensor_shards": num_tensor_shards,
        "fwd_rt_per_microbatch": fwd_rt_per_microbatch,
        "bwd_rt_per_microbatch": bwd_rt_per_microbatch,
        "total_model_size_params": total_model_size_params,
        "pipe_msg_size_bytes": pipe_msg_size_bytes,
        "dp_allreduce_size_bytes": dp_allreduce_size_bytes,
        "tp_allreduce_size_bytes": tp_allreduce_size_bytes,
        "device": device,
        "backend": backend
    }
    
    runtime_rows = []
    pp_comm_rows = []
    tp_comm_rows = []
    
    for rank, json_str in hybrid_3d_section.mpi_all_prints["ccutils_rank_json"].rank_outputs.items():
        try:
            parsed = json.loads(json_str)
        except (json.JSONDecodeError, KeyError):
            continue
        
        runtimes = parsed.get("runtimes", [])
        pp_comm_times = parsed.get("pp_comm_time", [])
        tp_comm_times = parsed.get("tp_comm_time", [])
        dp_comm_times = parsed.get("dp_comm_time", [])
        hostname = parsed.get("hostname")
        stage_id = parsed.get("stage_id")
        dp_id = parsed.get("dp_id")
        tp_id = parsed.get("tp_id")
        
        num_runs = len(runtimes)
        
        # Calculate operations per run
        pp_ops_per_run = len(pp_comm_times) // num_runs if num_runs > 0 else 0
        tp_ops_per_run = len(tp_comm_times) // num_runs if num_runs > 0 else 0
        
        # Runtime DataFrame: one row per run
        for run_idx in range(num_runs):
            dp_comm_time = dp_comm_times[run_idx] if run_idx < len(dp_comm_times) else None
            
            runtime_row = {
                **global_metrics,
                "rank": rank,
                "hostname": hostname,
                "stage_id": stage_id,
                "dp_id": dp_id,
                "tp_id": tp_id,
                "run": run_idx,
                "runtime": runtimes[run_idx],
                "dp_comm_time": dp_comm_time
            }
            runtime_rows.append(runtime_row)
            
            # PP Communication DataFrame: one row per pp_comm operation
            pp_start_idx = run_idx * pp_ops_per_run
            pp_end_idx = pp_start_idx + pp_ops_per_run
            
            for op_idx, pp_time in enumerate(pp_comm_times[pp_start_idx:pp_end_idx]):
                pp_comm_row = {
                    **global_metrics,
                    "rank": rank,
                    "hostname": hostname,
                    "stage_id": stage_id,
                    "dp_id": dp_id,
                    "tp_id": tp_id,
                    "run": run_idx,
                    "op_idx": op_idx,
                    "pp_comm_time": pp_time
                }
                pp_comm_rows.append(pp_comm_row)
            
            # TP Communication DataFrame: one row per tp_comm operation
            tp_start_idx = run_idx * tp_ops_per_run
            tp_end_idx = tp_start_idx + tp_ops_per_run
            
            for op_idx, tp_time in enumerate(tp_comm_times[tp_start_idx:tp_end_idx]):
                tp_comm_row = {
                    **global_metrics,
                    "rank": rank,
                    "hostname": hostname,
                    "stage_id": stage_id,
                    "dp_id": dp_id,
                    "tp_id": tp_id,
                    "run": run_idx,
                    "op_idx": op_idx,
                    "tp_comm_time": tp_time
                }
                tp_comm_rows.append(tp_comm_row)
    
    runtime_df = pd.DataFrame(runtime_rows)
    pp_comm_df = pd.DataFrame(pp_comm_rows)
    tp_comm_df = pd.DataFrame(tp_comm_rows)
    
    return (runtime_df, pp_comm_df, tp_comm_df)

def main():
    """
    Main function to parse stdout and save metrics as Parquet files.
    """
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Extract metrics from completed jobs and save to Parquet files")
    parser.add_argument("filename", type=str, help="Output filename (without extension)")
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=["dp", "fsdp", "dp_pp", "dp_pp_tp", "dp_pp_ep"],
        help="List of strategies to process (default: all known strategies)"
    )
    args = parser.parse_args()
    output_filename = args.filename
    strategies_to_run = args.strategies
    base_results_dir = Path("results")
    
    # Mapping strategy name -> extraction function
    strategy_extractors = {
        "dp": extract_dp_metrics_df,
        "fsdp": extract_fsdp_metrics_df,
        "dp_pp": extract_hybrid_2d_metrics_df,
        "dp_pp_tp": extract_hybrid_3d_metrics_df
    }
    
    # Meaningful suffixes for multi-DataFrame strategies
    multi_df_suffixes = {
        "fsdp": ["runtime", "commtime"],
        "dp_pp": ["runtime", "pp_comm"],
        "dp_pp_tp": ["runtime", "pp_comm", "tp_comm"]
    }
    
    jobs = sbm.jobs_list(status=[sbm.Status.COMPLETED])
    
    for strategy in strategies_to_run:
        if strategy not in strategy_extractors:
            print(f"Warning: No extractor defined for strategy '{strategy}', skipping.")
            continue
        
        extractor = strategy_extractors[strategy]
        strategy_data = []
        
        for job in jobs:
            job_output = job.get_stdout()
            mpi_parser = MPIOutputParser()
            parser_output = mpi_parser.parse_string(job_output)
            section = parser_output.get(strategy)
            
            if section:
                df_or_tuple = extractor(section, job.tag)
                strategy_data.append(df_or_tuple)
        
        if not strategy_data:
            print(f"No data found for strategy: {strategy}")
            continue
        
        strategy_dir = base_results_dir / strategy
        strategy_dir.mkdir(parents=True, exist_ok=True)
        
        first_item = strategy_data[0]
        if isinstance(first_item, tuple):
            # Multi-DataFrame strategy (FSDP, dp_pp, dp_pp_tp)
            combined_dfs = [
                pd.concat(dfs, ignore_index=True)
                for dfs in zip(*strategy_data)
            ]
            
            suffixes = multi_df_suffixes.get(strategy)
            if suffixes is None or len(suffixes) != len(combined_dfs):
                raise ValueError(
                    f"Missing or incorrect suffix configuration for strategy '{strategy}'"
                )
            
            for df, suffix in zip(combined_dfs, suffixes):
                output_file = strategy_dir / f"{output_filename}_{suffix}.parquet"
                df.to_parquet(output_file, compression="snappy", index=False)
                print(f"Saved {strategy} data to: {output_file}")
        else:
            # Single DataFrame strategy (DP)
            full_df = pd.concat(strategy_data, ignore_index=True)
            output_file = strategy_dir / f"{output_filename}.parquet"
            full_df.to_parquet(output_file, compression="snappy", index=False)
            print(f"Saved {strategy} data to: {output_file}")


if __name__ == "__main__":
    main()