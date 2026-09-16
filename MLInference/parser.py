#./build/bin/llama-bench -m ./Llama-3.2-3B-Instruct-Q4_K_M.gguf -p 0 -n 0 -pg 128,64 -r 3 -o json

import json
import sys
from pathlib import Path
from typing import Dict, Optional
import sbatchman as sbm

sys.path.append(str(Path(__file__).parent.parent / "common" / "energy"))
from ncm_parser import parse_ncm_tot_energy_print


def parse(job: sbm.Job) -> Optional[Dict[str, Dict]]:
    """Parse llama-bench JSON stdout into core metrics and samples."""
    if not job.tag.startswith("llamacpp_") or job.status != sbm.Status.COMPLETED.value:
        return None

    data = {k: v for k, v in (job.variables or {}).items()}
    data["cluster"] = job.cluster_name
    data["tot_runtime"] = job.get_run_time()
    stdout = job.get_stdout()

    if not stdout:
        return None

    try:
        runs = json.loads(stdout)
    except json.JSONDecodeError:
        return None

    parsed_runs = []

    for run in runs:
        run_metrics = {
            # Core Hardware & Backend
            "cpu_info": run.get("cpu_info"),
            # "gpu_info": run.get("gpu_info"),
            "backends": run.get("backends"),
            
            # Model Specs
            "model": run.get("model_type"),
            "model_size_bytes": run.get("model_size"),
            
            # Runtime / Execution Config
            "n_batch": run.get("n_batch"),
            "n_threads": run.get("n_threads"),
            "n_gpu_layers": run.get("n_gpu_layers"),
            "n_prompt": run.get("n_prompt", 0),
            "n_gen": run.get("n_gen", 0),
            
            # Summary Metrics
            "avg_ts": run.get("avg_ts"),
            "stddev_ts": run.get("stddev_ts"),
            "avg_ns": run.get("avg_ns"),
            "stddev_ns": run.get("stddev_ns"),
            
            # Detailed Sample Data
            "samples_ns": run.get("samples_ns", []),
            "samples_ts": run.get("samples_ts", []),
        }
        parsed_runs.append(run_metrics)

    data["runs"] = parsed_runs

    # Energy parsing
    tot_energy = parse_ncm_tot_energy_print(stdout)
    if tot_energy:
        data["tot_energy_J"] = tot_energy

    data["detailed_energy"] = "no"

    return {"llamacpp": data}