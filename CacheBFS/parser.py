import pprint
import re
import sys
from pathlib import Path
from copy import copy
from typing import List, Dict, Any
import sbatchman as sbm

sys.path.append(str(Path(__file__).parent.parent / "common"))
from energy.energy_parsers import get_tot_energy_measurement
from utils.utils import remove_energy_logs, remove_useless_job_meta


def parse_openmp_output(stdout: str) -> List[Dict[str, Any]]:
    """
    Parses OpenMP output format:
    Number of threads: 2
    run=1,0.864411
    """
    threads = None
    threads_match = re.search(r"Number of threads:\s*(\d+)", stdout)
    if threads_match:
        threads = int(threads_match.group(1))

    rows = []
    for line in stdout.splitlines():
        match = re.search(r"run=(\d+),\s*([\d\.\-eE]+)", line)
        if match:
            rows.append({
                "threads": threads,
                "run": int(match.group(1)),
                "time": float(match.group(2))
            })

    return rows


def parse_gap_output(stdout: str) -> List[Dict[str, Any]]:
    """
    Parses GAP Benchmark Suite output format into trial rows with graph metadata.
    """
    common_info: Dict[str, Any] = {}

    read_match = re.search(r"Read Time:\s*([\d\.\-eE]+)", stdout)
    if read_match:
        common_info["read_time"] = float(read_match.group(1))

    build_match = re.search(r"Build Time:\s*([\d\.\-eE]+)", stdout)
    if build_match:
        common_info["build_time"] = float(build_match.group(1))

    graph_match = re.search(r"Graph has (\d+) nodes and (\d+) directed edges for degree:\s*(\d+)", stdout)
    if graph_match:
        common_info["nodes"] = int(graph_match.group(1))
        common_info["edges"] = int(graph_match.group(2))
        common_info["degree"] = int(graph_match.group(3))

    avg_match = re.search(r"Average Time:\s*([\d\.\-eE]+)", stdout)
    if avg_match:
        common_info["avg_time"] = float(avg_match.group(1))

    rows = []
    trial_idx = 1
    for line in stdout.splitlines():
        m = re.search(r"Trial Time:\s*([\d\.\-eE]+)", line)
        if m:
            row = copy(common_info)
            row["trial"] = trial_idx
            row["time"] = float(m.group(1))
            rows.append(row)
            trial_idx += 1

    return rows


def parse_mergedcsr_output(stdout: str) -> List[Dict[str, Any]]:
    """
    Parses CSV-like key=val formats used by mergedcsr, mutex, atomic, and cnalock:
    run_id=0,diameter=6797,threads=8,chunk_size=64,max_chunks=111,0.3441
    """
    rows = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue

        row: Dict[str, Any] = {}
        for part in line.split(","):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                k, v = k.strip(), v.strip()
                if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
                    row[k] = int(v)
                else:
                    try:
                        row[k] = float(v)
                    except ValueError:
                        row[k] = v
            else:
                try:
                    row["time"] = float(part)
                except ValueError:
                    pass

        if row:
            rows.append(row)

    return rows


def parse(job: sbm.Job) -> dict | None:
    """
    Parse benchmark stdout into structured metrics.
    Returns {table_name: row(s)} where row(s) can be a single dict or a list of dicts.
    """
    if job.category == "compile" or job.status != sbm.Status.COMPLETED.value:
        return None

    meta = {k: v for k, v in (job.variables or {}).items()}
    meta["cluster"] = job.cluster_name
    meta["tot_runtime"] = job.get_run_time()
    meta = remove_useless_job_meta(meta)

    raw_stdout = job.get_stdout()
    if not raw_stdout:
        return None

    stdout = remove_energy_logs(raw_stdout)

    tot_energy = get_tot_energy_measurement(job, raw_stdout)
    if tot_energy:
        meta["tot_energy_J"] = tot_energy

    category = (job.category or "").lower()

    if category == "openmp":
        parsed_rows = parse_openmp_output(stdout)
    elif category == "gap":
        parsed_rows = parse_gap_output(stdout)
    elif category in ("mergedcsr", "mutex", "atomic", "cnalock"):
        parsed_rows = parse_mergedcsr_output(stdout)
    else:
        return None

    if not parsed_rows:
        return {}

    # Copy meta into each row dict
    if isinstance(parsed_rows, list):
        enriched_rows = [{**meta, **row} for row in parsed_rows]
    else:
        enriched_rows = {**meta, **parsed_rows}

    table_name = 'bfs' #category
    return {table_name: enriched_rows}