"""
parse_results_baseline.py
==========================
Collects all completed sbatchman baseline jobs, parses their stdout as multiple CSVs,
and writes every measurement into a single Parquet file (with multiple tables) 
via import_export.

Baseline jobs are identified by a tag starting with "baseline_".
Each job's stdout is expected to be processed into dictionary of CSV strings.

Summary printed to stdout:
  - Per job: outcome (ok / no_data / bad_csv).
  - Global totals.
  - List of every problematic job with the reason.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from compact_csv import compact_all
from parsers import stdout_to_csv_multi

sys.path.append(str(Path(__file__).parent.parent / "common"))
import import_export
import sbatchman as sbm
import time

def format_bytes(size_bytes, binary=False, precision=2, space_between_size_and_unit=False):
    """
    Convert a size in bytes into a human-readable string.

    Args:
        size_bytes (int or float): Size in bytes
        binary (bool): If True, use binary units (KiB, MiB, GiB).
                        If False, use SI units (KB, MB, GB)
        precision (int): Number of decimal places

    Returns:
        str: Human-readable string
    """
    if size_bytes < 0:
        raise ValueError("size_bytes must be non-negative")

    if binary:
        # Binary prefixes: 1024
        units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
        factor = 1024.0
    else:
        # SI prefixes: 1000
        units = ["B", "KB", "MB", "GB", "TB", "PB"]
        factor = 1000.0

    size = float(size_bytes)
    for unit in units:
        if size < factor:
            return f"{size:.{precision}f}{' ' if space_between_size_and_unit else ''}{unit}"
        size /= factor

    return f"{size:.{precision}f}{' ' if space_between_size_and_unit else ''}{units[-1]}"

# ---------------------------------------------------------------------------
# Where to write the result
# ---------------------------------------------------------------------------

OUT_DIR = Path("results")

# ---------------------------------------------------------------------------
# CSV parsing 
# ---------------------------------------------------------------------------

def _parse_csv_multi(csv_dict: dict[str, str]) -> dict[str, pd.DataFrame] | None:
    """
    Parses a dictionary of CSV-formatted strings into a dictionary of DataFrames.
    """
    if not csv_dict:
        return None
        
    dfs = {}
    for key, text in csv_dict.items():
        text = text.strip()
        if not text:
            continue
        try:
            df = pd.read_csv(io.StringIO(text))
            if not df.empty and df.columns.tolist() != []:
                dfs[key] = df
        except Exception:
            pass
            
    return dfs if dfs else None


# ---------------------------------------------------------------------------
# Outcome labels
# ---------------------------------------------------------------------------

OUTCOME_OK      = "ok"
OUTCOME_NO_DATA = "no_data"
OUTCOME_BAD_CSV = "bad_csv"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cluster_name = sbm.get_cluster_name()
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    out_file = OUT_DIR / f"DLNetBenchBaseline_{cluster_name}_data.parquet"

    all_jobs = sbm.jobs_list(
        status=[sbm.Status.COMPLETED],
        from_active=True,
        from_archived=False,
    )

    # Keep only baseline jobs
    jobs = [j for j in all_jobs if str(j.tag).startswith("baseline_")]

    if not jobs:
        print("No completed baseline jobs found.")
        return

    print(f"Found {len(jobs)} completed baseline job(s).\n")

    # Accumulated pairs for import_export
    pairs: list[tuple[dict[str, Any], dict[str, pd.DataFrame]]] = []

    # Summary bookkeeping
    issues: list[tuple[str, str, str, str]] = []   # (job_id, tag, outcome, detail)
    total_ok = 0

    job_summaries: list[dict] = []
    jobs_runtimes: dict = {}
    jobs_stdout_size: dict = {}
    jobs_stdout_size_compact: dict = {}
    jobs_stdout_compaction_time: dict = {}

    for job in jobs:
        stdout = job.get_stdout()
        tag    = str(job.tag)

        print(f"  job_id={job.job_id}  tag={tag}  runtime={job.get_run_time()}")

        if stdout is None or stdout.strip() == "":
            issues.append((str(job.job_id), tag, OUTCOME_NO_DATA, "stdout is empty"))
            job_summaries.append({"job_id": job.job_id, "tag": tag, "outcome": OUTCOME_NO_DATA})
            continue
        
        nodes_line = None 
        try:
            # Extract and remove "Allocated nodes: <nodes>" line if present
            if "Allocated nodes: " in stdout:
                lines = stdout.splitlines()
                filtered_lines = []
                for line in lines:
                    if line.startswith("Allocated nodes:"):
                        nodes_line = line
                    else:
                        filtered_lines.append(line)
                stdout = '\n'.join(filtered_lines)

            csv_dict,_ = stdout_to_csv_multi(stdout)
            dfs = _parse_csv_multi(csv_dict)
        except Exception as e:
            dfs = None
            issues.append((
                str(job.job_id), tag, OUTCOME_BAD_CSV,
                f"Parser failed with error: {str(e)}",
            ))
            job_summaries.append({"job_id": job.job_id, "tag": tag, "outcome": OUTCOME_BAD_CSV})
            continue

        if dfs is None:
            issues.append((
                str(job.job_id), tag, OUTCOME_BAD_CSV,
                f"could not parse CSV outputs; first 120 chars: {stdout[:120].strip()!r}",
            ))
            job_summaries.append({"job_id": job.job_id, "tag": tag, "outcome": OUTCOME_BAD_CSV})
            continue

        # Good data — build flat metadata from the job
        meta: dict[str, Any] = {
            "sbm_job_id":  job.job_id,
            "sbm_tag":     tag,
            "cluster":     cluster_name,
            "tot_runtime": job.get_run_time(),
            # set at launch time in launch_baseline_singlenode.py
            "strategy":    (job.variables or {}).get("strategy"),
            "gpus":        (job.variables or {}).get("gpus"),
            "nodes":       (job.variables or {}).get("nodes"),
            "comm_lib":    (job.variables or {}).get("comm_lib"),
            "gpu_model":   (job.variables or {}).get("gpu_model"),
            "nodelist":    nodes_line.removeprefix("Allocated nodes: ") if nodes_line else None,
        }

        # Instead of wrapping in {"measurements": df}, pass the entire dfs dict
        pairs.append((meta, dfs))
        job_key = f"{cluster_name}__{meta.get('strategy')}__{meta.get('gpus')}__{meta.get('gpu_model')}"
        jobs_runtimes[job_key] = job.get_run_time()
        jobs_stdout_size[job_key] = job.get_stdout_path().stat().st_size
        out = job.get_stdout()
        start_time = time.time()
        compacted = compact_all(out, warn_within_rank=True)
        elapsed = time.time() - start_time
        jobs_stdout_compaction_time[job_key] = elapsed
        jobs_stdout_size_compact[job_key] = len(compacted)
        total_ok += 1
        job_summaries.append({"job_id": job.job_id, "tag": tag, "outcome": OUTCOME_OK})
        
    print()
    print("=== Runtimes ===")
    print(jobs_runtimes)
    print()
    
    print()
    print("=== Stdout sizes ===")
    print({k: format_bytes(s) for k, s in jobs_stdout_size.items()})
    print('--- Compact ---')
    print({k: format_bytes(s) for k, s in jobs_stdout_size_compact.items()})
    print('--- Python compact time [s] ---')
    print({k: (s, format_bytes(jobs_stdout_size[k])) for k, s in jobs_stdout_compaction_time.items()})
    print('--- Ratios ---')
    print({k: f'{jobs_stdout_size_compact[k]/s*100.0:.2f}%' for k, s in jobs_stdout_size.items()})
    print('--- Assuming 1 compact run of ALL (generous upperbound) ---')
    print(format_bytes(sum([int(s) for s in jobs_stdout_size_compact.values()])))
    print()

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    col_job = max(len("job_id"), max((len(str(r["job_id"])) for r in job_summaries), default=0))
    col_tag = max(len("tag"),    max((len(str(r["tag"]))    for r in job_summaries), default=0))

    header  = f"{'job_id':<{col_job}}  {'tag':<{col_tag}}  outcome"
    divider = "-" * (col_job + col_tag + 12)

    print()
    print("=== Per-job summary ===")
    print(header)
    print(divider)
    for r in job_summaries:
        print(f"{str(r['job_id']):<{col_job}}  {str(r['tag']):<{col_tag}}  {r['outcome']}")
    print(divider)
    print(f"TOTAL: {len(jobs)} jobs  —  ok={total_ok}  issues={len(issues)}")
    print()

    # ------------------------------------------------------------------
    # Issues detail
    # ------------------------------------------------------------------
    if issues:
        print(f"=== Issues ({len(issues)} total) ===")
        for job_id, tag, outcome, detail in issues:
            print(f"  [{outcome:<10}]  job={job_id}  tag={tag}")
            print(f"               {detail}")
        print()
    else:
        print("No issues — all baseline jobs produced clean data.\n")

    # ------------------------------------------------------------------
    # Write Parquet
    # ------------------------------------------------------------------
    if not pairs:
        print("Nothing to write — no valid data collected.")
        return

    # ------------------------------------------------------------------
    # Preview collected data
    # ------------------------------------------------------------------
    preview_rows = []
    for meta, dfs in pairs:
        # We target 'main' since the parser guarantees 'main' is generated for all strategies
        if "main" in dfs:
            df = dfs["main"].copy()
            for k, v in meta.items():
                df[k] = v
            preview_rows.append(df)

    if preview_rows:
        preview = pd.concat(preview_rows, ignore_index=True)
        print("=== Preview of collected 'main' data ===")
        print(preview.to_string())
        print()
    else:
        print("No 'main' DataFrames were generated to preview.\n")

    print(f"Writing {len(pairs)} job(s) to {out_file} ...")
    import_export.write_multiple_to_parquet(pairs, out_file)
    print(f"Done.  Output: {out_file.resolve()}")


if __name__ == "__main__":
    main()