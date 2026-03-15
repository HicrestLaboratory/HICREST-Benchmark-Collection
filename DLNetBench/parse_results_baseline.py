"""
parse_results_baseline.py
==========================
Collects all completed sbatchman baseline jobs, parses their stdout as CSV,
and writes every measurement into a single Parquet file via import_export.

Baseline jobs are identified by a tag starting with "baseline_".
Each job's stdout is expected to be a raw CSV (header + data rows).

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

sys.path.append(str(Path(__file__).parent.parent / "common"))
import import_export
import sbatchman as sbm

# ---------------------------------------------------------------------------
# Where to write the result
# ---------------------------------------------------------------------------

OUT_DIR = Path("results")

# ---------------------------------------------------------------------------
# CSV parsing  (same logic as aggregate_runs.py)
# ---------------------------------------------------------------------------

def _parse_csv(stdout: str) -> pd.DataFrame | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        df = pd.read_csv(io.StringIO(text))
        if df.empty or df.columns.tolist() == []:
            return None
        return df
    except Exception:
        return None


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

    for job in jobs:
        stdout = job.get_stdout()
        tag    = str(job.tag)

        print(f"  job_id={job.job_id}  tag={tag}  runtime={job.get_run_time()}")

        if stdout.strip() == "":
            issues.append((str(job.job_id), tag, OUTCOME_NO_DATA, "stdout is empty"))
            job_summaries.append({"job_id": job.job_id, "tag": tag, "outcome": OUTCOME_NO_DATA})
            continue

        df = _parse_csv(stdout)
        if df is None:
            issues.append((
                str(job.job_id), tag, OUTCOME_BAD_CSV,
                f"could not parse CSV; first 120 chars: {stdout[:120].strip()!r}",
            ))
            job_summaries.append({"job_id": job.job_id, "tag": tag, "outcome": OUTCOME_BAD_CSV})
            continue

        # Good data — build flat metadata from the job
        meta: dict[str, Any] = {
            "sbm_job_id":  job.job_id,
            "sbm_tag":     tag,
            "cluster":     cluster_name,
            "tot_runtime": str(job.get_run_time()),
            # set at launch time in launch_baseline_singlenode.py
            "strategy":    (job.variables or {}).get("strategy"),
            "gpus":        (job.variables or {}).get("gpus"),
            "nodes":       (job.variables or {}).get("nodes"),
        }

        pairs.append((meta, {"measurements": df}))
        total_ok += 1
        job_summaries.append({"job_id": job.job_id, "tag": tag, "outcome": OUTCOME_OK})

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

    print(f"Writing {len(pairs)} job(s) to {out_file} ...")
    import_export.write_multiple_to_parquet(pairs, out_file)
    print(f"Done.  Output: {out_file.resolve()}")


if __name__ == "__main__":
    main()