"""
aggregate_runs.py
=================
Collects all completed sbatchman jobs, parses their scheduler output,
and writes every measurement into a single Parquet file via import_export.

Each sbatchman job corresponds to one scheduler invocation (one pattern run).
Within it, the scheduler may have executed many job repetitions.  Each
repetition that produced parseable CSV data becomes one (metadata, DataFrame)
pair written to Parquet.

Summary printed to stdout:
  - Per sbatchman job: how many runs were found, how many had good data,
    how many failed or had no data.
  - Global totals.
  - List of every problematic run with the reason.
"""

from __future__ import annotations

import io
from pprint import pprint
import sys
from pathlib import Path
from typing import Any
import re
from warnings import warn
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent / "common"))
from compact_csv import compact_all
import import_export
import sbatchman as sbm
from parsers import parse_scheduler_output, stdout_file_to_csv_multi

# ---------------------------------------------------------------------------
# Where to write the result
# ---------------------------------------------------------------------------

OUT_DIR = Path("results")

# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def _parse_csv(stdout: str) -> dict[str, pd.DataFrame] | None:
    text = stdout.strip()
    if not text:
        return None

    dfs = {}
    current_name = None
    current_lines = []

    def flush():
        nonlocal current_name, current_lines
        if current_name is not None and current_lines:
            csv_content = "\n".join(current_lines).strip()
            if csv_content:
                try:
                    # Replace commas outside of {} dict literals with a safe separator
                    def replace_commas(line: str) -> str:
                        result = []
                        depth = 0
                        for ch in line:
                            if ch == '{':
                                depth += 1
                            elif ch == '}':
                                depth -= 1
                            if ch == ',' and depth == 0:
                                result.append('\t')
                            else:
                                result.append(ch)
                        return ''.join(result)

                    safe_lines = [replace_commas(l) for l in csv_content.splitlines()]
                    safe_content = "\n".join(safe_lines)
                    df = pd.read_csv(io.StringIO(safe_content), sep='\t')
                    if not df.empty:
                        dfs[current_name] = df
                except Exception as e:
                    print(f"Failed to parse section '{current_name}': {e}")
        current_name = None
        current_lines = []

    for line in text.splitlines():
        if line.startswith("###"):
            flush()
            name = line.replace("###", "").strip()
            current_name = name if name else None
            current_lines = []
        elif current_name is not None:
            current_lines.append(line)

    flush()
    return dfs if dfs else None

# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def _build_metadata(sbm_job: sbm.Job, run: dict[str, Any]) -> dict[str, Any]:
    """
    Combine sbatchman-level job metadata with the per-run metadata from the
    scheduler output into a single flat dict that uniquely identifies the run.
    """
    return {
        # sbatchman level
        "sbm_job_id":    sbm_job.job_id,
        "sbm_tag":       sbm_job.tag,
        "cluster":       sbm_job.cluster_name,
        "tot_runtime":   str(sbm_job.get_run_time()),
        # scheduler run level
        "uid":           run["uid"],
        "job_name":      run["job_name"],
        "repetition":    run["repetition"],
        "resources":     ",".join(str(r) for r in run["resources"]),
        "app":           run["app"],
        "start_ts":      run["start_ts"],
        "finished_at":   run["finished_at"],
        "exit_code":     run["exit_code"],
    }


# ---------------------------------------------------------------------------
# Per-run outcome labels (used in the summary)
# ---------------------------------------------------------------------------

OUTCOME_OK        = "ok"
OUTCOME_NONZERO   = "nonzero_exit"
OUTCOME_NO_DATA   = "no_data"
OUTCOME_BAD_CSV   = "bad_csv"
OUTCOME_EXCEPTION = "exception"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cluster_name = sbm.get_cluster_name()
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    out_file = OUT_DIR / f"DLNetBenchConcurrent_{cluster_name}_data.parquet"

    jobs = sbm.jobs_list(
        status=[sbm.Status.COMPLETED],
        from_active=True,
        from_archived=False,
    )

    # walltime = 600
    # print(f"Found {len(jobs)} completed job(s) in sbatchman.  Filtering for walltime={walltime} seconds and excluding baselines...")
    jobs = [j for j in jobs if not j.tag.startswith('baseline')]
    print(f"{len(jobs)} job(s) remain after excluding job tags that start with 'baseline'.")
    # jobs = [j for j in jobs if j.variables.get('walltime') == walltime]
    # print(f"{len(jobs)} job(s) remain after walltime filtering.")
    # for j in jobs:
    #     print(f"{j.command}")

    if not jobs:
        print("No completed jobs found.")
        return

    # Accumulated (metadata, dataframes) pairs for import_export
    pairs: list[tuple[dict, dict[str, pd.DataFrame]]] = []

    # Summary bookkeeping
    # issues: list of (sbm_job_id, uid, outcome, detail)
    issues: list[tuple[str, str, str, str, str]] = []
    total_runs   = 0
    total_ok     = 0

    # Per-sbatchman-job summary rows (printed as a table)
    job_summaries: list[dict] = []

    for sbm_job in jobs:
        raw_stdout = sbm_job.get_stdout()
        print(f'Job tag={sbm_job.tag}  runtime={sbm_job.get_run_time()}')

        try:
            runs, log_lines = parse_scheduler_output(raw_stdout)
            REQUIRED_RUN_KEYS = {"uid", "job_name", "repetition", "resources",
                     "app", "start_ts", "finished_at", "exit_code",
                     "success", "stdout", "stderr"}
            for run in runs:
                missing = REQUIRED_RUN_KEYS - run.keys()
                if missing:
                    warn(f"Run dict missing keys: {missing}")
        except Exception as exc:
            issues.append((str(sbm_job.job_id), str(sbm_job.tag), "<all>", OUTCOME_EXCEPTION,
                           f"parse_scheduler_output raised: {exc}"))
            job_summaries.append({
                "sbm_job_id": sbm_job.job_id,
                "sbm_tag":    sbm_job.tag,
                "runs":       "?",
                "ok":         0,
                "nonzero":    0,
                "no_data":    0,
                "bad_csv":    0,
                "exception":  1,
            })
            continue

        n_ok = n_nonzero = n_no_data = n_bad_csv = 0

        for run in runs:
            total_runs += 1
            uid = run["uid"]

            # --- exit code check ---
            if not run["success"]:
                n_nonzero += 1
                issues.append((
                    str(sbm_job.job_id), str(sbm_job.tag), uid, OUTCOME_NONZERO,
                    f"exit_code={run['exit_code']}  stderr={run['stderr'][:120].strip()!r}",
                ))
                # Still attempt to parse whatever data was written before the failure
                # (the scheduler already fell back to raw on transform errors)

            stdout = str(run["stdout"]).strip()
            
            if stdout == "":
                n_no_data += 1
                issues.append((str(sbm_job.job_id), str(sbm_job.tag), uid, OUTCOME_NO_DATA, "stdout is empty"))
                continue
            
            if stdout.startswith('stdout: '):
                # This is a path to raw output file
                stdout_lines = stdout.splitlines()
                stdout_path = Path(stdout_lines[0].strip().removeprefix('stdout: '))
                # TODO check stderr
                # stderr_path = Path(stdout_lines[1].strip().removeprefix('stderr: '))
                stdout = compact_all(stdout_path.read_text(errors="replace"))
                
                # dict_df = {}
                # for k, v in dict_tmp.items():
                #     df = pd.read_csv(io.StringIO(v))
                #     if not df.empty:
                #         dict_df[k] = df
                
            # TODO handle non compressed case
            
            dict_df = _parse_csv(stdout)


            if dict_df is None:
                print(f"\033[92mUID: {uid}\033[0m")
                print(run['stdout'])
                n_bad_csv += 1
                issues.append((
                    str(sbm_job.job_id), str(sbm_job.tag), uid, OUTCOME_BAD_CSV,
                    f"could not parse CSV; first 120 chars: {run['stdout'][:120].strip()!r}",
                ))
                continue

            # --- good data ---
            if run["success"]:
                n_ok     += 1
                total_ok += 1

            meta = _build_metadata(sbm_job, run)
            pairs.append((meta, dict_df))
            # for key in dict_df.keys():
                # print(key)
                # pairs.append((meta, {key: dict_df[key]}))

        job_summaries.append({
            "sbm_job_id": sbm_job.job_id,
            "sbm_tag":    sbm_job.tag,
            "runs":       len(runs),
            "ok":         n_ok,
            "nonzero":    n_nonzero,
            "no_data":    n_no_data,
            "bad_csv":    n_bad_csv,
            "exception":  0,
        })

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    col_w = {
        "sbm_job_id": max(len("sbm_job_id"), max((len(str(r["sbm_job_id"])) for r in job_summaries), default=0)),
        "sbm_tag":    max(len("sbm_tag"), max((len(str(r["sbm_tag"])) for r in job_summaries), default=0)),
        "runs":       5,
        "ok":         4,
        "nonzero":    9,
        "no_data":    8,
        "bad_csv":    8,
        "exception":  10,
    }
    header = (
        f"{'sbm_job_id':<{col_w['sbm_job_id']}}  "
        f"{'sbm_tag':<{col_w['sbm_tag']}}  "
        f"{'runs':>{col_w['runs']}}  "
        f"{'ok':>{col_w['ok']}}  "
        f"{'nonzero':>{col_w['nonzero']}}  "
        f"{'no_data':>{col_w['no_data']}}  "
        f"{'bad_csv':>{col_w['bad_csv']}}  "
        f"{'exception':>{col_w['exception']}}"
    )
    divider = "-" * len(header)

    print()
    print("=== Per-job summary ===")
    print(header)
    print(divider)
    for r in job_summaries:
        print(
            f"{str(r['sbm_job_id']):<{col_w['sbm_job_id']}}  "
            f"{str(r['sbm_tag']):<{col_w['sbm_tag']}}  "
            f"{str(r['runs']):>{col_w['runs']}}  "
            f"{r['ok']:>{col_w['ok']}}  "
            f"{r['nonzero']:>{col_w['nonzero']}}  "
            f"{r['no_data']:>{col_w['no_data']}}  "
            f"{r['bad_csv']:>{col_w['bad_csv']}}  "
            f"{r['exception']:>{col_w['exception']}}"
        )
    print(divider)
    print(
        f"{'TOTAL':<{col_w['sbm_job_id']+col_w['sbm_tag']+2}}  "
        f"{total_runs:>{col_w['runs']}}  "
        f"{total_ok:>{col_w['ok']}}  "
        f"{sum(r['nonzero']   for r in job_summaries):>{col_w['nonzero']}}  "
        f"{sum(r['no_data']   for r in job_summaries):>{col_w['no_data']}}  "
        f"{sum(r['bad_csv']   for r in job_summaries):>{col_w['bad_csv']}}  "
        f"{sum(r['exception'] for r in job_summaries):>{col_w['exception']}}"
    )
    print()

    # ------------------------------------------------------------------
    # Issues detail
    # ------------------------------------------------------------------
    if issues:
        print(f"=== Issues ({len(issues)} total) ===")
        for sbm_id, tag, uid, outcome, detail in issues:
            print(f"  [{outcome:<12}]  job={sbm_id}  {tag=}  run={uid}")
            print(f"               {detail}")
        print()
    else:
        print("No issues found — all runs produced clean data.\n")

    # ------------------------------------------------------------------
    # Write Parquet
    # ------------------------------------------------------------------
    if not pairs:
        print("Nothing to write — no valid data collected.")
        return

    print(f"Writing {len(pairs)} run(s) to {out_file} ...")
    # for i in range(10):
    #     print(pairs[i][0])
    #     for k in pairs[i][1].keys():
    #         print(f'{k} --> {pairs[i][1][k].columns} (len {len(pairs[i][1][k])})')
    #     print()
    import_export.write_multiple_to_parquet(pairs, out_file)
    print(f"Done.  Output: {out_file.resolve()}")
    print()

    # Optionally describe contents
    # import_export.describe_pairs_content(pairs, verbose=True)


if __name__ == "__main__":
    main()
