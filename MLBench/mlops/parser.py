import csv
import io
from typing import Any, Dict, List, Optional
import sbatchman as sbm

# CSV tables emitted by the mlops micro-benchmarks (see src/common.h:print_csv):
#   gemm:    op,M,K,N,dtype,threads,iter,ms,gflops
#   gemv:    op,K,N,dtype,threads,iter,ms,gflops
#   gelu:    op,rows,cols,dtype,threads,iter,ms,gbps
#   softmax: op,rows,cols,dtype,threads,iter,ms,gbps
TABLE_NAME = "mlops"

_INT_COLS = {"M", "K", "N", "rows", "cols", "threads", "iter"}
_FLOAT_COLS = {"ms", "gflops", "gbps"}


def _parse_mlops_csv(stdout: str) -> List[Dict[str, Any]]:
    """Parse mlops benchmark stdout (plain CSV, one row per timed iteration)."""
    reader = csv.DictReader(io.StringIO((stdout or "").strip()))
    rows: List[Dict[str, Any]] = []
    for raw in reader:
        if not raw.get("op"):
            continue
        row: Dict[str, Any] = {}
        for k, v in raw.items():
            if k is None:
                continue
            key = k.strip()
            val = v.strip() if isinstance(v, str) else v
            if key in _INT_COLS:
                try:
                    row[key] = int(float(val))  # tolerate "4096.0"
                except (TypeError, ValueError):
                    row[key] = val
            elif key in _FLOAT_COLS:
                try:
                    row[key] = float(val)
                except (TypeError, ValueError):
                    row[key] = val
            else:
                row[key] = val
        # Unified metric columns for easy cross-op plotting/filtering.
        if "gflops" in row:
            row["throughput"] = row["gflops"]
            row["throughput_unit"] = "gflops"
            row["metric"] = row["gflops"]
            row["metric_name"] = "gflops"
        elif "gbps" in row:
            row["throughput"] = row["gbps"]
            row["throughput_unit"] = "gbps"
            row["metric"] = row["gbps"]
            row["metric_name"] = "gbps"
        rows.append(row)
    return rows


def parse(job: sbm.Job) -> Optional[Dict[str, Any]]:
    """Parse one mlops benchmark job for `sbatchman visualize`.

    Returns ``{"mlops": [row, ...]}`` (one row per timed iteration, enriched
    with job metadata) or ``None`` when the job has no parsable mlops CSV.
    """
    if job.status != sbm.Status.COMPLETED:
        return None

    try:
        stdout = job.get_stdout()
    except Exception:
        return None
    if not stdout:
        return None

    iter_rows = _parse_mlops_csv(stdout)
    if not iter_rows:
        return None

    # Base metadata shared by every row of this job.
    base: Dict[str, Any] = dict(job.variables or {})
    base["cluster"] = job.cluster_name
    base["tag"] = job.tag
    base["job_id"] = job.job_id
    base["config_name"] = job.config_name
    try:
        base["tot_runtime"] = job.get_run_time()
    except Exception:
        pass

    enriched: List[Dict[str, Any]] = []
    for r in iter_rows:
        merged = dict(base)
        merged.update(r)
        enriched.append(merged)

    return {TABLE_NAME: enriched}

