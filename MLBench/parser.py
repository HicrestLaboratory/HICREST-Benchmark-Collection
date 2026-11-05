import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import csv
import io
import json
import re

import sbatchman as sbm

# import mlops.parser as mlops_parser
# import inference.parser as inference_parser

sys.path.append(str(Path(__file__).parent.parent / "common"))
from utils.utils import _remove_ncm_logs
from energy.energy_parsers import _parse_ncm_tot_energy_print
from utils.utils import remove_useless_job_meta


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


def mlops_parser_parse(job: sbm.Job, stdout: str) -> Optional[Dict[str, Any]]:
    """Parse one mlops benchmark job for `sbatchman visualize`.

    Returns ``{"mlops": [row, ...]}`` (one row per timed iteration, enriched
    with job metadata) or ``None`` when the job has no parsable mlops CSV.
    """
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
        for k in ['n', 'k', 'm']:
            if k in merged:
                del merged[k]
        enriched.append(merged)
    
    return {TABLE_NAME: enriched}

TABLE_NAME = "inference"

def _num(s: str):
    for cast in (int, float):
        try:
            return cast(s)
        except (TypeError, ValueError):
            pass
    return s

_SIZE_UNITS = {"B": 1, "KIB": 1024, "MIB": 1024**2, "GIB": 1024**3, "TIB": 1024**4,
               "KB": 1000, "MB": 1000**2, "GB": 1000**3, "TB": 1000**4}
_PARAM_UNITS = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}

def inference_parse_table(stdout: str) -> List[Dict[str, Any]]:
    """Parse llama-bench markdown table(s) into a list of JSON-serializable dicts."""
    lines = [l.strip() for l in (stdout or "").splitlines() if l.strip().startswith("|")]
    sep = re.compile(r"^\|\s*:?-{2,}")
    split = lambda l: [c.strip() for c in l.strip("|").split("|")]

    rows: List[Dict[str, Any]] = []
    header: Optional[List[str]] = None

    for i, line in enumerate(lines):
        if sep.match(line):
            header = split(lines[i - 1]) if i > 0 else None
            continue
        if i + 1 < len(lines) and sep.match(lines[i + 1]):
            continue  # this is a header line
        if header is None:
            continue
        cells = split(line)
        if len(cells) != len(header):
            continue

        row: Dict[str, Any] = {}
        for col, val in zip(header, cells):
            key = re.sub(r"[^\w]+", "_", col.lower()).strip("_")  # "t/s" -> "t_s"

            if key == "size":
                m = re.match(r"^([\d.]+)\s*([KMGT]?i?B)$", val, re.I)
                row["size_str"] = val
                row["size"] = int(float(m.group(1)) * _SIZE_UNITS[m.group(2).upper()]) if m else val
            elif key == "params":
                m = re.match(r"^([\d.]+)\s*([KMBT])$", val, re.I)
                row["params_str"] = val
                row["params"] = int(float(m.group(1)) * _PARAM_UNITS[m.group(2).upper()]) if m else val
            elif key == "t_s":
                mean, _, std = val.partition("±")
                row["tps"] = _num(mean.strip())
                row["tps_std"] = _num(std.strip()) if std.strip() else None
            elif key == "test":
                row["test"] = val
                if m := re.search(r"pp(\d+)", val): row["n_prompt"] = int(m.group(1))
                if m := re.search(r"tg(\d+)", val): row["n_gen"] = int(m.group(1))
                if m := re.search(r"d(\d+)\s*$", val): row["depth"] = int(m.group(1))
            else:
                row[key] = _num(val)  # model, backend, threads, ngl, fa, ...
        rows.append(row)

    return rows

_JSON_KEYS = ('cpu_info', 'samples_ts', 'samples_ns', 'model_filename', 'model_size')

def inference_parse(job: sbm.Job, stdout: str) -> Optional[Dict[str, List[Dict]]]:
    from_json = True
    try:
        json_data = json.loads(stdout)
    except json.JSONDecodeError:
        # Not JSON: assume llama-bench markdown table
        from_json = False
        json_data = inference_parse_table(stdout)

    if not isinstance(json_data, list) or not json_data:
        return None

    base = {k: v for k, v in (job.variables or {}).items()}
    base['cluster'] = job.cluster_name
    base['tot_runtime'] = job.get_run_time()

    rows = []
    for item in json_data:
        if not isinstance(item, dict):
            continue
        if from_json:
            # keep only the fields you care about from the JSON output
            item = {k: item.get(k) for k in _JSON_KEYS if k in item}
        rows.append({**base, **item})  # item columns override job variables

    return {TABLE_NAME: rows} if rows else None

def parse(job: sbm.Job) -> Optional[Dict[str, Any]]:
    if job.status != sbm.Status.COMPLETED.value or job.category == 'compile':
        return None
    # print(job.category)
    # print(job.tag)
    
    stdout = None
    try:
        stdout = job.get_stdout()
    except Exception:
        return None
    if not stdout:
        return None
    
    res = None

    # Small fix
    if job.category.startswith('gelu') and job.tag.startswith('softmax'):
        job.category = 'softmax'

    # TODO add energy
    if job.category.startswith(('gemv', 'gemm', 'gelu', 'softmax')):
        res = mlops_parser_parse(job, _remove_ncm_logs(stdout))
    elif job.category.startswith(('inference')):
        res = inference_parse(job, _remove_ncm_logs(stdout))
    else:
        print(f'WARNING: category "{job.category}" not recognized (tag {job.tag}).')
        return None
    
    # TODO separate table?
    # if res:
    #     tot_energy = parse_ncm_tot_energy_print(stdout)
    #     if tot_energy:
    #         for k in res.keys():
    #             res[k]['tot_energy_J'] = tot_energy
    
    if res:
        tables = list(res.keys())
        for k in tables:
            for i in range(len(res[k])):
                res[k][i] = remove_useless_job_meta(res[k][i])
                
    # print(res[list(res.keys())[0]][0].keys())
    # print('====================')
    return res