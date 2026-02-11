#!/usr/bin/env python3
from pathlib import Path
from typing import Any, Dict
import sbatchman as sbm
import sys
import pandas as pd
import re
import sbatchman as sbm
from metrics import METRICS_TO_EXTRACT

sys.path.append(str(Path(__file__).parent.parent.parent / 'common'))
from utils.utils import raise_none, dict_get

OUT_DIR = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_hpcg_output(text: str) -> dict:
    """
    Parse an HPCG text file into a nested dict.
    If a node has both a scalar value and children, the scalar is stored under '_value'.
    """
    root = {}

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key_part, value_part = line.split("=", 1)
        key = key_part.strip()
        value_raw = value_part.strip()

        # normalize empty value -> None
        if value_raw == "":
            value = None
        else:
            # try int, then float, else keep string
            if re.match(r"^-?\d+$", value_raw):
                value = int(value_raw)
            else:
                try:
                    value = float(value_raw)
                except ValueError:
                    value = value_raw

        keys = [k.strip() for k in key.split("::")]

        # walk/inset into nested dict, converting scalars -> dicts as needed
        node = root
        for k in keys[:-1]:
            if k not in node:
                node[k] = {}
            elif not isinstance(node[k], dict):
                # convert scalar leaf into a dict, preserve old scalar under '_value'
                node[k] = {"_value": node[k]}
            node = node[k]

        last = keys[-1]
        if last in node:
            if isinstance(node[last], dict):
                # already has children -> keep scalar under '_value'
                node[last]["_value"] = value
            else:
                # existing scalar (rare): overwrite with new scalar value
                node[last] = value
        else:
            node[last] = value

    return root


def _get_section_value(parsed: dict, section: str, key_candidates):
    if not isinstance(parsed, dict):
        return None
    sec = parsed.get(section)
    if not isinstance(sec, dict):
        return None

    if isinstance(key_candidates, str):
        key_candidates = [key_candidates]
    for cand in key_candidates:
        if cand in sec:
            return sec[cand]

    lower_map = {k.lower(): k for k in sec.keys()}
    for cand in key_candidates:
        found = lower_map.get(cand.lower())
        if found:
            return sec[found]
    return None


def collect_metrics(parsed: dict) -> dict:
    out = {}

    processes = _get_section_value(parsed, "Machine Summary", ["Distributed Processes"])
    threads = _get_section_value(
        parsed, "Machine Summary", ["Threads per processes", "Threads per process"]
    )
    try:
        out["processes"] = int(processes) if processes is not None else None
    except (ValueError, TypeError):
        out["processes"] = None
    try:
        out["threads"] = int(threads) if threads is not None else None
    except (ValueError, TypeError):
        out["threads"] = None

    if out.get("processes") is not None and out.get("threads") is not None:
        out["total_cores"] = out["processes"] * out["threads"]
    else:
        out["total_cores"] = None

    for m in METRICS_TO_EXTRACT:
        value = _get_section_value(parsed, m["section"], m["candidates"])
        out[m["out_key"]] = value

    final_section = parsed.get("Final Summary")
    if isinstance(final_section, dict):
        out["final_result_valid"] = (
            final_section.get("Result")
            or final_section.get("HPCG result is VALID with a GFLOP/s rating of")
            or final_section.get("Results are valid but execution time (sec) is")
        )
        out["final_result_valid"] = out["final_result_valid"] is not None and out["final_result_valid"] > 0
    else:
        out["final_result_valid"] = None

    return out


def parse_job(j: sbm.Job) -> Dict[Any, Any]:
    lines = str(raise_none(j.get_stdout(), "stdout")).splitlines()
    hpcg_section_start_idx = lines.index('HPCG-Benchmark')
    hpcg_section = '\n'.join(lines[hpcg_section_start_idx:])

    if not j.variables:
        raise Exception(f'job "{j}" has no variables')

    meta = {
        "partition": j.variables.get("partition", "unknown"),
        "nodes": j.variables["nodes"],
        "cluster": sbm.get_cluster_name(),
    }
    parsed_hpcg_output = parse_hpcg_output(hpcg_section)
    hpcg_metrics = collect_metrics(parsed_hpcg_output)
    hpcg_metrics_keys = [
        "threads",
        "total_cores",
        "mem",
        "global_nx",
        "global_ny",
        "global_nz",
        "num_equations",
        "time_tot",
        "gflops",
        "gflops_opt",
        "final_result_valid",
    ]
    for k in hpcg_metrics_keys:
        meta[k] = dict_get(hpcg_metrics, k)

    return meta


def main():
    jobs = sbm.jobs_list(
        status=[sbm.Status.COMPLETED], from_active=True, from_archived=False
    )
    out_file = OUT_DIR / f"hpcg_nvidia_{sbm.get_cluster_name()}_data.csv"
    pd.DataFrame([parse_job(j) for j in jobs]).to_csv(out_file, index=False)
    print(f'Results saved to {out_file}')



if __name__ == "__main__":
    main()
