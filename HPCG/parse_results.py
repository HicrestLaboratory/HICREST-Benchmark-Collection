#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sbatchman as sbm
import sys
import pandas as pd
import re
import sbatchman as sbm
from metrics import METRICS_TO_EXTRACT

sys.path.append(str(Path(__file__).parent.parent / 'common'))
import ccutils.parser.ccutils_parser as ccutils_parser
import import_export
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


def extract_metrics_dict(dp_section) -> Dict[str, pd.DataFrame]:
    dfs = {
        "dotp": [],
        "spmv_halo": [],
        "waxpby": [],
        "cg_times": [],
        "mg": [],
        "halo_precond": [],
    }

    rank_outputs = dp_section.mpi_all_prints["ccutils_rank_json"].rank_outputs
    for rank, json_str in rank_outputs.items():
        parsed = json.loads(json_str)
        for iter_key in sorted(parsed.keys(), key=int):
            iter_data = parsed[iter_key]
            spmv_list = iter_data.get("spmv", [])
            halo_kernels = iter_data.get("halo_kernels", [])
            exchange_halo_list = iter_data.get("exchange_halo", [])
            halo_msg_sizes = iter_data.get("halo_msg_sizes", [])

            max_len = max(
                len(iter_data.get("dotp", [])),
                len(iter_data.get("dotp_allreduce", [])),
                len(iter_data.get("waxpby", [])),
                len(iter_data.get("mg", [])),
                1,
            )
            # DOTP
            for idx in range(max_len):
                dfs["dotp"].append(
                    {
                        "rank": rank,
                        "run": int(iter_key),
                        "dotp": (
                            iter_data.get("dotp", [None] * max_len)[idx]
                            if idx < len(iter_data.get("dotp", []))
                            else None
                        ),
                        "dotp_allreduce": (
                            iter_data.get("dotp_allreduce", [None] * max_len)[idx]
                            if idx < len(iter_data.get("dotp_allreduce", []))
                            else None
                        ),
                    }
                )
            # SPMV + HALO (only halo_kernel == "SPMV")
            spmv_counter = 0
            for halo_idx, kernel in enumerate(halo_kernels):
                if kernel == "SPMV":
                    dfs["spmv_halo"].append(
                        {
                            "rank": rank,
                            "run": int(iter_key),
                            "spmv": (
                                spmv_list[spmv_counter]
                                if spmv_counter < len(spmv_list)
                                else None
                            ),
                            "exchange_halo": (
                                exchange_halo_list[halo_idx]
                                if halo_idx < len(exchange_halo_list)
                                else None
                            ),
                            "halo_msg_size_bytes": (
                                halo_msg_sizes[halo_idx]
                                if halo_idx < len(halo_msg_sizes)
                                else None
                            ),
                        }
                    )
                    spmv_counter += 1
            # MG (standalone)
            mg_list = iter_data.get("mg", [])
            for idx, val in enumerate(mg_list):
                dfs["mg"].append({"rank": rank, "run": int(iter_key), "mg": val})
            # HALO (preconditioning only)
            for halo_idx, kernel in enumerate(halo_kernels):
                if kernel and "preconditioning_" in str(kernel):
                    dfs["halo_precond"].append(
                        {
                            "rank": rank,
                            "run": int(iter_key),
                            "exchange_halo": (
                                exchange_halo_list[halo_idx]
                                if halo_idx < len(exchange_halo_list)
                                else None
                            ),
                            "halo_msg_size_bytes": (
                                halo_msg_sizes[halo_idx]
                                if halo_idx < len(halo_msg_sizes)
                                else None
                            ),
                        }
                    )
            # WAXPBY
            waxpby_list = iter_data.get("waxpby", [])
            for idx, val in enumerate(waxpby_list):
                dfs["waxpby"].append(
                    {"rank": rank, "run": int(iter_key), "waxpby": val}
                )
            # CG TIMES (one row per rank x run)
            dfs["cg_times"].append(
                {
                    "rank": rank,
                    "run": int(iter_key),
                    "cg_times": iter_data.get("cg_times", [None])[0],
                }
            )

    final_df_dict = {}
    for key in dfs:
        final_df_dict[key] = pd.DataFrame(dfs[key]).reset_index(drop=True)

    return final_df_dict


def parse_job(j: sbm.Job) -> Tuple[Dict[Any, Any], Dict[str, pd.DataFrame]]:
    stdout = raise_none(j.get_stdout(), "stdout")
    res = ccutils_parser.parse_ccutils_output(stdout)
    cg_section = dict_get(res, "cg")
    hpcg_section = dict_get(res, "hpcg_output")

    if not j.variables:
        raise Exception(f'job "{j}" has no variables')

    meta = {
        "partition": j.variables.get("partition", "unknown"),
        "nodes": dict_get(cg_section.json_data, "world_size"),
        "cluster": sbm.get_cluster_name(),
    }
    parsed_hpcg_output = parse_hpcg_output(hpcg_section.raw_text)
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

    return meta, extract_metrics_dict(cg_section)


def main():
    jobs = sbm.jobs_list(
        status=[sbm.Status.COMPLETED], from_active=True, from_archived=False
    )
    
    for j in jobs:
        print(f'[runtime] tag: {j.tag} runtime[s]: {j.get_run_time()}')
    print()
    
    meta_df_pairs = []
    for j in jobs:
        try:
            meta_df_pairs.append(parse_job(j))
        except Exception as e:
            print(f'Error: Could not parse tag {j.tag}')
            print('Reason: ' + str(e))
            print()
            
    out_file = OUT_DIR / f"hpcg_{sbm.get_cluster_name()}_data.parquet"

    import_export.describe_pairs_content(meta_df_pairs, verbose=False)
    import_export.write_multiple_to_parquet(meta_df_pairs, out_file)


if __name__ == "__main__":
    main()
