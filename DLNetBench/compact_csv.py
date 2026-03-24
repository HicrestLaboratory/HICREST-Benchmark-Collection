import csv
import io
from collections import defaultdict

import numpy as np
import pandas as pd


_SCHEMA = {
    ("dp", "main"): {
        "metric_cols": ["runtime", "barrier_time", "comm_time", "throughput"],
    },
    ("fsdp", "main"): {
        "metric_cols": ["runtime", "allgather", "barrier", "throughput"],
    },
    ("fsdp", "allgather"): {
        "metric_cols": ["comm_time", "wait_time"],
    },
    ("fsdp", "reduce_scatter"): {
        "metric_cols": ["reduce_scatter_time"],
    },
    ("dp_pp", "main"): {
        "metric_cols": ["runtime", "dp_comm_time", "throughput"],
    },
    ("dp_pp", "pp_comm"): {
        "metric_cols": ["pp_comm_time"],
    },
    ("dp_pp_tp", "main"): {
        "metric_cols": ["runtime", "dp_comm_time", "throughput"],
    },
    ("dp_pp_tp", "pp_comm"): {
        "metric_cols": ["pp_comm_time"],
    },
    ("dp_pp_tp", "tp_comm"): {
        "metric_cols": ["tp_comm_time"],
    },
    ("dp_pp_ep", "main"): {
        "metric_cols": ["runtime", "dp_ep_comm_time", "dp_comm_time", "throughput"],
    },
    ("dp_pp_ep", "pp_comm"): {
        "metric_cols": ["pp_comm_time"],
    },
    ("dp_pp_ep", "ep_comm"): {
        "metric_cols": ["ep_comm_time"],
    },
}


def _iqr_outliers_np(vals: np.ndarray, ranks: list, fence: float = 3.0, min_iqr: float = 1e-6) -> dict:
    if len(vals) < 1:
        return {}
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = max(q3 - q1, min_iqr)
    mask = (vals < q1 - fence * iqr) | (vals > q3 + fence * iqr)
    return {f"r{r}": float(v) for r, v, m in zip(ranks, vals, mask) if m}


def _outliers_pct(vals: np.ndarray, ranks: list, pct_threshold: float = 25.0) -> dict:
    if len(vals) < 1:
        return {}
    median = np.median(vals)
    if median == 0:
        return {}
    deviation = np.abs(vals - median) / median * 100  # % deviation from median
    mask = deviation > pct_threshold
    return {f"r{r}": float(v) for r, v, m in zip(ranks, vals, mask) if m}


def compact_csv(csv_string: str, strategy_name: str, df_name: str, warn_within_rank: bool = False) -> dict:
    schema = _SCHEMA.get((strategy_name, df_name))
    if schema is None:
        raise ValueError(f"Unknown (strategy, df_name) pair: ({strategy_name!r}, {df_name!r})")

    metric_cols = schema["metric_cols"]

    # parse in one shot
    df = pd.read_csv(io.StringIO(csv_string))

    # step 1: mean per rank across all other dimensions (run_id, microbatch_id, phase, etc.)
    per_rank_std  = df.groupby("rank", sort=False)[metric_cols].std() if warn_within_rank else None
    per_rank_mean = df.groupby("rank", sort=False)[metric_cols].mean()
    per_rank = per_rank_mean

    vals_matrix = per_rank.values
    ranks       = per_rank.index.tolist()

    row = {}
    for i, col in enumerate(metric_cols):
        vals = vals_matrix[:, i]

        if warn_within_rank:
            high_var = {}
            for rank, mean_val, std_val in zip(per_rank_mean.index, per_rank_mean[col], per_rank_std[col]):
                if mean_val and abs(mean_val) > 1e-9:
                    pct = abs(std_val / mean_val) * 100
                    if pct > _VARIANCE_WARN_PCT:
                        high_var[f"r{rank}"] = round(pct, 2)

        outlier_dict = _outliers_pct(vals, ranks)

        if outlier_dict:
            outlier_mask = np.array([r in outlier_dict for r in ranks])
            clean_vals = vals[~outlier_mask]
        else:
            clean_vals = vals

        row[f"{col}_mean"]     = float(clean_vals.mean()) if len(clean_vals) else float(vals.mean())
        row[f"{col}_std"]      = float(clean_vals.std())  if len(clean_vals) else float(vals.std())
        row[f"{col}_n"]        = len(clean_vals)
        row[f"{col}_outliers"] = outlier_dict
        if warn_within_rank:
            row[f"{col}_within_rank_warns"] = high_var

    return row


def serialize_value(v):
    if isinstance(v, dict):
        if not v:
            return ""
        return ";".join(f"{k}:{val:.6f}" for k, val in v.items())
    return str(v)


def compact_all(stdout: str, warn_within_rank: bool = False) -> str:
    from parsers import stdout_to_csv_multi

    csv_dict, strategy_name = stdout_to_csv_multi(stdout)
    # print()
    # print('='*80)
    # # print(strategy_name)
    # print(f'{csv_dict.keys()=}')
    # print(f'{csv_dict['main']}')

    blocks = []
    for df_name, csv_string in csv_dict.items():
        row = compact_csv(csv_string, strategy_name, df_name, warn_within_rank)

        # serialize: one CSV with a single data row
        headers = list(row.keys())
        values = [serialize_value(v) for v in row.values()]
        # for h in headers:
            # if h == 'throughput_mean' or h == 'throughput_outliers':
            #     print(f'{df_name=}  {h=}')
            #     print(values)
            #     print('-'*80)
        blocks.append(f"###{df_name}\n" + ",".join(headers) + "\n" + ",".join(values))

    # print('='*80)
    # print()
    
    return "\n###\n".join(blocks)


if __name__ == "__main__":
    import sys
    import time

    filename = sys.argv[1]
    with open(filename, "r") as f:
        stdout = f.read()
    start_time = time.time()
    compacted = compact_all(stdout)
    end_time = time.time()
    print(f"Compaction took {end_time - start_time:.2f} seconds")
    print(compacted)

