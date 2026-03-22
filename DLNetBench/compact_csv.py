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


def _iqr_outliers_np(vals: np.ndarray, ranks: list) -> dict:
    if len(vals) < 4:
        return {}
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    mask = (vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)
    return {f"r{r}": float(v) for r, v, m in zip(ranks, vals, mask) if m}


def compact_csv(csv_string: str, strategy_name: str, df_name: str) -> dict:
    schema = _SCHEMA.get((strategy_name, df_name))
    if schema is None:
        raise ValueError(f"Unknown (strategy, df_name) pair: ({strategy_name!r}, {df_name!r})")

    metric_cols = schema["metric_cols"]

    # parse in one shot
    df = pd.read_csv(io.StringIO(csv_string))

    # step 1: mean per rank across all other dimensions (run_id, microbatch_id, phase, etc.)
    per_rank = df.groupby("rank", sort=False)[metric_cols].mean()

    # step 2: aggregate across ranks → single row
    vals_matrix = per_rank.values                    # shape (n_ranks, n_metrics)
    ranks       = per_rank.index.tolist()

    row = {}
    for i, col in enumerate(metric_cols):
        vals = vals_matrix[:, i]
        row[f"{col}_mean"]     = float(vals.mean())
        row[f"{col}_std"]      = float(vals.std())
        row[f"{col}_outliers"] = _iqr_outliers_np(vals, ranks)

    return row


def compact_all(stdout: str) -> str:
    from parsers import stdout_to_csv_multi

    csv_dict, strategy_name = stdout_to_csv_multi(stdout)

    blocks = []
    for df_name, csv_string in csv_dict.items():
        row = compact_csv(csv_string, strategy_name, df_name)

        # serialize: one CSV with a single data row
        headers = list(row.keys())
        values  = [str(v) for v in row.values()]
        blocks.append(f"###{df_name}\n" + ",".join(headers) + "\n" + ",".join(values))

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

