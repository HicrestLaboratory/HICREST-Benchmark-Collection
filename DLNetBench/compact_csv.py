import csv
import io
from collections import defaultdict

from parsers import stdout_to_csv_multi


# ---------------------------------------------------------------------------
# Schema: for each (strategy, df_name) define:
#   - group_keys_per_rank  : columns that form the group when compacting PER rank
#                            (rank is kept as a key, run_id is consumed)
#   - group_keys_across    : columns that form the group when compacting ACROSS ranks
#                            (both rank and run_id are consumed)
#   - metric_cols          : columns to aggregate (mean/std/outliers)
# ---------------------------------------------------------------------------

_SCHEMA = {
    ("dp", "main"): {
        "group_keys_per_rank":  ["rank"],
        "group_keys_across":    [],
        "metric_cols":          ["runtime", "barrier_time", "comm_time", "throughput"],
    },
    ("fsdp", "main"): {
        "group_keys_per_rank":  ["rank"],
        "group_keys_across":    [],
        "metric_cols":          ["runtime", "allgather", "barrier", "throughput"],
    },
    ("fsdp", "allgather"): {
        "group_keys_per_rank":  ["rank", "unit_id", "phase"],
        "group_keys_across":    ["unit_id", "phase"],
        "metric_cols":          ["comm_time", "wait_time"],
    },
    ("fsdp", "reduce_scatter"): {
        "group_keys_per_rank":  ["rank", "unit_id"],
        "group_keys_across":    ["unit_id"],
        "metric_cols":          ["reduce_scatter_time"],
    },
    ("dp_pp", "main"): {
        "group_keys_per_rank":  ["rank"],
        "group_keys_across":    [],
        "metric_cols":          ["runtime", "dp_comm_time", "throughput"],
    },
    ("dp_pp", "pp_comm"): {
        "group_keys_per_rank":  ["rank", "microbatch_id"],
        "group_keys_across":    ["microbatch_id"],
        "metric_cols":          ["pp_comm_time"],
    },
    ("dp_pp_tp", "main"): {
        "group_keys_per_rank":  ["rank"],
        "group_keys_across":    [],
        "metric_cols":          ["runtime", "dp_comm_time", "throughput"],
    },
    ("dp_pp_tp", "pp_comm"): {
        "group_keys_per_rank":  ["rank", "microbatch_id", "phase"],
        "group_keys_across":    ["microbatch_id", "phase"],
        "metric_cols":          ["pp_comm_time"],
    },
    ("dp_pp_tp", "tp_comm"): {
        "group_keys_per_rank":  ["rank", "microbatch_id", "phase"],
        "group_keys_across":    ["microbatch_id", "phase"],
        "metric_cols":          ["tp_comm_time"],
    },
    ("dp_pp_ep", "main"): {
        "group_keys_per_rank":  ["rank"],
        "group_keys_across":    [],
        "metric_cols":          ["runtime", "dp_ep_comm_time", "dp_comm_time", "throughput"],
    },
    ("dp_pp_ep", "pp_comm"): {
        "group_keys_per_rank":  ["rank", "microbatch_id", "phase"],
        "group_keys_across":    ["microbatch_id", "phase"],
        "metric_cols":          ["pp_comm_time"],
    },
    ("dp_pp_ep", "ep_comm"): {
        "group_keys_per_rank":  ["rank", "microbatch_id", "phase"],
        "group_keys_across":    ["microbatch_id", "phase"],
        "metric_cols":          ["ep_comm_time"],
    },
}

# ---------------------------------------------------------------------------
# IQR outlier detection
# ---------------------------------------------------------------------------

def _iqr_outliers(values, ids):
    """
    Given parallel lists `values` (float) and `ids` (any hashable),
    return a dict {id: value} for every point outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    With fewer than 4 points the fence degenerates, so we return {} in that case.
    """
    if len(values) < 4:
        return {}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    q1 = sorted_vals[n // 4]
    q3 = sorted_vals[(3 * n) // 4]
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    return {id_: v for id_, v in zip(ids, values) if v < lo or v > hi}


# ---------------------------------------------------------------------------
# Core aggregation helper
# ---------------------------------------------------------------------------

def _aggregate(groups, group_keys, metric_cols, outlier_id_col):
    """
    groups : dict  key_tuple -> {metric_col -> [(id_val, float_val), ...]}
    Returns a list of row-dicts ready to be written as CSV.

    outlier_id_col : the column name used as the outlier identifier
                     (e.g. "run_id" for per-rank, "(run_id,rank)" for across)
    """
    rows = []
    for key_tuple, metric_data in sorted(groups.items()):
        row = dict(zip(group_keys, key_tuple if isinstance(key_tuple, tuple) else (key_tuple,)))
        for col in metric_cols:
            pairs = metric_data[col]          # [(id, value), ...]
            ids   = [p[0] for p in pairs]
            vals  = [p[1] for p in pairs]
            n     = len(vals)
            mean  = sum(vals) / n if n else float("nan")
            std   = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5 if n > 1 else 0.0
            outliers = _iqr_outliers(vals, ids)   # {id: value}
            row[f"{col}_mean"]     = mean
            row[f"{col}_std"]      = std
            row[f"{col}_outliers"] = outliers      # dict kept as Python object
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compact_csv(csv_string, strategy_name, df_name):
    """
    Compact a CSV produced by stdout_to_csv_multi in two ways:

      1. per_rank  – group by (rank, <extra keys>), aggregate metrics across run_ids.
                     Outlier identifiers are run_ids.

      2. across    – group by (<extra keys> without rank), aggregate metrics
                     across all run_ids × ranks.
                     Outlier identifiers are (run_id, rank) tuples.

    Returns
    -------
    dict with keys "per_rank" and "across", each being a list of row-dicts.
    The *_outliers fields are plain Python dicts  {identifier: value}  so the
    caller can format them however they like (JSON, pretty-print, …).

    Example (dp_pp / main)
    ----------------------
    Input rows (subset):
        run_id  rank  runtime  dp_comm_time  throughput
        0       0     1.10     0.20          100
        1       0     1.12     0.21          99
        2       0     1.80     0.22          95        <- outlier
        0       1     1.05     0.19          102

    per_rank output row for rank=0:
        {
          "rank": "0",
          "runtime_mean": 1.34,   "runtime_std": 0.32,
          "runtime_outliers": {2: 1.80},
          "dp_comm_time_mean": …, …,
          "throughput_mean": …,   …,
        }

    across output row (single row, no group keys):
        {
          "runtime_mean": …,
          "runtime_outliers": {(2, "0"): 1.80, …},
          …
        }
    """
    schema = _SCHEMA.get((strategy_name, df_name))
    if schema is None:
        raise ValueError(f"Unknown (strategy, df_name) pair: ({strategy_name!r}, {df_name!r})")

    group_keys_per_rank = schema["group_keys_per_rank"]   # includes "rank"
    group_keys_across   = schema["group_keys_across"]     # excludes "rank"
    metric_cols         = schema["metric_cols"]

    # ── parse CSV ───────────────────────────────────────────────────────────
    reader = csv.DictReader(io.StringIO(csv_string))
    rows   = list(reader)

    # ── build per-rank groups ────────────────────────────────────────────────
    # key = tuple of group_keys_per_rank values
    # value = {metric_col: [(run_id, float_val), ...]}
    per_rank_groups = defaultdict(lambda: defaultdict(list))
    across_groups   = defaultdict(lambda: defaultdict(list))

    for row in rows:
        run_id = row["run_id"]
        rank   = row["rank"]

        # --- per-rank key (includes rank, excludes run_id) ---
        pr_key = tuple(row[k] for k in group_keys_per_rank)
        if len(pr_key) == 1:
            pr_key = pr_key[0]          # keep scalar for single-key groups

        # --- across key (excludes rank and run_id) ---
        ac_key = tuple(row[k] for k in group_keys_across) if group_keys_across else ()

        for col in metric_cols:
            val = float(row[col])
            per_rank_groups[pr_key][col].append((run_id, val))
            across_groups[ac_key][col].append(((run_id, rank), val))

    # ── aggregate ────────────────────────────────────────────────────────────
    per_rank_rows = _aggregate(per_rank_groups, group_keys_per_rank, metric_cols, "run_id")
    across_rows   = _aggregate(across_groups,   group_keys_across,   metric_cols, "(run_id,rank)")

    return {"per_rank": per_rank_rows, "across": across_rows}


def compact_all(stdout: str) -> str:
    '''
    Convenience wrapper: compact every df in the dict returned by stdout_to_csv_multi.

    Returns
    -------
    str  Compact CSV string with all aggregated results.
    '''
    csv_dict, strategy_name = stdout_to_csv_multi(stdout)
    
    blocks = []
    for df_name, csv_string in csv_dict.items():
        across_rows = compact_csv(csv_string, strategy_name, df_name)["across"]
        
        # rebuild as CSV string
        if not across_rows:
            continue
        headers = list(across_rows[0].keys())
        lines = [",".join(headers)]
        for row in across_rows:
            lines.append(",".join(str(v) for v in row.values()))
        
        blocks.append(f"###{df_name}\n" + "\n".join(lines))
    
    return "\n###\n".join(blocks)


