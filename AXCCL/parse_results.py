import re
import sys
from typing import Any, Dict, Tuple
import pandas as pd
import sbatchman as sbm
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "common"))
import import_export
from utils.utils import raise_none

OUT_DIR = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_job_stdout(text: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      raw_df: per-iteration samples
      avg_df: [Average] summary rows

    raw_df columns:
      - transfer_size_B
      - time_s
      - bandwidth_GiB_s
      - iteration

    avg_df columns:
      - transfer_size_B
      - time_s
      - bandwidth_GiB_s
      - error
    """

    raw_rows = []
    avg_rows = []

    pattern = re.compile(
        r"""
        ^(?P<avg>\[Average\])?\s*
        Transfer\ size\s*\(B\):\s*(?P<size>\d+),\s*
        Transfer\ Time\s*\(s\):\s*(?P<time>[0-9.]+),\s*
        Bandwidth\s*\(GiB/s\):\s*(?P<bw>[0-9.]+)
        (?:,\s*Iteration\s*(?P<iter>\d+))?
        (?:,\s*Error:\s*(?P<error>\d+))?
        """,
        re.VERBOSE
    )

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        m = pattern.search(line)
        if not m:
            continue

        row = {
            "transfer_size_B": int(m.group("size")),
            "time_s": float(m.group("time")),
            "bandwidth_GiB_s": float(m.group("bw")),
        }

        if m.group("avg"):
            row["error"] = (
                int(m.group("error"))
                if m.group("error") is not None
                else None
            )
            avg_rows.append(row)
        else:
            row["iteration"] = int(m.group("iter"))
            raw_rows.append(row)

    raw_df = pd.DataFrame(raw_rows)
    avg_df = pd.DataFrame(avg_rows)

    return raw_df, avg_df

def parse_job(j: sbm.Job) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    raw_df, avg_df = parse_job_stdout(raise_none(j.get_stdout(), "stdout"))
    vars = raise_none(j.variables, "job variables")
    meta = {k: vars[k] for k in ["implementation", "peering", "buff_cycle"]}
    meta['tag'] = j.tag
    meta["cluster"] = j.cluster_name
    
    return meta, {
        'raw': raw_df,
        'avg': avg_df,
    }
    
def main():
    jobs = sbm.jobs_list(status=[sbm.Status.COMPLETED], from_active=True, from_archived=False)
    cluster_name = sbm.get_cluster_name()

    meta_df_pairs = [parse_job(j) for j in jobs]
    out_file = OUT_DIR / f"hicrest-axccl_{cluster_name}_data.parquet"
    import_export.describe_pairs_content(meta_df_pairs, verbose=True)
    import_export.write_multiple_to_parquet(meta_df_pairs, out_file)


if __name__ == "__main__":
    main()
