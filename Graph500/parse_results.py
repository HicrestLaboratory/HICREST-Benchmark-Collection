import re
import sys
from pathlib import Path
from typing import Any, Dict, Tuple
import pandas as pd
import sbatchman as sbm

sys.path.append(str(Path(__file__).parent.parent / 'common'))
import ccutils.parser.ccutils_parser as ccutils_parser
import import_export
from utils.utils import raise_none, dict_get

OUT_DIR = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NODES_MAP = None


def parse_job(
    j: sbm.Job, run_indices=range(64)
) -> Tuple[Dict[Any, Any], Dict[str, pd.DataFrame]]:
    """
    Returns: list of DataFrames with added jobid/run columns
    """
    stdout = raise_none(j.get_stdout(), "stdout")
    res = ccutils_parser.parse_ccutils_output(stdout)

    # Map rank → node number
    ranks_nodes_map = {}
    ranks_nodes_map_raw = {}
    nodes = dict_get(res, "node_names")
    nodes = raise_none(nodes.get_mpi_print("node_names"), "node_names")

    for r in raise_none(nodes.get_all_ranks(), "nodes.get_all_ranks()"):
        node_str = raise_none(nodes.get_rank_output(r), f"node for rank {r}")
        ranks_nodes_map_raw[r] = node_str
        if j.cluster_name == "leonardo":
            ranks_nodes_map[r] = int(node_str.split(".")[0][4:])
        else:
            ranks_nodes_map[r] = node_str

    details = dict_get(res, "detailed_results")
    packet_bw = dict_get(details.mpi_all_prints, "packet_bandwidth")
    barrier_times = dict_get(details.mpi_all_prints, "barrier_times")
    general = dict_get(res, "general_results").raw_text

    meta = {}
    vars = raise_none(j.variables, "job variables")
    for k in ["nodes", "edgefactor", "scale", "partition"]:
        meta[k] = vars[k]
    meta["buffer_size"] = vars["bin"].split("_")[-1]
    meta["cluster"] = j.cluster_name
    meta["rank_node_map"] = ranks_nodes_map_raw
    teps = -1
    for line in general.strip().splitlines():
        if "harmonic_mean_TEPS" in line:
            line = re.subn(r"\s{2,}", " ", line)[0]
            teps = float(line.split(" ")[-1])
            continue
    meta["teps"] = teps

    out_packets_dfs = []
    out_barrier_dfs = []
    for run_i in run_indices:
        rows = []
        if int(vars["nodes"]) > 1:
            for dest in packet_bw.get_all_ranks():
                rank_output = packet_bw.get_rank_output(dest)
                if not rank_output:
                    continue

                for msg in rank_output.splitlines()[run_i].strip().split(" "):
                    if not msg:
                        continue
                    src, size, t = msg.split(",")
                    rows.append(
                        [
                            int(src),
                            int(dest),
                            int(size),
                            float(t),
                        ]
                    )
                    
            df = pd.DataFrame(rows, columns=["src", "dest", "size", "time"])
            df["run"] = run_i
            # Clean negative times
            neg = df["time"] < 0
            df.loc[neg, "time"] = 0.0
            if not df.empty:
                out_packets_dfs.append(df)
            
            rows = []
            for rank in barrier_times.get_all_ranks():
                rank_output = barrier_times.get_rank_output(rank)
                if not rank_output:
                    continue
                for time in rank_output.strip().split(' '):
                    rows.append(
                        [
                            int(rank),
                            float(time),
                        ]
                    )
                df = pd.DataFrame(rows, columns=["rank", "time"])
                df["run"] = run_i
                if not df.empty:
                    out_barrier_dfs.append(df)


    return meta, {
        "packets": (
            pd.concat(out_packets_dfs, ignore_index=True)
            if len(out_packets_dfs) > 0
            else pd.DataFrame()
        ),
        "barrier": (
            pd.concat(out_barrier_dfs, ignore_index=True)
            if len(out_barrier_dfs) > 0
            else pd.DataFrame()
        )
    }


def main():
    global NODES_MAP
    jobs = sbm.jobs_list(
        status=[sbm.Status.COMPLETED], from_active=True, from_archived=False
    )
    cluster_name = sbm.get_cluster_name()

    meta_df_pairs = [parse_job(j) for j in jobs]
    out_file = OUT_DIR / f"graph500_{cluster_name}_data.parquet"
    import_export.describe_pairs_content(meta_df_pairs, verbose=True)
    import_export.write_multiple_to_parquet(meta_df_pairs, out_file)


if __name__ == "__main__":
    main()
