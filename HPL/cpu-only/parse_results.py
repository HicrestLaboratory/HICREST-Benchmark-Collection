from typing import List
from pathlib import Path
import pandas as pd
import sbatchman as sbm


def parse_hpl_outputs(jobs: List[sbm.Job]):
    """
    Parse HPL benchmark jobs into a pandas DataFrame and save as CSV.

    Parameters
    ----------
    jobs : list of Job
        Each job stdout is the full output of one HPL run.
    csv_file : str
        Path where the CSV file will be written.

    Returns
    -------
    df : pandas.DataFrame
        Parsed benchmark results.
    """
    records = []

    header_line = "T/V                N    NB     P     Q               Time                 Gflops"

    for job in jobs:
        output = job.get_stdout()
        if output is None:
            print(f"Warning: job has no stdout\n{job}")
            continue

        lines = output.splitlines()

        # Find the exact header line
        try:
            start_idx = next(
                i for i, line in enumerate(lines) if line.strip() == header_line
            )
        except StopIteration:
            continue  # no valid table in this output

        # Table rows start after dashed line
        for line in lines[start_idx + 2 :]:
            if not line.strip():
                break  # stop at blank line
            if line.startswith("=") or line.startswith("-"):
                break  # stop at another separator
            if line.startswith("HPL_pdgesv()"):
                break  # stop at the end of table marker

            parts = line.split()
            if len(parts) < 7:
                continue

            config: sbm.SlurmConfig = job.get_job_config()
            partition = config.partition
            nodes = job.variables["nodes"]
            cpus = 0
            for v in config.env or []:
                if "OMP_NUM_THREADS" in v:
                    cpus = int(v.split("=")[1])
                    break

            record = {
                "cluster": job.cluster_name,
                "partition": partition,
                "nodes": nodes,
                "cpus_per_node": cpus,
                "tot_cpus": nodes * cpus,
                "T/V": parts[0],
                "N": int(parts[1]),
                "NB": int(parts[2]),
                "P": int(parts[3]),
                "Q": int(parts[4]),
                "Time": float(parts[5]),
                "Gflops": float(parts[6]),
            }
            records.append(record)

    df = pd.DataFrame(
        records,
        columns=[
            "partition",
            "nodes",
            "cpus_per_node",
            "tot_cpus",
            "T/V",
            "N",
            "NB",
            "P",
            "Q",
            "Time",
            "Gflops",
        ],
    )
    return df


if __name__ == "__main__":
    df = parse_hpl_outputs(sbm.jobs_list(status=[sbm.Status.COMPLETED]))
    df["cluster"] = sbm.get_cluster_name()
    df.sort_values(["nodes", "partition"], inplace=True)
    print(df)

    OUT_CSV = Path(f"results/hpl_results_{sbm.get_cluster_name()}.csv")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Results saved to {OUT_CSV.absolute()}")
