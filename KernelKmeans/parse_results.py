from pathlib import Path
import sys
import pandas as pd

import sbatchman as sbm

sys.path.append(str(Path(__file__).parent.parent / "common"))
from ccutils.parser.ccutils_parser import parse_ccutils_output


OUT_DIR = Path("results")


def main():
    cluster_name = sbm.get_cluster_name()
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    out_file = OUT_DIR / f"PopcornKernelKmeans_{cluster_name}_data.csv"

    jobs = sbm.jobs_list(
        status=[sbm.Status.COMPLETED],
        from_active=True,
        from_archived=False,
    )

    rows = []

    for j in jobs:
        parsed = parse_ccutils_output(j.get_stdout())

        if parsed is None:
            continue

        cpus = next((v for k, v in j.variables.items() if "cpus" in k.lower()), None)
        compiler = j.variables['compiler']
        implementation = j.variables['implementation']
        board = j.config_name.split('_')[0]
        config = parsed["config"].get_global_json()
        timers = parsed["timers"].get_global_json()

        if config is None or timers is None:
            continue

        # Number of iterations = length of any timer list
        # (assuming all timers have same length)
        n_iters = len(next(iter(timers.values())))

        for i in range(n_iters):
            row = {}

            # Add config (constant per run)
            for k, v in config.items():
                row[k] = v

            # Add job metadata
            row["job_id"] = j.job_id
            row["cluster"] = cluster_name
            row["board"] = board
            row["cpus"] = cpus
            row["compiler"] = compiler
            row["impl"] = implementation
            row["iteration"] = i

            # Add timers (per iteration)
            for timer_name, values in timers.items():
                if i < len(values):
                    row[timer_name] = values[i]
                else:
                    row[timer_name] = None

            rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Define column groups
    meta_cols = ["job_id", "cluster", "cpus", "compiler", "impl"]
    iter_cols = ["iteration"]

    # Config keys (sorted for consistency)
    config_cols = sorted([
        c for c in df.columns
        if c not in meta_cols + iter_cols
        and not isinstance(df[c].iloc[0], (list, dict))
        and c not in ["iteration"]
        and not any(c == t for t in ["argmin_assign", "distances_compute", "init", "score_compute", "total", "v_matrix_update"])
    ])

    # Timer columns (everything else numeric & per-iteration)
    timer_cols = sorted([
        c for c in df.columns
        if c not in meta_cols + iter_cols + config_cols
    ])

    # Final order
    ordered_cols = meta_cols + config_cols + iter_cols + timer_cols

    df = df[ordered_cols]

    # Save CSV
    df.to_csv(out_file, index=False)

    print(f"Saved {len(df)} rows to {out_file}")


if __name__ == "__main__":
    main()