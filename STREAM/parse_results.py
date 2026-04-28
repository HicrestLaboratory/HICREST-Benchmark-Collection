from pathlib import Path
import re
import pandas as pd
import sbatchman as sbm

OUT_DIR = Path("results")


STREAM_FUNCS = ["Copy", "Scale", "Add", "Triad"]


def parse_stream_stdout(stdout: str) -> dict:
    """
    Parse STREAM benchmark stdout into structured metrics.
    """
    data = {}

    # Parse benchmark table rows like:
    # Copy:          378231.3     0.002850     0.002839     0.002871
    row_re = re.compile(
        r"^(Copy|Scale|Add|Triad):\s+"
        r"([\d.]+)\s+"
        r"([\d.]+)\s+"
        r"([\d.]+)\s+"
        r"([\d.]+)",
        re.MULTILINE,
    )

    for match in row_re.finditer(stdout):
        func = match.group(1).lower()
        data[f"{func}_rate_mb_s"] = float(match.group(2))
        data[f"{func}_avg_time_s"] = float(match.group(3))
        data[f"{func}_min_time_s"] = float(match.group(4))
        data[f"{func}_max_time_s"] = float(match.group(5))

    # Optional metadata from stdout
    m = re.search(r"This system uses (\d+) bytes per array element", stdout)
    if m:
        data["bytes_per_element"] = int(m.group(1))

    m = re.search(r"Array size = (\d+) \(elements\)", stdout)
    if m:
        data["array_size_elements"] = int(m.group(1))

    m = re.search(r"Memory per array = ([\d.]+) MiB", stdout)
    if m:
        data["memory_per_array_mib"] = float(m.group(1))

    m = re.search(r"Total memory required = ([\d.]+) MiB", stdout)
    if m:
        data["total_memory_required_mib"] = float(m.group(1))

    return data


def main():
    system = sbm.get_cluster_name()

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    out_file = OUT_DIR / f"STREAM_{system}_data.csv"

    jobs = sbm.jobs_list()
    rows = []

    for job in jobs:
        parsed = parse_stream_stdout(job.get_stdout())

        # Skip jobs that don't look like STREAM output
        if not parsed:
            print(f"Skipping non-STREAM job: {job.config_name}")
            continue

        row = {
            "system": system,
            "status": str(job.status),
        }

        # Add sbatch/job variables
        row.update(job.variables)
        del row['array_mem']

        # Add parsed benchmark metrics
        row.update(parsed)

        rows.append(row)

    if not rows:
        print("No STREAM benchmark results found.")
        return

    df = pd.DataFrame(rows)

    # Optional: sort nicely if columns exist
    sort_cols = [c for c in ["partition", "compiler", "ncpus"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    df.to_csv(out_file, index=False)

    print(f"Wrote {len(df)} experiments to {out_file}")


if __name__ == "__main__":
    main()