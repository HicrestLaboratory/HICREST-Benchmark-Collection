from typing import List, Optional
from pathlib import Path
import re
import pandas as pd
import sbatchman as sbm


# ------------------------------------------------------------
# Regex patterns
# ------------------------------------------------------------

_RE_COMPUTING = re.compile(r"^Computing (.+?) time using (\d+) reps")

_RE_STATS = re.compile(
    r"^\s*(\d+)\s+"
    r"([\d.eE+\-]+)\s+"
    r"([\d.eE+\-]+)\s+"
    r"([\d.eE+\-]+)\s+"
    r"([\d.eE+\-]+)\s+"
    r"([\d.eE+\-]+)\s+"
    r"(\d+)"
)

_RE_REF_MEAN = re.compile(
    r"^(.+?)\s+mean time\s+=\s+([\d.eE+\-]+)\s+microseconds\s+\+/-\s+([\d.eE+\-]+)"
)

_RE_REF_MEDIAN = re.compile(
    r"^(.+?)\s+median time\s+=\s+([\d.eE+\-]+)\s+microseconds"
)

_RE_TEST_TIME = re.compile(
    r"^(.+?)\s+time\s+=\s+([\d.eE+\-]+)\s+microseconds\s+\+/-\s+([\d.eE+\-]+)"
)

_RE_TEST_OVHD = re.compile(
    r"^(.+?)\s+overhead\s+=\s+([\d.eE+\-]+)\s+microseconds\s+\+/-\s+([\d.eE+\-]+)"
)

_RE_TEST_MEDOVHD = re.compile(
    r"^(.+?)\s+median_ovrhd\s+=\s+([\d.eE+\-]+)\s+microseconds"
)

_RE_THREADS = re.compile(r"^\s*(\d+)\s+thread\(s\)")
_RE_OUTERREPS = re.compile(r"^\s*(\d+)\s+outer repetitions")

# schedbench_4cpus
# arraybench_243_4cpus
_RE_TAG = re.compile(r"^(\w+)(?:_(\d+))?_(\d+)cpus$")


# ------------------------------------------------------------
# Core stdout parser
# ------------------------------------------------------------

def _parse_stdout(text, benchmark, device, cores, size):

    lines = text.splitlines()

    threads = cores
    outerreps = 20

    for line in lines:

        m = _RE_THREADS.match(line.strip())
        if m:
            threads = int(m.group(1))

        m = _RE_OUTERREPS.match(line.strip())
        if m:
            outerreps = int(m.group(1))

    records = []

    block_name = None
    innerreps = None
    kind = None
    expect_stats = False
    cur_stats = {}
    pending = {}

    def flush():

        nonlocal block_name, innerreps, kind, cur_stats, pending

        if block_name is None:
            return

        records.append({

            "device": device,
            "benchmark": benchmark,
            "array_size": size,

            "cores": cores,
            "threads": threads,

            "outerreps": outerreps,
            "innerreps": innerreps,

            "block": block_name,
            "kind": kind,

            "sample_size": cur_stats.get("sample_size"),
            "mean_us": cur_stats.get("mean"),
            "median_us": cur_stats.get("median"),
            "min_us": cur_stats.get("min"),
            "max_us": cur_stats.get("max"),
            "stddev_us": cur_stats.get("stddev"),
            "outliers": cur_stats.get("outliers"),

            "ref_mean_us": pending.get("ref_mean"),
            "ref_mean_ci": pending.get("ref_mean_ci"),
            "ref_median_us": pending.get("ref_median"),

            "time_us": pending.get("time"),
            "time_ci": pending.get("time_ci"),

            "overhead_us": pending.get("overhead"),
            "overhead_ci": pending.get("overhead_ci"),
            "median_overhead_us": pending.get("median_ovhd"),
        })

        block_name = None
        innerreps = None
        kind = None
        cur_stats = {}
        pending = {}

    for raw in lines:

        line = raw.strip()
        if not line:
            continue

        m = _RE_COMPUTING.match(line)
        if m:

            flush()

            block_name = m.group(1).strip()
            innerreps = int(m.group(2))

            kind = "reference" if "reference time" in block_name.lower() else "test"

            continue

        if line.startswith("Sample_size"):
            expect_stats = True
            continue

        if expect_stats:

            m = _RE_STATS.match(line)

            if m:

                cur_stats = {
                    "sample_size": int(m.group(1)),
                    "mean": float(m.group(2)),
                    "median": float(m.group(3)),
                    "min": float(m.group(4)),
                    "max": float(m.group(5)),
                    "stddev": float(m.group(6)),
                    "outliers": int(m.group(7)),
                }

            expect_stats = False
            continue

        m = _RE_REF_MEAN.match(line)
        if m:

            pending["ref_mean"] = float(m.group(2))
            pending["ref_mean_ci"] = float(m.group(3))

            continue

        m = _RE_REF_MEDIAN.match(line)
        if m:

            pending["ref_median"] = float(m.group(2))

            flush()

            continue

        m = _RE_TEST_TIME.match(line)

        if m and kind == "test":

            pending["time"] = float(m.group(2))
            pending["time_ci"] = float(m.group(3))

            continue

        m = _RE_TEST_OVHD.match(line)

        if m and kind == "test":

            pending["overhead"] = float(m.group(2))
            pending["overhead_ci"] = float(m.group(3))

            continue

        m = _RE_TEST_MEDOVHD.match(line)

        if m and kind == "test":

            pending["median_ovhd"] = float(m.group(2))

            flush()

            continue

    flush()

    return records


# ------------------------------------------------------------
# Main parser
# ------------------------------------------------------------

def parse_openmp_bench_outputs(jobs):

    all_records = []

    for job in jobs:

        m = _RE_TAG.match(job.tag)

        if not m:
            print("[WARN] skipping", job.tag)
            continue

        benchmark = m.group(1)

        size = m.group(2)
        size = int(size) if size else None

        cores = int(m.group(3))

        device = job.config_name.split("_")[0]

        output = job.get_stdout()

        if output is None:
            continue

        records = _parse_stdout(output, benchmark, device, cores, size)

        print("Parsed", len(records), "blocks ←", job.tag)

        all_records.extend(records)

    return pd.DataFrame(all_records)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":

    jobs = sbm.jobs_list(status=[sbm.Status.COMPLETED])

    df = parse_openmp_bench_outputs(jobs)

    if df.empty:
        print("No results found")
        exit()

    df.sort_values(
        ["benchmark", "device", "array_size", "cores", "block"],
        inplace=True
    )

    # --------------------------------------------------------
    # Save ONE CSV PER BENCHMARK
    # --------------------------------------------------------

    output_dir = Path("benchmark_csv")
    output_dir.mkdir(exist_ok=True)

    for benchmark, subdf in df.groupby("benchmark"):

        outfile = output_dir / f"{benchmark}_results.csv"

        subdf.to_csv(outfile, index=False)

        print("Saved:", outfile)

    print("\nAll benchmark CSV files generated.")