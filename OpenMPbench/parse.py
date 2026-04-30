"""
parse_openmp_bench.py
=====================
Retrieves raw stdout files produced by the EPCC OpenMP MicroBenchmark Suite
(v4.0) from SbatchMan jobs and converts them into a single tidy CSV that the
plot script can consume.

Expected job tag format:  {benchmark}_{hw}_{ncpus}cpus
  e.g.  syncbench_pioneer_8cpus
        schedbench_bananaf3_4cpus
        arraybench_arriesgado_2cpus
        taskbench_pioneer_16cpus

Benchmark stdout structure (produced by common.c)
--------------------------------------------------
Running OpenMP benchmark version 4.0
    <N> thread(s)
    <outerreps> outer repetitions
    <targettesttime> test time (microseconds)
    <delaylength> delay length (iterations)
    <delaytime> delay time (microseconds)

--------------------------------------------------------
Computing <block_name> time using <innerreps> reps

Sample_size       Mean       Median     Min        Max        StdDev     Outliers
 20               1.234567   1.200000   1.100000   1.500000   0.080000   0

<block_name> mean time    = X microseconds +/- Y      <- reference block only
<block_name> median time  = X microseconds            <- reference block only

  -- OR (for test blocks) --

<test_name> time         = X microseconds +/- Y
<test_name> overhead     = X microseconds +/- Y
<test_name> median_ovrhd = X microseconds

Output CSV columns
------------------
hw, benchmark, cores, block, kind,
mean_us, median_us, min_us, max_us, stddev_us, outliers,
time_us, time_ci,
overhead_us, overhead_ci,
median_overhead_us,
innerreps, outerreps, threads
"""

import re
import sys
import argparse
from pathlib import Path
import pandas as pd
import sbatchman as sbm

# ---------------------------------------------------------------------------
# Regex patterns  (match against individual lines)
# ---------------------------------------------------------------------------

# Header printed by init()
_RE_THREADS   = re.compile(r"^\s*(\d+)\s+thread\(s\)")
_RE_OUTERREPS = re.compile(r"^\s*(\d+)\s+outer repetitions")

# Section divider printed by printheader()
_RE_COMPUTING = re.compile(
    r"^Computing (.+?) time using (\d+) reps"
)

# Stats table printed by stats()
_RE_STATS_HDR = re.compile(r"^Sample_size")
_RE_STATS_ROW = re.compile(
    r"^\s*(\d+)\s+"                  # sample_size
    r"([\d.eE+\-]+)\s+"              # mean
    r"([\d.eE+\-]+)\s+"              # median
    r"([\d.eE+\-]+)\s+"              # min
    r"([\d.eE+\-]+)\s+"              # max
    r"([\d.eE+\-]+)\s+"              # stddev
    r"(\d+)"                          # outliers
)

# Footer lines printed by printreferencefooter()
_RE_REF_TIME   = re.compile(
    r"^(.+?)\s+mean time\s+=\s+([\d.eE+\-]+)\s+microseconds\s+\+/-\s+([\d.eE+\-]+)"
)
_RE_REF_MEDIAN = re.compile(
    r"^(.+?)\s+median time\s+=\s+([\d.eE+\-]+)\s+microseconds"
)

# Footer lines printed by printfooter()
_RE_TEST_TIME    = re.compile(
    r"^(.+?)\s+time\s+=\s+([\d.eE+\-]+)\s+microseconds\s+\+/-\s+([\d.eE+\-]+)"
)
_RE_TEST_OVHD    = re.compile(
    r"^(.+?)\s+overhead\s+=\s+([\d.eE+\-]+)\s+microseconds\s+\+/-\s+([\d.eE+\-]+)"
)
_RE_TEST_MEDOVHD = re.compile(
    r"^(.+?)\s+median_ovrhd\s+=\s+([\d.eE+\-]+)\s+microseconds"
)

# Job tag  e.g.  syncbench_pioneer_8cpus
_RE_TAG = re.compile(r"^(\w+)_(\w+)_(\d+)cpus$")


# ---------------------------------------------------------------------------
# Per-file parser
# ---------------------------------------------------------------------------

def parse_stdout(text: str, benchmark: str, hw: str, cores: int) -> list[dict]:
    """
    Parse the full stdout of one benchmark run.
    Returns a list of record dicts (one per measured block).
    """
    lines = text.splitlines()

    threads   = cores   # fallback; overwritten from header if found
    outerreps = 20      # default

    records: list[dict] = []

    # State for the current block being accumulated
    block_name  = None
    innerreps   = None
    stats_next  = False   # True when next non-empty line is the stats data row
    cur_stats   = {}      # filled from the stats row
    kind        = None    # 'reference' or 'test'

    # Pending footer values (accumulated across 1-3 footer lines)
    pending: dict = {}

    def flush_pending():
        """Commit a completed record from pending + cur_stats."""
        nonlocal pending, cur_stats, block_name, innerreps, kind
        if not block_name:
            return
        rec = {
            "hw":         hw,
            "benchmark":  benchmark,
            "cores":      cores,
            "threads":    threads,
            "outerreps":  outerreps,
            "innerreps":  innerreps,
            "block":      block_name,
            "kind":       kind,
            # stats table
            "mean_us":    cur_stats.get("mean"),
            "median_us":  cur_stats.get("median"),
            "min_us":     cur_stats.get("min"),
            "max_us":     cur_stats.get("max"),
            "stddev_us":  cur_stats.get("stddev"),
            "outliers":   cur_stats.get("outliers"),
            # footer
            "time_us":             pending.get("time"),
            "time_ci":             pending.get("time_ci"),
            "overhead_us":         pending.get("overhead"),
            "overhead_ci":         pending.get("overhead_ci"),
            "median_overhead_us":  pending.get("median_ovhd"),
        }
        records.append(rec)
        pending    = {}
        cur_stats  = {}
        block_name = None
        innerreps  = None
        kind       = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        # ── Global header ──────────────────────────────────────────────
        m = _RE_THREADS.match(line)
        if m:
            threads = int(m.group(1))
            continue

        m = _RE_OUTERREPS.match(line)
        if m:
            outerreps = int(m.group(1))
            continue

        # ── New computing block starts ─────────────────────────────────
        m = _RE_COMPUTING.match(line)
        if m:
            # flush whatever was being built before
            flush_pending()
            block_name = m.group(1).strip()
            innerreps  = int(m.group(2))
            stats_next = False
            # Decide kind: reference blocks contain "reference time"
            kind = "reference" if "reference time" in block_name.lower() else "test"
            continue

        # ── Stats table header ─────────────────────────────────────────
        if _RE_STATS_HDR.match(line):
            stats_next = True
            continue

        # ── Stats table data row ───────────────────────────────────────
        if stats_next:
            m = _RE_STATS_ROW.match(line)
            if m:
                cur_stats = {
                    "mean":     float(m.group(2)),
                    "median":   float(m.group(3)),
                    "min":      float(m.group(4)),
                    "max":      float(m.group(5)),
                    "stddev":   float(m.group(6)),
                    "outliers": int(m.group(7)),
                }
                stats_next = False
            continue

        # ── Reference footer ───────────────────────────────────────────
        m = _RE_REF_TIME.match(line)
        if m:
            pending["time"]    = float(m.group(2))
            pending["time_ci"] = float(m.group(3))
            continue

        m = _RE_REF_MEDIAN.match(line)
        if m:
            pending["median_ovhd"] = float(m.group(2))
            # Reference block ends here
            flush_pending()
            continue

        # ── Test footer ────────────────────────────────────────────────
        m = _RE_TEST_TIME.match(line)
        if m and kind == "test":
            pending["time"]    = float(m.group(2))
            pending["time_ci"] = float(m.group(3))
            continue

        m = _RE_TEST_OVHD.match(line)
        if m and kind == "test":
            pending["overhead"]    = float(m.group(2))
            pending["overhead_ci"] = float(m.group(3))
            continue

        m = _RE_TEST_MEDOVHD.match(line)
        if m and kind == "test":
            pending["median_ovhd"] = float(m.group(2))
            # Test block ends here
            flush_pending()
            continue

    # Flush anything remaining at EOF
    flush_pending()
    return records


# ---------------------------------------------------------------------------
# SbatchMan retrieval
# ---------------------------------------------------------------------------

def load_from_sbatchman() -> pd.DataFrame:
    """Pull all completed jobs and parse their stdout."""
    jobs = sbm.jobs_list(
        from_active=True,
        from_archived=False,
        status=[sbm.Status.COMPLETED],
    )
    all_records = []
    for job in jobs:
        m = _RE_TAG.match(job.tag)
        if not m:
            print(f"[WARN] Skipping job with unrecognised tag: {job.tag!r}")
            continue
        benchmark, hw, cores = m.group(1), m.group(2), int(m.group(3))
        try:
            text = Path(job.get_stdout_path()).read_text(errors="replace")
        except Exception as e:
            print(f"[WARN] Could not read stdout for {job.tag}: {e}")
            continue
        records = parse_stdout(text, benchmark, hw, cores)
        if not records:
            print(f"[WARN] No records parsed from {job.tag}")
        all_records.extend(records)
        print(f"  Parsed {len(records):3d} blocks  ←  {job.tag}")

    return pd.DataFrame(all_records)


# ---------------------------------------------------------------------------
# Local file retrieval  (for offline / testing use)
# ---------------------------------------------------------------------------

def load_from_files(paths: list[Path]) -> pd.DataFrame:
    """
    Parse a list of raw stdout files.
    Each file must be named  {benchmark}_{hw}_{ncpus}cpus.txt
      e.g.  syncbench_pioneer_8cpus.txt
    """
    all_records = []
    for p in paths:
        m = _RE_TAG.match(p.stem)
        if not m:
            print(f"[WARN] Skipping file with unrecognised name: {p.name!r}")
            continue
        benchmark, hw, cores = m.group(1), m.group(2), int(m.group(3))
        text    = p.read_text(errors="replace")
        records = parse_stdout(text, benchmark, hw, cores)
        all_records.extend(records)
        print(f"  Parsed {len(records):3d} blocks  ←  {p.name}")
    return pd.DataFrame(all_records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Parse EPCC OpenMP benchmark stdout into a tidy CSV.",
    )
    p.add_argument(
        "--files", nargs="+", type=Path, metavar="FILE",
        help="Raw stdout text files to parse (named <bench>_<hw>_<N>cpus.txt). "
             "If omitted, data is fetched from SbatchMan.",
    )
    p.add_argument(
        "--output", "-o", type=Path,
        default=Path("openmp_bench_results.csv"),
        help="Destination CSV path (default: openmp_bench_results.csv).",
    )
    p.add_argument(
        "--tests-only", action="store_true",
        help="Drop reference-block rows from the output CSV.",
    )
    p.add_argument(
        "--overhead-only", action="store_true",
        help="Keep only rows that have a non-null overhead_us value.",
    )
    return p


def main():
    args = build_parser().parse_args()

    if args.files:
        df = load_from_files(args.files)
    else:
        df = load_from_sbatchman()

    if df.empty:
        print("[ERROR] No data was parsed. Check your files / job tags.")
        sys.exit(1)

    # Optional filters
    if args.tests_only:
        df = df[df["kind"] == "test"]
    if args.overhead_only:
        df = df[df["overhead_us"].notna()]

    # Derived convenience columns
    df["benchmark_test"] = df["benchmark"] + "/" + df["block"]

    # Sort for readability
    df.sort_values(["hw", "benchmark", "cores", "block"], inplace=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\n✅  Saved {len(df)} rows → {args.output}")

    # Quick summary
    print("\nRow counts per (hw, benchmark):")
    print(df.groupby(["hw", "benchmark"])["block"].count().to_string())


if __name__ == "__main__":
    main()