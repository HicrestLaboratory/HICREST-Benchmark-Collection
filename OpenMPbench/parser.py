import pprint
import re
import sys
from pathlib import Path
from copy import copy
import sbatchman as sbm
from typing import List, Optional, Dict

sys.path.append(str(Path(__file__).parent.parent / "common"))
from energy.energy_parsers import get_tot_energy_measurement
from utils.utils import remove_energy_logs, remove_useless_job_meta

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

_RE_REF_MEDIAN = re.compile(r"^(.+?)\s+median time\s+=\s+([\d.eE+\-]+)\s+microseconds")

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


def clean_kind(block_name: str, is_ref: bool) -> str:
    """Strips array sizes and standardizes baseline to 'REFERENCE'."""
    if is_ref:
        return "REFERENCE"
    # Remove parenthesized size if present, e.g. "PRIVATE (1024)" -> "PRIVATE"
    name = re.sub(r"[\(\[\{].*?[\)\]\}]", "", block_name).strip()
    # Remove trailing numeric array size, e.g. "FIRSTPRIVATE 2187" -> "FIRSTPRIVATE"
    name = re.sub(r"\s+\d+\s*$", "", name).strip()
    return name


def parse_stdout(text, benchmark, cluster, cores, size):
    lines = remove_energy_logs(text).splitlines()

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
    is_ref = False
    expect_stats = False
    cur_stats = {}
    pending = {}

    def flush():
        nonlocal block_name, innerreps, is_ref, cur_stats, pending

        if block_name is None:
            return

        records.append({
            "cluster": cluster,
            "benchmark": benchmark,
            "array_size": size,
            "cores": cores,
            "threads": threads,
            "outerreps": outerreps,
            "innerreps": innerreps,
            "kind": clean_kind(block_name, is_ref),
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
            "speedup": None,
        })

        block_name = None
        innerreps = None
        is_ref = False
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
            is_ref = "reference" in block_name.lower()
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
        if m and not is_ref:
            pending["time"] = float(m.group(2))
            pending["time_ci"] = float(m.group(3))
            continue

        m = _RE_TEST_OVHD.match(line)
        if m and not is_ref:
            pending["overhead"] = float(m.group(2))
            pending["overhead_ci"] = float(m.group(3))
            continue

        m = _RE_TEST_MEDOVHD.match(line)
        if m and not is_ref:
            pending["median_ovhd"] = float(m.group(2))
            flush()
            continue

    flush()

    # Extract baseline baseline time from the REFERENCE entry
    ref_rec = next((r for r in records if r["kind"] == "REFERENCE"), None)
    baseline_time = None
    if ref_rec:
        baseline_time = ref_rec.get("ref_mean_us") or ref_rec.get("mean_us")

    # Calculate speedup (T_ref / T_test) for non-REFERENCE kinds
    for r in records:
        if r["kind"] != "REFERENCE":
            test_time = r.get("time_us") or r.get("mean_us")
            if baseline_time is not None and test_time and test_time > 0:
                r["speedup"] = baseline_time / test_time

    return records


def parse(job: sbm.Job) -> Optional[Dict[str, Dict | List[Dict]]]:
    """
    Parse OpenMPbench benchmark stdout into structured metrics.
    """
    if job.category == "compile" or job.status != sbm.Status.COMPLETED.value:
        return None

    meta = {k: v for k, v in (job.variables or {}).items()}
    meta["cluster"] = job.cluster_name
    meta["tot_runtime"] = job.get_run_time()
    meta = remove_useless_job_meta(meta)
    benchmark = meta["benchmark"]
    size = None
    if benchmark.startswith("arraybench"):
        benchmark, size = benchmark.split("_")
    stdout = job.get_stdout()

    if not stdout:
        return None

    records = parse_stdout(stdout, benchmark, meta["cluster"], meta["ncpus"], size)

    tot_energy = get_tot_energy_measurement(job, stdout)
    if tot_energy:
        meta["tot_energy_J"] = tot_energy

    res = []
    for r in records:
        meta_copy = copy(meta)
        meta_copy.update(r)
        res.append(meta_copy)

    return {benchmark: res}