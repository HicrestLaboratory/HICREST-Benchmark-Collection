from collections import defaultdict
from dataclasses import dataclass, asdict
import re
import sys
from pathlib import Path
import sbatchman as sbm
from typing import Any, List, Optional, Dict

sys.path.append(str(Path(__file__).parent.parent / "common" / "energy"))
from ncm_parser import parse_ncm_tot_energy_print #, parse_ncm_energy_log


BENCHMARK_LINE_PATTERN = re.compile(
    r"\s*"
    r"(?P<threads>\d+)\s+"
    r"(?P<duration_sec>\d+)\s+"
    r"(?P<total_entries>\d+)\s+"
    r"(?P<avg_entries>\d+(?:\.\d+)?)\s+"
    r"(?P<std_dev>\d+(?:\.\d+)?)\s+"
    r"(?P<rsd_pct>\d+(?:\.\d+)?)%\s*"
)


@dataclass(slots=True)
class BenchmarkMetrics:
    threads: int
    duration_sec: int
    total_entries: int
    avg_entries: float
    std_dev: float
    rsd_pct: float


def parse_benchmark_line(line: str) -> Optional[BenchmarkMetrics]:
    """Parses a log line into a structured BenchmarkMetrics object."""
    match = BENCHMARK_LINE_PATTERN.match(line)
    if not match:
        return None

    data = match.groupdict()
    return BenchmarkMetrics(
        threads=int(data["threads"]),
        duration_sec=int(data["duration_sec"]),
        total_entries=int(data["total_entries"]),
        avg_entries=float(data["avg_entries"]),
        std_dev=float(data["std_dev"]),
        rsd_pct=float(data["rsd_pct"]),
    )


def parse(job: sbm.Job) -> Optional[Dict[str, Dict | List[Dict]]]:
    """
    Parse locks benchmark stdout into structured metrics.
    """
    if job.category == 'compile' or job.status != sbm.Status.COMPLETED.value:
        return None

    meta = {k:v for k,v in (job.variables or {}).items()}
    meta['system'] = job.cluster_name
    meta['tot_runtime'] = job.get_run_time()
    lock = meta['lock']
    stdout = job.get_stdout()

    if not stdout:
        return None
    
    tot_energy = parse_ncm_tot_energy_print(stdout)
    if tot_energy:
        meta['tot_energy_J'] = tot_energy

    res = defaultdict(list)
    for line in stdout.splitlines():
        metrics = parse_benchmark_line(line.strip())
        if metrics:
            print(metrics)
            metrics = asdict(metrics)
            metrics.update(meta)
            res[lock].append(metrics)
            continue
        
    return res
