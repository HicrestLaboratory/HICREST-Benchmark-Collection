import re
import sbatchman as sbm
from typing import Optional, Dict

def parse(job: sbm.Job) -> Optional[Dict[str, Dict]]:
    """
    Parse STREAM benchmark stdout into structured metrics.
    """
    if not job.tag.startswith('stream_') or job.status != sbm.Status.COMPLETED.value:
        return None

    data = {k:v for k,v in (job.variables or {}).items()}
    stdout = job.get_stdout()

    if not stdout:
        return None

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

    return { 'stream': data }
