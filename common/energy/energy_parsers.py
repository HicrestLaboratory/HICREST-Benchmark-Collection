import csv
import io
import os
import re
from pathlib import Path
from typing import List, Optional, Union
import sbatchman as sbm

# ... NCM_LOG_RE and parse_ncm_energy_log unchanged ...

def _parse_ncm_tot_energy_print(stdout: str) -> Optional[float]:
    # Example target line: "STABLE ENERGY MEASUREMENT: 11469 0.0 0xff 35.3dC 12.128V 0.1989A 47.316J"
    for line in stdout.splitlines():
        if line.startswith('STABLE ENERGY MEASUREMENT'):
            return float(line.split(': ')[1].split(' ')[-1][:-1])
    return None


def _parse_mc_tot_energy_print(stdout: str) -> Optional[float]:
    """
    Parse the table between 'START ENERGY MEASUREMENTS' and 'END ENERGY MEASUREMENTS'
    and return the sum of the `energy` column (J) over all BMCs.
    """
    in_block = False
    header: Optional[List[str]] = None
    total = 0.0
    found = False

    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith('START ENERGY MEASUREMENTS'):
            in_block = True
            header = None
            continue
        if line.startswith('END ENERGY MEASUREMENTS'):
            break
        if not in_block or not line:
            continue

        tokens = line.split()
        if header is None:
            header = tokens
            if 'energy' not in header:
                return None
            continue

        # Data rows start with a dataframe index column, so align from the right
        offset = len(tokens) - len(header)
        idx = header.index('energy') + offset
        if offset < 0 or idx >= len(tokens):
            continue
        try:
            total += float(tokens[idx])
            found = True
        except ValueError:
            continue

    return total if found else None


def _parse_mn5_tot_energy(base_path: Path) -> Optional[float]:
    """
    Sum node energy (J) from the EAR metrics CSV file(s) in the job directory.
    Energy per row = DC_NODE_POWER_W * TIME_SEC; rows (nodes/steps) are summed.
    """
    files = sorted(base_path.glob('metrics_*.csv'))
    if not files:
        return None

    total = 0.0
    found = False
    for file in files:
        with open(file, newline='') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                try:
                    power = float(row['DC_NODE_POWER_W'])
                    time_s = float(row['TIME_SEC'])
                except (KeyError, TypeError, ValueError):
                    continue
                total += power * time_s
                found = True

    return total if found else None


def _parse_isarco_tot_energy_print(file_path: Union[str, os.PathLike]) -> Optional[float]:
    """
    Parses an AMDuProfCLI power profile CSV/report file and returns the total energy consumed in Joules.
    Total Energy (J) = Sum of package power across all sockets (W) * Sampling Interval (s).
    """
    path = Path(file_path)
    if not path.is_file():
        return None

    interval_sec: Optional[float] = None
    in_records = False
    package_col_indices: list[int] = []
    total_energy_joules = 0.0
    valid_records = 0

    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue

                first_cell = row[0].strip()

                # 1. Extract sampling interval before reaching PROFILE RECORDS
                if not in_records:
                    if first_cell.startswith("Sampling Interval:") and len(row) > 1:
                        match = re.search(r"(\d+(?:\.\d+)?)\s*milli-seconds", row[1])
                        if match:
                            interval_sec = float(match.group(1)) / 1000.0
                    elif first_cell == "PROFILE RECORDS":
                        if interval_sec is None:
                            return None
                        in_records = True
                    continue

                # 2. Parse the header row immediately following PROFILE RECORDS
                if not package_col_indices:
                    header = [col.strip() for col in row]
                    package_col_indices = [
                        idx for idx, col in enumerate(header)
                        if re.fullmatch(r"socket\d+-package-power", col)
                    ]
                    if not package_col_indices:
                        return None
                    continue

                # 3. Parse numeric record rows
                if not first_cell.isdigit():
                    if valid_records > 0:
                        break
                    continue

                try:
                    step_power_watts = sum(float(row[idx].strip()) for idx in package_col_indices)
                    total_energy_joules += step_power_watts * interval_sec
                    valid_records += 1
                except (ValueError, IndexError):
                    continue
    except OSError:
        return None

    return total_energy_joules if valid_records > 0 else None


def get_tot_energy_measurement(job: sbm.Job, stdout: Optional[str] = None) -> Optional[float]:
    """Return the total energy consumed by the job, in joules."""
    if not stdout:
        stdout = job.get_stdout()
    if not stdout:
        return None
    
    if job.cluster_name.startswith(('dalek', 'bsc-hca')):
        return _parse_ncm_tot_energy_print(stdout)
    elif job.cluster_name.startswith('mc'):
        return _parse_mc_tot_energy_print(stdout)
    elif job.cluster_name.startswith('MN5'):
        return _parse_mn5_tot_energy(job.get_job_base_path())
    elif job.cluster_name.startswith('isarco'):
        return _parse_isarco_tot_energy_print(job.get_job_base_path() / job.tag / 'timechart.csv')
    else:
        print(f'WARNING: no support for energy measurement parsing for cluster "{job.cluster_name}"')
        return None