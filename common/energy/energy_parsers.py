import csv
import re
from pathlib import Path
from typing import List, Optional
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


def get_tot_energy_measurement(job: sbm.Job) -> Optional[float]:
    """Return the total energy consumed by the job, in joules."""
    if job.cluster_name.startswith(('dalek', 'bsc-hca')):
        return _parse_ncm_tot_energy_print(job.get_stdout())
    elif job.cluster_name.startswith('mc'):
        return _parse_mc_tot_energy_print(job.get_stdout())
    elif job.cluster_name.startswith('MN5'):
        return _parse_mn5_tot_energy(job.get_job_base_path())
    else:
        print(f'WARNING: no support for energy measurement parsing for cluster "{job.cluster_name}"')
        return None