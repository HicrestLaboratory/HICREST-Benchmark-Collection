
import csv
import io
import sys
from pathlib import Path
from typing import List, Optional, Dict

import sbatchman as sbm

sys.path.append(str(Path(__file__).parent.parent / "common"))
from energy.energy_parsers import get_tot_energy_measurement
from utils.utils import remove_useless_job_meta, remove_energy_logs


def parse(job: sbm.Job) -> Optional[Dict[str, List[Dict]]]:
    """
    Parse STREAM benchmark CSV stdout into individual raw measurements.
    """
    if job.category == 'compile' or job.status != sbm.Status.COMPLETED.value:
        return None

    metadata = dict(job.variables or {})
    metadata['system'] = job.cluster_name
    metadata['tot_runtime'] = job.get_run_time()
    metadata = remove_useless_job_meta(metadata)

    stdout = job.get_stdout()
    if not stdout:
        return None

    tot_energy = get_tot_energy_measurement(job)
    if tot_energy:
        metadata['tot_energy_J'] = tot_energy

    reader = csv.DictReader(io.StringIO(remove_energy_logs(stdout)))
    results = []

    for row in reader:
        try:
            data = metadata.copy()
            data.update({
                'group': row['group'],
                'config': row['config'],
                'threads': int(row['threads']),
                'kernel': row['kernel'].lower(),
                'stat': row['stat'].lower(),
                'iteration': int(row['iteration']),
                'time_s': float(row['time_s']),
                'GBps': float(row['GBps']),
            })
            results.append(data)

        except (KeyError, ValueError, TypeError):
            continue

    if not results:
        return None

    return {'stream': results}
