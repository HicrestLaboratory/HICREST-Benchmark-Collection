import re
import sys
from pathlib import Path
import pandas as pd
import sbatchman as sbm
from typing import Optional, Dict

sys.path.append(str(Path(__file__).parent.parent / "common" / "energy"))
from ncm_parser import parse_ncm_tot_energy_print #, parse_ncm_energy_log

# ---------------------------
# Regex for NPB output
# ---------------------------

RE_TIME = re.compile(r"Time in seconds\s*=\s*([\d.]+)")
RE_MOPS = re.compile(r"Mop/s total\s*=\s*([\d.]+)")
RE_CLASS = re.compile(r"Class\s*=\s*(\w)")
RE_VERIFY = re.compile(r"Verification\s*=\s*(\w+)")


def parse(job: sbm.Job) -> Optional[Dict[str, Dict]]:
    if not job.tag.startswith('NPB-OMP_') or job.status != sbm.Status.COMPLETED.value:
        return None

    res = {k:v for k,v in (job.variables or {}).items()}
    res['cluster'] = job.cluster_name
    res['tot_runtime'] = job.get_run_time()
    stdout = job.get_stdout()

    if not stdout:
        return None

    time = None
    mops = None
    verify = None

    for line in stdout.splitlines():

        if not res.get('time'):
            m = RE_TIME.search(line)
            if m:
                res['time'] = float(m.group(1))

        if not res.get('mops'):
            m = RE_MOPS.search(line)
            if m:
                res['mops'] = float(m.group(1))

        if not res.get('verify'):
            m = RE_VERIFY.search(line)
            if m:
                res['verify'] = m.group(1)

    tot_energy = parse_ncm_tot_energy_print(stdout)
    if tot_energy:
        res['tot_energy_J'] = tot_energy

    return { 'npb': res }
