import sys
from pathlib import Path
import sbatchman as sbm
import numpy as np
import pandas as pd
import warnings
from typing import Optional, Dict

sys.path.append(str(Path(__file__).parent.parent / "common" / "energy"))
from energy_parsers import parse_ncm_energy_log


def extract_numeric_matrix(path: Path, expected_cols: int) -> np.ndarray:
    """Reads log file starting after 'Passive mode:', parses numeric rows,

    pads incomplete rows with NaN, and skips non-numeric header/footer lines.
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    # Locate line right after "Passive mode:"
    start_idx = 0
    for idx, line in enumerate(lines):
        if line.startswith("Passive mode:"):
            start_idx = idx + 2
            break

    rows = []
    for line in lines[start_idx:]:
        tokens = line.strip().split()
        if not tokens:
            continue
        try:
            # Parse line tokens into floats
            nums = [float(t) for t in tokens]
            # Pad missing columns with NaN if row is incomplete
            if len(nums) < expected_cols:
                nums += [np.nan] * (expected_cols - len(nums))
            rows.append(nums[:expected_cols])
        except ValueError:
            # Skip non-numeric header lines (e.g. 'fuse', 'stride') or extra trailing log lines
            continue

    if not rows:
        return np.empty((0, expected_cols))

    return np.array(rows)


def parse_random_chase(path):
    data = extract_numeric_matrix(path, expected_cols=2)
    return pd.DataFrame(data, columns=['x', 'y']).assign(program='random-chase')


def parse_linear_chase(path):
    data = extract_numeric_matrix(path, expected_cols=2)
    return pd.DataFrame(data, columns=['x', 'y']).assign(program='linear-chase')


def parse_fused_chase(path):
    # Expect 9 columns: 1 stride (x) + 8 fuse factors
    raw = extract_numeric_matrix(path, expected_cols=9)
    if raw.size == 0:
        return pd.DataFrame(columns=['x', 'y', 'fuse', 'program'])

    stride = raw[:, 0]
    dfs = []
    for fuse in range(8):  # fuse factors 1-8
        y = raw[:, fuse + 1]
        df = pd.DataFrame({
            'x': stride,
            'y': y,
            'fuse': fuse + 1
        })
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


PARSERS = {
    'random-chase': parse_random_chase,
    'linear-chase': parse_linear_chase,
    'fused-linear-chase': parse_fused_chase,
    'fused-random-chase': parse_fused_chase,
}

def parse(job: sbm.Job) -> Optional[Dict[str, Dict]]:
    chase_type = job.category or '_'.split(job.tag)[0]
    if chase_type == 'compile' or job.status not in [sbm.Status.COMPLETED.value, sbm.Status.TIMEOUT.value]:
        print(f"Skipping JOB {job.tag} - {chase_type} - {job.status}")
        return None

    matched_parser = PARSERS.get(chase_type)
    if matched_parser is None:
        print(f"Skipping JOB {job.tag} - no matched parser")
        return None

    meta = {k:v for k,v in (job.variables or {}).items()}
    meta['cluster'] = job.cluster_name
    meta['tot_runtime'] = job.get_run_time()
    for k in ['energy_measure_start', 'energy_measure_stop']:
        del meta[k]

    # print(job.get_stdout())

    df = matched_parser(job.get_stdout_path())
    for k, v in meta.items():
        df[k] = v
    # print(data)
    # print('-'*80)
    # print(df)
    # exit()
    return { chase_type.replace('-', '_'): df.to_dict('records') }

    # If available, include energy measurements
    # energy_log: Path = job.get_job_base_path() / 'energy.log'
    # if energy_log.exists():
    #     res[f'energy_{job.tag}'] = parse_ncm_energy_log(energy_log)
    #     res['stream']['energy'] = 'with_energy'
