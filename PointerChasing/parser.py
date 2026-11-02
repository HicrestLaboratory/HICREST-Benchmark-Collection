import sys
from pathlib import Path
import sbatchman as sbm
import numpy as np
import pandas as pd
from typing import Optional, Dict

sys.path.append(str(Path(__file__).parent.parent / "common"))
from energy.energy_parsers import get_tot_energy_measurement
from utils.utils import remove_useless_job_meta, remove_energy_logs


def extract_numeric_matrix(text: str, expected_cols: int) -> np.ndarray:
    """Parses log text starting after 'Passive mode:', parses numeric rows,
    pads incomplete rows with NaN, and skips non-numeric header/footer lines.
    """
    rows = []
    for line in text.splitlines():
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


def parse_random_chase(text: str) -> pd.DataFrame:
    data = extract_numeric_matrix(text, expected_cols=2)
    return pd.DataFrame(data, columns=['x', 'y']).assign(program='random-chase')


def parse_linear_chase(text: str) -> pd.DataFrame:
    data = extract_numeric_matrix(text, expected_cols=2)
    return pd.DataFrame(data, columns=['x', 'y']).assign(program='linear-chase')


def parse_fused_chase(text: str) -> pd.DataFrame:
    # Expect 9 columns: 1 stride (x) + 8 fuse factors
    raw = extract_numeric_matrix(text, expected_cols=9)
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
    meta = remove_useless_job_meta(meta)
    
    stdout = job.get_stdout()
    
    if not stdout:
        return None

    # print(job.get_stdout())
    tot_energy = get_tot_energy_measurement(job, stdout)
    meta["tot_energy_J"] = tot_energy

    df = matched_parser(remove_energy_logs(stdout))
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
