import sys
from pathlib import Path
import sbatchman as sbm
import numpy as np
import pandas as pd
from typing import Optional, Dict

sys.path.append(str(Path(__file__).parent.parent / "common" / "energy"))
from ncm_parser import parse_ncm_energy_log


def parse_random_chase(path):
  data = np.genfromtxt(path, skip_header=2, usecols=(0, 1))
  return pd.DataFrame(data, columns=['x', 'y']).assign(program='random-chase')


def parse_linear_chase(path):
  data = np.genfromtxt(path, skip_header=2, usecols=(0, 1))
  return pd.DataFrame(data, columns=['x', 'y']).assign(program='linear-chase')


def parse_fused_linear_chase(path):
  try:
    raw = np.genfromtxt(path, skip_header=4)
  except ValueError:
    raw = np.genfromtxt(path, skip_header=4, skip_footer=1)

  stride = raw[:, 0]
  dfs = []
  for fuse in range(8):  # fuse factors 1-8
    y = raw[:, fuse + 1]
    df = pd.DataFrame({
      'x': stride,
      'y': y,
      'fuse': fuse + 1,
      'program': 'fused-linear-chase',
    })
    dfs.append(df)
  return pd.concat(dfs, ignore_index=True)


PARSERS = {
  'random-chase': parse_random_chase,
  'linear-chase': parse_linear_chase,
  'fused-linear-chase': parse_fused_linear_chase
  # TODO 'fused-random-chase': parse_fused_random_chase
}

def parse(job: sbm.Job) -> Optional[Dict[str, Dict]]:
  if job.status != sbm.Status.COMPLETED.value:
    return None
  
  matched_parser = None
  chase_type = None
  for prefix in PARSERS.keys():
    if job.tag.startswith(prefix):
      chase_type = prefix
      matched_parser = PARSERS[prefix]
      break

  if matched_parser is None:
    return None

  data = {k:v for k,v in (job.variables or {}).items()}
  data['cluster'] = job.cluster_name
  data['tot_runtime'] = job.get_run_time()
  
  df = matched_parser(job.get_stdout_path())
  print(data)
  print('-'*80)
  print(df.to_dict())
  return { chase_type: df.to_dict() }
    
  # If available, include energy measurements
  # energy_log: Path = job.get_job_base_path() / 'energy.log'
  # if energy_log.exists():
  #     res[f'energy_{job.tag}'] = parse_ncm_energy_log(energy_log)
  #     res['stream']['energy'] = 'with_energy'
