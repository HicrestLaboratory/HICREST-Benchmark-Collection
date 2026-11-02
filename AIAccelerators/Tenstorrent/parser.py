"""One parser shared by SbatchMan visualization and the CSV collector."""
import json
import math
import re
from itertools import product
from pathlib import Path
import yaml

RESULT_RE = re.compile(r'^HICREST_RESULT\s+(\{.*\})\s*$', re.MULTILINE)
BASE = Path(__file__).resolve().parent
TAG_RE = re.compile(
    r'^tt_(?:matrix|energy)_(square|short_wide|tall_narrow)_(gemm|gemv|gelu|softmax)_'
    r'(\d+)_(\d+)_(\d+)_(bfp8|bf16|fp32)_(lofi|hifi2|hifi4|none)_(interleaved|sharded)$'
)


def expanded_jobs(filename="jobs.yaml"):
    """Expand the plain list variables used by jobs.yaml, as SbatchMan does."""
    schedule = yaml.safe_load((BASE / filename).read_text())
    for job in schedule['jobs']:
        for entry in job['config_jobs']:
            variables = {**schedule.get('variables', {}),
                         **job.get('variables', {}), **entry.get('variables', {})}
            used = set(re.findall(r'\{(\w+)\}', entry['tag'] + ' ' + entry['command']))
            names = sorted(used)
            for values in product(*(variables[name] for name in names)):
                selected = dict(zip(names, values))
                yield {'tag': entry['tag'].format(**selected),
                       'command': entry['command'].format(**selected)}


def scheduled_cases(filename="jobs.yaml"):
    """Read the campaign expanded from jobs.yaml's shared variables."""
    result = {}
    for entry in expanded_jobs(filename):
        tag = entry['tag']
        match = TAG_RE.fullmatch(tag)
        if match is None or tag in result:
            raise ValueError(f'Invalid or duplicate matrix job tag: {tag}')
        family, operation, m, n, k, dtype, fidelity, layout = match.groups()
        result[tag] = dict(shape_family=family, operation=operation,
                       M=int(m), N=int(n), K=int(k), dtype=dtype,
                       math_fidelity=fidelity, memory_layout=layout)
    return result


CASE_MAP = scheduled_cases()
ENERGY_CASE_MAP = scheduled_cases("jobs_energy.yaml")
ALL_CASES = {**CASE_MAP, **ENERGY_CASE_MAP}


def parse_stdout(stdout, tag, metadata=None):
    if tag not in ALL_CASES:
        return None
    matches = RESULT_RE.findall(stdout or '')
    if len(matches) != 1:
        raise ValueError(f'{tag}: expected one HICREST_RESULT, found {len(matches)}')
    row = dict(metadata or {})
    row.update(json.loads(matches[0]))
    expected = ALL_CASES[tag]
    for key in ('M', 'N', 'K', 'dtype', 'memory_layout'):
        value = row.get(key)
        agrees = value is None or (float(value) == expected[key] if key in ('M', 'N', 'K') else str(value) == str(expected[key]))
        if not agrees:
            if key == 'M' and expected['operation'] == 'gemv' and int(row[key]) == 32:
                row['reported_M'] = row[key]
            else:
                raise ValueError(f'{tag}: result {key} disagrees with scheduled shape')
    if 'operation' in row:
        row['kernel_operation'] = row['operation']
    row.update(expected)
    us = float(row['device_time_us'])
    if not math.isfinite(us) or us <= 0:
        raise ValueError(f'{tag}: invalid device time')
    matrix = row['operation'] in ('gemm', 'gemv')
    # Logical work only: GEMV remains M=1 regardless of tile padding.
    row['throughput'] = (2*row['M']*row['N']*row['K'] if matrix else row['M']*row['N']) / us / 1000
    row['throughput_unit'] = 'GFLOP/s' if matrix else 'Gelem/s'
    row['tag'] = tag
    return row


def parse(job, stdout=None):
    if job.tag not in ALL_CASES:
        return None
    if str(job.status).lower() not in ('completed', 'status.completed'):
        return None
    row = parse_stdout(job.get_stdout() if stdout is None else stdout, job.tag,
                       {**(job.variables or {}), 'cluster': job.cluster_name,
                        'job_id': job.job_id, 'sbatchman_runtime_s': job.get_run_time()})
    return {'tenstorrent': row}
