"""One parser shared by SbatchMan visualization and the CSV collector."""
import json
import math
import re
from matrix_suite import cases, case_id

RESULT_RE = re.compile(r'^HICREST_RESULT\s+(\{.*\})\s*$', re.MULTILINE)
CASE_MAP = {case_id(c): c for c in cases()}


def parse_stdout(stdout, tag, metadata=None):
    if tag not in CASE_MAP:
        return None
    matches = RESULT_RE.findall(stdout or '')
    if len(matches) != 1:
        raise ValueError(f'{tag}: expected one HICREST_RESULT, found {len(matches)}')
    row = dict(metadata or {})
    row.update(json.loads(matches[0]))
    expected = CASE_MAP[tag]
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
    if job.tag not in CASE_MAP:
        return None
    if str(job.status).lower() not in ('completed', 'status.completed'):
        return None
    row = parse_stdout(job.get_stdout() if stdout is None else stdout, job.tag,
                       {**(job.variables or {}), 'cluster': job.cluster_name,
                        'job_id': job.job_id, 'sbatchman_runtime_s': job.get_run_time()})
    return {'tenstorrent': row}
