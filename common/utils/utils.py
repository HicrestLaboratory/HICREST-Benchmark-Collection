def dict_get(d, key):
    r = d.get(key)
    if r is None:
        raise KeyError(f"{key} not found")
    return r


def raise_none(v, msg):
    if v is None:
        raise ValueError(f"{msg} not found")
    return v


def remove_ncm_logs(stdout: str) -> str:
    res = []
    for s in stdout.splitlines():
        if s.startswith((
            '$ ',
            'Passive mode: ',
            'Sending SIGINT',
            'Still alive after',
            'WARNING: process',
            'STABLE ENERGY'
        )):
            continue
        res.append(s)
    return '\n'.join(res)


def remove_energy_logs(stdout: str) -> str:
    return remove_ncm_logs(stdout)


def remove_useless_job_meta(data: dict) -> dict:
    res = {}
    for k, v in data.items():
        if k not in ['energy_measure_before_exec', 'energy_measure_start', 'energy_measure_stop']:
            res[k] = v
    return res