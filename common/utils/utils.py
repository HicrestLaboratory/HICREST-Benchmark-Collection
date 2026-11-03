import re
from typing import Optional

def dict_get(d, key):
    r = d.get(key)
    if r is None:
        raise KeyError(f"{key} not found")
    return r


def raise_none(v, msg):
    if v is None:
        raise ValueError(f"{msg} not found")
    return v

def remove_between_inline(text: str, start_delim: str, end_delim: Optional[str] = None) -> str:
    """
    Removes text between delimiters anywhere within the string (inclusive).
    Leaves surrounding line content intact if delimiters are inline.
    """
    if end_delim is None:
        end_delim = start_delim
        
    pattern = re.escape(start_delim) + r'.*?' + re.escape(end_delim)
    return re.sub(pattern, '', text, flags=re.DOTALL)


def remove_between_lines(text: str, start_delim: str, end_delim: Optional[str] = None) -> str:
    """
    Removes entire lines from the line containing start_delim 
    through the line containing end_delim (inclusive).
    """
    if end_delim is None:
        end_delim = start_delim

    lines = text.splitlines(keepends=True)
    result = []
    inside_block = False

    for line in lines:
        if not inside_block:
            if start_delim in line:
                inside_block = True
                # Handle edge case where both delimiters are on the exact same line
                if end_delim in line:
                    inside_block = False
            else:
                result.append(line)
        else:
            if end_delim in line:
                inside_block = False

    return "".join(result)

def _remove_examon_logs(stdout: str) -> str:
    return remove_between_lines(stdout, 'START ENERGY MEASUREMENTS', 'END ENERGY MEASUREMENTS')
    
def _remove_ncm_logs(stdout: str) -> str:
    res = []
    for s in stdout.splitlines():
        if s.startswith((
            '$ ',
            'Passive mode: ',
            'Sending SIGINT',
            'Still alive after',
            'WARNING: process',
            'STABLE ENERGY',
            'Monitor exited after SIGINT',
        )):
            continue
        res.append(s)
    return '\n'.join(res)


def remove_energy_logs(stdout: str) -> str:
    return _remove_ncm_logs(_remove_examon_logs(stdout))


def remove_useless_job_meta(data: dict) -> dict:
    res = {}
    for k, v in data.items():
        if k not in ['energy_measure_before_exec', 'energy_measure_start', 'energy_measure_stop']:
            res[k] = v
    return res