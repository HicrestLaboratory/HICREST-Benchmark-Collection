import sys
from pathlib import Path
from typing import Any, Dict, Optional

import sbatchman as sbm

import mlops.parser as mlops_parser
import inference.parser as inference_parser

sys.path.append(str(Path(__file__).parent.parent / "common"))
from energy.ncm_parser import parse_ncm_tot_energy_print
from utils.utils import remove_ncm_logs, remove_useless_job_meta

def parse(job: sbm.Job) -> Optional[Dict[str, Any]]:
    if job.status != sbm.Status.COMPLETED.value or job.category == 'compile':
        return None
    # print(job.category)
    # print(job.tag)
    
    stdout = None
    try:
        stdout = job.get_stdout()
    except Exception:
        return None
    if not stdout:
        return None
    
    res = None

    # TODO add energy
    if job.category.startswith(('gemv', 'gemm', 'gelu', 'softmax')):
        res = mlops_parser.parse(job, remove_ncm_logs(stdout))
    elif job.category.startswith(('inference')):
        res = inference_parser.parse(job, remove_ncm_logs(stdout))
    else:
        print(f'WARNING: category "{job.category}" not recognized (tag {job.tag}).')
        return None
    
    # TODO separate table?
    # if res:
    #     tot_energy = parse_ncm_tot_energy_print(stdout)
    #     if tot_energy:
    #         for k in res.keys():
    #             res[k]['tot_energy_J'] = tot_energy
    
    if res:
        tables = list(res.keys())
        for k in tables:
            for i in range(len(res[k])):
                res[k][i] = remove_useless_job_meta(res[k][i])
                
    # print(res[list(res.keys())[0]][0].keys())
    # print('====================')
    return res