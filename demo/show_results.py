import json
import re
import sys
from pathlib import Path

import sbatchman as sbm

sys.path.append(str(Path(__file__).parent.parent / "common" / "ccutils" / "parser"))
from ccutils_parser import MPIOutputParser



########################
#      DLNetBench      #
########################

def extract_throughput(job: sbm.Job) -> dict:
    parser = MPIOutputParser()
    res = parser.parse_file(job.get_stdout_path())
    strategy_name = list(res.keys())[0]
    res = res[strategy_name]
    rank_outputs = res.mpi_all_prints["ccutils_rank_json"].rank_outputs
    throughputs = []
    for rank, json_str in rank_outputs.items():
        parsed = json.loads(json_str)
        rank_throughput = parsed.get("throughputs", [])[2]
        throughputs.append(rank_throughput)
    data = {
        'strategy': strategy_name,
        'throughput': min(throughputs),
    }
    data.update(res.get_global_json())
    return data

df_dlnetbench = sbm.jobs_to_dataframe(
    status=[sbm.Status.COMPLETED],
    job_filter=lambda j: j.config_name.startswith('DLNetBench'),
    extractors=[extract_throughput],
    include_job_variables=True,
    include_job_fields=True,
)

if not df_dlnetbench.empty:
    print('='*20 + '  DLNetBench  ' + '='*20 + '\n')
    print(f'Available columns: {df_dlnetbench.columns}')
    print(df_dlnetbench
        .sort_values(['strategy', 'model', 'nodes'])
        [['strategy', 'model', 'nodes', 'throughput']])




########################
#       Graph500       #
########################

def extract_teps(job: sbm.Job) -> dict:
    parser = MPIOutputParser()
    res = parser.parse_file(job.get_stdout_path())
    general = res["general_results"].raw_text
    teps = -1
    for line in general.strip().splitlines():
        if "harmonic_mean_TEPS" in line:
            line = re.subn(r"\s{2,}", " ", line)[0]
            teps = float(line.split(" ")[-1])
            break
    return {'teps': teps}

df_graph500 = sbm.jobs_to_dataframe(
    status=[sbm.Status.COMPLETED],
    job_filter=lambda j: j.config_name.startswith('Graph500'),
    extractors=[extract_teps],
    include_job_variables=True,
    include_job_fields=True,
)

if not df_graph500.empty:
    print('\n\n' + '='*20 + '  Graph500  ' + '='*20 + '\n')
    print(f'Available columns: {df_graph500.columns}')
    print(df_graph500
        .sort_values(['scale', 'edgefactor', 'buffer_size', 'nodes'])
        [['scale', 'edgefactor', 'buffer_size', 'nodes', 'teps']])