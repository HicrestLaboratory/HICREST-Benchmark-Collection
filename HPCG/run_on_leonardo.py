import argparse
import warnings
import sbatchman as sbm
import sys
import pprint
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent.parent))
from py_utils.utils.utils import dict_get

sys.path.append(str(Path(__file__).parent.parent / 'machines' / 'Leonardo'))
import nodelists_generator as leo_gen

NODES=[1, 2, 4, 8, 16]
PARTITION = 'boost_usr_prod'
MAX_CANDIDATES = 1000
NODELIST_RETRY_TIME_S = 5
DO_NODELIST_RANKING = False

NODELIST_TYPE_EMULATING_NANJING = 'emulating_nanjing'
NODELIST_TYPE_DIFFERENT_DISTANCES = 'different_distances'
NODELIST_TYPE_EMULATING_HAICGU = 'emulating_haicgu'

def gen_config_name(nodes: int) -> str:
    return f"{PARTITION}_{nodes}nodes"

def get_nodelist(nodelist_type: str, nodes: int):
    nodelist = None
    do_sleep = False
    while nodelist is None:
        print(f'Trying to find a {nodes}-nodes "{nodelist_type}" list...')
        if do_sleep:
            do_sleep = True
            time.sleep(NODELIST_RETRY_TIME_S)
        
        # This will ensure idle nodes are refreshed with `sinfo` at every try
        generator = leo_gen.LeonardoNodelistGenerator(
            verify_with_sinfo=True,
            sinfo_states=['idle'],
            sinfo_partitions=['boost_usr_prod']
        )
            
        if nodelist_type == NODELIST_TYPE_EMULATING_NANJING:
            nodelist = leo_gen.get_nodelists_emulating_nanjing(generator, PARTITION, nodes, do_rank_nodelists=DO_NODELIST_RANKING)
        elif nodelist_type == NODELIST_TYPE_DIFFERENT_DISTANCES:
            nodelist = leo_gen.get_nodelists_different_distances(generator, PARTITION, nodes, do_rank_nodelists=DO_NODELIST_RANKING)
        elif nodelist_type == NODELIST_TYPE_EMULATING_HAICGU:
            nodelist = leo_gen.get_nodelists_emulating_haicgu(generator, PARTITION, nodes, do_rank_nodelists=DO_NODELIST_RANKING)
        else:
            raise Exception(f'Invalid nodelist type "{nodelist_type}"')
        
    return nodelist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--jobs',
        type=str,
        help='Path to jobs YAML file',
        default='',
    )
    parser.add_argument(
        '--nodelist-type',
        '-t',
        type=str,
        choices=[NODELIST_TYPE_EMULATING_HAICGU, NODELIST_TYPE_EMULATING_NANJING, NODELIST_TYPE_DIFFERENT_DISTANCES],
        help='The desired node placement',
        required=True,
    )
    args = parser.parse_args()
    
    jobs_by_node_dict = None
    if args.jobs and len(args.jobs) > 0:
        jobs = sbm.launch_jobs_from_file(args.jobs, dry_run=True) # The dry run is not necessary
        jobs_by_node_dict = {}
        for job in jobs:
            if not job.variables:
                warnings.warn(f'Job {job} has no variables, SKIPPING.')
                continue
            nodes = dict_get(job.variables, 'nodes')
            if not jobs_by_node_dict.get(nodes):
                jobs_by_node_dict[nodes] = []
                
            jobs_by_node_dict[nodes].append(job)

    if jobs_by_node_dict:
        print('==== Jobs by nodes ====')
        for k, v in jobs_by_node_dict.items():
            print(f'---> {k} nodes')
            for j in v:
                print(j)
            if len(jobs_by_node_dict[k]) > 10:
                warnings.warn('This job list might be too long...')
            print()
            
        for nodes, jobs in jobs_by_node_dict.items():
            nodelist = get_nodelist(args.nodelist_type, nodes)
            if nodelist is None:
                raise Exception('should not happen')

            config_name = gen_config_name(nodes)
            sbm.create_slurm_config(
                name=config_name,
                cluster_name='leonardo',
                partition=str(PARTITION),
                account='try25_HNS',
                nodes=str(nodes),
                ntasks=str(nodes),
                cpus_per_task=1,
                time="00:10:00",
                gpus=0,
                nodelist=nodelist,
                qos="normal",
                modules=[
                    "gcc/12.2.0",
                    "openmpi/4.1.6--gcc--12.2.0-cuda-12.2"
                ],
                env=[
                    "OMP_PROC_BIND=true",
                    "OMP_NUM_THREADS=32",
                    "NCCL_IB_SL=1",
                    "UCX_IB_SL=1",
                ],
                # custom_headers=[
                #     "#SBATCH --mail-type=END",
                #     "#SBATCH --mail-user=thomas.pasquali@unitn.it"
                # ],
                overwrite=True,
            )

            for j in jobs:
                sbm.job_submit(j)
    


if __name__ == "__main__":
    main()