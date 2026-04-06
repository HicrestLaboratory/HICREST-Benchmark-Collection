#!/usr/bin/env python3
"""
parse_results.py — Data parsing and slowdown computation for DLNetBench.

For each (strategy, nodes, placement) combination this module:
  1. Parses per-rank data from raw ccutils stdout files.
  2. Applies adaptive warm-up skipping: if a rank recorded >= 5 iterations
     the first 3 are dropped (DP strategies have slow warm-up); otherwise
     the default --skip-first value is used.
  3. Computes the median throughput per rank, then takes the minimum across
     all ranks (the bottleneck rank limits distributed training speed).
  4. Calculates slowdown = baseline_min / concurrent_min.
     Ratios > 1.0 indicate the concurrent run was slower than isolated
     (congestion-induced slowdown).

Data sources (under each system's backup dir):
  SbatchMan/experiments/<system>/   isolated baselines  (metadata.yaml + stdout.log)
  workerpool_out/                   concurrent runs       (*.stdout files)

Public API
----------
  parse_baselines(backup_dir, skip_first, system_name)  -> dict
  parse_concurrent(backup_dir, skip_first, system_name) -> list[dict]
  compute_slowdowns(baselines, concurrent)               -> dict
  min_throughput_across_ranks(ranks, skip_first, adaptive_skip) -> float | None
"""

from pathlib import Path
from pprint import pprint
import re
from collections import defaultdict
import sys
from typing import Dict, List, Optional, Union
from warnings import warn
import sbatchman as sbm
import pandas as pd

from data_types import GPUS_PER_NODE_MAP, SYSTEM_ORDER, Baseline, ConcurrentRun, MeasurementStats, Model, Placement, RunKey, RunMeasurements, RunMetrics, SlowdownStats, Strategy, parse_placement
from parsers import parse_scheduler_output, stdout_file_to_csv_multi, stdout_to_csv_multi
from command_map import get_model_from_command

sys.path.append(str(Path(__file__).parent.parent / "common"))
from JobPlacer.cli_wrapper import JobPlacer

SYSTEMS = ["jupiter", "leonardo", "nvl72", "alps", "dgxA100", "lumi", "intel"]
SBM_SYSTEM_NAME_MAP = {"dgxA100": "baldo", "intel": "enea"}

SBM_SYSTEM_ARCHIVES = {
    'alps': ['dppp_ccs', 'duplicates', 'preliminary_tests'],
    'jupiter': ['DP_baselines_run1', 'DP_baselines_run2', 'baseline_correct_compute', 'baselines_prod'],
}

# ============================================================================
#  Raw stdout parsing
# ============================================================================

def parse_model_name_from_stdout(filepath):
    """Return model_name from stdout JSON metadata in the run file."""
    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            m = re.search(r'"model_name"\s*:\s*"([^"]+)"', line)
            if m:
                return m.group(1)
    return None


# ============================================================================
#  Baseline parsing
# ============================================================================


def get_system_jobs(system: str) -> List[sbm.Job]:
    jobs = sbm.jobs_list(
        cluster_name=SBM_SYSTEM_NAME_MAP.get(system, system),
        status=[sbm.Status.COMPLETED, sbm.Status.TIMEOUT, sbm.Status.CANCELLED]
    )
    
    for a in SBM_SYSTEM_ARCHIVES.get(system, []):
        a_jobs = sbm.jobs_list(
            cluster_name=SBM_SYSTEM_NAME_MAP.get(system, system),
            status=[sbm.Status.COMPLETED, sbm.Status.TIMEOUT, sbm.Status.CANCELLED],
            archive_name=a,
            from_active=False,
            from_archived=True,
        )
        print(f'  Added {len(a_jobs)} from {system.upper()} archive {a}')
        jobs.extend(a_jobs)
    
    return jobs 
    
    
def parse_proxy_stdout_data(stdout: Union[str, Path], gpus: int) -> Union[None, RunMeasurements]:
    df_dict, _ = (
        stdout_to_csv_multi(stdout, return_dataframes=True)
        if isinstance(stdout, str)
        else stdout_file_to_csv_multi(stdout, return_dataframes=True)
    )
    return RunMeasurements.from_df_dict(df_dict, n_ranks=gpus) if 'main' in df_dict else None
    
                    

def parse_baselines(systems=SYSTEMS) -> Dict[str, List[Baseline]]:
    """
    Parse all isolated baseline experiments from SbatchMan output.

    Parameters
    ----------
    systems    : set[str]
        System names.
    """
    baselines = defaultdict(list)
    
    print(f'Loading baselines for systems: {systems}')
    for system in systems:
        print(f'  Loading system: {system}')
        jobs = get_system_jobs(system)
        
        for j_i, job in enumerate(jobs):
            if j_i % 10 == 0:
                print(f'  job {j_i:<3} of {len(jobs)}')
            v = job.variables or {}
            
            if system == "nvl72" and v.get("gpu_model", "").upper() != "GB300":
                continue
              
            if not job.tag.startswith('baseline'): 
                if system in ['leonardo', 'lumi'] and "nccl_default" in job.tag:
                    strategy = Strategy("FSDP")
                    nodes = int(v.get("nodes", 1))
                    gpus = int(v.get("gpus", nodes * GPUS_PER_NODE_MAP[system]))
                    comm_lib = "nccl" if system == 'leonardo' else "rccl"
                    gpu_model = "A100" if system == 'leonardo' else "MI250X"
                    placement_class = Placement("na")
                    model_name = v.get("fsdp_model")
                else:
                    continue
            else:
                strategy  = Strategy(str(v.get("strategy")).strip('orig'))
                nodes     = int(v.get("nodes", 1))
                comm_lib  = v.get("comm_lib")
                gpu_model = v.get("gpu_model")
                gpus      = int(v.get("gpus", nodes * GPUS_PER_NODE_MAP[system]))
                placement_class = parse_placement(v.get("placement_class") or v.get("placement", "na"))
                model_name = v.get("model") or v.get("model_name")
            
            
            stdout = job.get_stdout()
            if not stdout:
                print('WARNING: could not find baseline stdout:')
                print(job)
                continue
    
            if not model_name:
                # print(job.command)
                model_name = get_model_from_command(job.command)
            
            if strategy == 'DP' and (model_name == 'vit-l' or 'vit-l' in job.command):
                continue
            
            if not model_name:
                model_name = parse_model_name_from_stdout(stdout)
            model = Model(model_name)
                
            
            if not all([strategy, nodes, gpus, placement_class, model_name, comm_lib, gpu_model]):
                print('WARNING: incomplete baseline meta:')
                print(job)
                print(f'{[strategy, nodes, gpus, placement_class, model_name, comm_lib, gpu_model]=}')
                continue

            baseline = Baseline(
                system=system,
                strategy=strategy,
                model=model,
                comm_lib=comm_lib,
                gpu_model=gpu_model,
                gpus=gpus,
                nodes=nodes,
                placement_class=placement_class,
                in_reservation=bool(placement_class) and placement_class != Placement.NA and system in ['leonardo', 'jupiter'], # TODO fix properly
            )

            try:
                baseline.data = parse_proxy_stdout_data(stdout, gpus)
                baselines[system].append(baseline)    
            except Exception as e:
                if str(e) != 'No ccutils sections found in stdout':
                    print(baseline.display())
                    raise e
        
    print('Loaded baselines (system -> number of baselines):')
    print({s: len(b) for s, b in baselines.items()})
    
    return baselines


def build_baselines_dict(baselines: Dict[str, List[Baseline]]) -> Dict[RunKey, RunMetrics]:
    """
    Builds a baseline dict merging duplicate entries via MeasurementStats.merge().

    Rather than deduplicating by keeping the best result (which cherry-picks the
    luckiest run, violating Hoefler & Belli 2015 Rule 2: "report all results,
    not just the best"), duplicate keys are merged by pooling their raw filtered
    values and recomputing all statistics. This gives a more representative and
    statistically robust baseline by using all available data.
    """
    res: Dict[RunKey, RunMetrics] = {}

    for b in baselines.values():
        for baseline in b:
            key = baseline.get_id_tuple()
            throughput = baseline.get_throughput()

            if throughput is None:
                print(f'WARNING: no throughput for baseline {key}, skipping')
                continue

            if key not in res:
                res[key] = RunMetrics(
                    throughput=throughput,
                    comm_relevance=baseline.get_comm_relevance(),
                )
            else:
                existing_cr = res[key].comm_relevance
                new_cr      = baseline.get_comm_relevance()
                if existing_cr is None:
                    merged_cr = new_cr
                elif new_cr is None:
                    merged_cr = existing_cr
                else:
                    merged_cr = existing_cr.merge(new_cr)

                print(f'Merging {key.system} {key.display()}')
                res[key] = RunMetrics(
                    throughput=res[key].throughput.merge(throughput),
                    comm_relevance=merged_cr,
                )
                print()

    return res


def get_baselines_dataframe(baseline_dict: Dict[RunKey, RunMetrics]):
    if not baseline_dict:
        print("  (no baseline entries)\n")
        return None

    data = []
    for k, v in sorted(baseline_dict.items()):
        t_stats = v.throughput
        
        data.append({
            "system": k.system,
            "strategy": k.strategy,
            "model": k.model,
            "gpus": k.gpus,
            "placement": k.placement_class,
            "throughput_min": t_stats.min if t_stats else None,
            "throughput_max": t_stats.max if t_stats else None,
            "throughput_median": t_stats.median if t_stats else None,
            "throughput_mean": t_stats.mean if t_stats else None,
            "throughput_geomean": t_stats.geomean if t_stats else None,
            "throughput_std": t_stats.std if t_stats else None,
            "comm_relevance": v.comm_relevance.ratio * 100.0 if v.comm_relevance else None,
            "comm_relevance_ci_low":  v.comm_relevance.ci_low  * 100.0 if v.comm_relevance else None,
            "comm_relevance_ci_high": v.comm_relevance.ci_high * 100.0 if v.comm_relevance else None,
            "comm_relevance_max": (v.comm_relevance.sync.max/v.comm_relevance.runtime.max) * 100.0 if v.comm_relevance else None,
        })

    df = pd.DataFrame(data)
    df["system"] = pd.Categorical(df["system"], categories=SYSTEM_ORDER, ordered=True)
    return df.sort_values(by=["system", "strategy", "model", "gpus", "placement"])

    
    

# ============================================================================
#  Concurrent-run parsing
# ============================================================================    

# Filename convention for concurrent runs:
#   <Strategy>_g<GPUs>_n<Nodes>_<Placement>[_<AppID>]_rep<N>_<worker>.stdout
# Example: DP_g8_n2_intra-l1_28_rep7_worker3.stdout
_CONC_RE = re.compile(
    r"((?:[A-Za-z]+\+)*[A-Za-z]+)_g(\d+)_n(\d+)_([A-Za-z\d-]+)_.*rep(\d+)_"
)

# ---------------------------------------------------------------------------
# Per-run outcome labels (used in the summary)
# ---------------------------------------------------------------------------

OUTCOME_OK        = "ok"
OUTCOME_NONZERO   = "nonzero_exit"
OUTCOME_NO_DATA   = "no_data"
OUTCOME_BAD_CSV   = "bad_csv"
OUTCOME_EXCEPTION = "exception"

PLACERS_CACHE = {}

def get_job_placer(system: str) -> Union[JobPlacer, None]:
    if system not in ['leonardo', 'jupiter', 'alps', 'lumi']:
        return None
    
    if system not in PLACERS_CACHE:
        topology_file=f'../common/JobPlacer/{system}_topo.txt'
        sinfo_file=f'../common/JobPlacer/{system}_sinfo.txt'
        if system == 'lumi':
            topology_file = None
            sinfo_file = None
        topology_toml_file=None
        if system.lower() in ['lumi', 'alps']:
            topology_toml_file=f'../common/JobPlacer/systems/{system.upper()}.toml'
            
        PLACERS_CACHE[system] = JobPlacer(
            system=system,
            topology_file=topology_file,
            topology_toml_file=topology_toml_file,
            sinfo_file=sinfo_file
        )
        
    return PLACERS_CACHE[system]

def parse_concurrent(systems=SYSTEMS) -> Dict[str, List[ConcurrentRun]]:
    """
    Parse all concurrent-execution stdout files from the workerpool output.
    """
    concurrent = defaultdict(list)
    
    print(f'Loading concurrent runs for systems: {systems}')
    for system in systems:
        print(f'  Loading system: {system}')
        jobs = get_system_jobs(system)
        
        # Summary bookkeeping
        # issues: list of (sbm_job_id, uid, outcome, detail)
        issues: list[tuple[str, str, str, str, str]] = []
        total_runs   = 0
        total_ok     = 0

        # Per-sbatchman-job summary rows (printed as a table)
        job_summaries: list[dict] = []
        
        for j_i, job in enumerate(jobs):
            if j_i % 10 == 0:
                print(f'  job {j_i:<3} of {len(jobs)}')
            
            # if str(job.job_id) == str(38409415):
            #     print('LUI!!!')
            
            if job.tag.startswith('baseline'):
                continue
            
            # These are some older test experiments
            if job.tag.startswith('experiments_'):
                continue
            
            if system == "nvl72" and "GB300" not in job.tag:
                continue
            
            stdout = job.get_stdout()                
            if not stdout:
                print('WARNING: could not find concurrent stdout:')
                print(job)
                continue
            
            try:
                runs, lines = parse_scheduler_output(stdout)
                placement_strategy = None                
                for l in lines:
                    if 'Placement strategy: ' in l:
                        match = re.search(r"(?<=Placement strategy: )(\w+)(?=,)", l)
                        if match:
                            placement_strategy = match.group(1)
                        else:
                            raise Exception("Could not parse placement strategy with regex")
                if not placement_strategy:
                    raise Exception("Could not find placement strategy in workerpool out")
                
                allocation_stats = None
                if placement_strategy != 'device':
                    placer = get_job_placer(system)
                    if placer:
                        allocations = {}
                        for l in lines:
                            if 'Assigned ' in l:
                                name, nodes = l.split(' → ')
                                name = name.split('/')[1]
                                nodes = eval(nodes)
                                allocations[name] = nodes
                                
                        out = None
                        # out = Path('plots/placements')
                        # out.mkdir(parents=True, exist_ok=True)
                        # out /= job.tag
                        # out = out.with_suffix('.svg')
                        try:
                            allocation_stats = placer.get_allocation_stats(allocations, out_svg=out)
                        except Exception as e:
                            print(e)
                            raise e
                        # print(f'{allocation_stats=}')
                
                REQUIRED_RUN_KEYS = {
                    "uid", "job_name", "repetition", "resources",
                    "app", "start_ts", "finished_at", "exit_code",
                    "success", "stdout", "stderr"
                }
                for run in runs:
                    missing = REQUIRED_RUN_KEYS - run.keys()
                    if missing:
                        warn(f"Run dict missing keys: {missing}")
            except Exception as exc:
                issues.append((str(job.job_id), str(job.tag), "<all>", OUTCOME_EXCEPTION, f"parse_scheduler_output raised: {exc}"))
                job_summaries.append({
                    "sbm_job_id": job.job_id,
                    "sbm_tag":    job.tag,
                    "runs":       "?",
                    "ok":         0,
                    "nonzero":    0,
                    "no_data":    0,
                    "bad_csv":    0,
                    "exception":  1,
                })
                continue
            
            tot_nodes = 1
            if job.variables:
                tot_nodes = job.variables.get('nodes', 1)
            all_rep0_runs = [r for r in runs if r['repetition']==0]
            pattern = []
            strategies = []
            placement_classes = []
            for r in all_rep0_runs:
                pattern.append(len(r['resources']))
                m = _CONC_RE.match(r['uid'])
                if not m:
                    warn(f'Could not parse run uid: {r["uid"]}')
                    print(job)
                    exit(1)
                strategy, gpus, nodes, placement_class, rep = m.groups()
                strategies.append(Strategy(strategy))
                placement_classes.append(parse_placement(placement_class))
                
            is_placement_device = placement_strategy == 'device'
            if is_placement_device:
                nodes = 1
                gpus = sum(pattern)
            else:
                nodes = sum(pattern)
                gpus = nodes * GPUS_PER_NODE_MAP[system]
                pattern = [p* GPUS_PER_NODE_MAP[system] for p in pattern]
                
            concurrent_run = ConcurrentRun(
                system=job.cluster_name,
                job_id=job.job_id,
                tag=job.tag,
                tot_runtime=job.get_run_time(),
                placements=placement_classes,
                pattern=pattern,
                strategies=strategies,
                nodes=nodes,
                gpus=gpus,
                allocation_stats=allocation_stats,
            )
            
            if concurrent_run.gpus / sum(pattern) < 0.8:
                warn(f'Large mismatch between {tot_nodes*GPUS_PER_NODE_MAP[system]=} and {sum(pattern)*GPUS_PER_NODE_MAP[system]=} -> {concurrent_run.gpus / sum(pattern)}')
            
            n_ok = n_nonzero = n_no_data = n_bad_csv = 0

            multi_runs = defaultdict(list)
            for run in runs:
                total_runs += 1
                uid = run["uid"]
                m = _CONC_RE.match(uid)
                if not m:
                    warn(f'Could not parse run uid: {uid}')
                    exit(1)
                strategy, gpus, nodes, placement_class, rep = m.groups()
                model = get_model_from_command(run["app"])

                # --- exit code check ---
                if not run["success"]:
                    n_nonzero += 1
                    issues.append((
                        str(job.job_id), str(job.tag), uid, OUTCOME_NONZERO,
                        f"exit_code={run['exit_code']}  stderr={run['stderr'][:120].strip()!r}",
                    ))
                    # Still attempt to parse whatever data was written before the failure
                    # (the scheduler already fell back to raw on transform errors)

                stdout = str(run["stdout"]).strip()
                
                if stdout == "":
                    n_no_data += 1
                    issues.append((str(job.job_id), str(job.tag), uid, OUTCOME_NO_DATA, "stdout is empty"))
                    continue
                
                if stdout.startswith('stdout: '):
                    # This is a path to raw output file
                    stdout_lines = stdout.splitlines()
                    stdout_path = Path(stdout_lines[0].strip().removeprefix('stdout: '))
                    try:
                        # FIXME shall we exclude the first run here as well?
                        measurements = parse_proxy_stdout_data(stdout_path, int(gpus))
                        multi_runs[RunKey(
                            system=system,
                            strategy=Strategy(strategy),
                            model=Model(model),
                            gpus=int(gpus),
                            placement_class=parse_placement(placement_class)
                        )].append(measurements)
                        
                        n_ok      += 1
                        total_ok  += 1
                    except FileNotFoundError:
                        n_no_data += 1
                    except Exception as e:
                        n_bad_csv += 1
                        if str(e) not in ['No ccutils sections found in stdout', 'Section \'dp\' has no END marker']:
                            print('PARSE ERROR:')
                            print(e)
                            raise e
                else:
                    warn('Parser not implemented for "if not stdout.startswith(\'stdout: \')"')
                
                    

            job_summaries.append({
                "sbm_job_id": job.job_id,
                "sbm_tag":    job.tag,
                "runs":       len(runs),
                "ok":         n_ok,
                "nonzero":    n_nonzero,
                "no_data":    n_no_data,
                "bad_csv":    n_bad_csv,
                "exception":  0,
            })
            concurrent_run.multi_runs = multi_runs
            concurrent[system].append(concurrent_run)
            
            # print(concurrent_run.display())
            # print()
            
        # ------------------------------------------------------------------
        # Summary table
        # ------------------------------------------------------------------
        col_w = {
            "sbm_job_id": max(len("sbm_job_id"), max((len(str(r["sbm_job_id"])) for r in job_summaries), default=0)),
            "sbm_tag":    max(len("sbm_tag"), max((len(str(r["sbm_tag"])) for r in job_summaries), default=0)),
            "runs":       5,
            "ok":         4,
            "nonzero":    9,
            "no_data":    8,
            "bad_csv":    8,
            "exception":  10,
        }
        header = (
            f"{'sbm_job_id':<{col_w['sbm_job_id']}}  "
            f"{'sbm_tag':<{col_w['sbm_tag']}}  "
            f"{'runs':>{col_w['runs']}}  "
            f"{'ok':>{col_w['ok']}}  "
            f"{'nonzero':>{col_w['nonzero']}}  "
            f"{'no_data':>{col_w['no_data']}}  "
            f"{'bad_csv':>{col_w['bad_csv']}}  "
            f"{'exception':>{col_w['exception']}}"
        )
        divider = "-" * len(header)

        print()
        print(f"================== Concurrent runs of {system}:  Per-job summary ==================")
        print(header)
        print(divider)
        for r in job_summaries:
            print(
                f"{str(r['sbm_job_id']):<{col_w['sbm_job_id']}}  "
                f"{str(r['sbm_tag']):<{col_w['sbm_tag']}}  "
                f"{str(r['runs']):>{col_w['runs']}}  "
                f"{r['ok']:>{col_w['ok']}}  "
                f"{r['nonzero']:>{col_w['nonzero']}}  "
                f"{r['no_data']:>{col_w['no_data']}}  "
                f"{r['bad_csv']:>{col_w['bad_csv']}}  "
                f"{r['exception']:>{col_w['exception']}}"
            )
        print(divider)
        print(
            f"{'TOTAL':<{col_w['sbm_job_id']+col_w['sbm_tag']+2}}  "
            f"{total_runs:>{col_w['runs']}}  "
            f"{total_ok:>{col_w['ok']}}  "
            f"{sum(r['nonzero']   for r in job_summaries):>{col_w['nonzero']}}  "
            f"{sum(r['no_data']   for r in job_summaries):>{col_w['no_data']}}  "
            f"{sum(r['bad_csv']   for r in job_summaries):>{col_w['bad_csv']}}  "
            f"{sum(r['exception'] for r in job_summaries):>{col_w['exception']}}"
        )
        print()

        # ------------------------------------------------------------------
        # Issues detail
        # ------------------------------------------------------------------
        # if issues:
        #     print(f"=== Issues ({len(issues)} total) ===")
        #     for sbm_id, tag, uid, outcome, detail in issues:
        #         print(f"  [{outcome:<12}]  job={sbm_id}  {tag=}  run={uid}")
        #         print(f"               {detail}")
        #     print()
        # else:
        #     print("No issues found — all runs produced clean data.\n")

    return concurrent


# ============================================================================
#  Slowdown computation
# ============================================================================

def compute_slowdowns(
    baselines: Dict[RunKey, RunMetrics],
    concurrent: Dict[str, List[ConcurrentRun]],
) -> None:
    """
    Match each concurrent run to its isolated baseline and compute the
    congestion impact (slowdown ratio).

    Following Hoefler & Belli 2015:
    - Rule 4: slowdown is a ratio — baseline and concurrent throughputs are
      aggregated separately via MeasurementStats.merge(), and the ratio is
      computed once from the merged components.
    - Section 3.1.3: the CI of the slowdown is derived from the nonparametric
      CIs of both components (conservative: ci_low/ci_high, optimistic: ci_high/ci_low).
    - Section 3.1.3: outlier removal is handled inside MeasurementStats, and
      n_outliers_removed is preserved through merging.
    """
    unmatched = set()
    no_throughput = set()

    for system, conc_runs in concurrent.items():
        for conc_run in conc_runs:
            slowdowns: Dict[RunKey, SlowdownStats] = {}

            for key, runs in conc_run.multi_runs.items():
                baseline_metrics = baselines.get(key)
                if not baseline_metrics or not baseline_metrics.throughput:
                    unmatched.add(key)
                    continue

                # merge all concurrent measurements for this key
                merged_concurrent: Optional[MeasurementStats] = None
                n_merged = 0
                for m_i, measure in enumerate(runs):
                    t = measure.get_throughput()
                    if not t:
                        no_throughput.add((key, m_i))
                        continue
                    merged_concurrent = t if merged_concurrent is None else merged_concurrent.merge(t)
                    n_merged += 1

                if merged_concurrent is None:
                    continue

                slowdowns[key] = SlowdownStats(
                    baseline=baseline_metrics.throughput,
                    concurrent=merged_concurrent,
                    n_measurements=n_merged,
                )

            conc_run.slowdowns = slowdowns

    if unmatched:
        print(f"  WARNING: {len(unmatched)} combos had no matching baseline:")
        for u in sorted(unmatched, key=str):
            print(f"    {u}")
    if no_throughput:
        print(f"  WARNING: {len(no_throughput)} combos had no throughput:")
        for u in sorted(no_throughput, key=str):
            print(f"    {u}")