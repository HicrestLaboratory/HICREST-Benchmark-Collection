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

from dataclasses import dataclass, field
import io
from pathlib import Path
from pprint import pprint
import re
from statistics import geometric_mean, median
from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Tuple, Union
from warnings import warn
import sbatchman as sbm
import numpy as np
import pandas as pd

from experiments_generator import PLACEMENT_CLASS_DEFS
from plot_commons import GPUS_PER_NODE_MAP
from parsers import parse_scheduler_output, stdout_file_to_csv_multi, stdout_to_csv_multi
from command_map import get_model_from_command

SYSTEMS = ["jupiter", "leonardo", "nvl72", "alps", "dgxA100", "lumi", "intel"]
SBM_SYSTEM_NAME_MAP = {"dgxA100": "baldo", "intel": "enea"}

SBM_SYSTEM_ARCHIVES = {
    'alps': ['dp_vitl', 'dppp_ccs', 'duplicates', 'preliminary_tests'],
    'jupiter': ['DP_baselines_run1', 'DP_baselines_run2', 'baseline_correct_compute', 'baselines_prod'],
}

# ============================================================================
#  Placement mapping
# ============================================================================

# Mapping from concurrent-run filename placement tags to the corresponding
# baseline placement class stored in SbatchMan metadata.yaml files.
# This lets us pair each concurrent run with the correct isolated baseline.
PLACEMENT_MAP = {
    "intra-l1":              "INTRA_L1_RANDOM",
    "intra-group":           "INTRA_GROUP_RANDOM",
    "inter-group":           "INTER_GROUP_RANDOM",
    "intra-group-same-l1-2": "INTRA_GROUP_SAME_L1_2",
    "inter-group-same-l1-2": "INTER_GROUP_SAME_L1_2",
    "intra-group-same-l1-4": "INTRA_GROUP_SAME_L1_4",
    "inter-group-same-l1-4": "INTER_GROUP_SAME_L1_4",
    "na": "na",
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

@dataclass
class RunMeasurements:
    n_ranks: int
    _df: pd.DataFrame  # main DataFrame, kept as-is

    @classmethod
    def from_df_dict(cls, df_dict: Dict[str, pd.DataFrame], n_ranks: int) -> 'RunMeasurements':
        return cls(n_ranks=n_ranks, _df=df_dict['main'])

    def _get_niter(self) -> int:
        niter, remainder = divmod(len(self._df), self.n_ranks)
        if remainder != 0:
            print(f'WARNING len(df) is not divisible by n_ranks: {len(self._df)=} {self.n_ranks=}')
        return niter

    def get_throughput_min_of_medians(self, skip_first_n: int = 1) -> Union[Tuple[float, float, float], None]:
        """
        Aggregates per-rank iterations using min
        Returns (min, max, median) of these mins
        """
        mins = []
        for _, rank_df in self._df.groupby('rank'):
            usable = rank_df['throughput'].iloc[skip_first_n:]
            if not usable.empty:
                mins.append(usable.min())
            else:
                print('WARNING no usable throughputs in job')
        return (float(min(mins)), float(max(mins)), float(median(mins))) if mins else None

    def get_comm_relevance(self, skip_first_n: int = 1) -> Union[Tuple[float, float, float], None]:
        """
        Aggregates per-rank iterations using max
        Returns (min, max, median) of these maxes
        """
        
        if 'sync_time' not in self._df.columns:
            return None

        maxes = []
        for _, rank_df in self._df.groupby('rank'):
            usable = rank_df.iloc[skip_first_n:]
            if not usable.empty:
                maxes.append((usable['sync_time'] / usable['runtime']).max())
            else:
                print('WARNING no usable sync_time in job')

        return (float(min(maxes)), float(max(maxes)), float(median(maxes))) if maxes else None
    
    
class RunKey(NamedTuple):
    """Canonical identifier for a training run configuration."""
    system:          str
    strategy:        str
    model:           str
    gpus:            int
    placement_class: str

    def display(self) -> str:
        """Human-readable label used in plot titles and table columns."""
        return f"{self.strategy}/{self.model}/{self.gpus}g/{self.placement_class}"

    def short(self) -> str:
        """Compact label used in legend entries and bar-chart x-ticks."""
        return f"{self.strategy}\n{self.model}\n{self.gpus}g\n{self.placement_class}"

class RunMetrics(NamedTuple):
    throughput:     Union[Tuple[float, float, float], None]
    comm_relevance: Union[Tuple[float, float, float], None]

@dataclass
class Baseline:
    system: str
    comm_lib: str
    gpu_model: str
    strategy: str
    model: str
    gpus: int
    nodes: int
    placement_class: Union[str,None]
    
    in_reservation: bool
    
    data: Union[RunMeasurements, None] = field(init=False)
    
    def display(self) -> str:
        """Human-readable representation of the baseline configuration."""
        return (
            f"Baseline({self.system}, {self.strategy}, {self.model}, "
            f"gpus={self.gpus}, nodes={self.nodes}, placement={self.placement_class}, "
            f"in_reservation={self.in_reservation})"
        )
    
    
    def get_id_tuple(self) -> RunKey:
        return RunKey(
            system=self.system,
            strategy=self.strategy,
            model=self.model,
            gpus=self.gpus,
            placement_class=self.placement_class or 'na'
        )
    
    def get_throughput_min_of_medians(self):
        """
        Aggregates per-rank iterations using min
        Returns (min, max, median) of these mins
        """
        return self.data.get_throughput_min_of_medians() if self.data else None
    
    def get_comm_relevance(self):
        """
        Aggregates per-rank iterations using max
        Returns (min, max, median) of these maxes
        """
        return self.data.get_comm_relevance() if self.data else None
    
    # TODO add more class methods
    

def get_system_jobs(system: str) -> List[sbm.Job]:
    jobs = sbm.jobs_list(
        cluster_name=SBM_SYSTEM_NAME_MAP.get(system, system),
        status=[sbm.Status.COMPLETED, sbm.Status.TIMEOUT]
    )
    
    for a in SBM_SYSTEM_ARCHIVES.get(system, []):
        a_jobs = sbm.jobs_list(
            cluster_name=SBM_SYSTEM_NAME_MAP.get(system, system),
            status=[sbm.Status.COMPLETED, sbm.Status.TIMEOUT],
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
            
            if not job.tag.startswith('baseline'):
                continue
            
            if system == "nvl72" and v.get("gpu_model", "").upper() != "GB300":
                continue

            strategy  = str(v.get("strategy")).strip('orig')
            nodes     = int(v.get("nodes", 1))
            comm_lib  = v.get("comm_lib")
            gpu_model = v.get("gpu_model")
            gpus      = int(v.get("gpus", nodes * GPUS_PER_NODE_MAP[system]))
            placement_class = v.get("placement_class") or v.get("placement", "na")
            model_name = v.get("model") or v.get("model_name")
            
            
            stdout = job.get_stdout()
            if not stdout:
                print('WARNING: could not find baseline stdout:')
                print(job)
                continue
    
            if not model_name:
                model_name = get_model_from_command(job.command)
            if not model_name:
                model_name = parse_model_name_from_stdout(stdout)
                
            if strategy == 'DP' and model_name == 'vit-l':
                continue
            
            if not all([strategy, nodes, gpus, placement_class, model_name, comm_lib, gpu_model]):
                print('WARNING: incomplete baseline meta:')
                print(job)
                print(f'{[strategy, nodes, gpus, placement_class, model_name, comm_lib, gpu_model]=}')
                continue

            baseline = Baseline(
                system=system,
                strategy=strategy,
                model=model_name,
                comm_lib=comm_lib,
                gpu_model=gpu_model,
                gpus=gpus,
                nodes=nodes,
                placement_class=placement_class,
                in_reservation=placement_class and placement_class != 'na' and system in ['leonardo', 'jupiter'], # TODO fix properly
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
    res = {}
    for b in baselines.values():
        for baseline in b:
            res[baseline.get_id_tuple()] = RunMetrics(
                throughput=baseline.get_throughput_min_of_medians(),
                comm_relevance=baseline.get_comm_relevance()
            )
    return res

    
def _table(rows: list[list[str]], header: list[str], title: str = "", indent: int = 2) -> None:
    """Print a plain-text aligned table to stdout."""
    all_rows   = [header] + rows
    col_widths = [max(len(str(r[c])) for r in all_rows) for c in range(len(header))]
    pad        = " " * indent
    sep        = pad + "  ".join("-" * w for w in col_widths)
    if title:
        total_w = sum(col_widths) + 2 * (len(col_widths) - 1)
        print(f"\n{pad}{title:=^{total_w}}")
    print(pad + "  ".join(str(h).ljust(w) for h, w in zip(header, col_widths)))
    print(sep)
    for row in rows:
        print(pad + "  ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))
    print()


def print_baseline_table(baseline_dict: Dict[RunKey, RunMetrics]) -> None:
    """Print a summary table of all baseline values."""
    if not baseline_dict:
        print("  (no baseline entries)\n")
        return
    rows = []
    for k, v in sorted(baseline_dict.items()):
        rows.append([
            k.system, k.strategy, k.model, str(k.gpus), k.placement_class,
            ' / '.join([f'{t:7.1f}' for t in v.throughput]) if v.throughput else '(none)',
            ' / '.join([f'{r*100.0:7.1f}' for r in v.comm_relevance]) if v.comm_relevance else '(none)',
        ])
    _table(
        rows,
        header=["System", "Strategy", "Model", "GPUs", "Placement", "Throughput (min/max/median)", "Comm relevance (min/max/median)"],
        title=" Baseline T0 summary ",
    )

# ============================================================================
#  Concurrent-run parsing
# ============================================================================

PLACEMENT_CLASS_SCORES = {
    p: s
    for _, p, s in PLACEMENT_CLASS_DEFS
}
PLACEMENT_CLASS_SCORES['na'] = 0.0

class MultiRunKey(NamedTuple):
    strategy: str
    gpus: int
    placement_class: str
    
    def display(self):
        return f'{self.strategy:<10} / {self.gpus:<4} / {self.placement_class:<15}'
        
    

@dataclass
class ConcurrentRun:
    system: str
    gpus: int
    nodes: int
    job_id: int
    tag: str
    tot_runtime: Union[float, None]
    
    multi_runs: Dict[MultiRunKey, List[RunMeasurements]] = field(init=False)# FIXME
    
    pattern: List[int]
    strategies: List[str]
    placements: List[str]
    
    in_reservation: bool
    
    def get_distinct_strategies(self) -> set[str]:
        return set(self.strategies)
    
    def get_placement_score(self) -> float:
        score = 0.0
        for p in self.placements:
            score += PLACEMENT_CLASS_SCORES[p]
        return score / float(len(self.placements))
    
    def display(self) -> str:
        parts = [
            f"ConcurrentRun {self.system} - {self.job_id} - {self.tag}:",
            f"tot_gpus={self.gpus}  tot_nodes={self.nodes}  runtime={self.tot_runtime}  strategies={len(self.get_distinct_strategies())}  placement_score={self.get_placement_score()}",
            *[f'  {k.display():<50} -> {len(v)} repetitions' for k, v in self.multi_runs.items()],
        ]
        return '\n'.join(parts)
    

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

def parse_concurrent(systems=SYSTEMS) -> Dict[str, List[ConcurrentRun]]:
    """
    Parse all concurrent-execution stdout files from the workerpool output.
    """
    concurrent = defaultdict(list)
    
    print(f'Loading concurrent runs for systems: {systems}')
    for system in systems:
        print(f'  Loading system: {system}')
        jobs = get_system_jobs(system)
        
        # Accumulated (metadata, dataframes) pairs for import_export
        pairs: list[tuple[dict, dict[str, pd.DataFrame]]] = []
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
            
            if job.tag.startswith('baseline'):
                continue
            
            if system == "nvl72" and "GB300" not in job.tag:
                continue
            
            stdout = job.get_stdout()
            if not stdout:
                print('WARNING: could not find concurrent stdout:')
                print(job)
                continue
            
            try:
                runs, _ = parse_scheduler_output(stdout)
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
                    exit(1)
                strategy, gpus, nodes, placement_class, rep = m.groups()
                strategies.append(strategy)
                placement_classes.append(placement_class)
                
            concurrent_run = ConcurrentRun(
                system=job.cluster_name,
                job_id=job.job_id,
                tag=job.tag,
                tot_runtime=job.get_run_time(),
                placements=placement_classes,
                pattern=pattern,
                strategies=strategies,
                nodes=sum(pattern),
                gpus=sum(pattern) * GPUS_PER_NODE_MAP[system],
                in_reservation=False, # FIXME
            )
            if concurrent_run.gpus != sum(pattern):
                warn(f'Mismatch between {tot_nodes*GPUS_PER_NODE_MAP[system]=} and {sum(pattern)*GPUS_PER_NODE_MAP[system]=}')
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
                        multi_runs[MultiRunKey(
                            strategy=strategy,
                            gpus=int(gpus),
                            placement_class=placement_class
                        )].append(measurements)
                        
                        n_ok      += 1
                        total_ok  += 1
                    except Exception as e:
                        n_bad_csv += 1
                        if str(e) != 'No ccutils sections found in stdout':
                            print('PARSE ERROR:')
                            print(e)
                            raise e
                        pass
                else:
                    warn('Parser not implemented for "if not stdout.startswith(\'stdout: \')"')
                
                    
                concurrent_run.multi_runs = multi_runs

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
            
            print(concurrent_run.display())
            print()
            
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
        print("=== Per-job summary ===")
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

def compute_slowdowns(baselines, concurrent):
    """
    Match each concurrent run to its isolated baseline and compute the
    congestion impact (slowdown ratio).

    Matching uses (strategy, nodes, model_name) and maps the concurrent
    placement name to the baseline's placement class via PLACEMENT_MAP.

    Parameters
    ----------
    baselines : dict
        ``{(strategy, nodes, placement, model_name): throughput}`` from
        ``parse_baselines``.
    concurrent : list of dict
        Concurrent run records from ``parse_concurrent``.

    Returns
    -------
    dict
        ``{(strategy, nodes, placement, model_name): [ratio, ...]}``

        ratio = baseline_throughput / concurrent_throughput.
        Values > 1.0 indicate congestion-induced slowdown.
    """
    slowdowns = defaultdict(list)
    unmatched = set()
    skipped_na = 0

    for run in concurrent:
        placement = run["placement"]

        baseline_placement = PLACEMENT_MAP.get(placement)
        if baseline_placement is None:
            skipped_na += 1
            continue

        model_name = run.get("model_name", "unknown")
        bkey = (run["strategy"], run["nodes"], baseline_placement, model_name)
        t0 = baselines.get(bkey)
        if t0 is None or t0 == 0:
            unmatched.add(bkey)
            continue

        sigma = t0 / run["throughput"]
        cat = (run["strategy"], run["nodes"], placement, model_name)
        slowdowns[cat].append(sigma)

    if skipped_na:
        print(
            f"  Skipped {skipped_na} concurrent runs with unmapped placement "
            f"(e.g. 'na')"
        )
    if unmatched:
        print(f"  WARNING: {len(unmatched)} combos had no matching baseline:")
        for u in sorted(unmatched, key=str):
            print(f"    {u}")

    return slowdowns