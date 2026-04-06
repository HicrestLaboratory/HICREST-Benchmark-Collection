from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
import re
from statistics import geometric_mean, mean, median, stdev
import sys
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
import pandas as pd
import numpy as np
from scipy.stats import trim_mean
from experiments_generator import PLACEMENT_CLASS_DEFS

sys.path.append(str(Path(__file__).parent.parent / "common"))
from JobPlacer.cli_wrapper import PlacementStats

GPUS_PER_NODE_MAP = {
    'dgxA100':      8,
    'jupiter':      4,
    'leonardo':     4,
    'nvl72':        4,
    'alps':         4,
    'lumi':         8,
    'intel':        8,
}

SYSTEM_NAMES_MAP = {
    'dgxA100':      'DGX A100',
    'jupiter':      'JUPITER',
    'leonardo':     'Leonardo',
    'nvl72':        'NVL 72',
    'alps':         'Alps (Daint)',
    'lumi':         'LUMI',
    'intel':        'Cresco',
}

PLACEMENT_CLASS_SCORES = {
    p: s
    for _, p, s in PLACEMENT_CLASS_DEFS
}
PLACEMENT_CLASS_SCORES['na'] = 0.0

class Strategy(StrEnum):
    DP = "DP"
    FSDP = "FSDP"
    DP_PP = "DP+PP"
    DP_PP_TP = "DP+PP+TP"
    DP_PP_EXPERT = "DP+PP+Expert"
    
    def __str__(self):
        return self.value
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"
    
    def short(self):
        return {
            Strategy.DP: "D",
            Strategy.FSDP: "FSDP",
            Strategy.DP_PP: "DP",
            Strategy.DP_PP_TP: "DPT",
            Strategy.DP_PP_EXPERT: "DPE",
        }[self]

    def color(self):
        return {
            Strategy.DP: "#1f77b4",       # blue
            Strategy.FSDP: "#ff7f0e",    # orange
            Strategy.DP_PP: "#2ca02c",   # green
            Strategy.DP_PP_TP: "#d62728",# red
            Strategy.DP_PP_EXPERT: "#9467bd", # purple
        }[self]

    def marker(self):
        return {
            Strategy.DP: "o",
            Strategy.FSDP: "s",
            Strategy.DP_PP: "D",
            Strategy.DP_PP_TP: "^",
            Strategy.DP_PP_EXPERT: "P",
        }[self]

    def linestyle(self):
        return {
            Strategy.DP: "-",
            Strategy.FSDP: "--",
            Strategy.DP_PP: "-.",
            Strategy.DP_PP_TP: ":",
            Strategy.DP_PP_EXPERT: (0, (3, 1, 1, 1)),  # custom dash pattern
        }[self]


class Model(StrEnum):
    VIT_H = "vit-h"
    VIT_L = "vit-l"
    LLAMA3_8B = "llama3-8b"
    MINERVA_7B = "minerva-7b"
    LLAMA3_70B = "llama3-70b"
    MIXTRAL_8X7B = "mixtral-8x7b"
    
    def __str__(self):
        return self.value
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"
    
    def short(self):
        return {
            Model.VIT_H: "ViT-H",
            Model.VIT_L: "ViT-L",
            Model.LLAMA3_8B: "LaM-8",
            Model.LLAMA3_70B: "LaM-70",
            Model.MINERVA_7B: "Minv",
            Model.MIXTRAL_8X7B: "Mxt",
        }[self]

    def marker(self):
        return {
            Model.VIT_H: "o",
            Model.LLAMA3_8B: "s",
            Model.MINERVA_7B: "D",
            Model.LLAMA3_70B: "^",
            Model.MIXTRAL_8X7B: "X",
        }[self]

    def color(self):
        # Optional: softer palette if model is the main grouping
        return {
            Model.VIT_H: "#4c72b0",
            Model.LLAMA3_8B: "#55a868",
            Model.MINERVA_7B: "#c44e52",
            Model.LLAMA3_70B: "#8172b2",
            Model.MIXTRAL_8X7B: "#ccb974",
        }[self]


class Placement(StrEnum):
    INTRA_L1_RANDOM = "intra-l1"
    INTRA_GROUP_RANDOM = "intra-group"
    INTER_GROUP_RANDOM = "inter-group"
    INTRA_GROUP_SAME_L1_2 = "intra-group-same-l1-2"
    INTER_GROUP_SAME_L1_2 = "inter-group-same-l1-2"
    INTRA_GROUP_SAME_L1_4 = "intra-group-same-l1-4"
    INTER_GROUP_SAME_L1_4 = "inter-group-same-l1-4"
    NA = "na"
    
    def __str__(self):
        return self.value
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"
    
    def display(self, new_line=True, short=False):
        if short:
            p = {
                Placement.INTRA_L1_RANDOM: "L1",
                Placement.INTRA_GROUP_RANDOM: "Intra Gr",
                Placement.INTER_GROUP_RANDOM: "Multi Gr",
                Placement.INTRA_GROUP_SAME_L1_2: "Intra Gr 2N/S",
                Placement.INTER_GROUP_SAME_L1_2: "Multi Gr 2N/S",
                Placement.INTRA_GROUP_SAME_L1_4: "Intra Gr 4N/S",
                Placement.INTER_GROUP_SAME_L1_4: "Multi Gr 4N/S",
                Placement.NA: "N/A",
            }[self]
        else:
            p = {
                Placement.INTRA_L1_RANDOM: "Intra L1",
                Placement.INTRA_GROUP_RANDOM: "Intra Group",
                Placement.INTER_GROUP_RANDOM: "Inter Group",
                Placement.INTRA_GROUP_SAME_L1_2: "Intra Group\n2 Nodes/Switch",
                Placement.INTER_GROUP_SAME_L1_2: "Inter Group\n2 Nodes/Switch",
                Placement.INTRA_GROUP_SAME_L1_4: "Intra Group\n4 Nodes/Switch",
                Placement.INTER_GROUP_SAME_L1_4: "Inter Group\n4 Nodes/Switch",
                Placement.NA: "N/A",
            }[self]
        return p if new_line else p.replace('\n', ' ')

    def linestyle(self):
        return {
            Placement.INTRA_L1_RANDOM: "-",
            Placement.INTRA_GROUP_RANDOM: "--",
            Placement.INTER_GROUP_RANDOM: ":",
            Placement.INTRA_GROUP_SAME_L1_2: "-.",
            Placement.INTER_GROUP_SAME_L1_2: (0, (5, 1)),
            Placement.INTRA_GROUP_SAME_L1_4: (0, (3, 1, 1, 1)),
            Placement.INTER_GROUP_SAME_L1_4: (0, (1, 1)),
            Placement.NA: "-",
        }[self]

    def marker(self):
        return {
            Placement.INTRA_L1_RANDOM: "o",
            Placement.INTRA_GROUP_RANDOM: "s",
            Placement.INTER_GROUP_RANDOM: "D",
            Placement.INTRA_GROUP_SAME_L1_2: "^",
            Placement.INTER_GROUP_SAME_L1_2: "v",
            Placement.INTRA_GROUP_SAME_L1_4: "P",
            Placement.INTER_GROUP_SAME_L1_4: "X",
            Placement.NA: "o",
        }[self]

def parse_placement(value: str) -> Placement:
    norm = re.sub(r"[-_\s]+", "-", value.strip().lower())

    for p in Placement:
        if norm == p.value or norm == p.name.lower().replace("_", "-"):
            return p

    raise ValueError(f"Unknown placement: {value}")

# Utils

def ensure_strategy(x: Union[str, Strategy]) -> Strategy:
    return x if isinstance(x, Strategy) else Strategy(x)

def ensure_model(x: Union[str, Model]) -> Model:
    return x if isinstance(x, Model) else Model(x)

def ensure_placement(x: Union[str, Placement]) -> Placement:
    return x if isinstance(x, Placement) else parse_placement(x)

# FOR PLOTS

SYSTEM_ORDER = [
    "dgxA100",
    "nvl72",
    "intel",
    "leonardo",
    "alps",
    "jupiter",
    "lumi"
]

STRATEGY_ORDER = [
    Strategy.DP,
    Strategy.FSDP,
    Strategy.DP_PP,
    Strategy.DP_PP_TP,
    Strategy.DP_PP_EXPERT,
]

PLACEMENT_ORDER = [
    Placement.NA,
    Placement.INTRA_L1_RANDOM,
    Placement.INTRA_GROUP_RANDOM,
    Placement.INTRA_GROUP_SAME_L1_2,
    Placement.INTRA_GROUP_SAME_L1_4,
    Placement.INTER_GROUP_RANDOM,
    Placement.INTER_GROUP_SAME_L1_2,
    Placement.INTER_GROUP_SAME_L1_4,
]

# Classes
@dataclass
class MeasurementStats:
    min: float
    max: float
    median: float
    mean: float
    geomean: float
    std: float
    per_rank_std_min: Optional[float] = None  # min of per-rank stds (trimmed mean only)
    per_rank_std_max: Optional[float] = None  # max of per-rank stds (trimmed mean only)
    
    def get(self, stat: str) -> float:
        """
        Returns the requested statistic value.
        
        Args:
            stat: One of 'min', 'max', 'median', 'mean', 'geomean', 'std'
            
        Returns:
            The requested statistic value
            
        Raises:
            ValueError: If the stat name is not available
        """
        if not hasattr(self, stat):
            raise ValueError(f"Unknown statistic: {stat}. Available: min, max, median, mean, geomean, std")
        return getattr(self, stat)


def _compute_measurements_stats(
    values, per_rank_stds: Optional[list] = None
) -> Union[MeasurementStats, None]:
    if not values:
        return None
    values = np.array(values)
    return MeasurementStats(
        min=float(np.min(values)),
        max=float(np.max(values)),
        median=float(np.median(values)),
        mean=float(np.mean(values)),
        geomean=float(geometric_mean(values)),
        std=float(np.std(values)),
        per_rank_std_min=min(per_rank_stds) if per_rank_stds else None,
        per_rank_std_max=max(per_rank_stds) if per_rank_stds else None,
    )

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
    

    def get_throughput_median(self, skip_first_n: int = 1) -> Union[MeasurementStats, None]:
        """
        Aggregates per-rank iterations using median per rank, then min across ranks.
        Returns (min, max, median, mean, geomean, std) of these per-rank medians.
        """
        per_rank_medians = []
        for _, rank_df in self._df.groupby('rank'):
            usable = rank_df['throughput'].iloc[skip_first_n:]
            if not usable.empty:
                per_rank_medians.append(float(np.median(usable)))
            else:
                print('WARNING no usable throughputs in job')
        return _compute_measurements_stats(per_rank_medians)


    def get_throughput_trimmed_mean(
        self, skip_first_n: int = 1, trim_fraction: float = 0.1
    ) -> Union[MeasurementStats, None]:
        """
        Aggregates per-rank iterations using trimmed mean per rank, then min across ranks.
        trim_fraction: fraction to cut from each tail (e.g. 0.1 = 10% each side).
        - std: spread across per-rank trimmed means (meaningful with many ranks)
        - within_std: mean of per-rank stds computed on the trimmed window (reflects
                    within-rank variance, more meaningful with few ranks)
        """
        per_rank_tmeans = []
        per_rank_stds = []
        for _, rank_df in self._df.groupby('rank'):
            usable = rank_df['throughput'].iloc[skip_first_n:].to_numpy()
            if len(usable) == 0:
                print('WARNING no usable throughputs in job')
                continue
            sorted_vals = np.sort(usable)
            n = len(sorted_vals)
            n_trim = int(np.floor(trim_fraction * n))
            trimmed = sorted_vals[n_trim: n - n_trim] if n_trim > 0 else sorted_vals
            per_rank_tmeans.append(float(trim_mean(usable, proportiontocut=trim_fraction)))
            per_rank_stds.append(float(np.std(trimmed)))

        return _compute_measurements_stats(per_rank_tmeans, per_rank_stds=per_rank_stds)

    def get_comm_relevance(self, skip_first_n: int = 1) -> Union[MeasurementStats, None]:
        """
        Aggregates per-rank iterations using max
        Returns (min, max, median, mean, geomean, std) of these maxes
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

        return _compute_measurements_stats(maxes)
    
    
class RunKey(NamedTuple):
    """Canonical identifier for a training run configuration."""
    system:          str
    strategy:        Strategy
    model:           Model
    gpus:            int
    placement_class: Placement

    def display(self) -> str:
        """Human-readable label used in plot titles and table columns."""
        return f"{self.strategy}/{self.model}/{self.gpus}g/{self.placement_class}"

    def short(self) -> str:
        """Compact label used in legend entries and bar-chart x-ticks."""
        return f"{self.strategy}\n{self.model}\n{self.gpus}g\n{self.placement_class}"

class RunMetrics(NamedTuple):
    throughput:     Union[MeasurementStats, None]
    comm_relevance: Union[MeasurementStats, None]

@dataclass
class Baseline:
    system: str
    comm_lib: str
    gpu_model: str
    strategy: Strategy
    model: Model
    gpus: int
    nodes: int
    placement_class: Union[Placement, None]
    
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
            gpus=int(self.gpus),
            placement_class=self.placement_class or Placement.NA
        )
    
    def get_throughput(self):
        """
        Aggregates per-rank iterations using min
        Returns (min, max, median, mean, geomean, std) of these mins
        """
        return self.data.get_throughput_trimmed_mean() if self.data else None
    
    def get_comm_relevance(self):
        """
        Aggregates per-rank iterations using max
        Returns (min, max, median, mean, geomean, std) of these maxes
        """
        return self.data.get_comm_relevance() if self.data else None
    
    # TODO add more class methods
    
    
@dataclass
class ConcurrentRun:
    system: str
    gpus: int
    nodes: int
    job_id: int
    tag: str
    tot_runtime: Union[float, None]
    
    multi_runs: Dict[RunKey, List[RunMeasurements]] = field(init=False)
    slowdowns: Dict[RunKey, List[float]] = field(init=False)
    
    pattern: List[int]
    strategies: List[Strategy]
    placements: List[Placement]
    
    allocation_stats: Union[PlacementStats, None]
    
    def is_in_reservation(self) -> bool:
        # FIXME this is specific for what we have
        # TODO make sure is correct
        by_system = self.system in ['leonardo', 'jupiter', 'intel', 'dgxA100', 'nvl72']
        less_than_three_groups = None
        if self.allocation_stats:
            less_than_three_groups = len(self.allocation_stats.distinct_groups) < 3
        return by_system and (less_than_three_groups is None or less_than_three_groups)
    
    def _summarize_pattern(self) -> str:
        """Summarize the pattern list as a compact string (e.g., '2x8' for [8,8])."""
        if not self.pattern:
            return ""
        counts = Counter(self.pattern)
        return "+".join(f"{count}@{size}" for size, count in sorted(counts.items()))
    
    def get_metrics_tuple(self):
        return self.system, self.gpus, self._summarize_pattern(), self.get_distinct_strategies(), self.get_placement_score()
    
    def get_distinct_strategies(self) -> set[Strategy]:
        return set(self.strategies)
    
    def get_placement_score(self) -> float:
        score = 0.0
        for p in self.placements:
            score += PLACEMENT_CLASS_SCORES[p]
        return score / float(len(self.placements))
    
    def display(self, include_runs=True) -> str:
        runtime = f'{self.tot_runtime:.1f}' if self.tot_runtime else 't/o'
        parts = [
            f"ConcurrentRun {self.system} - {self.job_id} - {self.tag} - {'' if self.is_in_reservation() else 'NO'} resv:",
            f"    tot_gpus={self.gpus}  tot_nodes={self.nodes}  {runtime=}",
        ]
        strategies=self.get_distinct_strategies()
        if strategies:
            parts.append(f'    {len(strategies)=}')
        placement_score=self.get_placement_score()
        if placement_score:
            parts.append(f'    placement_score={placement_score:.1f}')
        if include_runs:
            parts.extend([f'  {k.display():<50} -> {len(v)} repetitions' for k, v in self.multi_runs.items()])
        return '\n'.join(parts)