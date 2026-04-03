from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
import re
from statistics import median
from typing import Dict, List, NamedTuple, Tuple, Union
import pandas as pd
from experiments_generator import PLACEMENT_CLASS_DEFS

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
    'jupiter':      'Jupiter',
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
            Strategy.DP_PP: "D+P",
            Strategy.DP_PP_TP: "D+P+T",
            Strategy.DP_PP_EXPERT: "D+P+E",
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
    
    def display(self, new_line=True):
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


# FOR PLOTS

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

    def get_throughput(self, skip_first_n: int = 1) -> Union[Tuple[float, float, float], None]:
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
    throughput:     Union[Tuple[float, float, float], None]
    comm_relevance: Union[Tuple[float, float, float], None]

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
        Returns (min, max, median) of these mins
        """
        return self.data.get_throughput() if self.data else None
    
    def get_comm_relevance(self):
        """
        Aggregates per-rank iterations using max
        Returns (min, max, median) of these maxes
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
    
    in_reservation: bool
    
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
            f"ConcurrentRun {self.system} - {self.job_id} - {self.tag}:",
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