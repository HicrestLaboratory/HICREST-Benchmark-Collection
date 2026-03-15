"""
command_map.py
==============
Maps (strategy, num_gpus, comm_lib) -> full DLNetBench command string.
"""

from __future__ import annotations
from experiments_generator import STRATEGY_DEFS

FEASIBLE_GPU_COUNTS: dict[str, frozenset[int]] = {
    strategy[0]: frozenset(strategy[1]) for strategy in STRATEGY_DEFS
}

_EXECUTABLES: dict[str, str] = {
    "DP":           "cpp/data_parallel/dp",
    "FSDP":         "cpp/data_parallel/fsdp",
    "DP+PP":        "cpp/hybrid_parallel/hybrid_2d",
    "DP+PP+Expert": "cpp/hybrid_parallel/hybrid_3d_moe",
    "DP+PP+TP":     "cpp/hybrid_parallel/hybrid_3d",
}

_PARAMS: dict[str, callable] = {
    "DP":           lambda g: "vit-h 50 ./DLNetBench",
    "FSDP":         lambda g: f"llama3-8b 16 {g if g < 8 else 8} ./DLNetBench",
    "DP+PP":        lambda g: f"minerva-7b {2 if g==4 else 4 if g==8 else 8} 16 ./DLNetBench",
    "DP+PP+Expert": lambda g: "mixtral-8x7b 4 16 8 ./DLNetBench",
    "DP+PP+TP":     lambda g: "llama3-70b 80 16 4 ./DLNetBench",
}

_STRATEGIES_NUM_RUNS: dict[str, tuple[int, int]] = {
    "DP":           (1, 5), # 1.1s * 6 = 6.6s
    "FSDP":         (1, 3), # 4s   * 4 = 16s
    "DP+PP":        (1, 3), # 3s   * 4 = 12s
    "DP+PP+Expert": (1, 2), # 45s  * 3 = 2m 15s
    "DP+PP+TP":     (1, 2), # 23s  * 3 = 1m 9s
}

def get_command(strategy: str, num_gpus: int, comm_lib: str, gpu_model:str = "B200") -> str:
    if strategy not in _PARAMS:
        raise ValueError(f"Unknown strategy '{strategy}'. Valid: {sorted(_PARAMS)}")
    if strategy not in _STRATEGIES_NUM_RUNS:
        raise ValueError(f"Unknown strategy number of runs '{strategy}'. Valid: {sorted(_STRATEGIES_NUM_RUNS)}")
    if num_gpus not in FEASIBLE_GPU_COUNTS[strategy]:
        raise ValueError(f"num_gpus={num_gpus} not feasible for '{strategy}'. "
                         f"Valid: {sorted(FEASIBLE_GPU_COUNTS[strategy])}")
    return f"./DLNetBench/bin/{comm_lib}/{_EXECUTABLES[strategy]} {_PARAMS[strategy](num_gpus)} -w {_STRATEGIES_NUM_RUNS[strategy][0]} -r {_STRATEGIES_NUM_RUNS[strategy][1]} -g {gpu_model}"


if __name__ == "__main__":
    tests = [
        ("DP", 2), ("DP", 16),
        ("FSDP", 4), ("FSDP", 8), ("FSDP", 32),
        ("DP+PP", 4), ("DP+PP", 8), ("DP+PP", 16),
        ("DP+PP+Expert", 64), ("DP+PP+TP", 320),
    ]
    print(f"{'Strategy':<15} {'GPUs':>5}  Command")
    print("-" * 100)
    for s, g in tests:
        print(f"{s:<15} {g:>5}  {get_command(s, g, 'nccl')}")
