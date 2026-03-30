"""
command_map.py
==============
Maps (strategy, num_gpus, comm_lib) -> full DLNetBench command string.
"""

from __future__ import annotations
from typing import List, Union
from experiments_generator import STRATEGY_DEFS, STRATEGY_DEFS_DGX_A100, STRATEGY_DEFS_EXTENDED

EXTRA_SRUN_FLAGS = {
    'alps': ['--mpi=pmix', '--cpu-bind=cores', '--accel-bind=g']
}

FEASIBLE_GPU_COUNTS: dict[str, frozenset[int]] = {
    strategy[0]: frozenset(strategy[1]) for strategy in STRATEGY_DEFS_EXTENDED
}

FEASIBLE_GPU_COUNTS_DGX_A100: dict[str, frozenset[int]] = {
    strategy[0]: frozenset(strategy[1]) for strategy in STRATEGY_DEFS_DGX_A100
}

_EXECUTABLES: dict[str, str] = {
    "DP":           "cpp/data_parallel/dp",
    "FSDP":         "cpp/data_parallel/fsdp",
    "DP+PP":        "cpp/hybrid_parallel/hybrid_2d",
    "DP+PP+TP":     "cpp/hybrid_parallel/hybrid_3d",
    "DP+PP+Expert": "cpp/hybrid_parallel/hybrid_3d_moe",
}

_STRATEGY_MODELS_MAP: dict[str, Union[str, List[str]]] = {
    "DP":           ["vit-h"],
    "FSDP":         ["minerva-7b"], # "llama3-8b"],
    "DP+PP":        ["llama3-8b"],  # "minerva-7b"],
    "DP+PP+TP":     ["llama3-70b"],
    "DP+PP+Expert": ["mixtral-8x7b"],
}

def get_default_model(strategy: str) -> str:
    model = _STRATEGY_MODELS_MAP[strategy]
    if isinstance(model, str):
        return model
    return model[0]

_PARAMS: dict[str, callable] = {
    "DP":           lambda g: [f"{m} 50 ./DLNetBench" for m in _STRATEGY_MODELS_MAP['DP']],
    "FSDP":         lambda g: [f"{m} 16 {g if g < 8 else 8} ./DLNetBench" for m in _STRATEGY_MODELS_MAP['FSDP']],
    "DP+PP":        lambda g: [f"{m} {2 if g <= 8 else 8} 16 ./DLNetBench" for m in _STRATEGY_MODELS_MAP['DP+PP']],
    "DP+PP+TP":     lambda g: [f"{m} 8 16 4 ./DLNetBench" for m in _STRATEGY_MODELS_MAP['DP+PP+TP']],
    "DP+PP+Expert": lambda g: [f"{m} 8 16 8 ./DLNetBench" for m in _STRATEGY_MODELS_MAP['DP+PP+Expert']],
}


_STRATEGIES_NUM_RUNS: dict[str, tuple[int, int]] = {
    "DP":           (1, 6), # 1.1s * 6 = 6.6s
    "FSDP":         (1, 4), # 4s   * 4 = 16s
    "DP+PP":        (1, 4), # 3s   * 4 = 12s
    "DP+PP+Expert": (1, 2), # 45s  * 3 = 2m 15s
    "DP+PP+TP":     (1, 2), # 23s  * 3 = 1m 9s
}

_STRATEGIES_NUM_RUNS_BX00: dict[str, tuple[int, int]] = {
    "DP":           (1, 10), # 1.1s * 6 = 6.6s
    "FSDP":         (1, 6),  # 4s   * 4 = 16s
    "DP+PP":        (1, 6),  # 3s   * 4 = 12s
    "DP+PP+Expert": (1, 4),  # 45s  * 3 = 2m 15s
    "DP+PP+TP":     (1, 4),  # 23s  * 3 = 1m 9s
}

def get_command(strategy: str, num_gpus: int, comm_lib: str, gpu_model: str, num_warmup_override: Union[int, None]=None, use_dgx:bool=False) -> List[str]:
    if strategy not in _PARAMS:
        raise ValueError(f"Unknown strategy '{strategy}'. Valid: {sorted(_PARAMS)}")
    
    if strategy not in _STRATEGIES_NUM_RUNS:
        raise ValueError(f"Unknown strategy number of runs '{strategy}'. Valid: {sorted(_STRATEGIES_NUM_RUNS)}")
    
    if use_dgx:
        if num_gpus not in FEASIBLE_GPU_COUNTS_DGX_A100[strategy]:
            raise ValueError(f"num_gpus={num_gpus} not feasible for '{strategy}'. "
                             f"Valid: {sorted(FEASIBLE_GPU_COUNTS_DGX_A100[strategy])}")
    else:
        if num_gpus not in FEASIBLE_GPU_COUNTS[strategy]:
            raise ValueError(f"num_gpus={num_gpus} not feasible for '{strategy}'. "
                             f"Valid: {sorted(FEASIBLE_GPU_COUNTS[strategy])}")

    if gpu_model in ["B200", "GB300", "GB200"]:
        num_runs = _STRATEGIES_NUM_RUNS_BX00[strategy]
    else:
        num_runs = _STRATEGIES_NUM_RUNS[strategy]
        
    if num_warmup_override is not None and num_warmup_override >= 0:
        num_runs = (num_warmup_override, num_runs[0] + num_runs[1])
        
    params = _PARAMS[strategy](num_gpus)
    if not isinstance(params, list):
        params = [params]
        
    return [f"./DLNetBench/bin/{comm_lib}/{_EXECUTABLES[strategy]} {par} -w {num_runs[0]} -r {num_runs[1]} -g {gpu_model}" for par in params]


def get_model_from_command(command: str):
    tokens = command.split()

    # Flatten all possible model names
    models = set()
    for v in _STRATEGY_MODELS_MAP.values():
        if isinstance(v, list):
            models.update(v)
        else:
            models.add(v)

    # Find the first token that matches a known model
    for t in tokens:
        if t in models:
            return t

    return None


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
        print(f"{s:<15} {g:>5}  {get_command(s, g, 'nccl', 'H200')}")
