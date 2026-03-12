"""
command_map.py
==============
Maps (strategy, num_gpus, comm_lib) -> full DLNetBench command string.
"""

from __future__ import annotations


FEASIBLE_GPU_COUNTS: dict[str, frozenset[int]] = {
    "DP":           frozenset([2, 4, 8, 16]),
    "FSDP":         frozenset([2, 4, 8, 16, 32]),
    "DP+PP":        frozenset([4, 8, 16, 32, 64]),
    "DP+PP+Expert": frozenset([64, 128, 192, 256, 320, 384, 448, 512]),
    "DP+PP+TP":     frozenset([320, 640, 960]),
}

_EXECUTABLES: dict[str, str] = {
    "DP":           "cpp/data_parallel/dp",
    "FSDP":         "cpp/data_parallel/fsdp",
    "DP+PP":        "cpp/hybrid_parallel/hybrid_2d",
    "DP+PP+Expert": "cpp/hybrid_parallel/hybrid_3d_moe",
    "DP+PP+TP":     "cpp/hybrid_parallel/hybrid_3d",
}

_PARAMS: dict[str, callable] = {
    "DP":           lambda g: "vit_h_16_bfloat16 50 ./DLNetBench",
    "FSDP":         lambda g: f"llama3_8b_16_bfloat16 16 {g if g < 8 else 8} ./DLNetBench",
    "DP+PP":        lambda g: f"minerva_7b_16_bfloat16 {2 if g==4 else 4 if g==8 else 8} 16 ./DLNetBench",
    "DP+PP+Expert": lambda g: "mixtral_8x7b_16_bfloat16 4 16 8 ./DLNetBench",
    "DP+PP+TP":     lambda g: "llama3_70b_16_bfloat16 80 16 4 ./DLNetBench",
}


def get_command(strategy: str, num_gpus: int, comm_lib: str) -> str:
    return f"echo '{strategy} with {num_gpus} GPUs and {comm_lib} comm_lib'"
    if strategy not in _PARAMS:
        raise ValueError(f"Unknown strategy '{strategy}'. Valid: {sorted(_PARAMS)}")
    if num_gpus not in FEASIBLE_GPU_COUNTS[strategy]:
        raise ValueError(f"num_gpus={num_gpus} not feasible for '{strategy}'. "
                         f"Valid: {sorted(FEASIBLE_GPU_COUNTS[strategy])}")
    return f"./DLNetBench/bin/{comm_lib}/{_EXECUTABLES[strategy]} {_PARAMS[strategy](num_gpus)}"


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