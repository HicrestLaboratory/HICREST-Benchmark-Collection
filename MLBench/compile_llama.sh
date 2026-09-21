#!/bin/bash
#
# compile_llama.sh — one build for the whole project: llama-bench (used by
# inference/) and the gemm/gemv/gelu/softmax benchmarks (mlops/). Both link
# against the SAME compiled `ggml` target from ONE cmake configure+build of
# llamacpp/ -- ggml's CPU kernels are only compiled once, not once per
# subfolder.
#
# Usage:
#   ./compile_llama.sh                        # cpu backend (default)
#   ./compile_llama.sh --backend blas         # system BLAS, auto-discovered
#   ./compile_llama.sh --backend blis \
#       --blas-lib /path/libblis.so --blas-include /path/include
#
# --blas-lib/--blas-include are optional (point at a specific local build);
# anything after `--` is forwarded straight to cmake as an escape hatch.

set -euo pipefail
cd "$(dirname "$0")"

backend="cpu"
blas_lib=""
blas_include=""
extra_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)      backend="$2";      shift 2 ;;
        --blas-lib)     blas_lib="$2";     shift 2 ;;
        --blas-include) blas_include="$2"; shift 2 ;;
        --)             shift; extra_args=("$@"); break ;;
        *) echo "error: unknown argument '$1'" >&2; exit 1 ;;
    esac
done

if [[ ! -d llamacpp ]]; then
    echo "error: llamacpp/ not found here — run 'git submodule update --init -- llamacpp' first" >&2
    exit 1
fi

cmake_args=(-DCMAKE_BUILD_TYPE=Release -DBACKEND="$backend")
[[ -n "$blas_lib"     ]] && cmake_args+=(-DBLAS_LIBRARIES="$blas_lib")
[[ -n "$blas_include" ]] && cmake_args+=(-DBLAS_INCLUDE_DIRS="$blas_include")

cmake -B build -S . "${cmake_args[@]}" "${extra_args[@]}"
cmake --build build --target llama-bench gemm gemv gelu softmax -j"$(nproc)"

echo "Built: build/bin/llama-bench, build/mlops/{gemm,gemv,gelu,softmax}"
