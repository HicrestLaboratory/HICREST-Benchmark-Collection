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
#   ./compile_llama.sh --build-dir build_cuda # custom build directory
#   ./compile_llama.sh --backend blas         # system BLAS, auto-discovered
#   ./compile_llama.sh --backend blis \
#       --blas-lib /path/libblis.so --blas-include /path/include
#
# --blas-lib/--blas-include are optional (point at a specific local build);
# anything after `--` is forwarded straight to cmake as an escape hatch.

set -euo pipefail
cd "$(dirname "$0")"

backend="cpu"
vendor_name=""
blas_lib=""
blas_include=""
build_dir="build"
extra_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)      backend="$2";      shift 2 ;;
        --blas-lib)     blas_lib="$2";     shift 2 ;;
        --blas-include) blas_include="$2"; shift 2 ;;
        --build-dir)    build_dir="$2";    shift 2 ;;
        --)             shift; extra_args=("$@"); break ;;
        *) echo "error: unknown argument '$1'" >&2; exit 1 ;;
    esac
done

if [[ "$backend" == "openblas" ]]; then
    backend="blas"
    vendor_name="OpenBLAS"
fi
if [[ "$backend" == "none" ]]; then
    backend="cpu"
    vendor_name=""
fi
if [[ "$backend" == "blis" ]]; then
    backend="blis"
    vendor_name="FLAME"
fi

if [[ ! -d llamacpp ]]; then
    echo "error: llamacpp/ not found here — run 'git submodule update --init -- llamacpp' first" >&2
    exit 1
fi

cmake_args=(-DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_BUILD_TYPE=Release -DBACKEND="$backend")
[[ -n "$blas_lib"     ]] && cmake_args+=(-DGGML_BLAS=ON -DGGML_BLAS_VENDOR="$vendor_name" -DBLAS_LIBRARIES="$blas_lib" -DCMAKE_INSTALL_RPATH="$(dirname "$blas_lib")" -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_EXE_LINKER_FLAGS="-L$(dirname "$blas_lib") -Wl,-rpath=$(dirname "$blas_lib") -Wl,-rpath-link=$(dirname "$blas_lib")")
[[ -n "$blas_include" ]] && cmake_args+=(-DGGML_BLAS=ON -DGGML_BLAS_VENDOR="$vendor_name" -DBLAS_INCLUDE_DIRS="$blas_include"  -DCMAKE_INSTALL_RPATH="$(dirname "$blas_lib")" -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON)
  
echo "Running:"
echo "cmake -B $build_dir -S . ${cmake_args[@]} ${extra_args[@]}"
echo "cmake --build $build_dir --target llama-bench gemm gemv gelu softmax -j$(nproc)"

cmake -B "$build_dir" -S . "${cmake_args[@]}" "${extra_args[@]}"
cmake --build "$build_dir" --target llama-bench gemm gemv gelu softmax -j"$(nproc)"

echo "Built: ${build_dir}/bin/llama-bench, ${build_dir}/mlops/{gemm,gemv,gelu,softmax}"