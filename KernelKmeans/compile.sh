#!/bin/bash

set -e

source ../common/compile/utils.sh

SUPPORTED_SYSTEMS=("bsc-hca")

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <system>"
    exit 1
fi

system="$1"
validate_argument "$system" "system" "${SUPPORTED_SYSTEMS[@]}"

# Define compilers per system
declare -A COMPILERS
# Format: "name:CXX:CC:extra_cmake_flags"

if [[ "$system" == "bsc-hca" ]]; then
    COMPILERS["clang"]="clang:clang++:clang:-DEXTRA_LIBS=flang_rt.runtime"
    COMPILERS["gcc"]="gcc:g++:gcc:"
else
    # Default
    COMPILERS["gcc"]="gcc:g++:gcc:"
fi

# ---- FUNCTIONS ----

load_modules() {
    local compiler="$1"
    echo

    if [[ "$system" == "bsc-hca" ]]; then
        module purge

        if [[ "$compiler" == "clang" ]]; then
            ml llvm/EPI-development cmake/3.28.1 openBLAS/ubuntu/0.3.29_llvmEPI1.0
        elif [[ "$compiler" == "gcc" ]]; then
            ml cmake/3.28.1 openBLAS/ubuntu/0.3.20_gcc10.3.0
        fi
    fi
}

get_openblas_dir() {
    if [[ -z "${OBLAS_LIBS:-}" ]]; then
        echo "ERROR: OBLAS_LIBS is not set. Did you load the module?"
        exit 1
    fi

    # Extract path after -L
    local lib_path="${OBLAS_LIBS#-L}"

    # Construct cmake dir
    local cmake_path="${lib_path}/cmake/openblas"

    if [[ ! -d "$cmake_path" ]]; then
        echo "ERROR: OpenBLAS CMake directory not found at $cmake_path"
        exit 1
    fi

    echo "$cmake_path"
}

build_target() {
    local name="$1"
    local cxx="$2"
    local cc="$3"
    local extra_flags="$4"

    build_dir="build_${system}_${name}"

    echo
    echo
    echo "==== Building with $name ===="
    echo "Build directory: $build_dir"

    rm -rf "$build_dir"

    OPENBLAS_DIR=$(get_openblas_dir)

    cmake \
        -DCMAKE_CXX_COMPILER="$cxx" \
        -DCMAKE_C_COMPILER="$cc" \
        -DOpenBLAS_DIR="$OPENBLAS_DIR" \
        $extra_flags \
        -B "$build_dir"

    cmake --build "$build_dir" --target popcornkmeans_openblas -j
    cmake --build "$build_dir" --target popcornkmeans_openmp -j
}

# ---- MAIN LOOP ----

for key in "${!COMPILERS[@]}"; do
    IFS=":" read -r name cxx cc extra <<< "${COMPILERS[$key]}"

    load_modules "$name"
    build_target "$name" "$cxx" "$cc" "$extra"
done

echo "All builds completed."