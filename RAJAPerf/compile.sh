#!/bin/bash

set -e

source ../common/compile/utils.sh
source ../common/compile/compilers.sh

SUPPORTED_SYSTEMS=("bsc-hca" "thea" "leonardo")
SUPPORTED_BOARDS=("default" "pioneer" "arriesgado" "bananaf3")

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <system> [<board>]"
    exit 1
fi

system="$1"
board="${2:-default}"

validate_argument "$system" "system" "${SUPPORTED_SYSTEMS[@]}"
validate_argument "$board" "board" "${SUPPORTED_BOARDS[@]}"

check_ccutils_installation

root_dir=$(pwd)

declare -A ARCH
ARCH["leonardo"]="native"
ARCH["thea"]="native"
ARCH["bsc-hca"]="rv64gcv"


build_rajaperf() {
    local name="$1"
    local cxx="$2"
    local cc="$3"
    local extra_libs="$4"
    local extra_link_opts="$5"

    build_dir="build_${name}_${board}"

    echo
    echo "==== Building RAJAPerf with $name ===="
    echo "Build directory: $build_dir"

    # Navigate to RAJAPerf directory
    cd RAJAPerf

    # Initialize submodules with depth 1 for faster cloning
    git submodule update --init --recursive --depth 1

    # Get source directory and set build path relative to RAJAPerf
    rajaperf_src="$(pwd)"
    rajaperf_build="${rajaperf_src}/../${build_dir}"

    mkdir -p "$rajaperf_build"
    cd "$rajaperf_build"

    echo "Compiling with command:"
    echo cmake "$rajaperf_src" \
        -DCMAKE_CXX_COMPILER="$cxx" \
        -DCMAKE_C_COMPILER="$cc" \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_OPENMP=On \
        -DRAJA_ENABLE_OPENMP=On \
        -DRAJA_ENABLE_CUDA=Off \
        -DRAJA_ENABLE_HIP=Off \
        -DRAJA_ENABLE_TARGET_OPENMP=Off \
        -DRAJA_PERFSUITE_USE_CALIPER=Off \
        -DCMAKE_CXX_FLAGS="-march=${ARCH[$system]} -O3 -ffast-math"

    cmake "$rajaperf_src" \
        -DCMAKE_CXX_COMPILER="$cxx" \
        -DCMAKE_C_COMPILER="$cc" \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_OPENMP=On \
        -DRAJA_ENABLE_OPENMP=On \
        -DRAJA_ENABLE_CUDA=Off \
        -DRAJA_ENABLE_HIP=Off \
        -DRAJA_ENABLE_TARGET_OPENMP=Off \
        -DRAJA_PERFSUITE_USE_CALIPER=Off \
        -DCMAKE_CXX_FLAGS="-march=${ARCH[$system]} -O3 -ffast-math"

    make -j"$(nproc)"

    # Return to original directory
    cd $root_dir
}

# ---- MAIN ----

compilers_var=$(get_compilers_for_system "$system")
declare -n compilers="$compilers_var"

for compiler_name in "${!compilers[@]}"; do
    config=$(get_compiler_config "$system" "$compiler_name")
    parse_compiler_config "$config" name cxx cc modules libs link_opts

    load_modules_for_compiler "$modules"
    set_compiler_env "$cxx" "$cc"
    
    build_rajaperf "$name" "$cxx" "$cc" "$libs" "$link_opts"
done

echo "RAJAPerf build completed."