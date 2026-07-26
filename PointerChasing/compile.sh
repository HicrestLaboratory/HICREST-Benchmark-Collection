#!/bin/bash

set -e

source ../common/compile/utils.sh
source ../common/compile/compilers.sh

SUPPORTED_SYSTEMS=("bsc-hca" "thea" "leonardo" "e4")

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <system>"
    exit 1
fi

system="$1"

validate_argument "$system" "system" "${SUPPORTED_SYSTEMS[@]}"

root_dir=$(pwd)


build() {
    local name="$1"
    local cxx="$2"
    local cc="$3"
    local extra_libs="$4"
    local extra_link_opts="$5"

    echo
    echo "==== Building PointerChasing with $name ===="
    echo "Build directory: $build_dir"

    cd pointer-chasing

    build_dir="bin/${name}"
    mkdir -p "$build_dir"

    echo "Compiling with command:"
    echo "make BINDIR=$build_dir CXX=$cxx -j$(nproc)"

    make clean
    make BINDIR=$build_dir CXX=$cxx -j"$(nproc)"

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
    
    build "$name" "$cxx" "$cc" "$libs" "$link_opts"
done

[[ -d bin ]] && rm -r bin
mv pointer-chasing/bin .

echo "PointerChasing build completed."
