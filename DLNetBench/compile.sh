#!/bin/bash

set -e

source ../common/compile/utils.sh

SUPPORTED_SYSTEMS=("cpu" "leonardo" "lumi" "baldo" "alps" "isarco")

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <system>"
    exit 1
fi

system="$1"
validate_argument "$system" "system" "${SUPPORTED_SYSTEMS[@]}"

# Setup JobPlacer (skip for nvidia and baldo)
if [[ "$system" != "nvidia" && "$system" != "baldo" ]]; then
    . ../common/compile/setup_job_placer.sh
else
    echo "Skipping JobPlacer for $system"
fi

check_ccutils_installation

echo "==== Compiling DLNetBench for ${system^^} ===="
[[ -d "DLNetBench" ]] || { echo "Error: DLNetBench not found. Please get the submodule (from the repo root: \`git submodule update --init DLNetBench/DLNetBench\`)."; exit 1; }

cd DLNetBench

if [[ $system == "alps" ]]; then
    check_alps_uenv
fi

make -f "Makefile.${system^^}" clean
make -f "Makefile.${system^^}"

echo "HICREST DLNetBench Ready!"
echo "Binaries saved in: 'DLNetBench/bin'"
