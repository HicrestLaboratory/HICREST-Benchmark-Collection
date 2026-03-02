#!/bin/bash

set -e

source ../common/compile/utils.sh

SUPPORTED_SYSTEMS=("cpu" "leonardo" "lumi" "baldo" "alps")

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <system>"
    exit 1
fi

system="$1"
validate_argument "$system" "system" "${SUPPORTED_SYSTEMS[@]}"

# Setup JobPlacer
. ../common/compile/setup_job_placer.sh

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