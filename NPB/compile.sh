#!/bin/bash

set -e

source ../common/compile/utils.sh

SUPPORTED_SYSTEMS=("leonardo" "bsc")

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <system>"
    exit 1
fi

system="$1"
validate_argument "$system" "system" "${SUPPORTED_SYSTEMS[@]}"

# Relative path to the NPB OpenMP directory
NPB_DIR="NPB/NPB3.4/NPB3.4-OMP"

# List of benchmarks (match directory names exactly)
BENCHMARKS=("BT" "CG" "EP" "FT" "IS" "LU" "SP" "MG")
CLASS="C"

echo "Compiling NPB Class $CLASS..."

sed -i "s/X'/Z'/g" "${NPB_DIR}/common/randi8.f"
cp "make.def.${system^^}" "${NPB_DIR}/config/make.def"
mkdir -p "${NPB_DIR}/bin"

# Change to the NPB directory
cd "$NPB_DIR" || { echo "Failed to change directory to $NPB_DIR"; exit 1; }

for bm in "${BENCHMARKS[@]}"; do
    echo "Compiling $bm (Class $CLASS)"
    make "$bm" CLASS="$CLASS"
done

echo "Done."