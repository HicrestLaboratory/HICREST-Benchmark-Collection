#!/bin/bash

# Relative path to the NPB OpenMP directory
NPB_DIR="NPB3.4/NPB3.4-OMP"

# List of benchmarks (match directory names exactly)
BENCHMARKS=("BT" "CG" "EP" "FT" "IS" "LU" "SP" "MG")
CLASS="C"

echo "Compiling NPB Class $CLASS..."

# Change to the NPB directory
cd "$NPB_DIR" || { echo "Failed to change directory to $NPB_DIR"; exit 1; }

for bm in "${BENCHMARKS[@]}"; do
    echo "Compiling $bm (Class $CLASS)"
    make "$bm" CLASS="$CLASS"
done

echo "Done."