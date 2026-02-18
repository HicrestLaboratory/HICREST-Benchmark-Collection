#!/bin/bash

set -e
SUPPORTED_SYSTEMS=("cpu" "leonardo" "lumi" "baldo")

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <system>"
    exit 1
fi

system="$1"

if [[ ! " ${SUPPORTED_SYSTEMS[@]} " =~ " ${system} " ]]; then
    echo "Error: Unsupported system '$system'. Supported systems: ${SUPPORTED_SYSTEMS[*]}"
    exit 1
fi


echo "==== Setting up JobPlacer ===="
echo "This requires a Rust compiler. If you don't have it, install it locally running:"
echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
[[ -d "../common/JobPlacer" ]] || { echo "Error: JobPlacer not found. Please get the submodule (from the repo root: \`git submodule update --init common/JobPlacer\`)."; exit 1; }

cd ../common/JobPlacer
cargo build --release
cd ../../DLNetBench

echo "JobPlacer Ready!"


echo "==== Compiling DLNetBench for $system ===="
[[ -d "DLNetBench" ]] || { echo "Error: DLNetBench not found. Please get the submodule (from the repo root: \`git submodule update --init DLNetBench/DLNetBench\`)."; exit 1; }

cd DLNetBench/cpp


case "$system" in
    leonardo)
        echo "Compiling for Leonardo"
        make -f Makefile.LEONARDO all
        ;;
    baldo)
        echo "Compiling for Baldo"
        make -f Makefile.BALDO all
        ;;
    lumi)
        echo "Compiling for Lumi"
        make -f Makefile.LUMI all
        ;;
    cpu)
        echo "Compiling for CPU-only system"
        make -f Makefile all
        ;;
esac

echo "HICREST DLNetBench Ready!"