#!/bin/bash

set -e
SUPPORTED_SYSTEMS=("leonardo")

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
cd ../../AXCCL

echo "JobPlacer Ready!"


echo "==== Compiling AXCCL for $system ===="
[[ -d "hicrest-axccl" ]] || { echo "Error: hicrest-axccl not found. Please get the submodule (from the repo root: \`git submodule update --init AXCCL/hicrest-axccl\`)."; exit 1; }

cd hicrest-axccl
./init.sh --system $system
source "configure/${system^^}_DEFAULT.conf"
make pingpong p2p a2a ar

echo "HICREST AXCCL Ready!"
