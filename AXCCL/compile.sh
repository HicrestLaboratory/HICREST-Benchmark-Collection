#!/bin/bash

set -e
SUPPORTED_SYSTEMS=("leonardo" "isarco" "alps")

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

if ! [ -x "$(command -v cargo)" ]; then
    echo "This requires a Rust compiler. If you don't have it, install it locally running:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi
[[ -d "../common/JobPlacer" ]] || { echo "Error: JobPlacer not found. Please get the submodule (from the repo root: \`git submodule update --init common/JobPlacer\`)."; exit 1; }

cd ../common/JobPlacer
cargo build --release
cd ../../AXCCL

echo "JobPlacer Ready!"


echo "==== Compiling AXCCL for $system ===="
[[ -d "hicrest-axccl" ]] || { echo "Error: hicrest-axccl not found. Please get the submodule (from the repo root: \`git submodule update --init AXCCL/hicrest-axccl\`)."; exit 1; }

TARGETS="pingpong p2p a2a ar"
cd hicrest-axccl

if [[ $system == "alps" ]]; then
    if [[ ! -d "/user-environment/linux-neoverse_v2" ]]; then
        # Note: NCCL 22.3 does not have A2A
        echo "Please pull the uenv image: 'uenv image pull prgenv-gnu/25.6:v2'"
        echo "Then, start a new session with that image: 'uenv start prgenv-gnu/25.6:v2 --view=modules'"
        exit 1
    fi
fi

./init.sh --system $system
source "configure/${system^^}_DEFAULT.conf"
make $TARGETS

echo "HICREST AXCCL Ready!"
echo "AXCCL binaries are in: 'hicrest-axccl/bin'"
