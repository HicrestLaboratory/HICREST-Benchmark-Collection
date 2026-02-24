#!/bin/bash

set -e

source ../common/compile/utils.sh

SUPPORTED_SYSTEMS=("leonardo" "isarco" "alps")

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <system>"
    exit 1
fi

system="$1"
validate_argument "$system" "system" "${SUPPORTED_SYSTEMS[@]}"

. ../common/compile/setup_job_placer.sh

echo "==== Compiling AXCCL for $system ===="
[[ -d "hicrest-axccl" ]] || { echo "Error: hicrest-axccl not found. Please get the submodule (from the repo root: \`git submodule update --init AXCCL/hicrest-axccl\`)."; exit 1; }

TARGETS="pingpong p2p a2a ar"
cd hicrest-axccl

if [[ $system == "alps" ]]; then
    check_alps_uenv
fi

./init.sh --system $system
source "configure/${system^^}_DEFAULT.conf"
make $TARGETS

echo "HICREST AXCCL Ready!"
echo "AXCCL binaries are in: 'hicrest-axccl/bin'"
