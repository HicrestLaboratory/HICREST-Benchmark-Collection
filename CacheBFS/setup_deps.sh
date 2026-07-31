#!/bin/bash

for impl in openmp mutex atomic cnalock; do
    git submodule update --init "MergedCSR_$impl"
    cd "MergedCSR_$impl"
    git submodule update --init --recursive distributed_mmio
    cd ..
done

cd MergedCSR_mutex
rm -rf gapbs
git clone https://github.com/sasso0101/gapbs.git

# cd MergedCSR_openmp
# git submodule update --init --recursive

# for impl in mutex atomic cnalock; do
#     if [[ ! -f "../MergedCSR_${impl}/distributed_mmio/README.md" ]]; then
#         rm -rf "../MergedCSR_${impl}/distributed_mmio"
#         ln -s "$(pwd)/distributed_mmio" "../MergedCSR_${impl}/distributed_mmio"
#     fi
# done