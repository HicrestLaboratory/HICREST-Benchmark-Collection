#!/bin/bash

set -e

BIN_DIR=../../bin

cd NetGraph500
if [[ ! -d ../../common/ccutils/ ]]; then
    echo "ccutils not found (common/ccutils). Please download it with 'git submodule update --init ccutils.'"
    exit 1
fi

cd src
mkdir -p $BIN_DIR

# Standard benchmark
make deep_clean
CFLAGS="-DBENCHPIN" PREPROCESSOR_FLAGS="-DAGGR_intra=32768 -DAGGR=32768" make graph500_reference_bfs
mv graph500_reference_bfs $BIN_DIR/graph500_bfs_32KiB

# Benchmarks that flushes the buffer more often: after 512 vertices instead of 8192
make clean
CFLAGS="-DBENCHPIN" PREPROCESSOR_FLAGS="-DAGGR_intra=2048 -DAGGR=2048" make graph500_reference_bfs
mv graph500_reference_bfs $BIN_DIR/graph500_bfs_2KiB

# Benchmarks that flushes the buffer less often
make clean
CFLAGS="-DBENCHPIN" PREPROCESSOR_FLAGS="-DAGGR_intra=262144 -DAGGR=262144" make graph500_reference_bfs
mv graph500_reference_bfs $BIN_DIR/graph500_bfs_256KiB

# Benchmarks that flushes the buffer even less often
make clean
CFLAGS="-DBENCHPIN" PREPROCESSOR_FLAGS="-DAGGR_intra=8388608 -DAGGR=8388608" make graph500_reference_bfs
mv graph500_reference_bfs $BIN_DIR/bin/graph500_bfs_8MiB

echo "INFO: currently only BFS implements custom metric correctly."