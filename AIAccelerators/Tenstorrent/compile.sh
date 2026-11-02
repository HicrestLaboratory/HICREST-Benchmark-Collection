#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/../AI-accelerators-benchmark/tenstorrent"
BUILD_DIR="${SOURCE_DIR}/build"

source "${SCRIPT_DIR}/../../common/compile/utils.sh"

print_header "Building the Tenstorrent benchmarks"

check_command cmake
check_file "${SOURCE_DIR}/CMakeLists.txt"
check_file "${SOURCE_DIR}/third_party/argparse/include/argparse/argparse.hpp"

: "${TT_METAL_HOME:?TT_METAL_HOME must point to the tt-metal checkout}"
: "${TT_METAL_RUNTIME_ROOT:?TT_METAL_RUNTIME_ROOT must be set}"

cmake \
    -S "${SOURCE_DIR}" \
    -B "${BUILD_DIR}" \
    -DTT_METAL_HOME="${TT_METAL_HOME}" \
    -DTT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT}"

cmake --build "${BUILD_DIR}" --parallel 8

check_file "${BUILD_DIR}/bin/gemm"
check_file "${BUILD_DIR}/bin/non_linearity"

print_success "Tenstorrent benchmarks built in ${BUILD_DIR}/bin"

