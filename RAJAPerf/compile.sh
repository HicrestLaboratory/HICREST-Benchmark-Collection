#!/bin/bash

set -e

## !! Please run this from a Pioneer board !!
# module load cmake/3.28.1 
# module load llvm/EPI-development
cd RAJAPerf
# git submodule update --init --recursive
git submodule update --init --recursive --depth 1
RAJAPERF_SRC="$(cd "$(dirname "$0")" && pwd)"
RAJAPERF_BUILD="${RAJAPERF_SRC}/build"

mkdir -p "$RAJAPERF_BUILD"
cd "$RAJAPERF_BUILD"

cmake .. \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_OPENMP=On \
  -DRAJA_ENABLE_OPENMP=On \
  -DRAJA_ENABLE_CUDA=Off \
  -DRAJA_ENABLE_HIP=Off \
  -DRAJA_ENABLE_TARGET_OPENMP=Off \
  -DRAJA_PERFSUITE_USE_CALIPER=Off \
  -DCMAKE_CXX_FLAGS="-march=rv64gcv -O3 -ffast-math"

  # -DCMAKE_CXX_FLAGS="-march=znver4 -mavx512f -O3"


make -j"$(nproc)"