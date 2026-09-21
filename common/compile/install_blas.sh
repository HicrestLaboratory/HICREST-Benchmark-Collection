#!/bin/bash

ARCH="$1"

OPENBLAS_VERSION="v0.3.34"
BLIS_VERSION="2.1"

#Toolchain and variables setup
if [ "$ARCH" == "sg2044" ]; then
    OPENBLAS_TARGET="RISCV64_ZVL128B"
    BLIS_TARGET="rv64iv"

    CC="${HOME}/software_env/llvm-EPI/riscv64/rvv1/bin/clang"
    CXX="${HOME}/software_env/llvm-EPI/riscv64/rvv1/bin/clang++"
    EPI_INC="${HOME}/software_env/llvm-EPI/riscv64/rvv1/include"
    EPI_LIB="${HOME}/software_env/llvm-EPI/riscv64/rvv1/lib"
    GCC_LIB="/usr/lib/gcc/riscv64-linux-gnu/13/"

    OPENBLAS_MAKE_OPTS="CC=${CC} -B${GCC_LIB} -L${GCC_LIB} -L${EPI_LIB} -I${EPI_INC} NOFORTRAN=1 USE_OPENMP=1 TARGET=$OPENBLAS_TARGET"
    BLIS_CONFIG_OPTS="CXX=${CXX} CC=${CC} CFLAGS=-I${EPI_INC} LDFLAGS=-B${GCC_LIB} -L${GCC_LIB} -L${EPI_LIB}"

elif [ "$ARCH" == "sg2042" ]; then
    CC="${HOME}/software_env/llvm-EPI/riscv64/rvv071/bin/clang"
    CXX="${HOME}/software_env/llvm-EPI/riscv64/rvv071/bin/clang++"

elif [ "$ARCH" == "grace" ]; then
    OPENBLAS_TARGET="NEOVERSEV2"
    BLIS_TARGET="auto"

    OPENBLAS_MAKE_OPTS="USE_OPENMP=1 TARGET=$OPENBLAS_TARGET"
    BLIS_CONFIG_OPTS=""

elif [ "$ARCH" == "sapphirerapids" ]; then
    OPENBLAS_TARGET="SAPPHIRERAPIDS"
    BLIS_TARGET="auto"

    OPENBLAS_MAKE_OPTS="USE_OPENMP=1 TARGET=$OPENBLAS_TARGET"
    BLIS_CONFIG_OPTS=""
else
    echo "No special vars for this architecture"
    BLIS_TARGET="auto"
    
    OPENBLAS_MAKE_OPTS="USE_OPENMP=1"
    BLIS_CONFIG_OPTS=""
fi



#OpenBLAS
git clone https://github.com/OpenMathLib/OpenBLAS.git
cd OpenBLAS
git checkout ${OPENBLAS_VERSION}
make ${OPENBLAS_MAKE_OPTS} -j
make install PREFIX="${HOME}/software_env/OpenBLAS-${ARCH}" ${OPENBLAS_MAKE_OPTS}
cd ..

#BLIS
git clone https://github.com/flame/blis.git
cd blis
git checkout ${BLIS_VERSION}
./configure --prefix=${HOME}/software_env/blis-${ARCH} ${BLIS_CONFIG_OPTS} --enable-multithreading=openmp --enable-cblas ${BLIS_TARGET}
make -j
make install
cd ..
