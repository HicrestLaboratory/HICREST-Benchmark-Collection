#!/bin/bash

ARCH="$1"

OPENBLAS_VERSION="v0.3.34"
BLIS_VERSION="2.1"

#Toolchain and variables setup
if [ "$ARCH" == "sg2044" ]; then
    OPENBLAS_TARGET="RISCV64_ZVL128B"
    BLIS_TARGET="rv64iv"

    CC="${HOME}/software_env/llvm-EPI-rvv1/bin/clang"
    CXX="${HOME}/software_env/llvm-EPI-rvv1/bin/clang++"
    EPI_INC="${HOME}/software_env/llvm-EPI-rvv1/include"
    EPI_LIB="${HOME}/software_env/llvm-EPI-rvv1/lib"
    GCC_LIB="/usr/lib/gcc/riscv64-openEuler-linux/12"

elif [ "$ARCH" == "sg2042" ]; then
    CC="${HOME}/software_env/llvm-EPI-rvv071/bin/clang"
    CXX="${HOME}/software_env/llvm-EPI-rvv071/bin/clang++"
else
    echo "Unknown architecture: $ARCH"
    exit 1
fi

if [ "$ARCH" == "sg2044" ]; then
    #OpenBLAS
    git clone https://github.com/OpenMathLib/OpenBLAS.git
    cd OpenBLAS
    git checkout ${OPENBLAS_VERSION}
    make CC="${CC} -B${GCC_LIB} -L${GCC_LIB} -L${EPI_LIB} -I${EPI_INC}" NOFORTRAN=1 USE_OPENMP=1 TARGET=$OPENBLAS_TARGET -j
    make install PREFIX="${HOME}/software_env/OpenBLAS-${ARCH}" CC="${CC} -B${GCC_LIB} -L${GCC_LIB} -L${EPI_LIB} -I${EPI_INC}" NOFORTRAN=1 USE_OPENMP=1 TARGET=RISCV64_ZVL128B
    cd ..

    #BLIS
    git clone https://github.com/flame/blis.git
    cd blis
    git checkout ${BLIS_VERSION}
    ./configure --prefix=${HOME}/software_env/blis-${ARCH} CXX="${CXX}" CC="${CC}" CFLAGS="-I{EPI_INC}" LDFLAGS="-B${GCC_LIB} -L${GCC_LIB} -L${EPI_LIB}" --enable-multithreading=openmp --enable-cblas ${BLIS_TARGET}
    make -j
    make install
    cd ..

elif [ "ARCH" == "sg2042" ]; then
    git clone https://github.com/OpenMathLib/OpenBLAS.git
    cd OpenBLAS
    git checkout ${OPENBLAS_VERSION}
    echo "For now nothing"
fi