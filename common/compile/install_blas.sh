#!/bin/bash
set -euo pipefail

ARCH=""
SKIP_OPENBLAS="${SKIP_OPENBLAS:-0}"
SKIP_BLIS="${SKIP_BLIS:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-openblas)
            SKIP_OPENBLAS=1
            shift
            ;;
        --skip-blis)
            SKIP_BLIS=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 <architecture> [--skip-openblas] [--skip-blis]"
            echo
            echo "Supported architectures:"
            echo "  sg2044"
            echo "  sg2042"
            echo "  grace"
            echo "  rspi5"
            echo "  sapphirerapids"
            echo "  jtsonx"
            echo "  minix"
            echo "  oh5"
            echo "  <generic>"
            echo
            echo "Options:"
            echo "  --skip-openblas    Skip building and installing OpenBLAS"
            echo "  --skip-blis        Skip building and installing BLIS"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            if [[ -z "${ARCH}" ]]; then
                ARCH="$1"
            else
                echo "Unexpected argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "${ARCH}" ]]; then
    echo "Usage: $0 <architecture> [--skip-openblas] [--skip-blis]"
    echo
    echo "Supported architectures:"
    echo "  sg2044"
    echo "  sg2042"
    echo "  grace"
    echo "  rspi5"
    echo "  sapphirerapids"
    echo "  jtsonx"
    echo "  minix"
    echo "  oh5"
    echo "  epyc9135"
    echo "  <generic>"
    echo
    echo "Use -h or --help for option details."
    exit 1
fi

# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

OPENBLAS_VERSION="v0.3.34"
BLIS_VERSION="2.1"

# ---------------------------------------------------------------------------
# Base directories
# ---------------------------------------------------------------------------

SOFTWARE_ENV="${HOME}/software_env"

OPENBLAS_SRC="${SOFTWARE_ENV}/src/OpenBLAS"
BLIS_SRC="${SOFTWARE_ENV}/src/blis"

# These are populated below.
OPENBLAS_TARGET=""
BLIS_TARGET=""

CC=""
CXX=""

# Compiler include/library paths
COMPILER_INC=""
COMPILER_LIB=""

# Additional architecture/vendor include/library paths
EXTRA_INC=""
EXTRA_LIB=""

# Build flags
CFLAGS=""
CXXFLAGS=""
LDFLAGS=""

# ---------------------------------------------------------------------------
# Architecture/toolchain configuration
# ---------------------------------------------------------------------------

case "${ARCH}" in

    sg2044)

    OPENBLAS_TARGET="RISCV64_ZVL128B"
    BLIS_TARGET="rv64iv"

    export CC="${HOME}/software_env/llvm-EPI/riscv64/rvv1/bin/clang"
    export CXX="${HOME}/software_env/llvm-EPI/riscv64/rvv1/bin/clang++"
    export HOSTCC=gcc

    export GCC_TRIPLE="$(gcc -dumpmachine)"
    export GCC_INSTALL_DIR="$(dirname "$(gcc -print-file-name=crtbeginS.o)")"

    export COMPILER_INC="$HOME/software_env/llvm-EPI/riscv64/rvv1/include"
    export COMPILER_LIB="$HOME/software_env/llvm-EPI/riscv64/rvv1/lib"

    export CFLAGS="--target=$GCC_TRIPLE \
    --gcc-install-dir=$GCC_INSTALL_DIR \
    -I$COMPILER_INC \
    -O3 -march=rv64gc -mepi \
    -fno-slp-vectorize \
    -mllvm -combiner-store-merging=0 \
    -mllvm -enable-mem-access-versioning=0 \
    -mllvm -disable-loop-idiom-memcpy \
    -mllvm -disable-loop-idiom-memset"

    export LDFLAGS="-L$GCC_INSTALL_DIR -L$COMPILER_LIB"
    export EXTRA_LIB="${GCC_INSTALL_DIR}:${COMPILER_LIB}"
    ;;


    sg2042)
        # Adjust these paths if the SG2042 toolchain has a different layout.
        OPENBLAS_TARGET="RISCV64_GENERIC"
        BLIS_TARGET="auto"

        CC="${HOME}/software_env/llvm-EPI/riscv64/rvv071/bin/clang"
        CXX="${HOME}/software_env/llvm-EPI/riscv64/rvv071/bin/clang++"

        COMPILER_INC="${HOME}/software_env/llvm-EPI/riscv64/rvv071/include"
        COMPILER_LIB="${HOME}/software_env/llvm-EPI/riscv64/rvv071/lib"

        GCC_LIB="/usr/lib/gcc/riscv64-linux-gnu/13"

        EXTRA_INC="${COMPILER_INC}"
        EXTRA_LIB="${COMPILER_LIB}:${GCC_LIB}"

        CFLAGS="-I${COMPILER_INC} -O3 -march=rv64gc -menable-experimental-extensions -mepi -fno-slp-vectorize -mllvm -combiner-store-merging=0 -mllvm -vectorizer-use-vp-strided-load-store -mllvm -enable-mem-access-versioning=0 -mllvm -disable-loop-idiom-memcpy -mllvm -disable-loop-idiom-memset -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize"
        CXXFLAGS="-I${COMPILER_INC} -O3 -march=rv64gc -menable-experimental-extensions -mepi -fno-slp-vectorize -mllvm -combiner-store-merging=0 -mllvm -vectorizer-use-vp-strided-load-store -mllvm -enable-mem-access-versioning=0 -mllvm -disable-loop-idiom-memcpy -mllvm -disable-loop-idiom-memset -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize"
        LDFLAGS="-L${COMPILER_LIB} -L${GCC_LIB} -B${GCC_LIB}"

        ;;

    grace)
        OPENBLAS_TARGET="NEOVERSEV2"
        BLIS_TARGET="auto"

        CC="${HOME}/software_env/gcc/aarch64/16.1.0/bin/gcc"
        CXX="${HOME}/software_env/gcc/aarch64/16.1.0/bin/g++"

        COMPILER_INC="${HOME}/software_env/gcc/aarch64/16.1.0/include"
        COMPILER_LIB="${HOME}/software_env/gcc/aarch64/16.1.0/lib"

        # NVIDIA NVPL
        NVPL_VERSION="26.5"
        NVPL_PREFIX="${SOFTWARE_ENV}/nvpl/${NVPL_VERSION}"

        NVPL_ARCHIVE="nvpl-linux-sbsa-${NVPL_VERSION}.tar.gz"
        NVPL_URL="https://developer.download.nvidia.com/compute/nvpl/${NVPL_VERSION}/local_installers/${NVPL_ARCHIVE}"

        if [[ ! -d "${NVPL_PREFIX}" ]]; then
            mkdir -p "${SOFTWARE_ENV}"

            wget -O "${NVPL_ARCHIVE}" "${NVPL_URL}"

            mkdir -p "${NVPL_PREFIX}"

            tar -xzf "${NVPL_ARCHIVE}" \
                -C "${NVPL_PREFIX}" \
                --strip-components=1

            rm -f "${NVPL_ARCHIVE}"
        fi

        EXTRA_INC="${NVPL_PREFIX}/include:${COMPILER_INC}"
        EXTRA_LIB="${NVPL_PREFIX}/lib:${COMPILER_LIB}"

        CFLAGS="-I${NVPL_PREFIX}/include"
        CXXFLAGS="-I${NVPL_PREFIX}/include"
        LDFLAGS="-L${NVPL_PREFIX}/lib -L${COMPILER_LIB}"

        ;;

    sapphirerapids)
        OPENBLAS_TARGET="SAPPHIRERAPIDS"
        BLIS_TARGET="auto"

        CC="icx"
        CXX="icx"

        COMPILER_INC="[EDIT ME]"
        COMPILER_LIB="[EDIT ME]"

        ;;
    rspi5)

        OPENBLAS_TARGET="CORTEXA76"
        BLIS_TARGET="cortexa57"

        CC="${HOME}/software_env/gcc/aarch64/16.1.0/bin/gcc"
        CXX="${HOME}/software_env/gcc/aarch64/16.1.0/bin/g++"

        COMPILER_INC="${HOME}/software_env/gcc/aarch64/16.1.0/include"
        COMPILER_LIB="${HOME}/software_env/gcc/aarch64/16.1.0/lib"

        EXTRA_INC="${COMPILER_INC}"
        EXTRA_LIB="${COMPILER_LIB}"

        CFLAGS="-I${COMPILER_INC}"
        CXXFLAGS="-I${COMPILER_INC}"
        LDFLAGS="-L${COMPILER_LIB}"

        ;;
    jtsonx)

        OPENBLAS_TARGET="ARMV8"
        BLIS_TARGET="auto"

        CC="${HOME}/software_env/gcc/aarch64/16.1.0/bin/gcc"
        CXX="${HOME}/software_env/gcc/aarch64/16.1.0/bin/g++"

        COMPILER_INC="${HOME}/software_env/gcc/aarch64/16.1.0/include"
        COMPILER_LIB="${HOME}/software_env/gcc/aarch64/16.1.0/lib"

        EXTRA_INC="${COMPILER_INC}"
        EXTRA_LIB="${COMPILER_LIB}"

        CFLAGS="-I${COMPILER_INC}"
        CXXFLAGS="-I${COMPILER_INC}"
        LDFLAGS="-L${COMPILER_LIB}"

        ;;
    minix)

        OPENBLAS_TARGET="ZEN"
        BLIS_TARGET="zen"

        CC="${HOME}/software_env/gcc/x86_64/16.1.0/bin/gcc"
        CXX="${HOME}/software_env/gcc/x86_64/16.1.0/bin/g++"

        COMPILER_INC="${HOME}/software_env/gcc/x86_64/16.1.0/include"
        COMPILER_LIB="${HOME}/software_env/gcc/x86_64/16.1.0/lib"

        # Intel oneAPI / MKL
        MKL_VERSION="2026.1.0"
        MKL_PREFIX="${SOFTWARE_ENV}/intel-mkl"

        MKL_INSTALLER="intel-oneapi-toolkit-2026.1.0.192_offline.sh"
        MKL_URL="https://registrationcenter-download.intel.com/akdlm/IRC_NAS/33cb2a22-ddf1-4aa9-8d68-1f5a118acaf2/${MKL_INSTALLER}"

        if [[ ! -d "${MKL_PREFIX}" ]]; then
            wget -O "${MKL_INSTALLER}" "${MKL_URL}"

            sh "./${MKL_INSTALLER}" \
                -a \
                --silent \
                --eula accept \
                --install-dir "${MKL_PREFIX}"

            rm -f "${MKL_INSTALLER}"
        fi

        EXTRA_INC="${MKL_PREFIX}/mkl/latest/include:${COMPILER_INC}"
        EXTRA_LIB="${MKL_PREFIX}/mkl/latest/lib/intel64:${COMPILER_LIB}"

        CFLAGS="-I${MKL_PREFIX}/mkl/latest/include"
        CXXFLAGS="-I${MKL_PREFIX}/mkl/latest/include"
        LDFLAGS="-L${MKL_PREFIX}/mkl/latest/lib/intel64 -L${COMPILER_LIB}"

        ;;
    oh5)

        OPENBLAS_TARGET="HASWELL"
        BLIS_TARGET="haswell"

        CC="${HOME}/software_env/gcc/x86_64/16.1.0/bin/gcc"
        CXX="${HOME}/software_env/gcc/x86_64/16.1.0/bin/g++"

        COMPILER_INC="${HOME}/software_env/gcc/x86_64/16.1.0/include"
        COMPILER_LIB="${HOME}/software_env/gcc/x86_64/16.1.0/lib"

        # Intel oneAPI / MKL
        MKL_VERSION="2026.1.0"
        MKL_PREFIX="${SOFTWARE_ENV}/intel-mkl"

        MKL_INSTALLER="intel-oneapi-toolkit-2026.1.0.192_offline.sh"
        MKL_URL="https://registrationcenter-download.intel.com/akdlm/IRC_NAS/33cb2a22-ddf1-4aa9-8d68-1f5a118acaf2/${MKL_INSTALLER}"

        if [[ ! -d "${MKL_PREFIX}" ]]; then
            wget -O "${MKL_INSTALLER}" "${MKL_URL}"

            sh "./${MKL_INSTALLER}" \
                -a \
                --silent \
                --eula accept \
                --install-dir "${MKL_PREFIX}"

            rm -f "${MKL_INSTALLER}"
        fi

        EXTRA_INC="${MKL_PREFIX}/mkl/latest/include:${COMPILER_INC}"
        EXTRA_LIB="${MKL_PREFIX}/mkl/latest/lib/intel64:${COMPILER_LIB}"

        CFLAGS="-I${MKL_PREFIX}/mkl/latest/include"
        CXXFLAGS="-I${MKL_PREFIX}/mkl/latest/include"
        LDFLAGS="-L${MKL_PREFIX}/mkl/latest/lib/intel64 -L${COMPILER_LIB}"

        ;;
    epyc9135)

        OPENBLAS_TARGET="ZEN"
        BLIS_TARGET="zen3"

        CC="${HOME}/software_env/gcc/x86_64/16.1.0/bin/gcc"
        CXX="${HOME}/software_env/gcc/x86_64/16.1.0/bin/g++"

        COMPILER_INC="${HOME}/software_env/gcc/x86_64/16.1.0/include"
        COMPILER_LIB="${HOME}/software_env/gcc/x86_64/16.1.0/lib"

        # Intel oneAPI / MKL
        MKL_VERSION="2026.1.0"
        MKL_PREFIX="${SOFTWARE_ENV}/intel-mkl"

        MKL_INSTALLER="intel-oneapi-toolkit-2026.1.0.192_offline.sh"
        MKL_URL="https://registrationcenter-download.intel.com/akdlm/IRC_NAS/33cb2a22-ddf1-4aa9-8d68-1f5a118acaf2/${MKL_INSTALLER}"

        if [[ ! -d "${MKL_PREFIX}" ]]; then
            wget -O "${MKL_INSTALLER}" "${MKL_URL}"

            sh "./${MKL_INSTALLER}" \
                -a \
                --silent \
                --eula accept \
                --install-dir "${MKL_PREFIX}"

            rm -f "${MKL_INSTALLER}"
        fi

        EXTRA_INC="${MKL_PREFIX}/mkl/latest/include:${COMPILER_INC}"
        EXTRA_LIB="${MKL_PREFIX}/mkl/latest/lib/intel64:${COMPILER_LIB}"

        CFLAGS="-I${MKL_PREFIX}/mkl/latest/include"
        CXXFLAGS="-I${MKL_PREFIX}/mkl/latest/include"
        LDFLAGS="-L${MKL_PREFIX}/mkl/latest/lib/intel64 -L${COMPILER_LIB}"

        ;;
    *)
        echo "No special toolchain configuration for '${ARCH}'."
        echo "Using the system compiler."

        OPENBLAS_TARGET=""
        BLIS_TARGET="auto"

        CC="${CC:-gcc}"
        CXX="${CXX:-g++}"

        COMPILER_INC="/usr/include"
        COMPILER_LIB="/usr/lib"

        EXTRA_INC="${COMPILER_INC}"
        EXTRA_LIB="${COMPILER_LIB}"

        CFLAGS="-I${COMPILER_INC}"
        CXXFLAGS="-I${COMPILER_INC}"
        LDFLAGS="-L${COMPILER_LIB}"

        ;;
esac

# ---------------------------------------------------------------------------
# Validate toolchain
# ---------------------------------------------------------------------------

echo "================================================================"
echo "Architecture : ${ARCH}"
echo "Compiler     : ${CC}"
echo "C++ Compiler : ${CXX}"
echo "BLAS target  : ${OPENBLAS_TARGET:-default}"
echo "BLIS target  : ${BLIS_TARGET}"
echo "Compiler inc : ${COMPILER_INC}"
echo "Compiler lib : ${COMPILER_LIB}"
echo "Extra inc    : ${EXTRA_INC}"
echo "Extra lib    : ${EXTRA_LIB}"
echo "Skip OpenBLAS: ${SKIP_OPENBLAS}"
echo "Skip BLIS    : ${SKIP_BLIS}"
echo "================================================================"

command -v "${CC}" >/dev/null 2>&1 || {
    echo "ERROR: C compiler not found: ${CC}"
    exit 1
}

command -v "${CXX}" >/dev/null 2>&1 || {
    echo "ERROR: C++ compiler not found: ${CXX}"
    exit 1
}

export LD_LIBRARY_PATH="${EXTRA_LIB}"
export LIBRARY_PATH="${LD_LIBRARY_PATH}"
echo "LD_LIBRARY_PATH=${EXTRA_LIB}"
echo '------------------'
$CXX -print-search-dirs
echo '------------------'
ld --verbose | grep SEARCH_DIR

# ---------------------------------------------------------------------------
# Structured installation directories
# 
#   library / target / version
#
# ---------------------------------------------------------------------------

OPENBLAS_ISA="$(arch)/${OPENBLAS_TARGET:-generic}"
BLIS_ISA="$(arch)/${BLIS_TARGET:-generic}"
if [[ "${BLIS_TARGET}" == "auto" ]]; then
    BLIS_ISA+="/${ARCH}"
fi

OPENBLAS_PREFIX="${SOFTWARE_ENV}/OpenBLAS/${OPENBLAS_ISA}/${OPENBLAS_VERSION}"
BLIS_PREFIX="${SOFTWARE_ENV}/blis/${BLIS_ISA}/${BLIS_VERSION}"

mkdir -p "${SOFTWARE_ENV}/src"

# ---------------------------------------------------------------------------
# OpenBLAS
# ---------------------------------------------------------------------------

if [[ "${SKIP_OPENBLAS}" -ne 1 ]]; then
    echo
    echo "================================================================"
    echo "Building OpenBLAS"
    echo "================================================================"
    echo "Install prefix: ${OPENBLAS_PREFIX}"

    FRESH_DOWNLOAD=0

    if [ -d "${OPENBLAS_SRC}" ]; then
        read -rp "Directory '${OPENBLAS_SRC}' already exists. [R]esume or [F]resh copy? (r/f) [default: r]: " CHOICE
        case "${CHOICE}" in
            [Ff]*)
                echo "Removing existing directory and cloning fresh repository..."
                rm -rf "${OPENBLAS_SRC}"
                FRESH_DOWNLOAD=1
                ;;
            *)
                echo "Resuming using existing directory..."
                cd "${OPENBLAS_SRC}"
                ;;
        esac
    else
        FRESH_DOWNLOAD=1
    fi

    if [ "${FRESH_DOWNLOAD}" -eq 1 ]; then
        git clone https://github.com/OpenMathLib/OpenBLAS.git "${OPENBLAS_SRC}"
        cd "${OPENBLAS_SRC}"
        git checkout "${OPENBLAS_VERSION}"
    fi

    OPENBLAS_MAKE_OPTS=(
        "CC=${CC}"
        "CFLAGS=${CFLAGS}"
        "LDFLAGS=${LDFLAGS}"
        "NOFORTRAN=1"
        "USE_OPENMP=1"
    )

    if [[ -n "${OPENBLAS_TARGET}" ]]; then
        OPENBLAS_MAKE_OPTS+=("TARGET=${OPENBLAS_TARGET}")
    fi

    echo "${OPENBLAS_MAKE_OPTS[@]}"

    make "${OPENBLAS_MAKE_OPTS[@]}" -j"$(nproc)"

    make install \
        PREFIX="${OPENBLAS_PREFIX}" \
        "${OPENBLAS_MAKE_OPTS[@]}"

    cd - >/dev/null
else
    echo
    echo "================================================================"
    echo "Skipping OpenBLAS build as requested."
    echo "================================================================"
fi

# ---------------------------------------------------------------------------
# BLIS
# ---------------------------------------------------------------------------

if [[ "${SKIP_BLIS}" -ne 1 ]]; then
    echo
    echo "================================================================"
    echo "Building BLIS"
    echo "================================================================"
    echo "Install prefix: ${BLIS_PREFIX}"

    FRESH_DOWNLOAD=0

    if [ -d "${BLIS_SRC}" ]; then
        read -rp "Directory '${BLIS_SRC}' already exists. [R]esume or [F]resh copy? (r/f) [default: r]: " CHOICE
        case "${CHOICE}" in
            [Ff]*)
                echo "Removing existing directory and cloning fresh repository..."
                rm -rf "${BLIS_SRC}"
                FRESH_DOWNLOAD=1
                ;;
            *)
                echo "Resuming using existing directory..."
                cd "${BLIS_SRC}"
                ;;
        esac
    else
        FRESH_DOWNLOAD=1
    fi

    if [ "${FRESH_DOWNLOAD}" -eq 1 ]; then
        git clone https://github.com/flame/blis.git "${BLIS_SRC}"
        cd "${BLIS_SRC}"
        git checkout "${BLIS_VERSION}"
    fi

    BLIS_CONFIG_OPTS=(
        "--prefix=${BLIS_PREFIX}"
        "CXX=${CXX}"
        "CC=${CC}"
        "CFLAGS=\"${CFLAGS}\""
        "LDFLAGS=\"${LDFLAGS}\""
        "--enable-multithreading=openmp"
        "--enable-cblas"
    )

    echo "${BLIS_CONFIG_OPTS[@]} ${BLIS_TARGET}"

    if [[ "${BLIS_TARGET}" == "auto" ]]; then
        ./configure "${BLIS_CONFIG_OPTS[@]}" auto
    else
        ./configure "${BLIS_CONFIG_OPTS[@]}" "${BLIS_TARGET}"
    fi

    make -j"$(nproc)"
    make install

    cd - >/dev/null
else
    echo
    echo "================================================================"
    echo "Skipping BLIS build as requested."
    echo "================================================================"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo
echo "================================================================"
echo "Build complete"
echo "================================================================"
echo
echo "OpenBLAS:"
if [[ "${SKIP_OPENBLAS}" -eq 1 ]]; then
    echo "  [SKIPPED]"
else
    echo "  ${OPENBLAS_PREFIX}"
    echo "  include: ${OPENBLAS_PREFIX}/include"
    echo "  lib:     ${OPENBLAS_PREFIX}/lib"
fi
echo
echo "BLIS:"
if [[ "${SKIP_BLIS}" -eq 1 ]]; then
    echo "  [SKIPPED]"
else
    echo "  ${BLIS_PREFIX}"
    echo "  include: ${BLIS_PREFIX}/include"
    echo "  lib:     ${BLIS_PREFIX}/lib"
fi
echo
echo "Compiler:"
echo "  CC:      ${CC}"
echo "  CXX:     ${CXX}"
echo "  include: ${COMPILER_INC}"
echo "  lib:     ${COMPILER_LIB}"
echo "================================================================"