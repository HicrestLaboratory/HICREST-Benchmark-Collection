#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 -a <arch> -m <mpi> [-v <version>]"
    echo
    echo "  -a  Architecture: x86_64 | arm64-sbsa"
    echo "  -m  MPI implementation: openmpi | mpich"
    echo "  -v  Version (default: 25.09.06)"
    echo
    echo "Example:"
    echo "  $0 -a x86_64 -m openmpi"
    exit 1
}

# Defaults
VERSION="25.09.06"

while getopts "a:m:v:h" opt; do
    case "$opt" in
        a) ARCH="$OPTARG" ;;
        m) MPI="$OPTARG" ;;
        v) VERSION="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

: "${ARCH:?Missing -a <arch>}"
: "${MPI:?Missing -m <mpi>}"

# Normalize inputs
ARCH_LOWER=$(echo "$ARCH" | tr '[:upper:]' '[:lower:]')
MPI_LOWER=$(echo "$MPI" | tr '[:upper:]' '[:lower:]')

case "$ARCH_LOWER" in
    x86_64|arm64-sbsa) ;;
    *)
        echo "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

case "$MPI_LOWER" in
    openmpi|mpich) ;;
    *)
        echo "Unsupported MPI implementation: $MPI"
        exit 1
        ;;
esac

BASE_URL="https://developer.download.nvidia.com/compute/nvidia-hpc-benchmarks/redist"
PKG_NAME="nvidia_hpc_benchmarks_${MPI_LOWER}-linux-${ARCH_LOWER}-${VERSION}-archive.tar.xz"
URL="${BASE_URL}/nvidia_hpc_benchmarks_${MPI_LOWER}/linux-${ARCH_LOWER}/${PKG_NAME}"

echo "Downloading NVIDIA HPL benchmark:"
echo "  Architecture : $ARCH_LOWER"
echo "  MPI          : $MPI_LOWER"
echo "  Version      : $VERSION"
echo "  URL          : $URL"
echo

wget -c "$URL"

echo
echo "Download complete:"
echo "  $PKG_NAME"
