#!/usr/bin/env bash
set -euo pipefail

source ../common/compile/compilers.sh
source ../common/compile/utils.sh

SUPPORTED_SYSTEMS=("leonardo" "e4" "thea")

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <system>"
    exit 1
fi

system="$1"
validate_argument "$system" "system" "${SUPPORTED_SYSTEMS[@]}"

cd STREAM
make clean

# ---- Parameter space ----
MEM_SIZES_MB=(8 512)  # desired memory footprint per array
NTIMES=(15)
TYPES=(float double uint32 int32)
OFFSETS=(0)

# ---- Output directory ----
OUT_DIR="../bin"
mkdir -p "$OUT_DIR"

compilers_var=$(get_compilers_for_system "$system")
declare -n COMPILERS="$compilers_var"

# ---- Helper: type size in bytes ----
sizeof_type() {
  case "$1" in
    float)    echo 4 ;;
    double)   echo 8 ;;
    uint32)   echo 4 ;;
    int32)    echo 4 ;;
    *)
      echo "Unknown type: $1" >&2
      exit 1
      ;;
  esac
}

for key in "${!COMPILERS[@]}"; do
    IFS=":" read -r name cxx cc modules libs link_opts <<< "${COMPILERS[$key]}"

    echo "==== Compiler: $name ===="

    out_dir="$OUT_DIR/$name"
    mkdir -p "$out_dir"

    load_modules_for_compiler "$modules"
    set_compiler_env "$cxx" "$cc"

    CFLAGS="-O3 -fopenmp"
    if [[ $name == "icx" ]]; then
        CFLAGS="-O3 -qopenmp"
    fi

    # your existing parameter sweep stays unchanged
    for MEM_MB in "${MEM_SIZES_MB[@]}"; do
        for R in "${NTIMES[@]}"; do
            for TYPE in "${TYPES[@]}"; do
                for OFFSET in "${OFFSETS[@]}"; do

                    TYPE_SIZE=$(sizeof_type "$TYPE")
                    BYTES=$((MEM_MB * 1024 * 1024))
                    ARRAY_SIZE=$((BYTES / TYPE_SIZE))

                    VARS="-DSTREAM_ARRAY_SIZE=${ARRAY_SIZE} -DNTIMES=${R} -DSTREAM_TYPE_${TYPE^^} -DOFFSET=${OFFSET}"

                    BIN_NAME="${out_dir}/stream_${name}_${TYPE}_M${MEM_MB}_R${R}_O${OFFSET}"

                    make clean >/dev/null
                    make CFLAGS="$CFLAGS" CUSTOM_PREPROCESSOR_VARS="$VARS" stream_c_custom
                    mv stream_c_custom "$BIN_NAME"

                done
            done
        done
    done
done