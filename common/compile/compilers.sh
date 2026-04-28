#!/usr/bin/env bash

# ==========================================
# Compiler registry
# Format:
# name:CXX:CC:modules:semicolon-separated
# ==========================================

# ---------- BSC-HCA ----------
declare -A COMPILERS_bsc_hca
COMPILERS_bsc_hca["gcc"]="gcc:g++:gcc:cmake/3.28.1;openBLAS/ubuntu/0.3.20_gcc10.3.0"

COMPILERS_bsc_hca["clang"]="clang:clang++:clang:cmake/3.28.1;llvm/EPI-development;openBLAS/ubuntu/0.3.29_llvmEPI1.0"

# Specialization
COMPILERS_bsc_hca["clang-banana"]="clang:clang++:clang:cmake/3.28.1;llvm/EPI-development;openBLAS/ubuntu/0.3.30_vlen256_llvmEPI1.0"

export COMPILERS_bsc_hca


# ---------- THEA ----------
declare -A COMPILERS_thea
COMPILERS_thea["gcc"]="gcc:g++:gcc:gcc/14.3.0"

export COMPILERS_thea


# ---------- LEONARDO ----------
declare -A COMPILERS_leonardo
COMPILERS_leonardo["clang"]="clang:clang++:clang:llvm/14.0.6--gcc--12.2.0-cuda-12.2"
COMPILERS_leonardo["gcc"]="gcc:g++:gcc:gcc/12.2.0"

export COMPILERS_leonardo


# ==========================================
# API
# ==========================================

get_compilers_for_system() {
    local system="$1"
    local var="COMPILERS_${system//-/_}"

    if ! declare -p "$var" &>/dev/null; then
        echo "ERROR: Unknown system '$system'" >&2
        exit 1
    fi

    echo "$var"
}

load_modules_for_compiler() {
    local modules_str="$1"

    module purge 2>/dev/null || true

    IFS=";" read -ra modules <<< "$modules_str"
    
    echo "Loading modules: ${modules[@]}" 
    
    for m in "${modules[@]}"; do
        [[ -n "$m" ]] && ml "$m"
    done
}

set_compiler_env() {
  local cxx="$1"
  local cc="$2"

  export CXX="$cxx"
  export CC="$cc"

  echo "Using CC=$CC, CXX=$CXX"
}
