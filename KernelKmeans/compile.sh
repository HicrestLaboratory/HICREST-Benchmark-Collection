#!/bin/bash

# Popcorn K-means Multi-System Build Script
# Pure Makefile-based compilation system
# Supports: bsc-hca, thea, leonardo
# Variants: default, pioneer, arriesgado, bananaf3

set -euo pipefail

source ../common/compile/compilers.sh
source ../common/compile/utils.sh

# ============================================================================
# CONFIGURATION
# ============================================================================

SUPPORTED_SYSTEMS=("bsc-hca" "thea" "leonardo")
SUPPORTED_VARIANTS=("default" "pioneer" "arriesgado" "bananaf3")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# BLAS DETECTION
# ============================================================================

detect_blas_vendor() {
    if [[ -n "${NVPL_HOME:-}" ]]; then
        echo "NVPL"
    elif [[ -n "${MKLROOT:-}" ]]; then
        echo "Intel10_64lp"
    elif [[ -n "${OPENBLAS_HOME:-}" ]]; then
        echo "OpenBLAS"
    else
        echo ""
    fi
}

get_blas_dir() {
    local candidates=(
        "NVPL_HOME"
        "MKLROOT"
        "BLAS_HOME"
        "OBLAS_LIBS"
        "OPENBLAS_LIB"
        "OPENBLAS_HOME"
    )

    local raw path prefix

    for var in "${candidates[@]}"; do
        raw="${!var:-}"
        [[ -z "$raw" ]] && continue

        # Extract path from linker flags if needed
        if [[ "$raw" =~ -L([^[:space:]]+) ]]; then
            path="${BASH_REMATCH[1]}"
        else
            path="$raw"
        fi

        path="$(realpath "$path" 2>/dev/null || echo "$path")"

        # Normalize prefix/lib relationship
        case "$(basename "$path")" in
            lib | lib64)
                prefix="$(dirname "$path")"
                ;;
            *)
                prefix="$path"
                ;;
        esac

        if [[ -d "$prefix" ]]; then
            echo "$prefix"
            return 0
        fi
    done

    print_error "Could not determine BLAS installation path."
    exit 1
}

get_blas_libs() {
    local raw="${OBLAS_LIBS:-${OPENBLAS_LIB:-}}"

    if [[ -n "$raw" ]]; then
        echo "$raw"
        return 0
    fi

    if [[ -n "${NVPL_HOME:-}" ]]; then
        echo "${NVPL_HOME}/lib/libnvpl_blas_ilp64_gomp.so"
        return 0
    fi

    if [[ -n "${MKLROOT:-}" ]]; then
        echo "${MKLROOT}/lib/intel64/libmkl_rt.so"
        return 0
    fi

    if [[ -n "${OPENBLAS_HOME:-}" ]]; then
        echo "${OPENBLAS_HOME}/lib/libopenblas.so"
        return 0
    fi

    print_error "Could not determine BLAS libraries"
    exit 1
}

get_blas_include_dirs() {
    if [[ -n "${NVPL_HOME:-}" ]]; then
        echo "${NVPL_HOME}/include"
    elif [[ -n "${MKLROOT:-}" ]]; then
        echo "${MKLROOT}/include"
    elif [[ -n "${OPENBLAS_HOME:-}" ]]; then
        echo "${OPENBLAS_HOME}/include"
    fi
}

# ============================================================================
# BUILD LOGIC
# ============================================================================

build_with_compiler() {
    local system="$1"
    local variant="$2"
    local compiler_name="$3"

    # Get compiler configuration
    local config=$(get_compiler_config "$system" "$compiler_name")
    if [[ -z "$config" ]]; then
        print_warning "Compiler '$compiler_name' not configured for system '$system'. Skipping."
        return 0
    fi

    # Parse config: "cxx cc modules libs link_opts"
    local parts=($config)
    local cxx_compiler="${parts[0]}"
    local cc_compiler="${parts[1]}"
    local modules="${parts[2]:-}"
    local extra_libs="${parts[3]:-}"
    local link_opts="${parts[4]:-}"

    # Build directory
    local build_dir="build_${system}_${compiler_name}_${variant}"

    print_header "Building: $system / $compiler_name / $variant"
    print_info "Build directory: $build_dir"
    print_info "CXX compiler: $cxx_compiler"
    print_info "CC compiler: $cc_compiler"

    # Clean previous build
    rm -rf "$build_dir"
    mkdir -p "$build_dir"

    # Load modules if available
    if [[ -n "$modules" ]]; then
        print_info "Modules: $modules"
        load_modules_for_compiler "$modules"
    fi

    # Detect BLAS
    local blas_vendor=$(detect_blas_vendor)
    local blas_libs=$(get_blas_libs)
    local blas_include=$(get_blas_include_dirs)
    local blas_dir=$(get_blas_dir)

    [[ -n "$blas_vendor" ]] && print_info "BLAS Vendor: $blas_vendor"
    print_info "BLAS Dir: $blas_dir"
    print_info "BLAS Libs: $blas_libs"
    [[ -n "$blas_include" ]] && print_info "BLAS Include: $blas_include"

    # Run make
    print_info "Starting build..."
    if ! make \
        -C "$build_dir" \
        BUILD_DIR="$build_dir" \
        SYSTEM="$system" \
        VARIANT="$variant" \
        COMPILER_NAME="$compiler_name" \
        CXX="$cxx_compiler" \
        CC="$cc_compiler" \
        BLAS_VENDOR="${blas_vendor:-}" \
        BLAS_LIBS="$blas_libs" \
        BLAS_INCLUDE_DIRS="${blas_include:-}" \
        BLAS_DIR="$blas_dir" \
        EXTRA_LIBS="$extra_libs" \
        EXTRA_LINK_OPTIONS="$link_opts" \
        SOURCE_DIR="$SCRIPT_DIR"; then
        print_error "Build failed for $system / $compiler_name / $variant"
        return 1
    fi

    print_success "Build completed: $build_dir/bin"
    echo ""
}

# ============================================================================
# VARIANT-SPECIFIC COMPILER FILTERING
# ============================================================================

should_build_compiler() {
    local system="$1"
    local variant="$2"
    local compiler="$3"

    # bananaf3 uses clang-banana, not clang
    if [[ "$variant" == "bananaf3" ]]; then
        [[ "$compiler" == "clang" ]] && return 1
        return 0
    fi

    # Other variants skip clang-banana
    [[ "$compiler" == "clang-banana" ]] && return 1

    # leonardo doesn't use clang
    if [[ "$system" == "leonardo" ]]; then
        [[ "$compiler" == "clang" ]] && return 1
    fi

    return 0
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    if [[ $# -eq 0 ]]; then
        cat << EOF
Popcorn K-means Build System

Usage: $0 <system> [<variant>]

Supported systems:  ${SUPPORTED_SYSTEMS[*]}
Supported variants: ${SUPPORTED_VARIANTS[*]}

Examples:
  $0 bsc-hca                    # Build all compilers, default variant
  $0 bsc-hca default            # Same as above
  $0 bsc-hca bananaf3           # Build all compilers, bananaf3 variant
  $0 leonardo default           # Build leonardo with default variant

Environment Variables:
  NVPL_HOME                     # NVIDIA Performance Libraries path
  MKLROOT                       # Intel MKL path
  OPENBLAS_HOME                 # OpenBLAS path
  CCUTILS_DIR                   # ccutils installation path

EOF
        exit 1
    fi

    local system="$1"
    local variant="${2:-default}"

    # Validation
    validate_argument "$system" "system" "${SUPPORTED_SYSTEMS[@]}"
    validate_argument "$variant" "variant" "${SUPPORTED_VARIANTS[@]}"

    # Check prerequisites
    check_command "make"
    check_command "bash"
    check_file "$SCRIPT_DIR/Makefile"

    print_header "Popcorn K-means Build System"
    print_info "System: $system"
    print_info "Variant: $variant"
    print_info "Working directory: $(pwd)"

    # Get compilers for system
    local compilers_var=$(get_compilers_for_system "$system")
    if [[ -z "$compilers_var" ]]; then
        print_error "Unknown system: $system"
        exit 1
    fi

    declare -n compilers="$compilers_var"

    # Build with each compiler
    local build_count=0
    local success_count=0
    local failed_builds=()

    for compiler_name in "${!compilers[@]}"; do
        # if ! should_build_compiler "$system" "$variant" "$compiler_name"; then
        #     print_info "Skipping compiler '$compiler_name' for variant '$variant'"
        #     continue
        # fi

        echo "$system" "$variant" "$compiler_name"

        ((build_count++))
        if build_with_compiler "$system" "$variant" "$compiler_name"; then
            ((success_count++))
        else
            failed_builds+=("$compiler_name")
        fi
    done

    # Summary
    print_header "Build Summary"
    print_info "Total builds: $build_count"
    print_success "Successful: $success_count"

    if [[ ${#failed_builds[@]} -gt 0 ]]; then
        print_error "Failed: ${#failed_builds[@]}"
        print_error "Failed compilers: ${failed_builds[*]}"
        exit 1
    fi

    print_success "All builds completed successfully! 🎉"
}

main "$@"