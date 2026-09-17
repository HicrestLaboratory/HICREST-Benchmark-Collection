#!/usr/bin/env bash
#
# install_compilers.sh
#
# Stage NATIVE Linux/GNU compiler toolchains into a shared filesystem.
#
# The login node is used only to download/extract packages.  The installed
# compiler executable must be native to the compute-node ISA:
#
#   x86_64  -> ELF x86-64
#   aarch64 -> ELF ARM aarch64
#   riscv64 -> ELF RISC-V
#
# No cross compiler is accepted.
#
# GCC binaries are supplied by conda-forge and installed directly into their
# final prefix.  IMPORTANT: conda environments must not be created in /tmp
# and then moved, because absolute prefixes may be embedded in packages.
#
# Clang is installed from conda-forge, which provides native packages for
# x86_64, aarch64, and riscv64.  This is important because upstream LLVM
# release archives do not publish a native Linux RISC-V64 binary.
#
# Usage:
#   ./install_compilers.sh --gcc 15.2.0
#   ./install_compilers.sh --clang 23.1.1
#   ./install_compilers.sh --all
#
# Optional:
#   ./install_compilers.sh --base /shared/software_env --all
#
# GCC:
#   MAMBA_EXE=/path/to/micromamba
#   GCC_CHANNEL=conda-forge
#
# Dependencies:
#   The script bootstraps micromamba and the required host-side utilities
#   into INSTALL_BASE when they are not already available. No root access
#   or system package manager is required.
#
#   A downloader is needed for the initial bootstrap: curl, wget, or python3.
#
set -Eeuo pipefail

INSTALL_BASE="${INSTALL_BASE:-${HOME}/software_env}"
DEFAULT_GCC_VERSION="${DEFAULT_GCC_VERSION:-15.2.0}"
DEFAULT_CLANG_VERSION="${DEFAULT_CLANG_VERSION:-23.1.1}"

GCC_CHANNEL="${GCC_CHANNEL:-conda-forge}"
CLANG_CHANNEL="${CLANG_CHANNEL:-conda-forge}"

TARGETS=(x86_64 aarch64 riscv64)

DOWNLOAD_DIR="${INSTALL_BASE}/downloads"
GCC_DIR="${INSTALL_BASE}/gcc"
CLANG_DIR="${INSTALL_BASE}/clang"
ENV_DIR="${INSTALL_BASE}/env"

TMP_DIRS=()

info()  { printf '\n==> %s\n' "$*"; }
step()  { printf '    %s\n' "$*"; }
warn()  { printf 'WARNING: %s\n' "$*" >&2; }
error() { printf 'ERROR: %s\n' "$*" >&2; }
die()   { error "$*"; exit 1; }

cleanup() {
    local d
    for d in "${TMP_DIRS[@]}"; do
        if [[ -n "${d}" && -d "${d}" ]]; then
            rm -rf -- "${d}"
        fi
    done
    return 0
}
trap cleanup EXIT

require_command() {
    command -v "$1" >/dev/null 2>&1 ||
        die "Required command '$1' was not found."
}

host_arch() {
    case "$(uname -m)" in
        x86_64|amd64) printf '%s' x86_64 ;;
        aarch64|arm64) printf '%s' aarch64 ;;
        riscv64) printf '%s' riscv64 ;;
        *) die "Unsupported host architecture: $(uname -m)" ;;
    esac
}

download_bootstrap() {
    local url="$1"
    local destination="$2"

    mkdir -p -- "$(dirname -- "${destination}")"

    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 5 --retry-delay 2 \
            --connect-timeout 30 --output "${destination}.partial" "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "${destination}.partial" "${url}"
    elif command -v python3 >/dev/null 2>&1; then
        python3 - "${url}" "${destination}.partial" <<'PY'
import sys
import urllib.request

url, destination = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(url) as response, open(destination, "wb") as out:
    while True:
        chunk = response.read(1024 * 1024)
        if not chunk:
            break
        out.write(chunk)
PY
    else
        die "Cannot bootstrap dependencies: curl, wget, or python3 is required."
    fi

    [[ -s "${destination}.partial" ]] ||
        die "Downloaded bootstrap archive is empty: ${url}"
    mv -- "${destination}.partial" "${destination}"
}

extract_tar() {
    local archive="$1"
    local directory="$2"

    mkdir -p -- "${directory}"

    if command -v tar >/dev/null 2>&1; then
        case "${archive}" in
            *.tar.bz2) tar -xjf "${archive}" -C "${directory}" ;;
            *.tar.xz)  tar -xJf "${archive}" -C "${directory}" ;;
            *.tar.gz|*.tgz) tar -xzf "${archive}" -C "${directory}" ;;
            *.tar.zst) tar --zstd -xf "${archive}" -C "${directory}" ;;
            *) die "Unsupported bootstrap archive format: ${archive}" ;;
        esac
    elif command -v python3 >/dev/null 2>&1; then
        python3 - "${archive}" "${directory}" <<'PY'
import sys
import tarfile

archive, directory = sys.argv[1], sys.argv[2]
with tarfile.open(archive, "r:*") as tar:
    tar.extractall(directory, filter="data")
PY
    else
        die "Cannot extract dependencies: tar or python3 is required."
    fi
}

bootstrap_micromamba() {
    local destination="${INSTALL_BASE}/bin/micromamba"
    local host=""
    local archive=""
    local temporary=""

    if [[ -x "${destination}" ]]; then
        printf '%s\n' "${destination}"
        return
    fi

    host="$(host_arch)"

    case "${host}" in
        x86_64)
            archive="micromamba-linux-64.tar.bz2"
            ;;
        aarch64)
            archive="micromamba-linux-aarch64.tar.bz2"
            ;;
        riscv64)
            archive="micromamba-linux-riscv64.tar.bz2"
            ;;
        *)
            die "Unsupported host architecture: ${host}"
            ;;
    esac

    info "micromamba was not found; bootstrapping a user-local copy." >&2

    temporary="${INSTALL_BASE}/.tmp-micromamba-bootstrap-$$"
    rm -rf -- "${temporary}"
    mkdir -p -- "${temporary}"
    TMP_DIRS+=("${temporary}")

    download_bootstrap \
        "https://micro.mamba.pm/api/micromamba/linux-${host}/latest" \
        "${temporary}/${archive}"

    extract_tar "${temporary}/${archive}" "${temporary}"

    [[ -x "${temporary}/bin/micromamba" ]] ||
        die "Bootstrapped archive did not contain bin/micromamba."

    mkdir -p -- "${INSTALL_BASE}/bin"
    cp -- "${temporary}/bin/micromamba" "${destination}"
    chmod +x -- "${destination}"

    printf '%s\n' "${destination}"
}

install_host_tools() {
    local mamba="$1"
    local tools_root="${INSTALL_BASE}/tools"

    # These packages are only for the login/host side of the installer.
    # They are deliberately installed separately from compiler prefixes.
    if [[ -x "${tools_root}/bin/curl" &&
          -x "${tools_root}/bin/tar" &&
          -x "${tools_root}/bin/file" &&
          -x "${tools_root}/bin/awk" &&
          -x "${tools_root}/bin/sed" &&
          -x "${tools_root}/bin/grep" &&
          -x "${tools_root}/bin/head" &&
          -x "${tools_root}/bin/find" &&
          -x "${tools_root}/bin/mktemp" ]]; then
        export PATH="${tools_root}/bin:${PATH}"
        return
    fi

    info "Installing missing host-side utilities into ${tools_root}"
    "${mamba}" create \
        --yes \
        --no-rc \
        --override-channels \
        --root-prefix "${DOWNLOAD_DIR}/micromamba-root" \
        --channel "${GCC_CHANNEL}" \
        --prefix "${tools_root}" \
        curl tar file gawk sed grep coreutils findutils

    export PATH="${tools_root}/bin:${PATH}"
}

check_requirements() {
    if [[ -n "${MAMBA_EXE:-}" ]]; then
        [[ -x "${MAMBA_EXE}" ]] ||
            die "MAMBA_EXE is not executable: ${MAMBA_EXE}"
    elif command -v micromamba >/dev/null 2>&1; then
        MAMBA_EXE="$(command -v micromamba)"
        export MAMBA_EXE
    else
        bootstrap_micromamba
        MAMBA_EXE="${INSTALL_BASE}/bin/micromamba"
        export MAMBA_EXE
    fi

    install_host_tools "${MAMBA_EXE}"

    local commands=(curl tar file awk sed grep head find mktemp)
    local cmd

    for cmd in "${commands[@]}"; do
        require_command "${cmd}"
    done
}

target_elf_regex() {
    case "$1" in
        x86_64)  printf '%s\n' 'ELF 64-bit.*x86-64' ;;
        aarch64) printf '%s\n' 'ELF 64-bit.*ARM aarch64' ;;
        riscv64) printf '%s\n' 'ELF 64-bit.*RISC-V' ;;
        *) die "Unsupported target ISA: $1" ;;
    esac
}

verify_native_binary() {
    local binary="$1"
    local target="$2"
    local description
    local regex

    [[ -f "${binary}" ]] ||
        die "Expected executable was not found: ${binary}"

    description="$(file -Lb "${binary}")"
    regex="$(target_elf_regex "${target}")"

    if ! grep -Eiq "${regex}" <<<"${description}"; then
        error "NATIVE ARCHITECTURE CHECK FAILED"
        error "Expected target: ${target}"
        error "Executable:      ${binary}"
        error "file(1):         ${description}"
        error "The archive/package is not a native compiler for ${target}."
        exit 1
    fi

    step "Verified ${target}: ${description}"
}

download() {
    local url="$1"
    local destination="$2"
    local partial="${destination}.partial"

    mkdir -p -- "$(dirname -- "${destination}")"

    if [[ -s "${destination}" ]]; then
        step "Using cached archive: ${destination}"
        return
    fi

    rm -f -- "${partial}"
    step "URL: ${url}"

    curl \
        --fail \
        --location \
        --retry 5 \
        --retry-delay 2 \
        --connect-timeout 30 \
        --output "${partial}" \
        "${url}"

    [[ -s "${partial}" ]] || die "Downloaded archive is empty: ${url}"
    mv -- "${partial}" "${destination}"
}

extract() {
    local archive="$1"
    local directory="$2"

    mkdir -p -- "${directory}"

    case "${archive}" in
        *.tar.xz)  tar -xJf "${archive}" -C "${directory}" ;;
        *.tar.gz|*.tgz) tar -xzf "${archive}" -C "${directory}" ;;
        *.tar.zst)
            tar --zstd -xf "${archive}" -C "${directory}" ;;
        *)
            die "Unsupported archive format: ${archive}" ;;
    esac
}

find_top_directory() {
    local directory="$1"
    local count
    local top

    mapfile -t _top_dirs < <(
        find "${directory}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n'
    )

    count="${#_top_dirs[@]}"
    [[ "${count}" -eq 1 ]] ||
        die "Expected exactly one top-level directory in ${directory}, found ${count}."

    top="${_top_dirs[0]}"
    printf '%s\n' "${top}"
}

# -----------------------------------------------------------------------------
# LLVM / Clang via conda-forge / micromamba
# -----------------------------------------------------------------------------

clang_conda_subdir() {
    case "$1" in
        x86_64)  printf '%s\n' linux-64 ;;
        aarch64) printf '%s\n' linux-aarch64 ;;
        riscv64) printf '%s\n' linux-riscv64 ;;
        *) die "Unsupported Clang target: $1" ;;
    esac
}

install_clang_target() {
    local version="$1"
    local target="$2"
    local destination="${CLANG_DIR}/${target}/${version}"
    local mamba
    local subdir
    local mamba_root

    if [[ -x "${destination}/bin/clang" ]]; then
        info "Clang ${version} / ${target} is already installed."
        verify_native_binary "${destination}/bin/clang" "${target}"

        [[ -x "${destination}/bin/clang++" ]] ||
            die "Existing Clang installation has no clang++: ${destination}"
        verify_native_binary "${destination}/bin/clang++" "${target}"
        return
    fi

    [[ ! -e "${destination}" ]] ||
        die "Installation directory exists but is incomplete: ${destination}"

    info "Preparing native Clang ${version} for ${target}"

    mamba="$(find_mamba)"
    subdir="$(clang_conda_subdir "${target}")"
    mamba_root="${DOWNLOAD_DIR}/micromamba-root"

    mkdir -p -- "${CLANG_DIR}/${target}" "${mamba_root}"

    step "Architecture: ${target}"
    step "Conda subdir: ${subdir}"
    step "Version:      ${version}"
    step "Channel:      ${CLANG_CHANNEL}"

    # Install directly into the final prefix.  Conda packages may contain
    # absolute prefix references, so do not create the environment elsewhere
    # and move it afterwards.
    #
    # clang provides the compiler implementation and clangxx provides the
    # C++ driver/package.  --platform selects the *native* target package
    # even when this installer is being run from a different login-node ISA.
    "${mamba}" create \
        --yes \
        --no-rc \
        --override-channels \
        --root-prefix "${mamba_root}" \
        --platform "${subdir}" \
        --channel "${CLANG_CHANNEL}" \
        --prefix "${destination}" \
        "clang=${version}" \
        "clangxx=${version}"

    verify_native_binary "${destination}/bin/clang" "${target}"
    verify_native_binary "${destination}/bin/clang++" "${target}"

    info "Native Clang installed."
    step "${destination}"
}

install_clang() {
    local version="$1"
    local target

    info "Installing native Clang ${version}"
    for target in "${TARGETS[@]}"; do
        install_clang_target "${version}" "${target}"
        create_clang_environment "${version}" "${target}"
    done
}

# -----------------------------------------------------------------------------
# GCC via conda-forge / micromamba
# -----------------------------------------------------------------------------

find_mamba() {
    if [[ -n "${MAMBA_EXE:-}" ]]; then
        [[ -x "${MAMBA_EXE}" ]] ||
            die "MAMBA_EXE is not executable: ${MAMBA_EXE}"
        printf '%s\n' "${MAMBA_EXE}"
        return
    fi

    if command -v micromamba >/dev/null 2>&1; then
        command -v micromamba
        return
    fi

    die \
        "GCC installation requires micromamba.\n" \
        "Install micromamba or set MAMBA_EXE=/path/to/micromamba."
}

gcc_conda_subdir() {
    case "$1" in
        x86_64)  printf '%s\n' linux-64 ;;
        aarch64) printf '%s\n' linux-aarch64 ;;
        riscv64) printf '%s\n' linux-riscv64 ;;
        *) die "Unsupported GCC target: $1" ;;
    esac
}

install_gcc_target() {
    local version="$1"
    local target="$2"
    local destination="${GCC_DIR}/${target}/${version}"
    local mamba
    local subdir
    local mamba_root

    if [[ -x "${destination}/bin/gcc" ]]; then
        info "GCC ${version} / ${target} is already installed."
        verify_native_binary "${destination}/bin/gcc" "${target}"

        [[ -x "${destination}/bin/g++" ]] ||
            die "Existing GCC installation has no g++: ${destination}"
        verify_native_binary "${destination}/bin/g++" "${target}"
        return
    fi

    [[ ! -e "${destination}" ]] ||
        die "Installation directory exists but is incomplete: ${destination}"

    info "Preparing native GCC ${version} for ${target}"

    mamba="$(find_mamba)"
    subdir="$(gcc_conda_subdir "${target}")"
    mamba_root="${DOWNLOAD_DIR}/micromamba-root"

    mkdir -p -- "${GCC_DIR}/${target}" "${mamba_root}"

    step "Architecture: ${target}"
    step "Conda subdir: ${subdir}"
    step "Version:      ${version}"
    step "Channel:      ${GCC_CHANNEL}"

    # DO NOT create this environment in /tmp and move it afterwards.
    # Conda packages can contain absolute prefix references.  Creating the
    # environment directly at its final location keeps those references valid.
    "${mamba}" create \
        --yes \
        --no-rc \
        --override-channels \
        --root-prefix "${mamba_root}" \
        --platform "${subdir}" \
        --channel "${GCC_CHANNEL}" \
        --prefix "${destination}" \
        "gcc=${version}" \
        "gxx=${version}"

    verify_native_binary "${destination}/bin/gcc" "${target}"
    verify_native_binary "${destination}/bin/g++" "${target}"

    info "Native GCC installed."
    step "${destination}"
}

install_gcc() {
    local version="$1"
    local target

    info "Installing native GCC ${version}"
    for target in "${TARGETS[@]}"; do
        install_gcc_target "${version}" "${target}"
        create_gcc_environment "${version}" "${target}"
    done
}

# -----------------------------------------------------------------------------
# Environment files
# -----------------------------------------------------------------------------

create_gcc_environment() {
    local version="$1"
    local target="$2"
    local directory="${ENV_DIR}/${target}"
    local file="${directory}/gcc-${version}.sh"

    mkdir -p -- "${directory}"

    cat >"${file}" <<EOF
# Native GCC ${version} for ${target}
# Generated by install_compilers.sh

export COMPILER_ISA="${target}"
export COMPILER_FAMILY="gcc"
export COMPILER_VERSION="${version}"

export GCC_HOME="${GCC_DIR}/${target}/${version}"
export PATH="\${GCC_HOME}/bin:\${PATH}"
EOF
}

create_clang_environment() {
    local version="$1"
    local target="$2"
    local directory="${ENV_DIR}/${target}"
    local file="${directory}/clang-${version}.sh"

    mkdir -p -- "${directory}"

    cat >"${file}" <<EOF
# Native Clang ${version} for ${target}
# Generated by install_compilers.sh

export COMPILER_ISA="${target}"
export COMPILER_FAMILY="clang"
export COMPILER_VERSION="${version}"

export CLANG_HOME="${CLANG_DIR}/${target}/${version}"
export PATH="\${CLANG_HOME}/bin:\${PATH}"
EOF
}

# -----------------------------------------------------------------------------
# Help / summary
# -----------------------------------------------------------------------------

print_help() {
    cat <<EOF

install_compilers.sh
====================

Install NATIVE GCC and/or Clang compiler binaries into a shared Linux/GNU
filesystem.  No cross compiler is accepted.

Targets:
  x86_64
  aarch64
  riscv64

Default installation:
  ${HOME}/software_env

Usage:
  ./install_compilers.sh --gcc 15.2.0
  ./install_compilers.sh --clang 23.1.1
  ./install_compilers.sh --gcc 15.2.0 --clang 23.1.1
  ./install_compilers.sh --all

Options:
  --gcc VERSION       Install native GCC for all targets.
  --clang VERSION     Install native Clang for all targets from conda-forge.
  --all               Install the default GCC and Clang versions.
  --base DIRECTORY    Change the installation root.
  --help              Show this help.

Environment:
  MAMBA_EXE           Path to micromamba, if it is not in PATH.
  GCC_CHANNEL         Conda channel for GCC (default: conda-forge).
  CLANG_CHANNEL       Conda channel for Clang (default: conda-forge).
  INSTALL_BASE        Installation root (default: \$HOME/software_env).

Bootstrap:
  If micromamba is not available, it is downloaded into
  \$INSTALL_BASE/bin/micromamba. Host-side utilities are installed into
  \$INSTALL_BASE/tools. No sudo/root access is required.

Result:
  \$INSTALL_BASE/
    gcc/<target>/<version>/
    clang/<target>/<version>/
    env/<target>/gcc-<version>.sh
    env/<target>/clang-<version>.sh
    downloads/

Example on a compute node:
  source "\$HOME/software_env/env/\$TARGET_ISA/gcc-15.2.0.sh"
  gcc --version
  g++ --version

Do not source this installer itself:
  ./install_compilers.sh --all

EOF
}

print_summary() {
    local gcc_version="$1"
    local clang_version="$2"

    printf '\n'
    printf '%s\n' '================================================================'
    printf '%s\n' 'Installation complete'
    printf '%s\n' '================================================================'
    printf '\nShared installation:\n  %s\n\n' "${INSTALL_BASE}"

    [[ -n "${gcc_version}" ]] && printf 'GCC:   %s\n' "${gcc_version}"
    [[ -n "${clang_version}" ]] && printf 'Clang: %s\n' "${clang_version}"

    printf '\nTarget ISAs:\n'
    printf '  - %s\n' "${TARGETS[@]}"

    printf '\nSelect a compiler on a compute node with:\n'
    [[ -n "${gcc_version}" ]] &&
        printf '  source "%s/env/${TARGET_ISA}/gcc-%s.sh"\n' \
            "${INSTALL_BASE}" "${gcc_version}"
    [[ -n "${clang_version}" ]] &&
        printf '  source "%s/env/${TARGET_ISA}/clang-%s.sh"\n' \
            "${INSTALL_BASE}" "${clang_version}"
    printf '\n'
}

main() {
    if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
        error "install_compilers.sh must be EXECUTED, not sourced."
        error "Use: ./install_compilers.sh --all"
        return 2
    fi

    local gcc_version=""
    local clang_version=""
    local do_gcc=0
    local do_clang=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --gcc)
                [[ $# -ge 2 ]] || die "--gcc requires a version."
                gcc_version="$2"
                do_gcc=1
                shift 2
                ;;
            --clang)
                [[ $# -ge 2 ]] || die "--clang requires a version."
                clang_version="$2"
                do_clang=1
                shift 2
                ;;
            --all)
                gcc_version="${DEFAULT_GCC_VERSION}"
                clang_version="${DEFAULT_CLANG_VERSION}"
                do_gcc=1
                do_clang=1
                shift
                ;;
            --base)
                [[ $# -ge 2 ]] || die "--base requires a directory."
                INSTALL_BASE="$2"
                DOWNLOAD_DIR="${INSTALL_BASE}/downloads"
                GCC_DIR="${INSTALL_BASE}/gcc"
                CLANG_DIR="${INSTALL_BASE}/clang"
                ENV_DIR="${INSTALL_BASE}/env"
                shift 2
                ;;
            --help|-h)
                print_help
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                print_help
                exit 2
                ;;
        esac
    done

    if [[ "${do_gcc}" -eq 0 && "${do_clang}" -eq 0 ]]; then
        print_help
        exit 2
    fi

    check_requirements

    mkdir -p -- \
        "${INSTALL_BASE}" \
        "${DOWNLOAD_DIR}" \
        "${GCC_DIR}" \
        "${CLANG_DIR}" \
        "${ENV_DIR}"

    printf '\n'
    printf '%s\n' '================================================================'
    printf '%s\n' 'Native compiler installer'
    printf '%s\n' '================================================================'
    printf 'Install base: %s\n\n' "${INSTALL_BASE}"
    printf 'Native targets:\n'
    printf '  - %s\n' "${TARGETS[@]}"
    printf '\n'

    if [[ "${do_gcc}" -eq 1 ]]; then
        install_gcc "${gcc_version}"
    fi

    if [[ "${do_clang}" -eq 1 ]]; then
        install_clang "${clang_version}"
    fi

    print_summary "${gcc_version}" "${clang_version}"
}

main "$@"