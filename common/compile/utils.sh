#!/bin/bash

validate_argument() {
    local value="$1"
    local name="$2"
    shift 2
    local valid_values=("$@")

    if [[ -z "$value" ]]; then
        echo "Usage: $0 <$name>"
        echo "Valid $name values: ${valid_values[*]}"
        exit 1
    fi

    for v in "${valid_values[@]}"; do
        if [[ "$v" == "$value" ]]; then
            return 0
        fi
    done

    echo "Error: Invalid $name '$value'."
    echo "Valid $name values: ${valid_values[*]}"
    exit 1
}

check_alps_uenv() {
    if [[ ! -d "/user-environment/linux-neoverse_v2" ]]; then
        echo "Please pull the uenv image: 'uenv image pull prgenv-gnu-openmpi/25.12:v1'"
        echo "Then, start a new session with that image: 'uenv start --view=modules prgenv-gnu-openmpi/25.12:v1'"
        echo "Then, load modules: 'ml gcc/14.3.0 cuda/12.9.1 nccl/2.28.9-1 openmpi/5.0.9 python/3.14.0 libfabric/2.3.1 aws-ofi-nccl/1.17.1'"
        exit 2
    fi
    # if [[ ! -d "/user-environment/linux-sles15-neoverse_v2" ]]; then
    #     echo "Please pull the uenv image: 'uenv image pull prgenv-gnu/24.7:v3'"
    #     echo "Then, start a new session with that image: 'uenv start prgenv-gnu/24.7:v3 --view=modules'"
    #     exit 2
    # fi
}

check_ccutils_installation() {
    if [[ -z "$CCUTILS_INCLUDE" ]]; then
        echo "Environment variable CCUTILS_INCLUDE not set. Make sure you have installed ccutils."
        echo "To install CCUTILS, run:"
        echo "    wget -qO- https://raw.githubusercontent.com/ThomasPasquali/ccutils/ccutils_json/install.sh | env bash"
        exit 3
    fi
}
