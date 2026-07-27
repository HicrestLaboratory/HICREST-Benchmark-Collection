#!/bin/bash

set -e

algorithms="MCS SpinLock HemLock PthreadLock Reciprocating FissileTicket"

rawhost=$(hostname)
outdir=$(hostname | tr '.-' '__')

mkdir -p "${rawhost}"

cflag="-Wall -Wextra -Werror -Wno-implicit-fallthrough \
-std=gnu11 -O3 -DNDEBUG -fno-reorder-functions -DPIN"

atomicinst=""

# ARM-specific tuning
if [ "${outdir}" = "algol" ] || [ "${outdir}" = "prolog" ]; then
    atomicinst="-mno-outline-atomics"
    cflag="${cflag} ${atomicinst} -DATOMIC"
fi

CC=gcc

command -v ${CC} >/dev/null 2>&1 || {
    echo "Compiler gcc not found"
    exit 1
}

if [ $# -gt 0 ]; then
    algorithms="$@"
fi

for algorithm in ${algorithms}; do
    exe="${algorithm}.out"

    echo "Compiling ${algorithm}..."

    ${CC} ${cflag} \
        ${affinity:+-D${affinity}} \
        ${atomic:+-D${atomic}} \
        ${atomicinst:+-DATOMICINST=${atomicinst}} \
        -DHOST_${outdir} \
        -DAlgorithm=${algorithm} \
        -DNCS_DELAY=20 \
        Harness.c -lpthread -lm -o "${exe}"

    echo "Created ${exe}"
done