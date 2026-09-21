#!/bin/bash
#
# run.sh — run llama-bench
#
# Usage:
#   ./run.sh -- -m model.gguf -p 512 -n 128 -t 4,8 -r 10

set -euo pipefail
cd "$(dirname "$0")"

llama_bench="../build/bin/llama-bench"
bench_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --llama-bench) llama_bench="$2"; shift 2 ;;
        --)            shift; bench_args=("$@"); break ;;
        *) echo "error: unknown argument '$1' (did you forget '--' before the llama-bench args?)" >&2; exit 1 ;;
    esac
done

if [[ ! -x "$llama_bench" ]]; then
    echo "error: '$llama_bench' not found/executable — run ../compile_llama.sh first" >&2
    exit 1
fi

# Execute llama-bench directly, replacing the current shell process
exec "$llama_bench" "${bench_args[@]}"