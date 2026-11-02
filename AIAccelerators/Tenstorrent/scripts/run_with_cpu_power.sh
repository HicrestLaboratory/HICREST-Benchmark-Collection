#!/usr/bin/env bash

set -uo pipefail

if (( $# < 3 )) || [[ "$2" != "--" ]]; then
    echo "Usage: run_with_cpu_power.sh TRACY_OUTPUT_DIR -- COMMAND [ARG ...]" >&2
    exit 2
fi

OUTPUT_DIR="$1"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
POWER_DIR="${ROOT_DIR}/power_discovery"

if [[ "${CPU_POWER_ENABLED:-1}" == "0" ]]; then
    exec "$@"
fi

[[ -f "${POWER_DIR}/perf_start.sh" ]] || {
    echo "Error: missing ${POWER_DIR}/perf_start.sh" >&2
    exit 2
}
[[ -f "${POWER_DIR}/perf_stop.sh" ]] || {
    echo "Error: missing ${POWER_DIR}/perf_stop.sh" >&2
    exit 2
}

mkdir -p "${OUTPUT_DIR}"
STATE_DIR="$(mktemp -d "${OUTPUT_DIR}/.cpu-power.XXXXXX")"
NODE="$(hostname -s)"
PID_FILE="${STATE_DIR}/${NODE}.perf.pid"
MONITOR_PID=""

stop_monitor() {
    if [[ -n "${MONITOR_PID}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
        (cd "${STATE_DIR}" && bash "${POWER_DIR}/perf_stop.sh") || \
            kill "${MONITOR_PID}" 2>/dev/null || true
        wait "${MONITOR_PID}" 2>/dev/null || true
    fi
}

cleanup() {
    stop_monitor
}
trap cleanup EXIT INT TERM

(cd "${STATE_DIR}" && exec bash "${POWER_DIR}/perf_start.sh") &
MONITOR_PID=$!
printf '%s\n' "${MONITOR_PID}" > "${PID_FILE}"

"$@"
command_status=$?

stop_monitor
MONITOR_PID=""
trap - EXIT INT TERM

LATEST_REPORT="$(
    find "${OUTPUT_DIR}/reports" -mindepth 1 -maxdepth 1 -type d \
        -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
)"

if [[ -n "${LATEST_REPORT}" && -f "${STATE_DIR}/perf_log" ]]; then
    cp -- "${STATE_DIR}/perf_log" "${LATEST_REPORT}/cpu_power.csv"
    cp -- "${STATE_DIR}/cpu_power_backend.txt" \
        "${LATEST_REPORT}/cpu_power_backend.txt" 2>/dev/null || true
    python "${SCRIPT_DIR}/summarize_cpu_power.py" \
        "${LATEST_REPORT}/cpu_power.csv" \
        "${LATEST_REPORT}/cpu_power_summary.json"
else
    echo "Warning: CPU power log or Tracy report directory was not produced" >&2
fi

rm -r "${STATE_DIR}"
exit "${command_status}"
