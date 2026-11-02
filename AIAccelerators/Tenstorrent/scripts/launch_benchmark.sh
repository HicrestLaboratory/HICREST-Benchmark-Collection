#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
    cat <<'EOF'
Usage:
  launch_benchmark.sh gemm [WRAPPER_OPTIONS] -- M K N [GEMM_OPTIONS]
  launch_benchmark.sh nonlinearity [WRAPPER_OPTIONS] -- H W [NONLINEARITY_OPTIONS]

Run exactly one Tenstorrent benchmark configuration, collect its Tracy output,
generate perf_report.csv and a summary CSV, measure device power in the
background, and print one machine-readable HICREST_RESULT JSON line.

Power measurement phases:
  pre          Idle measurement before the Tracy benchmark starts
  benchmark    Tracy-wrapped GEMM or nonlinearity execution
  post         Measurement after the benchmark process exits

Wrapper options:
  --result-id ID         Unique filesystem-safe result identifier (required)
  --output-root DIR      Result root
                         Default: <tenstorrent>/generated/sbatchman
  --build-bin DIR        Executable directory
                         Default: <tenstorrent>/build/bin
  --power-pre SEC        Measurement before benchmark starts (default: 5)
  --power-post SEC       Measurement after benchmark ends (default: 5)
  --power-period SEC     Requested power sampling period (default: 0.5)
  --no-power             Disable power measurement
  --cpu-power            Measure host CPU package energy with perf
  -h, --help             Show this help

Everything following -- is passed to exactly one benchmark executable.

Examples:
  ./scripts/launch_benchmark.sh gemm \
      --result-id tt_matmul_bf16_hifi4_s1024_g8x8 \
      --power-pre 5 \
      --power-post 5 \
      --power-period 0.5 \
      -- \
      1024 1024 1024 \
      --op matmul \
      --dtype bf16 \
      --math-fidelity hifi4 \
      --compute-grid-x 8 \
      --compute-grid-y 8 \
      --warmup 5 \
      --run 10

  ./scripts/launch_benchmark.sh nonlinearity \
      --result-id tt_gelu_bf16_s1024 \
      --power-pre 5 \
      --power-post 5 \
      --power-period 0.5 \
      -- \
      1024 1024 \
      --op gelu \
      --dtype bf16 \
      --warmup 5 \
      --run 10

Requirements:
  - The selected benchmark executable must already be built.
  - python -m tracy must be available.
  - tt-perf-report must be available.
  - pandas must be available to the selected Python command.
  - tt-smi must be available unless --no-power is used.
  - perf power/energy-pkg/ access is required when --cpu-power is used.
  - A result ID cannot be reused.
EOF
}

die() {
    echo "Error: $*" >&2
    exit 2
}

validate_nonnegative_number() {
    local name="$1"
    local value="$2"

    "${PYTHON_BIN}" - "${name}" "${value}" <<'PY'
import math
import sys

name = sys.argv[1]
text = sys.argv[2]

try:
    value = float(text)
except ValueError:
    raise SystemExit(f"Error: {name} must be a non-negative number")

if not math.isfinite(value) or value < 0:
    raise SystemExit(f"Error: {name} must be a non-negative number")
PY
}

validate_positive_number() {
    local name="$1"
    local value="$2"

    "${PYTHON_BIN}" - "${name}" "${value}" <<'PY'
import math
import sys

name = sys.argv[1]
text = sys.argv[2]

try:
    value = float(text)
except ValueError:
    raise SystemExit(f"Error: {name} must be greater than zero")

if not math.isfinite(value) or value <= 0:
    raise SystemExit(f"Error: {name} must be greater than zero")
PY
}

if (( $# == 0 )); then
    usage >&2
    exit 2
fi

BENCHMARK_KIND="$1"
shift

case "${BENCHMARK_KIND}" in
    gemm)
        EXECUTABLE_NAME="gemm"
        SUMMARY_OPERATION="matmul"
        ;;
    nonlinearity|non_linearity)
        BENCHMARK_KIND="nonlinearity"
        EXECUTABLE_NAME="non_linearity"
        SUMMARY_OPERATION=""
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        die "unknown benchmark '${BENCHMARK_KIND}'; expected gemm or nonlinearity"
        ;;
esac

WARMUP_ITERATIONS=5
MEASURED_ITERATIONS=10

RESULT_ID=""
OUTPUT_ROOT="${ROOT_DIR}/generated/sbatchman"
BUILD_BIN="${ROOT_DIR}/build/bin"

POWER_ENABLED=1
CPU_POWER_ENABLED=0
POWER_PRE_SECONDS=5
POWER_POST_SECONDS=5
POWER_PERIOD_SECONDS=0.5

while (( $# > 0 )); do
    case "$1" in
        --result-id)
            (( $# >= 2 )) || die "--result-id requires a value"
            RESULT_ID="$2"
            shift 2
            ;;
        --output-root)
            (( $# >= 2 )) || die "--output-root requires a value"
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --build-bin)
            (( $# >= 2 )) || die "--build-bin requires a value"
            BUILD_BIN="$2"
            shift 2
            ;;
        --power-pre)
            (( $# >= 2 )) || die "--power-pre requires a value"
            POWER_PRE_SECONDS="$2"
            shift 2
            ;;
        --power-post)
            (( $# >= 2 )) || die "--power-post requires a value"
            POWER_POST_SECONDS="$2"
            shift 2
            ;;
        --power-period)
            (( $# >= 2 )) || die "--power-period requires a value"
            POWER_PERIOD_SECONDS="$2"
            shift 2
            ;;
        --no-power)
            POWER_ENABLED=0
            shift
            ;;
        --cpu-power)
            CPU_POWER_ENABLED=1
            shift
            ;;
        --)
            shift
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown wrapper option '$1' (benchmark arguments must follow --)"
            ;;
    esac
done

[[ -n "${RESULT_ID}" ]] || die "--result-id is required"

[[ "${RESULT_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || \
    die "--result-id may contain only letters, digits, '.', '_', and '-'"

(( $# > 0 )) || die "benchmark arguments are required after --"

BENCHMARK_ARGUMENTS=("$@")

case "${BENCHMARK_KIND}" in
    gemm)
        (( ${#BENCHMARK_ARGUMENTS[@]} >= 3 )) || \
            die "gemm requires M K N after --"
        ;;
    nonlinearity)
        (( ${#BENCHMARK_ARGUMENTS[@]} >= 2 )) || \
            die "nonlinearity requires H W after --"

        for (( index=0; index < ${#BENCHMARK_ARGUMENTS[@]}; index++ )); do
            argument="${BENCHMARK_ARGUMENTS[index]}"

            case "${argument}" in
                --op)
                    value_index=$((index + 1))

                    (( value_index < ${#BENCHMARK_ARGUMENTS[@]} )) || \
                        die "--op requires a value"

                    SUMMARY_OPERATION="${BENCHMARK_ARGUMENTS[value_index]}"
                    break
                    ;;
                --op=*)
                    SUMMARY_OPERATION="${argument#*=}"
                    break
                    ;;
            esac
        done

        [[ "${SUMMARY_OPERATION}" =~ ^(relu|gelu|softmax)$ ]] || \
            die "nonlinearity requires --op relu, gelu, or softmax"
        ;;
esac

for (( index=0; index < ${#BENCHMARK_ARGUMENTS[@]}; index++ )); do
    argument="${BENCHMARK_ARGUMENTS[index]}"

    case "${argument}" in
        -w|--warmup)
            value_index=$((index + 1))

            (( value_index < ${#BENCHMARK_ARGUMENTS[@]} )) || \
                die "${argument} requires a value"

            WARMUP_ITERATIONS="${BENCHMARK_ARGUMENTS[value_index]}"
            ;;
        --warmup=*)
            WARMUP_ITERATIONS="${argument#*=}"
            ;;
        -r|--run)
            value_index=$((index + 1))

            (( value_index < ${#BENCHMARK_ARGUMENTS[@]} )) || \
                die "${argument} requires a value"

            MEASURED_ITERATIONS="${BENCHMARK_ARGUMENTS[value_index]}"
            ;;
        --run=*)
            MEASURED_ITERATIONS="${argument#*=}"
            ;;
    esac
done

[[ "${WARMUP_ITERATIONS}" =~ ^[0-9]+$ ]] || \
    die "--warmup must be a non-negative integer"

[[ "${MEASURED_ITERATIONS}" =~ ^[1-9][0-9]*$ ]] || \
    die "--run must be a positive integer"

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || \
    die "Python command not found: ${PYTHON_BIN}"

if (( POWER_ENABLED )); then
    validate_nonnegative_number "--power-pre" "${POWER_PRE_SECONDS}"
    validate_nonnegative_number "--power-post" "${POWER_POST_SECONDS}"
    validate_positive_number "--power-period" "${POWER_PERIOD_SECONDS}"
fi

if (( CPU_POWER_ENABLED )); then
    command -v perf >/dev/null 2>&1 || \
        die "perf is not available in PATH"
fi

EXECUTABLE="${BUILD_BIN}/${EXECUTABLE_NAME}"
REPORT_GROUP="${OUTPUT_ROOT}/${RESULT_ID}_reports"

[[ -x "${EXECUTABLE}" ]] || \
    die "benchmark executable not found or not executable: ${EXECUTABLE}"

command -v tt-perf-report >/dev/null 2>&1 || \
    die "tt-perf-report is not available in PATH"

"${PYTHON_BIN}" -c 'import pandas' >/dev/null 2>&1 || \
    die "pandas is not available to ${PYTHON_BIN}"

"${PYTHON_BIN}" -c 'import tracy' >/dev/null 2>&1 || \
    die "the tracy Python module is not available to ${PYTHON_BIN}"

if (( POWER_ENABLED )); then
    command -v tt-smi >/dev/null 2>&1 || \
        die "tt-smi is not available in PATH"
fi

[[ ! -e "${REPORT_GROUP}" ]] || \
    die "result directory already exists: ${REPORT_GROUP}"

mkdir -p "${REPORT_GROUP}"

echo "Running one ${BENCHMARK_KIND} configuration"
echo "Result ID: ${RESULT_ID}"
echo "Report directory: ${REPORT_GROUP}"
printf 'Command:'
printf ' %q' "${EXECUTABLE}" "${BENCHMARK_ARGUMENTS[@]}"
printf '\n'
echo "Warm-up iterations excluded from summary: ${WARMUP_ITERATIONS}"
echo "Measured iterations included in summary: ${MEASURED_ITERATIONS}"

if (( POWER_ENABLED )); then
    echo "Power measurement: enabled"
    echo "Power pre-window: ${POWER_PRE_SECONDS} s"
    echo "Power post-window: ${POWER_POST_SECONDS} s"
    echo "Power sampling period: ${POWER_PERIOD_SECONDS} s"
else
    echo "Power measurement: disabled"
fi
echo "CPU package-energy measurement: $([[ ${CPU_POWER_ENABLED} == 1 ]] && echo enabled || echo disabled)"

POWER_MONITOR_PID=""
POWER_PHASE_FILE=""
POWER_STOP_FILE=""

cleanup_power_monitor() {
    local original_status=$?

    trap - EXIT INT TERM

    if [[ -n "${POWER_MONITOR_PID}" ]] && \
       kill -0 "${POWER_MONITOR_PID}" 2>/dev/null; then
        if [[ -n "${POWER_STOP_FILE}" ]]; then
            touch "${POWER_STOP_FILE}" 2>/dev/null || true
        fi

        wait "${POWER_MONITOR_PID}" 2>/dev/null || true
    fi

    [[ -z "${POWER_PHASE_FILE}" ]] || rm -f "${POWER_PHASE_FILE}"
    [[ -z "${POWER_STOP_FILE}" ]] || rm -f "${POWER_STOP_FILE}"

    exit "${original_status}"
}

start_power_monitor() {
    local power_csv="${REPORT_GROUP}/power.csv"
    local power_summary="${REPORT_GROUP}/power_summary.json"
    local power_log="${REPORT_GROUP}/power_monitor.log"
    local monitor_status

    POWER_PHASE_FILE="${REPORT_GROUP}/.power_phase"
    POWER_STOP_FILE="${REPORT_GROUP}/.power_stop"

    rm -f \
        "${power_csv}" \
        "${power_summary}" \
        "${power_log}" \
        "${POWER_PHASE_FILE}" \
        "${POWER_STOP_FILE}"

    printf '%s\n' "pre" > "${POWER_PHASE_FILE}"

    "${PYTHON_BIN}" - \
        "${power_csv}" \
        "${power_summary}" \
        "${POWER_PHASE_FILE}" \
        "${POWER_STOP_FILE}" \
        "${POWER_PERIOD_SECONDS}" \
        >"${power_log}" 2>&1 <<'PY' &

import csv
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


power_csv = Path(sys.argv[1])
summary_json = Path(sys.argv[2])
phase_file = Path(sys.argv[3])
stop_file = Path(sys.argv[4])
period_s = float(sys.argv[5])

if period_s <= 0:
    raise SystemExit("Power sampling period must be greater than zero")

attempted = 0
failed = 0
overruns = 0

all_powers = []
all_read_times = []

phase_powers = {
    "pre": [],
    "benchmark": [],
    "post": [],
}


def extract_json(output):
    lines = output.splitlines()

    for index, line in enumerate(lines):
        if line.lstrip().startswith("{"):
            return json.loads("\n".join(lines[index:]))

    raise RuntimeError("No JSON object found in tt-smi output")


def get_phase():
    try:
        phase = phase_file.read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"

    return phase or "unknown"


def read_power():
    started = time.perf_counter()

    result = subprocess.run(
        ["tt-smi", "-s"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )

    read_time_s = time.perf_counter() - started
    payload = extract_json(result.stdout)

    devices = payload.get("device_info", [])

    if not devices:
        raise RuntimeError("tt-smi returned no devices")

    telemetry = devices[0].get("telemetry", {})

    if "power" not in telemetry:
        raise RuntimeError("tt-smi telemetry did not contain power")

    power_w = float(str(telemetry["power"]).strip())

    return power_w, read_time_s


def summarize_values(values):
    if not values:
        return {
            "samples": 0,
            "average_power_w": None,
            "power_stdev_w": None,
            "minimum_power_w": None,
            "maximum_power_w": None,
        }

    return {
        "samples": len(values),
        "average_power_w": statistics.fmean(values),
        "power_stdev_w": (
            statistics.pstdev(values)
            if len(values) >= 2
            else 0.0
        ),
        "minimum_power_w": min(values),
        "maximum_power_w": max(values),
    }


measurement_start = time.perf_counter()
next_sample_time = measurement_start

power_csv.parent.mkdir(parents=True, exist_ok=True)

with power_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)

    writer.writerow(
        [
            "sample",
            "timestamp_utc",
            "elapsed_s",
            "phase",
            "power_w",
            "read_time_ms",
            "scheduled_lateness_ms",
            "status",
            "error",
        ]
    )

    handle.flush()

    while not stop_file.exists():
        scheduled_time = next_sample_time
        sample_start = time.perf_counter()
        phase = get_phase()

        attempted += 1
        lateness_s = max(0.0, sample_start - scheduled_time)

        try:
            power_w, read_time_s = read_power()
            elapsed_s = time.perf_counter() - measurement_start

            all_powers.append(power_w)
            all_read_times.append(read_time_s)

            if phase in phase_powers:
                phase_powers[phase].append(power_w)

            writer.writerow(
                [
                    attempted,
                    datetime.now(timezone.utc).isoformat(),
                    f"{elapsed_s:.6f}",
                    phase,
                    f"{power_w:.6f}",
                    f"{read_time_s * 1000:.6f}",
                    f"{lateness_s * 1000:.6f}",
                    "ok",
                    "",
                ]
            )

            print(
                f"Power {attempted:4d} | "
                f"elapsed={elapsed_s:9.3f} s | "
                f"phase={phase:9s} | "
                f"power={power_w:7.2f} W | "
                f"read_time={read_time_s * 1000:8.2f} ms",
                flush=True,
            )

        except Exception as error:
            failed += 1
            elapsed_s = time.perf_counter() - measurement_start

            writer.writerow(
                [
                    attempted,
                    datetime.now(timezone.utc).isoformat(),
                    f"{elapsed_s:.6f}",
                    phase,
                    "",
                    "",
                    f"{lateness_s * 1000:.6f}",
                    "error",
                    str(error),
                ]
            )

            print(
                f"Power reading {attempted} failed: {error}",
                file=sys.stderr,
                flush=True,
            )

        handle.flush()

        next_sample_time += period_s
        sleep_s = next_sample_time - time.perf_counter()

        if sleep_s > 0:
            while sleep_s > 0 and not stop_file.exists():
                time.sleep(min(sleep_s, 0.05))
                sleep_s = next_sample_time - time.perf_counter()
        else:
            overruns += 1

            # Do not issue multiple immediate readings to catch up.
            next_sample_time = time.perf_counter()


measurement_duration_s = time.perf_counter() - measurement_start

summary = {
    "measurement_duration_s": measurement_duration_s,
    "requested_period_s": period_s,
    "requested_sampling_rate_hz": 1.0 / period_s,
    "attempted_readings": attempted,
    "successful_readings": len(all_powers),
    "failed_readings": failed,
    "period_overruns": overruns,
    "attempted_sampling_rate_hz": (
        attempted / measurement_duration_s
        if measurement_duration_s > 0
        else None
    ),
    "effective_sampling_rate_hz": (
        len(all_powers) / measurement_duration_s
        if measurement_duration_s > 0
        else None
    ),
    "average_reading_time_ms": (
        statistics.fmean(all_read_times) * 1000
        if all_read_times
        else None
    ),
    "minimum_reading_time_ms": (
        min(all_read_times) * 1000
        if all_read_times
        else None
    ),
    "maximum_reading_time_ms": (
        max(all_read_times) * 1000
        if all_read_times
        else None
    ),
    "overall": summarize_values(all_powers),
    "phases": {
        "pre": summarize_values(phase_powers["pre"]),
        "benchmark": summarize_values(phase_powers["benchmark"]),
        "post": summarize_values(phase_powers["post"]),
    },
}

summary_json.write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print()
print("Power measurement summary")
print("-------------------------")
print(f"Measurement duration:     {measurement_duration_s:.3f} s")
print(f"Requested period:         {period_s:.3f} s")
print(f"Attempted readings:       {attempted}")
print(f"Successful readings:      {len(all_powers)}")
print(f"Failed readings:          {failed}")
print(f"Period overruns:          {overruns}")

if all_read_times:
    print(
        "Average reading time:     "
        f"{statistics.fmean(all_read_times) * 1000:.3f} ms"
    )
    print(
        "Minimum reading time:     "
        f"{min(all_read_times) * 1000:.3f} ms"
    )
    print(
        "Maximum reading time:     "
        f"{max(all_read_times) * 1000:.3f} ms"
    )

print()
print("Power results")
print("-------------")

for phase_name in ("pre", "benchmark", "post"):
    values = phase_powers[phase_name]

    if values:
        average_power_w = statistics.fmean(values)
        power_stdev_w = (
            statistics.pstdev(values)
            if len(values) >= 2
            else 0.0
        )

        print(f"{phase_name.capitalize()} phase:")
        print(f"  Samples:               {len(values)}")
        print(f"  Average power:         {average_power_w:.3f} W")
        print(f"  Power standard dev.:   {power_stdev_w:.3f} W")
        print(f"  Minimum power:         {min(values):.3f} W")
        print(f"  Maximum power:         {max(values):.3f} W")
    else:
        print(f"{phase_name.capitalize()} phase: no successful samples")

if measurement_duration_s > 0:
    attempted_frequency_hz = attempted / measurement_duration_s
    successful_frequency_hz = len(all_powers) / measurement_duration_s
else:
    attempted_frequency_hz = 0.0
    successful_frequency_hz = 0.0

print()
print("Power measurement frequency")
print("---------------------------")
print(f"Requested frequency:       {1.0 / period_s:.3f} Hz")
print(f"Attempted frequency:       {attempted_frequency_hz:.3f} Hz")
print(f"Successful frequency:      {successful_frequency_hz:.3f} Hz")
print(f"Requested period:          {period_s:.3f} s")

if all_read_times:
    print(
        "Average tt-smi read time: "
        f"{statistics.fmean(all_read_times) * 1000:.3f} ms"
    )
PY

    POWER_MONITOR_PID=$!

    sleep 0.2

    if ! kill -0 "${POWER_MONITOR_PID}" 2>/dev/null; then
        set +e
        wait "${POWER_MONITOR_PID}"
        monitor_status=$?
        set -e

        POWER_MONITOR_PID=""

        cat "${power_log}" >&2

        die "power monitor failed to start with status ${monitor_status}"
    fi

    echo "Power monitor started in background with PID ${POWER_MONITOR_PID}"
}

stop_power_monitor() {
    local power_log="${REPORT_GROUP}/power_monitor.log"
    local monitor_status

    [[ -n "${POWER_MONITOR_PID}" ]] || return 0

    touch "${POWER_STOP_FILE}"

    set +e
    wait "${POWER_MONITOR_PID}"
    monitor_status=$?
    set -e

    POWER_MONITOR_PID=""

    rm -f "${POWER_PHASE_FILE}" "${POWER_STOP_FILE}"

    echo
    cat "${power_log}"

    if (( monitor_status != 0 )); then
        echo "Power monitor failed with status ${monitor_status}" >&2
        return "${monitor_status}"
    fi

    return 0
}

trap cleanup_power_monitor EXIT INT TERM

if (( POWER_ENABLED )); then
    start_power_monitor

    echo
    echo "Measuring pre-benchmark power for ${POWER_PRE_SECONDS} seconds..."
    sleep "${POWER_PRE_SECONDS}"

    printf '%s\n' "benchmark" > "${POWER_PHASE_FILE}"
fi

echo
echo "Starting Tracy benchmark..."
echo

run_tracy_benchmark() {
    if (( CPU_POWER_ENABLED )); then
        CPU_POWER_ENABLED=1 \
        bash "${SCRIPT_DIR}/run_with_cpu_power.sh" "${REPORT_GROUP}" -- \
            "${PYTHON_BIN}" -m tracy \
            -o "${REPORT_GROUP}" \
            -r \
            -p \
            "${EXECUTABLE}" "${BENCHMARK_ARGUMENTS[@]}"
    else
        "${PYTHON_BIN}" -m tracy \
            -o "${REPORT_GROUP}" \
            -r \
            -p \
            "${EXECUTABLE}" "${BENCHMARK_ARGUMENTS[@]}"
    fi
}

set +e
run_tracy_benchmark 2>&1 | tee "${REPORT_GROUP}/run.log"

benchmark_status=${PIPESTATUS[0]}
set -e

if (( POWER_ENABLED )); then
    printf '%s\n' "post" > "${POWER_PHASE_FILE}"

    echo
    echo "Benchmark process ended with status ${benchmark_status}."
    echo "Measuring post-benchmark power for ${POWER_POST_SECONDS} seconds..."

    sleep "${POWER_POST_SECONDS}"

    set +e
    stop_power_monitor
    power_monitor_status=$?
    set -e
else
    power_monitor_status=0
fi

trap - EXIT INT TERM

if (( benchmark_status != 0 )); then
    echo \
        "Benchmark failed with status ${benchmark_status}; partial reports remain in ${REPORT_GROUP}" \
        >&2
    exit "${benchmark_status}"
fi

if (( power_monitor_status != 0 )); then
    echo \
        "Benchmark succeeded, but power measurement failed with status ${power_monitor_status}" \
        >&2
    exit "${power_monitor_status}"
fi

mapfile -t OPS_FILES < <(
    find "${REPORT_GROUP}" \
        -type f \
        -name 'ops_perf_results_*.csv' \
        -print
)

if (( ${#OPS_FILES[@]} != 1 )); then
    die \
        "expected exactly one ops_perf_results_*.csv under ${REPORT_GROUP}, found ${#OPS_FILES[@]}"
fi

OPS_FILE="${OPS_FILES[0]}"
RUN_DIR="$(dirname "${OPS_FILE}")"
REPORTS_DIR="$(dirname "${RUN_DIR}")"
SUMMARY_CSV="${REPORTS_DIR%/}_summary.csv"

HOST_TIMES_FILE="${REPORT_GROUP}/.logs/tracy_ops_times.csv"
if [[ -f "${HOST_TIMES_FILE}" ]]; then
    cp -- "${HOST_TIMES_FILE}" "${RUN_DIR}/tracy_ops_times.csv"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_host_phases.py" \
        "${RUN_DIR}/tracy_ops_times.csv" \
        "${RUN_DIR}/host_phase_summary.csv"
else
    echo "Warning: Tracy host times not found: ${HOST_TIMES_FILE}" >&2
fi

[[ ! -e "${SUMMARY_CSV}" ]] || \
    die "summary already exists: ${SUMMARY_CSV}"

echo "Generating ${RUN_DIR}/perf_report.csv"

(
    cd "${RUN_DIR}"

    tt-perf-report "$(basename "${OPS_FILE}")" \
        --no-advice \
        --csv perf_report.csv
)

[[ -f "${RUN_DIR}/perf_report.csv" ]] || \
    die "tt-perf-report did not create ${RUN_DIR}/perf_report.csv"

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_reports.py" \
    --reports-dir "${REPORTS_DIR}" \
    --operation "${SUMMARY_OPERATION}" \
    --skip-warmup-runs 0 \
    --warmup-iterations "${WARMUP_ITERATIONS}" \
    --measured-iterations "${MEASURED_ITERATIONS}" \
    --csv-only

[[ -f "${SUMMARY_CSV}" ]] || \
    die "summary was not created: ${SUMMARY_CSV}"

"${PYTHON_BIN}" - \
    "${SUMMARY_CSV}" \
    "${RESULT_ID}" \
    "${BENCHMARK_KIND}" \
    "${SUMMARY_OPERATION}" \
    "${REPORT_GROUP}" \
    "${POWER_ENABLED}" \
    "${CPU_POWER_ENABLED}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path


def normalize(value):
    if value is None or value == "":
        return None

    lowered = value.lower()

    if lowered in {"true", "false"}:
        return lowered == "true"

    try:
        number = float(value)
    except ValueError:
        return value

    if not math.isfinite(number):
        return None

    if number.is_integer() and not any(
        mark in value.lower() for mark in (".", "e")
    ):
        return int(number)

    return number


summary_path = Path(sys.argv[1]).resolve()

with summary_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

if len(rows) != 1:
    raise SystemExit(
        f"Expected one summary row in {summary_path}, found {len(rows)}"
    )

result = {
    key: normalize(value)
    for key, value in rows[0].items()
}

report_dir = Path(sys.argv[5]).resolve()
power_enabled = sys.argv[6] == "1"
cpu_power_enabled = sys.argv[7] == "1"

result.update(
    {
        "result_id": sys.argv[2],
        "accelerator": "tenstorrent",
        "benchmark_kind": sys.argv[3],
        "operation": sys.argv[4],
        "measurement_source": "tracy",
        "report_dir": str(report_dir),
        "summary_csv": str(summary_path),
        "power_measurement_enabled": power_enabled,
        "cpu_power_measurement_enabled": cpu_power_enabled,
    }
)

if power_enabled:
    power_csv = report_dir / "power.csv"
    power_summary_path = report_dir / "power_summary.json"

    result["power_csv"] = str(power_csv)
    result["power_summary_json"] = str(power_summary_path)

    if power_summary_path.is_file():
        power_summary = json.loads(
            power_summary_path.read_text(encoding="utf-8")
        )

        benchmark_power = (
            power_summary
            .get("phases", {})
            .get("benchmark", {})
        )

        result["benchmark_power_samples"] = benchmark_power.get(
            "samples"
        )
        result["benchmark_average_power_w"] = benchmark_power.get(
            "average_power_w"
        )
        result["benchmark_power_stdev_w"] = benchmark_power.get(
            "power_stdev_w"
        )
        result["benchmark_minimum_power_w"] = benchmark_power.get(
            "minimum_power_w"
        )
        result["benchmark_maximum_power_w"] = benchmark_power.get(
            "maximum_power_w"
        )
        result["power_sampling_period_s"] = power_summary.get(
            "requested_period_s"
        )
        result["power_requested_sampling_rate_hz"] = power_summary.get(
            "requested_sampling_rate_hz"
        )
        result["power_attempted_sampling_rate_hz"] = power_summary.get(
            "attempted_sampling_rate_hz"
        )
        result["power_effective_sampling_rate_hz"] = power_summary.get(
            "effective_sampling_rate_hz"
        )
        result["power_average_reading_time_ms"] = power_summary.get(
            "average_reading_time_ms"
        )
        result["power_period_overruns"] = power_summary.get(
            "period_overruns"
        )

if cpu_power_enabled:
    cpu_report_dir = Path(result.get("run_dir") or report_dir)
    cpu_power_csv = cpu_report_dir / "cpu_power.csv"
    cpu_power_summary_path = cpu_report_dir / "cpu_power_summary.json"
    result["cpu_power_csv"] = str(cpu_power_csv)
    result["cpu_power_summary_json"] = str(cpu_power_summary_path)
    if cpu_power_summary_path.is_file():
        result.update(json.loads(cpu_power_summary_path.read_text(encoding="utf-8")))

print(
    "HICREST_RESULT "
    + json.dumps(
        result,
        sort_keys=True,
        allow_nan=False,
    )
)
PY

echo "Completed result ${RESULT_ID}"
