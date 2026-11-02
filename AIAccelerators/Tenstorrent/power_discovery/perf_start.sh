#!/usr/bin/env bash

# Records one package-energy delta (J) per second in ./perf_log. The historic
# filename is retained for compatibility with the benchmark launchers.
set -u

OUTFILE="perf_log"
BACKEND_FILE="cpu_power_backend.txt"
BACKEND="${CPU_POWER_BACKEND:-auto}"
PERF_EVENT="${PERF_ENERGY_EVENT:-power/energy-pkg/}"

if [[ "${BACKEND}" == "auto" ]]; then
    if [[ -d /sys/bus/event_source/devices/power ]] && command -v perf >/dev/null 2>&1; then
        BACKEND="perf"
    else
        BACKEND="sysfs"
    fi
fi

printf 'timestamp,energy\n' > "${OUTFILE}"

is_number() {
    [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

sample_perf() {
    printf 'perf %s\n' "${PERF_EVENT}" > "${BACKEND_FILE}"
    while true; do
        energy="$(LC_ALL=C perf stat -a -x, -e "${PERF_EVENT}" -- sleep 1 2>&1 | \
            awk -F, -v event="${PERF_EVENT}" '$3 == event && $1 ~ /^[0-9.]+$/ { print $1; exit }')"
        if is_number "${energy}"; then
            printf '%s,%s\n' "$(date +%s)" "${energy}" >> "${OUTFILE}"
        else
            echo "CPU energy sample unavailable for ${PERF_EVENT}; no invalid value was recorded" >&2
            sleep 1
        fi
    done
}

sample_sysfs() {
    local energy_path name range_path current range delta_uj total_uj index
    local -a energy_paths previous ranges

    for energy_path in /sys/class/powercap/intel-rapl:*/energy_uj; do
        [[ -r "${energy_path}" ]] || continue
        name="$(< "${energy_path%/energy_uj}/name")"
        [[ "${name}" == package-* ]] || continue
        energy_paths+=("${energy_path}")
        previous+=("$(< "${energy_path}")")
        range_path="${energy_path%/energy_uj}/max_energy_range_uj"
        if [[ -r "${range_path}" ]]; then
            ranges+=("$(< "${range_path}")")
        else
            ranges+=(0)
        fi
    done

    if (( ${#energy_paths[@]} == 0 )); then
        echo "No readable Intel RAPL package energy_uj counters were found" >&2
        exit 2
    fi

    printf 'sysfs intel-rapl package energy_uj\n' > "${BACKEND_FILE}"
    while true; do
        sleep 1
        total_uj=0
        for index in "${!energy_paths[@]}"; do
            current="$(< "${energy_paths[index]}")"
            delta_uj=$(( current - previous[index] ))
            if (( delta_uj < 0 )); then
                range="${ranges[index]}"
                (( range > 0 )) || {
                    echo "RAPL energy counter wrapped but max_energy_range_uj is unavailable" >&2
                    exit 2
                }
                delta_uj=$(( delta_uj + range ))
            fi
            previous[index]="${current}"
            total_uj=$(( total_uj + delta_uj ))
        done
        awk -v timestamp="$(date +%s)" -v energy_uj="${total_uj}" \
            'BEGIN { printf "%s,%.9f\n", timestamp, energy_uj / 1000000 }' >> "${OUTFILE}"
    done
}

case "${BACKEND}" in
    perf) sample_perf ;;
    sysfs) sample_sysfs ;;
    *)
        echo "Unsupported CPU_POWER_BACKEND '${BACKEND}'; use auto, perf, or sysfs" >&2
        exit 2
        ;;
esac
