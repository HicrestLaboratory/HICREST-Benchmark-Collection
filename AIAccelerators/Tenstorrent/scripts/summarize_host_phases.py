#!/usr/bin/env python3

"""Summarize benchmark host zones exported by Tracy."""

import argparse
from pathlib import Path

import pandas as pd


PHASES = (
    "CompileProgram",
    "device_input_creation",
    "host_input_creation",
    "host_to_device_transfer",
    "device_to_host_transfer",
    "first_run_dispatch",
    "first_run_synchronize",
    "layout_conversion_interleaved_to_sharded",
    "layout_conversion_row_major_to_tile",
    "steady_state_dispatch",
    "steady_state_synchronize",
    "warmup_dispatch",
    "warmup_synchronize",
)


def field_name(name):
    if name == "CompileProgram":
        return "compile"
    return name


def summarize(input_path):
    data = pd.read_csv(input_path, low_memory=False)
    required = {"name", "exec_time_ns"}
    missing = required.difference(data.columns)
    if missing:
        raise RuntimeError(
            f"{input_path} is missing columns: {', '.join(sorted(missing))}"
        )

    data["exec_time_us"] = pd.to_numeric(
        data["exec_time_ns"], errors="coerce"
    ) / 1000.0
    data = data[data["name"].astype(str).isin(PHASES)]

    result = {"host_phase_file": str(input_path.resolve())}
    for phase in PHASES:
        values = data.loc[data["name"] == phase, "exec_time_us"].dropna()
        prefix = field_name(phase)
        result[f"{prefix}_count"] = int(values.count())
        result[f"{prefix}_time_us"] = values.mean() if not values.empty else None
        result[f"{prefix}_time_sum_us"] = values.sum() if not values.empty else None
        result[f"{prefix}_time_min_us"] = values.min() if not values.empty else None
        result[f"{prefix}_time_max_us"] = values.max() if not values.empty else None

    def total(*names):
        values = [result.get(name) for name in names]
        if any(value is None for value in values):
            return None
        return sum(values)

    result["first_run_time_us"] = total(
        "first_run_dispatch_time_sum_us",
        "first_run_synchronize_time_sum_us",
    )
    result["steady_state_host_time_sum_us"] = total(
        "steady_state_dispatch_time_sum_us",
        "steady_state_synchronize_time_sum_us",
    )
    result["warmup_host_time_sum_us"] = total(
        "warmup_dispatch_time_sum_us",
        "warmup_synchronize_time_sum_us",
    )
    layout_values = data.loc[
        data["name"].astype(str).str.startswith("layout_conversion_"),
        "exec_time_us",
    ].dropna()
    result["layout_conversion_count"] = int(layout_values.count())
    result["layout_conversion_time_sum_us"] = (
        layout_values.sum() if not layout_values.empty else 0.0
    )
    result["cold_end_to_end_time_us"] = total(
        "host_input_creation_time_sum_us",
        "host_to_device_transfer_time_sum_us",
        "layout_conversion_time_sum_us",
        "first_run_time_us",
    )
    result["steady_state_end_to_end_time_us"] = total(
        "host_input_creation_time_sum_us",
        "host_to_device_transfer_time_sum_us",
        "layout_conversion_time_sum_us",
        "steady_state_host_time_sum_us",
        "device_to_host_transfer_time_sum_us",
    )
    result["measurement_mode"] = (
        "end-to-end" if result["host_to_device_transfer_count"] else "kernel"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Tracy tracy_ops_times.csv")
    parser.add_argument("output", type=Path, help="Output one-row CSV")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input does not exist: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summarize(args.input)]).to_csv(args.output, index=False)
    print(f"Wrote host phase summary: {args.output}")


if __name__ == "__main__":
    main()
