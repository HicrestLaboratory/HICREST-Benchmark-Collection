#!/usr/bin/env python3

"""Summarize package-energy samples produced by the CPU energy monitor."""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    if not {"timestamp", "energy"}.issubset(data.columns):
        parser.error(f"unexpected perf log columns in {args.input}")

    energy = pd.to_numeric(data["energy"], errors="coerce").dropna()
    timestamps = pd.to_numeric(data["timestamp"], errors="coerce").dropna()
    sample_count = int(energy.count())
    total_energy_j = float(energy.sum()) if sample_count else None
    backend_path = args.input.with_name("cpu_power_backend.txt")
    source = (
        backend_path.read_text(encoding="utf-8").strip()
        if backend_path.is_file()
        else "unknown CPU energy backend"
    )

    # perf_start.sh measures energy over one `sleep 1` interval per sample.
    measured_interval_s = float(sample_count) if sample_count else None
    summary = {
        "cpu_power_source": source,
        "cpu_power_sample_period_s": 1.0,
        "cpu_power_samples": sample_count,
        "cpu_energy_pkg_j": total_energy_j,
        "cpu_average_package_power_w": (
            total_energy_j / measured_interval_s
            if total_energy_j is not None and measured_interval_s
            else None
        ),
        "cpu_energy_pkg_mean_sample_j": (
            float(energy.mean()) if sample_count else None
        ),
        "cpu_energy_pkg_min_sample_j": (
            float(energy.min()) if sample_count else None
        ),
        "cpu_energy_pkg_max_sample_j": (
            float(energy.max()) if sample_count else None
        ),
        "cpu_power_timestamp_span_s": (
            float(timestamps.max() - timestamps.min())
            if len(timestamps) > 1
            else 0.0 if len(timestamps) == 1 else None
        ),
    }

    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote CPU power summary: {args.output}")


if __name__ == "__main__":
    main()
