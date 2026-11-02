#!/usr/bin/env python3

"""
Aggregate Tenstorrent GEMM, GEMM-scaling, Softmax, ReLU, or GELU reports.

Input layout
------------

--reports-dir must name one configuration folder containing timestamped run
directories. Each run directory must contain perf_report.csv and an
ops_perf_results_*.csv file. Supported folder names include:

    generated/reports/gemm_bfp8_lofi_reports
    generated/reports/gemm_scaling_bf16_hifi4_reports
    generated/reports/relu_bfp8_reports
    generated/reports/gelu_bf16_reports
    generated/reports/softmax_fp32_reports

Operation selection
-------------------

--operation auto infers matmul from a folder containing "gemm" or "matmul",
and infers softmax, relu, or gelu from the corresponding folder name. An
operation can also be selected explicitly.

Outputs
-------

The per-run aggregate CSV is written beside the reports directory as:

    <reports-directory-name>_summary.csv

For example:

    generated/reports/gemm_bfp8_lofi_reports_summary.csv

By default, the script also writes a Markdown summary and a long-form
operation-breakdown CSV. Pass --csv-only to generate only the aggregate CSV;
this is the mode used by run_and_collect_reports.sh.

The aggregate CSV includes exact mean-value columns for:

    NOC UTIL (%)
    MULTICAST NOC UTIL (%)
    NPE CONG IMPACT (%)
    DRAM BW UTIL (%)
    Packer Efficiency Avg (%)
    Packer Engine Util Avg (%)
    Unpacker Write Efficiency Avg (%)
    Unpacker-to-Math Data Flow Avg (%)

Warm-up handling
----------------

The default is --skip-warmup-runs 1. The integrated collection workflow uses
--skip-warmup-runs 0 so every generated run is included.

Examples
--------

CSV-only summary with operation inferred from the folder name:

    python scripts/summarize_reports_all.py \
        --reports-dir generated/reports/gelu_bf16_reports \
        --operation auto \
        --skip-warmup-runs 0 \
        --csv-only

Full GEMM summary with the default one-run warm-up exclusion:

    python scripts/summarize_reports_all.py \
        --reports-dir generated/reports/gemm_bfp8_lofi_reports \
        --operation matmul
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


OPERATION_CONFIG = {
    "matmul": {
        "report_op_code": "MatmulDeviceOperation",
        "display_name": "Matmul",
    },
    "softmax": {
        "report_op_code": "SoftmaxDeviceOperation",
        "display_name": "Softmax",
    },
    "relu": {
        "report_op_code": "UnaryDeviceOperation",
        "display_name": "ReLU",
    },
    "gelu": {
        "report_op_code": "UnaryDeviceOperation",
        "display_name": "GELU",
    },
}


MICROARCHITECTURAL_COLUMNS = [
    "Avg FPU util on full grid (%)",
    "PM FPU UTIL (%)",
    "FPU Util Min (%)",
    "FPU Util Median (%)",
    "FPU Util Max (%)",
    "Thread 0 Stall Rate Avg (%)",
    "Thread 1 Stall Rate Avg (%)",
    "Thread 2 Stall Rate Avg (%)",
    "Math Scoreboard Stall Rate Avg (%)",
    "Data Hazard Stall Rate Avg (%)",
    "Fidelity Stall Rate Avg (%)",
    "Math Pipeline Utilization Avg (%)",
    "FPU Execution Efficiency Avg (%)",
    "MATH Instrn Avail Rate Avg (%)",
    "T1 Instrn Issue Rate Avg (%)",
    "Math-to-Pack Handoff Efficiency Avg (%)",
    "NOC UTIL (%)",
    "MULTICAST NOC UTIL (%)",
    "NPE CONG IMPACT (%)",
    "DRAM BW UTIL (%)",
    "PM REQ I BW",
    "PM REQ O BW",
    "Packer Efficiency Avg (%)",
    "Packer Engine Util Avg (%)",
    "Unpacker Write Efficiency Avg (%)",
    "Unpacker-to-Math Data Flow Avg (%)",
    "Pack Idle Wait T2 Avg (%)",
    "Unpack Idle Wait T0 Avg (%)",
]

FINAL_REPORT_COLUMNS = {
    "NOC UTIL (%)",
    "MULTICAST NOC UTIL (%)",
    "NPE CONG IMPACT (%)",
    "DRAM BW UTIL (%)",
    "Packer Efficiency Avg (%)",
    "Packer Engine Util Avg (%)",
    "Unpacker Write Efficiency Avg (%)",
    "Unpacker-to-Math Data Flow Avg (%)",
}


def numeric_series(df, column_name):
    """
    Return a numeric Series for a DataFrame column.

    Missing columns return an empty Series.
    """
    if column_name not in df.columns:
        return pd.Series(dtype=float)

    return pd.to_numeric(
        df[column_name],
        errors="coerce",
    ).dropna()


def mean_col(df, column_name):
    values = numeric_series(df, column_name)

    if values.empty:
        return None

    return values.mean()


def sum_col(df, column_name):
    values = numeric_series(df, column_name)

    if values.empty:
        return None

    return values.sum()


def summarize_col(df, column_name):
    values = numeric_series(df, column_name)

    if values.empty:
        return None

    return {
        "min": values.min(),
        "mean": values.mean(),
        "median": values.median(),
        "max": values.max(),
    }


def fmt(value, unit="", digits=3):
    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:.{digits}f}{unit}"


def sanitize_name(value):
    result = str(value).strip().lower()

    replacements = {
        " ": "_",
        "/": "_",
        "\\": "_",
        "=>": "_to_",
        "(": "",
        ")": "",
        "[": "",
        "]": "",
        "%": "pct",
        "-": "_",
        ":": "_",
        ",": "_",
    }

    for old, new in replacements.items():
        result = result.replace(old, new)

    result = re.sub(r"_+", "_", result)

    return result.strip("_")


def infer_operation(reports_dir, requested_operation):
    """
    Infer the operation from the report directory name when auto mode
    is selected.
    """
    if requested_operation != "auto":
        return requested_operation

    directory_name = reports_dir.name.lower()

    if "softmax" in directory_name:
        return "softmax"

    if "gelu" in directory_name:
        return "gelu"

    if "relu" in directory_name:
        return "relu"

    if "matmul" in directory_name:
        return "matmul"

    if "gemm" in directory_name:
        return "matmul"

    raise RuntimeError(
        "Could not infer the operation from the reports directory name. "
        "Pass --operation matmul, softmax, relu, or gelu."
    )


def find_run_files(run_dir):
    """
    Locate perf_report.csv and ops_perf_results_*.csv in a run folder.
    """
    perf_path = run_dir / "perf_report.csv"
    ops_files = sorted(
        run_dir.glob("ops_perf_results_*.csv")
    )

    if not perf_path.exists():
        return None, None

    if not ops_files:
        return None, None

    if len(ops_files) > 1:
        print(
            f"Warning: {run_dir} contains multiple "
            "ops_perf_results CSV files."
        )
        print(f"Using: {ops_files[0]}")

    return perf_path, ops_files[0]


def read_host_phase_summary(run_dir):
    """Read optional phase-resolved host timings for one run."""
    path = run_dir / "host_phase_summary.csv"
    if not path.is_file():
        return {}

    data = pd.read_csv(path, low_memory=False)
    if len(data) != 1:
        raise RuntimeError(f"Expected one row in {path}, found {len(data)}")

    result = data.iloc[0].to_dict()
    result["host_phase_summary_file"] = str(path)
    return result


def read_cpu_power_summary(run_dir):
    """Read optional perf package-energy summary for one run."""
    path = run_dir / "cpu_power_summary.json"
    if not path.is_file():
        return {}
    result = json.loads(path.read_text(encoding="utf-8"))
    result["cpu_power_csv"] = str(run_dir / "cpu_power.csv")
    result["cpu_power_summary_json"] = str(path)
    return result


def normalize_operation_code(value):
    """
    Remove annotations from an operation code.

    Example:

        UnaryDeviceOperation (in0:dram_interleaved)

    becomes:

        UnaryDeviceOperation
    """
    value = str(value).strip()

    return re.sub(
        r"\s*\([^)]*\)\s*$",
        "",
        value,
    )


def classify_operation(op_code):
    """
    Assign broad operation categories similar to tt-perf-report.
    """
    normalized = normalize_operation_code(op_code)

    category_mapping = {
        "MatmulDeviceOperation": "Compute",
        "SoftmaxDeviceOperation": "Compute",
        "UnaryDeviceOperation": "Compute",
        "RandDeviceOperation": "Other",
        "TypecastDeviceOperation": "TM",
    }

    return category_mapping.get(
        normalized,
        "Other",
    )


def get_operation_rows(
    df,
    column_name,
    operation_name,
    report_op_code,
    source_name,
):
    """
    Select rows belonging to the requested benchmark operation.
    """
    if column_name not in df.columns:
        raise RuntimeError(
            f"Could not find {column_name!r} in {source_name}"
        )

    rows = df[
        df[column_name]
        .astype(str)
        .str.contains(
            report_op_code,
            case=False,
            regex=False,
            na=False,
        )
    ].copy()

    if rows.empty:
        available = sorted(
            df[column_name]
            .dropna()
            .astype(str)
            .unique()
        )

        available_text = ", ".join(
            available[:10]
        )

        if len(available) > 10:
            available_text += ", ..."

        raise RuntimeError(
            f"No {report_op_code} rows found for "
            f"{operation_name!r} in {source_name}. "
            f"Available OP Codes: {available_text}"
        )

    return rows


def select_measured_iterations(
    operation_perf_df,
    operation_ops_df,
    warmup_iterations=0,
    measured_iterations=None,
):
    """
    Remove in-process warm-up operation entries from one Tracy report.

    The benchmark emits one target-operation row per warm-up or measured
    iteration. The two profiler CSVs are ordered and aligned using the global
    call count so both contribute statistics for exactly the same iterations.

    When measured_iterations is None and warmup_iterations is zero, return the
    original rows unchanged. This preserves the historical behaviour for
    existing callers that do not use the new CLI options.
    """
    if warmup_iterations < 0:
        raise RuntimeError(
            "warmup_iterations cannot be negative."
        )

    if (
        measured_iterations is not None
        and measured_iterations <= 0
    ):
        raise RuntimeError(
            "measured_iterations must be greater than zero."
        )

    if (
        warmup_iterations == 0
        and measured_iterations is None
    ):
        return (
            operation_perf_df,
            operation_ops_df,
            len(operation_perf_df),
        )

    perf_call_count = "Global Call Count"
    ops_call_count = "GLOBAL CALL COUNT"

    if perf_call_count not in operation_perf_df.columns:
        raise RuntimeError(
            f"Could not find {perf_call_count!r} in perf_report.csv; "
            "cannot identify warm-up iterations."
        )

    if ops_call_count not in operation_ops_df.columns:
        raise RuntimeError(
            f"Could not find {ops_call_count!r} in ops profiler CSV; "
            "cannot identify warm-up iterations."
        )

    perf_rows = operation_perf_df.copy()
    ops_rows = operation_ops_df.copy()

    perf_rows["_global_call_count"] = pd.to_numeric(
        perf_rows[perf_call_count],
        errors="raise",
    )
    ops_rows["_global_call_count"] = pd.to_numeric(
        ops_rows[ops_call_count],
        errors="raise",
    )

    perf_rows = perf_rows.sort_values(
        "_global_call_count"
    ).reset_index(drop=True)
    ops_rows = ops_rows.sort_values(
        "_global_call_count"
    ).reset_index(drop=True)

    perf_counts = perf_rows["_global_call_count"].tolist()
    ops_counts = ops_rows["_global_call_count"].tolist()

    perf_total_iterations = len(perf_rows)
    ops_total_iterations = len(ops_rows)

    if measured_iterations is None:
        expected_iterations = perf_total_iterations
    else:
        expected_iterations = warmup_iterations + measured_iterations

    # tt-perf-report honours steady_state:start and can therefore emit only the
    # measured target operations. The raw Tracy ops CSV retains the warm-ups.
    # Their global call counters belong to different profiler views and cannot
    # be compared directly in this valid configuration.
    if (
        measured_iterations is not None
        and perf_total_iterations == measured_iterations
        and ops_total_iterations == expected_iterations
    ):
        print(
            "perf_report.csv is already filtered to the measured operations; "
            f"dropping {warmup_iterations} warm-up rows only from "
            "ops_perf_results CSV."
        )
        return (
            perf_rows.drop(columns=["_global_call_count"]),
            ops_rows.iloc[warmup_iterations:].drop(
                columns=["_global_call_count"]
            ),
            expected_iterations,
        )

    if (
        measured_iterations is not None
        and perf_total_iterations == expected_iterations
        and ops_total_iterations == measured_iterations
    ):
        print(
            "ops_perf_results CSV is already filtered to the measured "
            f"operations; dropping {warmup_iterations} warm-up rows only "
            "from perf_report.csv."
        )
        return (
            perf_rows.iloc[warmup_iterations:].drop(
                columns=["_global_call_count"]
            ),
            ops_rows.drop(columns=["_global_call_count"]),
            expected_iterations,
        )

    if perf_total_iterations != ops_total_iterations:
        raise RuntimeError(
            "Target-operation row counts differ between perf_report.csv "
            f"({perf_total_iterations}) and ops_perf_results CSV "
            f"({ops_total_iterations})."
        )

    total_iterations = perf_total_iterations

    if measured_iterations is not None and total_iterations != expected_iterations:
        raise RuntimeError(
            f"Expected {expected_iterations} target-operation entries "
            f"({warmup_iterations} warm-up + {measured_iterations} measured), "
            f"but found {total_iterations}."
        )

    if warmup_iterations >= total_iterations:
        raise RuntimeError(
            f"Cannot skip {warmup_iterations} warm-up entries from "
            f"only {total_iterations} target-operation entries."
        )

    if perf_counts != ops_counts:
        print(
            "Warning: target-operation Global Call Count values differ "
            "between profiler views; aligning the equally sized, sorted "
            "target-operation rows by order."
        )

    measured_perf_rows = perf_rows.iloc[
        warmup_iterations:
    ].copy()
    measured_ops_rows = ops_rows.iloc[
        warmup_iterations:
    ].copy()

    measured_perf_rows = measured_perf_rows.drop(
        columns=["_global_call_count"]
    )
    measured_ops_rows = measured_ops_rows.drop(
        columns=["_global_call_count"]
    )

    return (
        measured_perf_rows,
        measured_ops_rows,
        total_iterations,
    )


def parse_matmul_size(operation_df):
    """
    Parse M, K, N from strings such as:

        MatmulDeviceOperation 4096 x 4096 x 4096
    """
    if "OP Code" not in operation_df.columns:
        return None, None, None

    for value in operation_df["OP Code"].astype(str):
        match = re.search(
            r"(\d+)\s*x\s*(\d+)\s*x\s*(\d+)",
            value,
            flags=re.IGNORECASE,
        )

        if match:
            return tuple(
                int(item)
                for item in match.groups()
            )

    return None, None, None


def parse_shape_from_text(value):
    """
    Best-effort parsing of tensor shapes from text.
    """
    text = str(value)

    bracket_match = re.search(
        r"[\[\(]([\d,\s]+)[\]\)]",
        text,
    )

    if bracket_match:
        numbers = [
            int(item)
            for item in re.findall(
                r"\d+",
                bracket_match.group(1),
            )
        ]

        if numbers:
            return tuple(numbers)

    x_match = re.search(
        r"(\d+(?:\s*x\s*\d+)+)",
        text,
        flags=re.IGNORECASE,
    )

    if x_match:
        numbers = [
            int(item)
            for item in re.findall(
                r"\d+",
                x_match.group(1),
            )
        ]

        if numbers:
            return tuple(numbers)

    return None


def parse_padded_logical_dimension(value):
    """
    Parse profiler dimensions formatted as PADDED[LOGICAL].

    Returns:

        padded_dimension, logical_dimension
    """
    match = re.fullmatch(
        r"\s*(\d+)\s*\[\s*(\d+)\s*\]\s*",
        str(value),
    )

    if not match:
        return None

    return tuple(
        int(item)
        for item in match.groups()
    )


def detect_logical_tensor_shape(operation_ops_df):
    """
    Read a logical tensor shape from raw Tracy PAD[LOGICAL] columns.

    TTNN profiler rows commonly represent a two-dimensional tensor as
    W=1, Z=1, Y=H, X=W. Leading singleton dimensions are omitted from the
    reported logical shape while meaningful trailing singleton dimensions are
    retained.
    """
    axes = ("W", "Z", "Y", "X")

    for tensor_prefix in ("INPUT_0", "OUTPUT_0"):
        columns = [
            f"{tensor_prefix}_{axis}_PAD[LOGICAL]"
            for axis in axes
        ]

        if not all(
            column in operation_ops_df.columns
            for column in columns
        ):
            continue

        discovered_shapes = []

        for _, row in operation_ops_df.iterrows():
            logical_dimensions = []

            for column in columns:
                parsed = parse_padded_logical_dimension(
                    row[column]
                )

                if parsed is None:
                    logical_dimensions = []
                    break

                _, logical_dimension = parsed
                logical_dimensions.append(
                    logical_dimension
                )

            if logical_dimensions:
                while (
                    len(logical_dimensions) > 1
                    and logical_dimensions[0] == 1
                ):
                    logical_dimensions.pop(0)

                discovered_shapes.append(
                    tuple(logical_dimensions)
                )

        if not discovered_shapes:
            continue

        unique_shapes = set(discovered_shapes)

        if len(unique_shapes) != 1:
            raise RuntimeError(
                f"Measured {tensor_prefix} rows contain inconsistent "
                f"logical shapes: {sorted(unique_shapes)}"
            )

        return discovered_shapes[0]

    return None


def detect_shape(operation_perf_df, operation_ops_df):
    """
    Try several profiler column names to locate a tensor shape.
    """
    candidate_columns = [
        "Input Shape",
        "INPUT SHAPE",
        "Input 0 Shape",
        "INPUT 0 SHAPE",
        "IN0 SHAPE",
        "Output Shape",
        "OUTPUT SHAPE",
        "Tensor Shape",
        "TENSOR SHAPE",
        "Shape",
        "SHAPE",
    ]

    for df in (
        operation_perf_df,
        operation_ops_df,
    ):
        for column_name in candidate_columns:
            if column_name not in df.columns:
                continue

            for value in (
                df[column_name]
                .dropna()
                .astype(str)
            ):
                shape = parse_shape_from_text(value)

                if shape:
                    return shape

    for df, op_column in (
        (operation_perf_df, "OP Code"),
        (operation_ops_df, "OP CODE"),
    ):
        if op_column not in df.columns:
            continue

        for value in (
            df[op_column]
            .dropna()
            .astype(str)
        ):
            shape = parse_shape_from_text(value)

            if shape:
                return shape

    return None


def shape_to_string(shape):
    if not shape:
        return "unknown"

    return "x".join(
        str(value)
        for value in shape
    )


def element_count(shape):
    if not shape:
        return None

    count = 1

    for dimension in shape:
        count *= dimension

    return count


def parse_dtype_and_math_fidelity(operation_perf_df):
    """
    Parse data type and math fidelity from the Math Fidelity column.

    Examples:

        HiFi4 BF16 x BF16 => BF16
        BF16 => BF16
        FP32 => BFP8
    """
    if "Math Fidelity" not in operation_perf_df.columns:
        return "unknown", "unknown", None

    values = (
        operation_perf_df["Math Fidelity"]
        .dropna()
        .astype(str)
    )

    if values.empty:
        return "unknown", "unknown", None

    value = values.iloc[0].strip()
    upper = value.upper()

    fidelity_match = re.search(
        r"\b(HIFI\d+|LOFI)\b",
        upper,
    )

    math_fidelity = (
        fidelity_match.group(1).lower()
        if fidelity_match
        else "unknown"
    )

    dtype_names = (
        "BF16",
        "BFLOAT16",
        "FP32",
        "FLOAT32",
        "BFP8",
        "BFLOAT8_B",
        "BFP4",
        "BFLOAT4_B",
        "FP16",
        "INT8",
    )

    dtype_pattern = "|".join(dtype_names)

    input_match = re.search(
        rf"\b({dtype_pattern})\b\s*(?:x|=>)",
        upper,
    )

    if input_match:
        dtype = input_match.group(1).lower()
    else:
        fallback_match = re.search(
            rf"\b({dtype_pattern})\b",
            upper,
        )

        dtype = (
            fallback_match.group(1).lower()
            if fallback_match
            else "unknown"
        )

    aliases = {
        "bfloat16": "bf16",
        "float32": "fp32",
        "bfloat8_b": "bfp8",
        "bfloat4_b": "bfp4",
    }

    dtype = aliases.get(dtype, dtype)

    bytes_per_element = {
        "bf16": 2,
        "fp16": 2,
        "fp32": 4,
        "bfp8": 1,
        "int8": 1,
        "bfp4": 0.5,
    }.get(dtype)

    return (
        math_fidelity,
        dtype,
        bytes_per_element,
    )


def normalize_math_fidelity(value):
    """Return a canonical profiler fidelity name or None."""
    match = re.search(
        r"\b(HIFI\d+|LOFI)\b",
        str(value).upper(),
    )

    if not match:
        return None

    return match.group(1).lower()


def parse_profiler_math_fidelity(operation_ops_df):
    """Read a consistent fidelity value from raw measured profiler rows."""
    column_name = "MATH FIDELITY"

    if column_name not in operation_ops_df.columns:
        return None

    fidelities = {
        fidelity
        for fidelity in (
            normalize_math_fidelity(value)
            for value in operation_ops_df[
                column_name
            ].dropna()
        )
        if fidelity is not None
    }

    if not fidelities:
        return None

    if len(fidelities) != 1:
        raise RuntimeError(
            "Measured target-operation rows contain inconsistent "
            f"MATH FIDELITY values: {sorted(fidelities)}"
        )

    return next(iter(fidelities))


def add_microarchitectural_statistics(
    row,
    operation_ops_df,
):
    """
    Add min, mean, median, and max hardware-counter values.
    """
    for column_name in MICROARCHITECTURAL_COLUMNS:
        stats = summarize_col(
            operation_ops_df,
            column_name,
        )

        if stats is None:
            if column_name in FINAL_REPORT_COLUMNS:
                row[column_name] = None
            continue

        safe_name = sanitize_name(column_name)

        row[f"{safe_name}_mean"] = stats["mean"]
        row[f"{safe_name}_min"] = stats["min"]
        row[f"{safe_name}_median"] = stats["median"]
        row[f"{safe_name}_max"] = stats["max"]

        if column_name in FINAL_REPORT_COLUMNS:
            row[column_name] = stats["mean"]


def summarize_all_operations(perf_df, run_name):
    """
    Build long-form stacked-report-style data for every operation.

    This includes setup operations such as RandDeviceOperation and
    TypecastDeviceOperation, not only the benchmark operation.
    """
    required_columns = [
        "OP Code",
        "Device Time",
    ]

    missing_columns = [
        column_name
        for column_name in required_columns
        if column_name not in perf_df.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Missing required columns in perf_report.csv: "
            + ", ".join(missing_columns)
        )

    data = perf_df.copy()

    data["device_time_us"] = pd.to_numeric(
        data["Device Time"],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            "OP Code",
            "device_time_us",
        ]
    )

    if data.empty:
        return pd.DataFrame()

    data["op_code_full"] = (
        data["OP Code"]
        .astype(str)
        .str.strip()
    )

    data["op_code"] = (
        data["op_code_full"]
        .apply(normalize_operation_code)
    )

    data["op_category"] = (
        data["op_code"]
        .apply(classify_operation)
    )

    if "FLOPs" in data.columns:
        data["flops_numeric"] = pd.to_numeric(
            data["FLOPs"],
            errors="coerce",
        )
    else:
        data["flops_numeric"] = float("nan")

    grouped_rows = []

    for op_code_full, group in data.groupby(
        "op_code_full",
        sort=False,
    ):
        device_time_sum_us = group[
            "device_time_us"
        ].sum()

        flops = group[
            "flops_numeric"
        ].dropna()

        if flops.empty:
            min_flops = None
            max_flops = None
            mean_flops = None
            std_flops = None
            weighted_mean_flops = None
        else:
            min_flops = flops.min()
            max_flops = flops.max()
            mean_flops = flops.mean()
            std_flops = flops.std(ddof=0)

            weighted_data = group[
                [
                    "flops_numeric",
                    "device_time_us",
                ]
            ].dropna()

            weight_sum = weighted_data[
                "device_time_us"
            ].sum()

            if (
                not weighted_data.empty
                and weight_sum > 0
            ):
                weighted_mean_flops = (
                    weighted_data["flops_numeric"]
                    * weighted_data["device_time_us"]
                ).sum() / weight_sum
            else:
                weighted_mean_flops = None

        normalized_op_code = normalize_operation_code(
            op_code_full
        )

        grouped_rows.append(
            {
                "run": run_name,
                "op_code": normalized_op_code,
                "op_code_full": op_code_full,
                "op_category": classify_operation(
                    normalized_op_code
                ),
                "device_time_sum_us": (
                    device_time_sum_us
                ),
                "op_count": len(group),
                "min_flops": min_flops,
                "max_flops": max_flops,
                "mean_flops": mean_flops,
                "std_flops": std_flops,
                "weighted_mean_flops": (
                    weighted_mean_flops
                ),
            }
        )

    result = pd.DataFrame(grouped_rows)

    total_device_time_us = result[
        "device_time_sum_us"
    ].sum()

    if total_device_time_us > 0:
        result["total_percent"] = (
            result["device_time_sum_us"]
            / total_device_time_us
            * 100
        )
    else:
        result["total_percent"] = 0.0

    return result.sort_values(
        "device_time_sum_us",
        ascending=False,
    ).reset_index(drop=True)


def summarize_run(
    run_dir,
    operation,
    warmup_iterations=0,
    measured_iterations=None,
):
    """
    Summarize one report folder.

    Returns:

        summary_row, operation_breakdown_df
    """
    perf_path, ops_path = find_run_files(run_dir)

    if perf_path is None or ops_path is None:
        return None

    perf_df = pd.read_csv(
        perf_path,
        low_memory=False,
    )

    ops_df = pd.read_csv(
        ops_path,
        low_memory=False,
    )

    operation_breakdown_df = summarize_all_operations(
        perf_df,
        run_dir.name,
    )

    config = OPERATION_CONFIG[operation]

    report_op_code = config["report_op_code"]
    display_name = config["display_name"]

    operation_perf_df = get_operation_rows(
        df=perf_df,
        column_name="OP Code",
        operation_name=operation,
        report_op_code=report_op_code,
        source_name="perf_report.csv",
    )

    operation_ops_df = get_operation_rows(
        df=ops_df,
        column_name="OP CODE",
        operation_name=operation,
        report_op_code=report_op_code,
        source_name=ops_path.name,
    )

    (
        operation_perf_df,
        operation_ops_df,
        total_operation_entries,
    ) = select_measured_iterations(
        operation_perf_df,
        operation_ops_df,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
    )

    (
        math_fidelity,
        dtype,
        bytes_per_element,
    ) = parse_dtype_and_math_fidelity(
        operation_perf_df
    )

    logical_shape = None

    if operation != "matmul":
        logical_shape = detect_logical_tensor_shape(
            operation_ops_df
        )

    shape = logical_shape or detect_shape(
        operation_perf_df,
        operation_ops_df,
    )

    profiler_math_fidelity = (
        parse_profiler_math_fidelity(
            operation_ops_df
        )
    )

    if profiler_math_fidelity is None:
        profiler_math_fidelity = (
            math_fidelity
            if math_fidelity != "unknown"
            else None
        )

    requested_math_fidelity = (
        math_fidelity
        if (
            operation == "matmul"
            and math_fidelity != "unknown"
        )
        else None
    )

    if (
        operation != "matmul"
        and profiler_math_fidelity is not None
    ):
        math_fidelity = profiler_math_fidelity

    elements = element_count(shape)

    device_time_us_mean = mean_col(
        operation_perf_df,
        "Device Time",
    )

    device_time_us_sum = sum_col(
        operation_perf_df,
        "Device Time",
    )

    kernel_ns_mean = mean_col(
        operation_ops_df,
        "DEVICE KERNEL DURATION [ns]",
    )

    kernel_ns_sum = sum_col(
        operation_ops_df,
        "DEVICE KERNEL DURATION [ns]",
    )

    host_ns_mean = mean_col(
        operation_ops_df,
        "HOST DURATION [ns]",
    )

    host_ns_sum = sum_col(
        operation_ops_df,
        "HOST DURATION [ns]",
    )

    op_to_op_ns_mean = mean_col(
        operation_ops_df,
        "OP TO OP LATENCY [ns]",
    )

    op_to_op_ns_sum = sum_col(
        operation_ops_df,
        "OP TO OP LATENCY [ns]",
    )

    operation_count = len(operation_perf_df)

    mean_device_time_s = (
        device_time_us_mean * 1e-6
        if (
            device_time_us_mean is not None
            and device_time_us_mean > 0
        )
        else None
    )

    operations_per_second = (
        1.0 / mean_device_time_s
        if mean_device_time_s is not None
        else None
    )

    elements_per_second = (
        elements / mean_device_time_s
        if (
            elements is not None
            and mean_device_time_s is not None
        )
        else None
    )

    row = {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "operation": operation,
        "operation_display_name": display_name,
        "report_op_code": report_op_code,
        "perf_file": str(perf_path),
        "ops_file": str(ops_path),
        "dtype": dtype,
        "math_fidelity": math_fidelity,
        "requested_math_fidelity": (
            requested_math_fidelity
        ),
        "profiler_math_fidelity": (
            profiler_math_fidelity
        ),
        "bytes_per_element": bytes_per_element,
        "shape": shape_to_string(shape),
        "logical_shape": shape_to_string(
            logical_shape
        ),
        "element_count": elements,
        "operation_count": operation_count,
        "total_operation_entries": (
            total_operation_entries
        ),
        "warmup_iterations": warmup_iterations,
        "measured_iterations": (
            measured_iterations
            if measured_iterations is not None
            else operation_count
        ),
        "device_time_us": device_time_us_mean,
        "device_time_sum_us": device_time_us_sum,
        "kernel_time_us": (
            kernel_ns_mean / 1000
            if kernel_ns_mean is not None
            else None
        ),
        "kernel_time_sum_us": (
            kernel_ns_sum / 1000
            if kernel_ns_sum is not None
            else None
        ),
        "host_time_us": (
            host_ns_mean / 1000
            if host_ns_mean is not None
            else None
        ),
        "host_time_sum_us": (
            host_ns_sum / 1000
            if host_ns_sum is not None
            else None
        ),
        "op_to_op_latency_us": (
            op_to_op_ns_mean / 1000
            if op_to_op_ns_mean is not None
            else None
        ),
        "op_to_op_latency_sum_us": (
            op_to_op_ns_sum / 1000
            if op_to_op_ns_sum is not None
            else None
        ),
        "operations_per_second": (
            operations_per_second
        ),
        "elements_per_second": (
            elements_per_second
        ),
        "reported_tflops": mean_col(
            operation_perf_df,
            "FLOPs",
        ),
        "reported_flops_percent": mean_col(
            operation_perf_df,
            "FLOPs %",
        ),
        "reported_dram_gbs": mean_col(
            operation_perf_df,
            "DRAM",
        ),
        "reported_dram_percent": mean_col(
            operation_perf_df,
            "DRAM %",
        ),
        "cores": mean_col(
            operation_perf_df,
            "Cores",
        ),
        "global_call_count": mean_col(
            operation_perf_df,
            "Global Call Count",
        ),
        "inner_dim_block_size": mean_col(
            operation_perf_df,
            "Inner Dim Block Size",
        ),
        "output_subblock_h": mean_col(
            operation_perf_df,
            "Output Subblock H",
        ),
        "output_subblock_w": mean_col(
            operation_perf_df,
            "Output Subblock W",
        ),
    }

    row.update(read_host_phase_summary(run_dir))
    row.update(read_cpu_power_summary(run_dir))

    if operation == "matmul":
        m, k, n = parse_matmul_size(
            operation_perf_df
        )

        row["M"] = m
        row["K"] = k
        row["N"] = n

        if (
            m is not None
            and k is not None
            and n is not None
        ):
            row["shape"] = f"{m}x{k}x{n}"
            row["matrix_size"] = f"{m}x{k}x{n}"

            flops = 2 * m * k * n

            row["flops"] = flops
            row["gflops"] = flops / 1e9

            if mean_device_time_s is not None:
                row["derived_tflops"] = (
                    flops / mean_device_time_s
                ) / 1e12

                row["matmuls_per_second"] = (
                    1.0 / mean_device_time_s
                )

            if bytes_per_element is not None:
                bytes_a = (
                    m * k * bytes_per_element
                )

                bytes_b = (
                    k * n * bytes_per_element
                )

                bytes_c = (
                    m * n * bytes_per_element
                )

                minimum_tensor_bytes = (
                    bytes_a
                    + bytes_b
                    + bytes_c
                )

                row[
                    "minimum_tensor_traffic_bytes"
                ] = minimum_tensor_bytes

                row[
                    "minimum_tensor_traffic_mib"
                ] = minimum_tensor_bytes / (
                    1024**2
                )

                if mean_device_time_s is not None:
                    minimum_tensor_gb = (
                        minimum_tensor_bytes
                        / 1e9
                    )

                    row[
                        "derived_minimum_gbs"
                    ] = (
                        minimum_tensor_gb
                        / mean_device_time_s
                    )
        else:
            row["matrix_size"] = "unknown"

    add_microarchitectural_statistics(
        row,
        operation_ops_df,
    )

    return row, operation_breakdown_df


def determine_sort_columns(df, operation):
    if operation == "matmul":
        required = ["M", "K", "N"]

        if all(
            column_name in df.columns
            for column_name in required
        ):
            return [
                "M",
                "K",
                "N",
                "run",
            ]

    if "element_count" in df.columns:
        values = pd.to_numeric(
            df["element_count"],
            errors="coerce",
        )

        if values.notna().any():
            return [
                "element_count",
                "run",
            ]

    return ["run"]


def make_base_name(df, operation):
    dtype = "unknown"

    if "dtype" in df.columns:
        values = (
            df["dtype"]
            .dropna()
            .astype(str)
        )

        if not values.empty:
            dtype = values.iloc[0]

    return (
        f"{sanitize_name(operation)}_"
        f"{sanitize_name(dtype)}"
    )


def append_runtime_summary(lines, df):
    lines.append("## Runtime Summary")
    lines.append("")
    lines.append(
        "| Metric | Mean | Min | Median | Max |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|"
    )

    runtime_columns = [
        "device_time_us",
        "device_time_sum_us",
        "kernel_time_us",
        "kernel_time_sum_us",
        "host_time_us",
        "host_time_sum_us",
        "op_to_op_latency_us",
        "op_to_op_latency_sum_us",
        "operations_per_second",
        "elements_per_second",
        "derived_tflops",
        "reported_tflops",
        "reported_flops_percent",
        "reported_dram_gbs",
        "reported_dram_percent",
        "cores",
    ]

    rows_written = 0

    for column_name in runtime_columns:
        if column_name not in df.columns:
            continue

        values = pd.to_numeric(
            df[column_name],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        lines.append(
            f"| {column_name} | "
            f"{fmt(values.mean())} | "
            f"{fmt(values.min())} | "
            f"{fmt(values.median())} | "
            f"{fmt(values.max())} |"
        )

        rows_written += 1

    if rows_written == 0:
        lines.append(
            "| No numeric runtime data | "
            "N/A | N/A | N/A | N/A |"
        )

    lines.append("")


def append_microarchitectural_summary(lines, df):
    lines.append(
        "## Microarchitectural Analysis"
    )
    lines.append("")
    lines.append(
        "| Counter | Mean | Min | Median | Max |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|"
    )

    mean_columns = [
        column_name
        for column_name in df.columns
        if column_name.endswith("_mean")
    ]

    rows_written = 0

    for mean_column in mean_columns:
        base_name = mean_column[:-5]

        values = pd.to_numeric(
            df[mean_column],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        min_column = f"{base_name}_min"
        median_column = f"{base_name}_median"
        max_column = f"{base_name}_max"

        minimum = (
            pd.to_numeric(
                df[min_column],
                errors="coerce",
            ).min()
            if min_column in df.columns
            else None
        )

        median = (
            pd.to_numeric(
                df[median_column],
                errors="coerce",
            ).median()
            if median_column in df.columns
            else None
        )

        maximum = (
            pd.to_numeric(
                df[max_column],
                errors="coerce",
            ).max()
            if max_column in df.columns
            else None
        )

        lines.append(
            f"| {base_name} | "
            f"{fmt(values.mean())} | "
            f"{fmt(minimum)} | "
            f"{fmt(median)} | "
            f"{fmt(maximum)} |"
        )

        rows_written += 1

    if rows_written == 0:
        lines.append(
            "| No microarchitectural counters found | "
            "N/A | N/A | N/A | N/A |"
        )

    lines.append("")


def write_markdown_report(
    df,
    output_path,
    reports_dir,
    operation,
    skipped_warmup_dirs,
):
    config = OPERATION_CONFIG[operation]

    display_name = config["display_name"]
    report_op_code = config["report_op_code"]

    lines = []

    lines.append(
        f"# Aggregate {display_name} Performance Report"
    )
    lines.append("")
    lines.append(
        f"Reports directory: `{reports_dir}`"
    )
    lines.append("")
    lines.append(
        f"Operation: **{display_name}**"
    )
    lines.append("")
    lines.append(
        f"Profiler OP Code: `{report_op_code}`"
    )
    lines.append("")
    lines.append(
        f"Number of runs summarized: **{len(df)}**"
    )
    lines.append("")
    lines.append(
        f"Warm-up runs skipped: "
        f"**{len(skipped_warmup_dirs)}**"
    )
    lines.append("")

    if skipped_warmup_dirs:
        lines.append(
            "Skipped warm-up directories:"
        )
        lines.append("")

        for run_dir in skipped_warmup_dirs:
            lines.append(
                f"- `{run_dir.name}`"
            )

        lines.append("")

    lines.append("## Per-Run Results")
    lines.append("")

    if operation == "matmul":
        lines.append(
            "| Run | Size | DType | Fidelity | Ops | "
            "Device us | Kernel us | Op gap us | "
            "Derived TFLOP/s | Reported TFLOP/s | "
            "DRAM GB/s |"
        )
        lines.append(
            "|---|---:|---|---|---:|---:|---:|---:|"
            "---:|---:|---:|"
        )

        for _, row in df.iterrows():
            lines.append(
                f"| {row.get('run', 'N/A')} | "
                f"{row.get('matrix_size', 'unknown')} | "
                f"{row.get('dtype', 'unknown')} | "
                f"{row.get('math_fidelity', 'unknown')} | "
                f"{fmt(row.get('operation_count'), digits=0)} | "
                f"{fmt(row.get('device_time_us'))} | "
                f"{fmt(row.get('kernel_time_us'))} | "
                f"{fmt(row.get('op_to_op_latency_us'))} | "
                f"{fmt(row.get('derived_tflops'))} | "
                f"{fmt(row.get('reported_tflops'))} | "
                f"{fmt(row.get('reported_dram_gbs'))} |"
            )
    else:
        lines.append(
            "| Run | Shape | Elements | DType | Ops | "
            "Device us | Kernel us | Host us | "
            "Op gap us | Ops/s | Elements/s | Cores |"
        )
        lines.append(
            "|---|---:|---:|---|---:|---:|---:|---:|"
            "---:|---:|---:|---:|"
        )

        for _, row in df.iterrows():
            lines.append(
                f"| {row.get('run', 'N/A')} | "
                f"{row.get('shape', 'unknown')} | "
                f"{fmt(row.get('element_count'), digits=0)} | "
                f"{row.get('dtype', 'unknown')} | "
                f"{fmt(row.get('operation_count'), digits=0)} | "
                f"{fmt(row.get('device_time_us'))} | "
                f"{fmt(row.get('kernel_time_us'))} | "
                f"{fmt(row.get('host_time_us'))} | "
                f"{fmt(row.get('op_to_op_latency_us'))} | "
                f"{fmt(row.get('operations_per_second'))} | "
                f"{fmt(row.get('elements_per_second'))} | "
                f"{fmt(row.get('cores'))} |"
            )

    lines.append("")

    append_runtime_summary(
        lines,
        df,
    )

    append_microarchitectural_summary(
        lines,
        df,
    )

    lines.append("## Notes")
    lines.append("")
    lines.append(
        f"- Only `{report_op_code}` rows are used for "
        "operation-specific runtime statistics."
    )
    lines.append(
        "- The operation-breakdown CSV includes all operation "
        "types, including random initialization and typecasts."
    )
    lines.append(
        "- Warm-up skipping is performed at the report-directory "
        "level before any CSV files are processed."
    )
    lines.append(
        "- Device Time is read from perf_report.csv."
    )
    lines.append(
        "- Kernel duration, host duration, op-to-op latency, "
        "and hardware counters are read from "
        "ops_perf_results_*.csv."
    )

    if operation == "matmul":
        lines.append(
            "- Matmul TFLOP/s is calculated as "
            "`2*M*K*N / device_time`."
        )
        lines.append(
            "- Minimum tensor traffic assumes "
            "A read + B read + C write."
        )
    else:
        lines.append(
            "- Matmul TFLOP/s formulas are not applied "
            "to this operation."
        )

    if operation in {"relu", "gelu"}:
        lines.append(
            "- ReLU and GELU both appear as "
            "`UnaryDeviceOperation`; the selected operation "
            "determines the report label."
        )

    lines.append("")

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Tenstorrent Matmul, Softmax, "
            "ReLU, or GELU report directories."
        )
    )

    parser.add_argument(
        "--reports-dir",
        required=True,
        help=(
            "Directory containing timestamped report folders."
        ),
    )

    parser.add_argument(
        "--operation",
        choices=[
            "auto",
            "matmul",
            "softmax",
            "relu",
            "gelu",
        ],
        default="auto",
        help=(
            "Operation represented by the reports. "
            "Default: infer from the reports-directory name."
        ),
    )

    parser.add_argument(
        "--skip-warmup-runs",
        type=int,
        default=1,
        help=(
            "Number of earliest sorted report directories to skip. "
            "Default: 1. Use 0 to include all runs."
        ),
    )

    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=0,
        help=(
            "Number of earliest target-operation entries to exclude "
            "inside each Tracy report. Default: 0, preserving the "
            "historical behaviour."
        ),
    )

    parser.add_argument(
        "--measured-iterations",
        type=int,
        default=None,
        help=(
            "Expected number of target-operation entries after warm-up. "
            "When provided, each report must contain exactly warm-up plus "
            "measured entries."
        ),
    )

    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Write only the per-run summary CSV.",
    )

    args = parser.parse_args()

    if args.skip_warmup_runs < 0:
        raise RuntimeError(
            "--skip-warmup-runs cannot be negative."
        )

    if args.warmup_iterations < 0:
        raise RuntimeError(
            "--warmup-iterations cannot be negative."
        )

    if (
        args.measured_iterations is not None
        and args.measured_iterations <= 0
    ):
        raise RuntimeError(
            "--measured-iterations must be greater than zero."
        )

    reports_dir = Path(
        args.reports_dir
    ).expanduser().resolve()

    if not reports_dir.exists():
        raise RuntimeError(
            f"Reports directory does not exist: "
            f"{reports_dir}"
        )

    if not reports_dir.is_dir():
        raise RuntimeError(
            f"Reports path is not a directory: "
            f"{reports_dir}"
        )

    operation = infer_operation(
        reports_dir,
        args.operation,
    )

    config = OPERATION_CONFIG[operation]

    print(
        f"Selected operation: "
        f"{config['display_name']} "
        f"({config['report_op_code']})"
    )

    all_run_directories = sorted(
        path
        for path in reports_dir.iterdir()
        if path.is_dir()
    )

    if not all_run_directories:
        raise RuntimeError(
            f"No report directories found under "
            f"{reports_dir}"
        )

    if (
        args.skip_warmup_runs
        >= len(all_run_directories)
    ):
        raise RuntimeError(
            f"Requested skipping "
            f"{args.skip_warmup_runs} warm-up runs, "
            f"but only {len(all_run_directories)} "
            "report directories were found."
        )

    skipped_warmup_dirs = all_run_directories[
        :args.skip_warmup_runs
    ]

    run_directories = all_run_directories[
        args.skip_warmup_runs:
    ]

    if skipped_warmup_dirs:
        print(
            f"Skipping {len(skipped_warmup_dirs)} "
            "warm-up run(s):"
        )

        for run_dir in skipped_warmup_dirs:
            print(f"  {run_dir.name}")
    else:
        print("No warm-up runs skipped.")

    rows = []
    operation_breakdown_frames = []

    for run_dir in run_directories:
        try:
            result = summarize_run(
                run_dir,
                operation,
                warmup_iterations=(
                    args.warmup_iterations
                ),
                measured_iterations=(
                    args.measured_iterations
                ),
            )
        except Exception as error:
            print(
                f"Skipping {run_dir}: {error}"
            )
            continue

        if result is None:
            print(
                f"Skipping {run_dir}: missing "
                "perf_report.csv or "
                "ops_perf_results_*.csv"
            )
            continue

        (
            row,
            operation_breakdown_df,
        ) = result

        rows.append(row)

        if (
            operation_breakdown_df is not None
            and not operation_breakdown_df.empty
        ):
            operation_breakdown_frames.append(
                operation_breakdown_df
            )

    if not rows:
        raise RuntimeError(
            f"No valid {operation} report directories "
            f"found under {reports_dir}"
        )

    df = pd.DataFrame(rows)

    sort_columns = determine_sort_columns(
        df,
        operation,
    )

    df = df.sort_values(
        sort_columns,
        na_position="last",
    )

    trailing_columns = [
        "run",
        "run_dir",
        "perf_file",
        "ops_file",
        "host_phase_file",
        "host_phase_summary_file",
        "cpu_power_csv",
        "cpu_power_summary_json",
    ]
    trailing_columns = [
        column
        for column in trailing_columns
        if column in df.columns
    ]
    leading_columns = [
        column
        for column in df.columns
        if column not in trailing_columns
    ]
    df = df[leading_columns + trailing_columns]

    output_parent = reports_dir.parent

    base_name = reports_dir.name


    output_csv = (
        output_parent
        / f"{base_name}_summary.csv"
    )

    output_md = (
        output_parent
        / f"{base_name}_summary.md"
    )

    operation_breakdown_csv = (
        output_parent
        / f"{base_name}_operation_breakdown.csv"
    )

    df.to_csv(
        output_csv,
        index=False,
    )

    print(
        f"Wrote CSV summary: {output_csv}"
    )

    if not args.csv_only:
        write_markdown_report(
            df=df,
            output_path=output_md,
            reports_dir=reports_dir,
            operation=operation,
            skipped_warmup_dirs=skipped_warmup_dirs,
        )
        print(f"Wrote Markdown report: {output_md}")

    if not args.csv_only and operation_breakdown_frames:
        operation_breakdown_df = pd.concat(
            operation_breakdown_frames,
            ignore_index=True,
        )

        operation_breakdown_df = (
            operation_breakdown_df.sort_values(
                [
                    "run",
                    "device_time_sum_us",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
        )

        operation_breakdown_df.to_csv(
            operation_breakdown_csv,
            index=False,
        )

        print(
            "Wrote operation breakdown: "
            f"{operation_breakdown_csv}"
        )
    elif not args.csv_only:
        print(
            "No operation-breakdown data was available."
        )

    print(
        f"Summarized {len(df)} measured runs."
    )

    print(
        f"Skipped {len(skipped_warmup_dirs)} "
        "warm-up runs."
    )


if __name__ == "__main__":
    main()
