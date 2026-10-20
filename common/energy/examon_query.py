import argparse
import sys

import pandas as pd

try:
    from iotdb.Session import Session
except ImportError:  # allow --self-test to run without apache-iotdb installed
    Session = None

# How to use
# Record start timestamp (milliseconds since Unix epoch)
#   START_TS=$(date +%s%3N)

# Run the benchmark
#   ./benchmark "$@"

# Record end timestamp
#   END_TS=$(date +%s%3N)

# Call the script
#   python3 examon_query.py $START_TS $END_TS --output [output_name] --summary_output [summary_name]
#
# Energy model: samples are taken every ~10 seconds and each sample is valid
# until the next one (zero-order hold). The gap before the first sample is
# charged at the idle power of the last sample at or before the start
# timestamp (the query looks back --idle-lookback-ms, default 20000 ms). The
# last sample is valid until the end timestamp, so the energy integrates
# exactly over the benchmark duration [start, end]:
#
#   E = P_idle*(t1-start) + sum(pi*(ti+1 - ti)) + pn*(end-tn)
#
# Run the built-in energy computation tests (no database connection needed):
#   python3 examon_query.py --self-test

# Create a session
ip = "137.204.45.25"
port_ = "16667"

#local IP
#ip = "192.168.1.201"
#port_ = "6667"

username_ = "mcquery" #ask emanuele.venieri2@unibo.it
password_ = "mcimone2165"


def compute_energy(samples, start_ms, end_ms):
    """Compute the energy over [start_ms, end_ms] with a piecewise-constant
    (zero-order hold) power model.

    samples: pd.Series of power readings (W) indexed by millisecond timestamps
    (they can extend before start_ms or after end_ms). NaN and 0 readings are
    treated as missing. Each valid reading is held until the next valid one.
    The gap before the first in-window reading is charged at the idle power of
    the last reading at or before start_ms when one exists, otherwise at the
    first in-window reading. The last in-window reading is held until end_ms.

    Returns a dict with sample_count, average_power, idle_power,
    first_sample_ms, last_sample_ms, head_gap_s, tail_gap_s and energy (W*s).
    Everything except sample_count is NA when there is no usable data.
    """
    na = {
        "sample_count": 0,
        "average_power": pd.NA,
        "idle_power": pd.NA,
        "first_sample_ms": pd.NA,
        "last_sample_ms": pd.NA,
        "head_gap_s": pd.NA,
        "tail_gap_s": pd.NA,
        "energy": pd.NA,
    }
    s = samples.dropna()
    s = s[s != 0]
    s = s[~s.index.duplicated(keep="last")]
    if s.empty:
        return na

    in_window = s[(s.index > start_ms) & (s.index < end_ms)]
    if in_window.empty:
        return na

    t = in_window.index.to_numpy(dtype=float)
    p = in_window.to_numpy(dtype=float)

    idle_series = s[s.index <= start_ms]
    if not idle_series.empty:
        idle_power = float(idle_series.iloc[-1])
        head_energy = idle_power * (t[0] - start_ms)
    else:
        idle_power = pd.NA
        print("warning: no sample at or before the start timestamp; "
              "holding the first sample backwards over the head gap")
        head_energy = p[0] * (t[0] - start_ms)

    middle_energy = float((p[:-1] * (t[1:] - t[:-1])).sum()) if len(t) > 1 else 0.0
    tail_energy = p[-1] * (end_ms - t[-1])

    return {
        "sample_count": int(in_window.shape[0]),
        "average_power": float(p.mean()),
        "idle_power": idle_power,
        "first_sample_ms": int(t[0]),
        "last_sample_ms": int(t[-1]),
        "head_gap_s": (t[0] - start_ms) / 1000.0,
        "tail_gap_s": (end_ms - t[-1]) / 1000.0,
        "energy": (head_energy + middle_energy + tail_energy) / 1000.0,
    }
def self_test():
    """Built-in tests for compute_energy; returns the number of failures."""
    failures = 0

    def series(ts, p):
        return pd.Series(p, index=ts, dtype=float)

    def close(a, b):
        return abs(a - b) <= 1e-9

    def check(name, condition):
        nonlocal failures
        print(f"[self-test] {name}: {'PASS' if condition else 'FAIL'}")
        if not condition:
            failures += 1

    # 10-second ticks with start/end mid-interval: head gap at idle power,
    # each sample held until the next one, last sample held until the end.
    r = compute_energy(series([0, 10000, 20000, 30000, 40000], [100, 200, 300, 400, 500]), 12000, 37000)
    check("energy 7400 W*s (200*8s + 300*10s + 400*7s)", close(r["energy"], 7400.0))
    check("idle power 200 W", close(r["idle_power"], 200.0))
    check("head/tail gaps 8s/7s", close(r["head_gap_s"], 8.0) and close(r["tail_gap_s"], 7.0))
    check("intervals cover the whole duration",
          close(r["head_gap_s"] + (r["last_sample_ms"] - r["first_sample_ms"]) / 1000.0 + r["tail_gap_s"], 25.0))

    # No sample at or before the start: fall back to holding the first sample backwards.
    r = compute_energy(series([20000, 30000, 40000], [300, 400, 500]), 12000, 37000)
    check("no idle sample -> NA + 8200 W*s (300*8s + 300*10s + 400*7s)",
          pd.isna(r["idle_power"]) and close(r["energy"], 8200.0))

    # Zero readings (sensor dropout) are dropped and the gap is bridged.
    r = compute_energy(series([0, 10000, 30000, 40000], [100, 200, 0, 500]), 12000, 42000)
    check("zero dropped -> 6600 W*s (200*28s + 500*2s)",
          r["sample_count"] == 1 and close(r["energy"], 6600.0) and close(r["head_gap_s"], 28.0))

    # NaN readings are dropped the same way.
    r = compute_energy(series([10000, 20000, 30000], [200, float("nan"), 400]), 12000, 37000)
    check("NaN dropped -> 6400 W*s (200*18s + 400*7s)",
          r["sample_count"] == 1 and close(r["energy"], 6400.0))

    # A single in-window sample is held over both gaps.
    r = compute_energy(series([10000, 20000], [200, 300]), 12000, 37000)
    check("single sample -> 6700 W*s (200*8s + 300*17s)",
          r["sample_count"] == 1 and close(r["energy"], 6700.0))

    # No usable data at all.
    r = compute_energy(series([], []), 12000, 37000)
    check("no data -> NA energy", r["sample_count"] == 0 and pd.isna(r["energy"]))

    print(f"[self-test] {failures} failure(s)")
    return failures


def main():
    parser = argparse.ArgumentParser(description="Query Examon metrics and export CSV data")
    parser.add_argument(
        "timestamp_start_ms",
        type=int,
        nargs="?",
        default=None,
        help="Start timestamp in milliseconds",
    )
    parser.add_argument(
        "timestamp_end_ms",
        type=int,
        nargs="?",
        default=None,
        help="End timestamp in milliseconds",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to the raw CSV output file",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Path to the summary CSV output file (defaults to <output>_summary.csv)",
    )
    parser.add_argument(
        "--idle-lookback-ms",
        type=int,
        default=20000,
        help="How far before the start timestamp to look for an idle power sample "
             "(default: 20000 ms, i.e. two 10-second sampling periods)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the built-in energy computation tests and exit",
    )
    args = parser.parse_args()

    if args.self_test:
        sys.exit(self_test())

    if args.timestamp_start_ms is None or args.timestamp_end_ms is None:
        parser.error("timestamp_start_ms and timestamp_end_ms are required")
    if Session is None:
        print("error: apache-iotdb is not installed (pip install apache-iotdb)", file=sys.stderr)
        sys.exit(1)

    timestamp_start = args.timestamp_start_ms
    timestamp_end = args.timestamp_end_ms
    output_path = args.output
    summary_output_path = args.summary_output
    idle_lookback_ms = args.idle_lookback_ms

    #timestamp_start =   1774340660000
    #timestamp_end =  1774342534000

    if timestamp_end <= timestamp_start:
        parser.error("end timestamp must be after the start timestamp")

    session = Session(ip, port_, username_, password_)
    session.open(False)

    # Query available metrics
    querys = ["root.mcimone.plugin.ipmi_pub.metric"]
    for query in querys:
        result = session.execute_query_statement(f"show child paths {query}")
        result = [x[0].replace(f"{query}.", "") for x in result.todf().values.tolist()]

        # print(f"\n available metrics for {query}: {result}")

    BMC_SELECTED = ["1", "2"]
    selected_metrics = ["Efuse_Power"]
    df = pd.DataFrame()
    summary_rows = []
    duration_seconds = (timestamp_end - timestamp_start) / 1000.0
    no_data_stats = compute_energy(pd.Series(dtype=float), timestamp_start, timestamp_end)
    for bmc in BMC_SELECTED:
        for metric in range(len(selected_metrics)):
            metric_name = selected_metrics[metric]
            # print(f"Retrieving bmc_{bmc} metric: {metric_name}")
            query = (
                f"SELECT {metric_name} "
                f"FROM root.mcimone.plugin.ipmi_pub.id.bmc_peak{bmc}.value "
                f"WHERE time >= {timestamp_start - idle_lookback_ms} "
                f"AND time < {timestamp_end}"
            )
            bmc_result = session.execute_query_statement(query)
            tmp = bmc_result.todf()
            if tmp.empty:
                print(f"No data returned for bmc_{bmc} metric: {metric_name}")
                row = {
                    "bmc": bmc,
                    "metric": metric_name,
                    "start_timestamp_ms": timestamp_start,
                    "end_timestamp_ms": timestamp_end,
                    "duration_s": duration_seconds,
                }
                row.update(no_data_stats)
                summary_rows.append(row)
                continue
            # Set index as first column
            tmp = tmp.set_index(tmp.columns[0])
            value_columns = [column for column in tmp.columns if column != tmp.index.name]
            if not value_columns:
                print(f"No value column found for bmc_{bmc} metric: {metric_name}")
                continue

            value_column = value_columns[0]
            series = pd.to_numeric(tmp[value_column], errors="coerce")
            df[f"bmc_{bmc}_{metric_name}"] = series

            stats = compute_energy(series, timestamp_start, timestamp_end)

            summary_rows.append(
                {
                    "bmc": bmc,
                    "metric": metric_name,
                    "start_timestamp_ms": timestamp_start,
                    "end_timestamp_ms": timestamp_end,
                    "duration_s": duration_seconds,
                    **stats,
                }
            )

    session.close()

    if output_path is not None:
        df.to_csv(output_path)

    if summary_output_path is None and output_path is not None:
        if output_path.lower().endswith(".csv"):
            summary_output_path = output_path[:-len(".csv")] + "_summary.csv"
        else:
            summary_output_path = output_path + "_summary.csv"

    summary_df = pd.DataFrame(summary_rows)

    if summary_output_path is not None:
        summary_df.to_csv(summary_output_path, index=False)

    # print(df.columns)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)


    print("START ENERGY MEASUREMENTS")
    print(summary_df)
    print("END ENERGY MEASUREMENTS")


if __name__ == "__main__":
    main()
