from iotdb.Session import Session
import argparse
import pandas as pd
import sys

# How to use
# Record start timestamp (milliseconds since Unix epoch)
#   START_TS=$(date +%s%3N)

# Run the benchmark
#   ./benchmark "$@"

# Record end timestamp
#   END_TS=$(date +%s%3N)

# Call the script
#   python3 examon_query.py $START_TS $END_TS --output [output_name] --summary_output [summary_name]

# Create a session
#ip = "137.204.45.25" 
#port_ = "16667"

#local IP
ip = "192.168.1.201"
port_ = "6667"

username_ = "[username]" #ask emanuele.venieri2@unibo.it
password_ = "[password]"

parser = argparse.ArgumentParser(description="Query Examon metrics and export CSV data")
parser.add_argument("timestamp_start_ms", type=int, help="Start timestamp in milliseconds")
parser.add_argument("timestamp_end_ms", type=int, help="End timestamp in milliseconds")
parser.add_argument(
    "--output",
    default="output.csv",
    help="Path to the raw CSV output file",
)
parser.add_argument(
    "--summary-output",
    default=None,
    help="Path to the summary CSV output file (defaults to <output>_summary.csv)",
)
args = parser.parse_args()

timestamp_start = args.timestamp_start_ms
timestamp_end = args.timestamp_end_ms
output_path = args.output
summary_output_path = args.summary_output

#timestamp_start =   1774340660000
#timestamp_end =  1774342534000

session = Session(ip, port_, username_, password_)
session.open(False)

# Query available metrics
querys = ["root.mcimone.plugin.ipmi_pub.metric"] 
for query in querys:
    result = session.execute_query_statement(f"show child paths {query}")
    result = [x[0].replace(f"{query}.", "") for x in result.todf().values.tolist()]

    print(f"\n available metrics for {query}: {result}")

BMC_SELECTED = ["1", "2"]
selected_metrics = ["Efuse_Power"]
df = pd.DataFrame()
summary_rows = []
duration_seconds = (timestamp_end - timestamp_start) / 1000.0

for bmc in BMC_SELECTED:
    for metric in range(len(selected_metrics)):
        metric_name = selected_metrics[metric]
        print(f"Retrieving bmc_{bmc} metric: {metric_name}")
        query = f"SELECT {metric_name} \
                FROM root.mcimone.plugin.ipmi_pub.id.bmc_peak{bmc}.value \
                WHERE time > {timestamp_start} \
                AND time < {timestamp_end}"
        bmc_result = session.execute_query_statement(query)
        tmp = bmc_result.todf()
        if tmp.empty:
            print(f"No data returned for bmc_{bmc} metric: {metric_name}")
            summary_rows.append(
                {
                    "bmc": bmc,
                    "metric": metric_name,
                    "start_timestamp_ms": timestamp_start,
                    "end_timestamp_ms": timestamp_end,
                    "duration_s": duration_seconds,
                    "sample_count": 0,
                    "average_power": pd.NA,
                    "energy": pd.NA,
                }
            )
            continue
        # Set index as first column
        tmp = tmp.set_index(tmp.columns[0])
        value_columns = [column for column in tmp.columns if column != tmp.index.name]
        if not value_columns:
            print(f"No value column found for bmc_{bmc} metric: {metric_name}")
            continue

        value_column = value_columns[0]
        df[f"bmc_{bmc}_{metric_name}"] = pd.to_numeric(tmp[value_column], errors="coerce")

        metric_series = pd.to_numeric(tmp[value_column], errors="coerce")
        valid_samples = metric_series.dropna()
        valid_samples = valid_samples[valid_samples != 0]
        average_power = valid_samples.mean()
        energy = average_power * duration_seconds if pd.notna(average_power) else pd.NA

        summary_rows.append(
            {
                "bmc": bmc,
                "metric": selected_metrics[metric],
                "start_timestamp_ms": timestamp_start,
                "end_timestamp_ms": timestamp_end,
                "duration_s": duration_seconds,
                "sample_count": int(valid_samples.count()),
                "average_power": average_power,
                "energy": energy,
            }
        )
        
session.close()

df.to_csv(output_path)

summary_df = pd.DataFrame(summary_rows)
if summary_output_path is None:
    if output_path.lower().endswith(".csv"):
        summary_output_path = output_path[:-4] + "_summary.csv"
    else:
        summary_output_path = output_path + "_summary.csv"

summary_df.to_csv(summary_output_path, index=False)

print(df.columns)
print(summary_df)

