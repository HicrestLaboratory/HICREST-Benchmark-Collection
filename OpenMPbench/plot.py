import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ------------------------------------------------
# Folder containing benchmark CSV files
# ------------------------------------------------

CSV_DIR = Path("benchmark_csv")

csv_files = list(CSV_DIR.glob("*_results.csv"))

if not csv_files:
    print("No benchmark CSV files found.")
    exit()


# ------------------------------------------------
# Plot scaling comparison across devices
# ------------------------------------------------

def plot_device_scaling(df, benchmark):

    plt.figure()

    for device in sorted(df["device"].unique()):

        sub = df[df["device"] == device]

        grouped = (
            sub.groupby("cores")["mean_us"]
            .mean()
            .reset_index()
            .sort_values("cores")
        )

        plt.plot(
            grouped["cores"],
            grouped["mean_us"],
            marker="o",
            label=device
        )

    plt.xlabel("Number of Cores")
    plt.ylabel("Mean Time (microseconds)")
    plt.title(f"{benchmark} Scaling Across Devices")

    plt.legend()
    plt.grid(True)

    outfile = f"{benchmark}_device_scaling.png"
    plt.savefig(outfile, bbox_inches="tight")

    print("Saved", outfile)

    plt.close()


# ------------------------------------------------
# Plot overhead comparison
# ------------------------------------------------

def plot_overhead(df, benchmark):

    plt.figure()

    for device in sorted(df["device"].unique()):

        sub = df[df["device"] == device]

        grouped = (
            sub.groupby("cores")["overhead_us"]
            .mean()
            .reset_index()
            .sort_values("cores")
        )

        plt.plot(
            grouped["cores"],
            grouped["overhead_us"],
            marker="o",
            label=device
        )

    plt.xlabel("Number of Cores")
    plt.ylabel("Overhead (microseconds)")
    plt.title(f"{benchmark} Overhead Comparison")

    plt.legend()
    plt.grid(True)

    outfile = f"{benchmark}_overhead_devices.png"
    plt.savefig(outfile, bbox_inches="tight")

    print("Saved", outfile)

    plt.close()


# ------------------------------------------------
# Block comparison across devices
# ------------------------------------------------

def plot_blocks(df, benchmark):

    pivot = df.pivot_table(
        index="block",
        columns="device",
        values="mean_us",
        aggfunc="mean"
    )

    pivot.plot(kind="bar")

    plt.xlabel("Benchmark Block")
    plt.ylabel("Mean Time (microseconds)")
    plt.title(f"{benchmark} Block Comparison Across Devices")

    plt.xticks(rotation=45)

    plt.grid(True)

    outfile = f"{benchmark}_blocks_devices.png"
    plt.savefig(outfile, bbox_inches="tight")

    print("Saved", outfile)

    plt.close()


# ------------------------------------------------
# Main loop
# ------------------------------------------------

for csv_file in csv_files:

    benchmark = csv_file.stem.replace("_results", "")

    print("\nProcessing", benchmark)

    df = pd.read_csv(csv_file)

    df = df[df["kind"] == "test"]

    plot_device_scaling(df, benchmark)
    plot_overhead(df, benchmark)
    plot_blocks(df, benchmark)


print("\nAll benchmark comparison plots generated.")