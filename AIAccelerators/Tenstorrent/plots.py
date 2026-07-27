import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


OUT_DIR = Path("plots")


def load_results(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]

    if not frames:
        raise RuntimeError("No Tenstorrent result CSV files were provided.")

    return pd.concat(frames, ignore_index=True)


def plot_device_time(dataframe: pd.DataFrame, output_stem: str) -> None:
    required = {"operation", "device_time_us"}

    if not required.issubset(dataframe.columns):
        return

    figure, axis = plt.subplots(figsize=(9, 5))

    for (operation, dtype), group in dataframe.groupby(
        ["operation", "dtype"],
        dropna=False,
    ):
        group = group.copy()
        group["problem_elements"] = pd.to_numeric(
            group.get("element_count"),
            errors="coerce",
        )
        group["device_time_us"] = pd.to_numeric(
            group["device_time_us"],
            errors="coerce",
        )
        group = group.dropna(
            subset=["problem_elements", "device_time_us"]
        ).sort_values("problem_elements")

        if group.empty:
            continue

        axis.plot(
            group["problem_elements"],
            group["device_time_us"],
            marker="o",
            label=f"{operation} | {dtype}",
        )

    axis.set_xscale("log", base=2)
    axis.set_xlabel("Logical problem elements")
    axis.set_ylabel("Mean device time [us]")
    axis.set_title("Tenstorrent operation time")
    axis.grid(True, linestyle=":", alpha=0.7)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUT_DIR / f"{output_stem}_device_time.png", dpi=200)
    plt.close(figure)


def plot_throughput(dataframe: pd.DataFrame, output_stem: str) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    plotted = False

    nonlinear = dataframe.copy()

    if "elements_per_second" in nonlinear.columns:
        nonlinear["throughput"] = pd.to_numeric(
            nonlinear["elements_per_second"],
            errors="coerce",
        ) / 1e9
        nonlinear = nonlinear.dropna(subset=["throughput"])

        for operation, group in nonlinear.groupby("operation"):
            axis.scatter(
                group["element_count"],
                group["throughput"],
                label=f"{operation} [Gelem/s]",
            )
            plotted = True

    if not plotted:
        plt.close(figure)
        return

    axis.set_xscale("log", base=2)
    axis.set_xlabel("Logical problem elements")
    axis.set_ylabel("Throughput [billion elements/s]")
    axis.set_title("Tenstorrent nonlinear throughput")
    axis.grid(True, linestyle=":", alpha=0.7)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUT_DIR / f"{output_stem}_throughput.png", dpi=200)
    plt.close(figure)


def main() -> None:
    argument_parser = argparse.ArgumentParser(
        description="Plot parsed Tenstorrent HICREST results."
    )
    argument_parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Result CSVs; defaults to results/tenstorrent_*_data.csv",
    )
    arguments = argument_parser.parse_args()

    paths = arguments.files or sorted(
        Path("results").glob("tenstorrent_*_data.csv")
    )

    dataframe = load_results(paths)
    output_stem = "__".join(path.stem.removesuffix("_data") for path in paths)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_device_time(dataframe, output_stem)
    plot_throughput(dataframe, output_stem)

    print(f"Plots written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

