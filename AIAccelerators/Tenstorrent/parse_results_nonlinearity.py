from result_writer import write_results


def main() -> None:
    write_results(
        "nonlinearity",
        lambda row: row.get("benchmark_kind") == "nonlinearity",
    )


if __name__ == "__main__":
    main()
