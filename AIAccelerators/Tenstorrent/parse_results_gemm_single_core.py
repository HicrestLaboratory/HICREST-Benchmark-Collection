from result_writer import is_grid, write_results


def main() -> None:
    write_results(
        "gemm_single_core",
        lambda row: (
            row.get("benchmark_kind") == "gemm"
            and is_grid(row, 1, 1)
        ),
    )


if __name__ == "__main__":
    main()
