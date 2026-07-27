import json
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd
import sbatchman as sbm

from parser import parse


OUT_DIR = Path("results")
ResultRow = Mapping[str, Any]
ResultFilter = Callable[[ResultRow], bool]


def csv_safe(value: Any) -> Any:
    """Serialize nested values while leaving CSV-friendly scalars intact."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def is_grid(row: ResultRow, x: int, y: int) -> bool:
    """Return whether a result used the requested compute grid."""
    try:
        return (
            int(row["compute_grid_x"]) == x
            and int(row["compute_grid_y"]) == y
        )
    except (KeyError, TypeError, ValueError):
        return False


def write_results(output_name: str, result_filter: ResultFilter) -> None:
    """Collect matching completed jobs and write one cluster-specific CSV."""
    jobs = sbm.jobs_list(
        status=[sbm.Status.COMPLETED],
        from_active=True,
        from_archived=True,
    )

    rows = []
    for job in jobs:
        parsed = parse(job)
        if parsed is None:
            continue

        result = parsed["tenstorrent"]
        if result_filter(result):
            rows.append({key: csv_safe(value) for key, value in result.items()})

    if not rows:
        print(f"No completed Tenstorrent {output_name} results found.")
        return

    dataframe = pd.DataFrame(rows)
    sort_columns = [
        column
        for column in (
            "operation",
            "dtype",
            "element_count",
            "M",
            "K",
            "N",
        )
        if column in dataframe.columns
    ]
    if sort_columns:
        dataframe = dataframe.sort_values(sort_columns, na_position="last")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cluster = sbm.get_cluster_name()
    output_path = OUT_DIR / f"tenstorrent_{output_name}_{cluster}_data.csv"
    dataframe.to_csv(output_path, index=False)
    print(f"Wrote {len(dataframe)} experiments to {output_path}")
