import json
from pathlib import Path
from typing import Any

import pandas as pd
import sbatchman as sbm

from parser import parse


OUT_DIR = Path("results")


def csv_safe(value: Any) -> Any:
    """Serialize nested values while leaving CSV-friendly scalars intact."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def main() -> None:
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

        rows.append(
            {
                key: csv_safe(value)
                for key, value in parsed["tenstorrent"].items()
            }
        )

    if not rows:
        print("No completed Tenstorrent HICREST_RESULT records found.")
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
        dataframe = dataframe.sort_values(
            sort_columns,
            na_position="last",
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cluster = sbm.get_cluster_name()
    output_path = OUT_DIR / f"tenstorrent_{cluster}_data.csv"
    dataframe.to_csv(output_path, index=False)

    print(f"Wrote {len(dataframe)} experiments to {output_path}")


if __name__ == "__main__":
    main()

