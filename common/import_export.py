import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import warnings
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def describe_pairs_content(
    pairs: List[Tuple[Dict[Any, Any], Dict[str, pd.DataFrame]]],
    verbose: bool = False,
    check_structure: bool = True,
    check_dtypes: bool = False,
) -> None:
    """
    Describe the content of `pairs` as it will be serialized by
    `write_multiple_to_parquet`.

    Args:
        pairs           : List of (metadata_dict, dict_of_dataframes) pairs
        verbose         : Verbose per-pair inspection
        check_structure : Enable structural consistency checks
        check_dtypes    : If True, check DataFrame column dtypes in addition
                          to column names

    Structural checks (if enabled):
        - Metadata: same keys and value types
        - DataFrame dicts: same DataFrame name sets
        - DataFrame schemas:
            * always: column names
            * optionally: column dtypes
    """

    n_pairs = len(pairs)
    print(f"Pairs count: {n_pairs}")

    if n_pairs == 0:
        print("No data to describe.")
        return

    # ------------------------------------------------------------------
    # Reference structures (pair 0)
    # ------------------------------------------------------------------
    ref_meta = pairs[0][0]
    ref_meta_structure = {k: type(v) for k, v in ref_meta.items()}

    ref_df_dict = pairs[0][1]
    ref_df_names = set(ref_df_dict.keys())

    ref_df_columns = {
        name: list(df.columns)
        for name, df in ref_df_dict.items()
    }

    ref_df_dtypes = {
        name: list(df.dtypes.items())
        for name, df in ref_df_dict.items()
    }

    # ------------------------------------------------------------------
    # Aggregates and mismatch tracking
    # ------------------------------------------------------------------
    total_dfs = 0
    total_rows = 0

    meta_structure_mismatch = False
    df_name_mismatch = False
    df_schema_mismatch = False

    meta_mismatch_example = None
    df_name_mismatch_example = None
    df_schema_mismatch_example = None

    # ------------------------------------------------------------------
    # Per-pair inspection
    # ------------------------------------------------------------------
    for pair_id, (meta, df_dict) in enumerate(pairs):
        total_dfs += len(df_dict)

        # ---- Structural checks ----
        if check_structure:
            # Metadata
            meta_structure = {k: type(v) for k, v in meta.items()}
            if meta_structure != ref_meta_structure:
                meta_structure_mismatch = True
                if meta_mismatch_example is None:
                    meta_mismatch_example = (pair_id, meta_structure)

            # DataFrame name set
            df_names = set(df_dict.keys())
            if df_names != ref_df_names:
                df_name_mismatch = True
                if df_name_mismatch_example is None:
                    df_name_mismatch_example = (pair_id, df_names)

        # ---- Per-DataFrame inspection ----
        for df_name, df in df_dict.items():
            total_rows += len(df)

            if check_structure and df_name in ref_df_columns:
                # Column-name check (always)
                if list(df.columns) != ref_df_columns[df_name]:
                    df_schema_mismatch = True
                    if df_schema_mismatch_example is None:
                        df_schema_mismatch_example = (
                            pair_id,
                            df_name,
                            list(df.columns),
                            list(df.dtypes.items()) if check_dtypes else None,
                        )

                # Optional dtype check
                elif check_dtypes:
                    if list(df.dtypes.items()) != ref_df_dtypes[df_name]:
                        df_schema_mismatch = True
                        if df_schema_mismatch_example is None:
                            df_schema_mismatch_example = (
                                pair_id,
                                df_name,
                                list(df.columns),
                                list(df.dtypes.items()),
                            )

        # ---- Verbose output ----
        if verbose:
            print("\n" + "-" * 80)
            print(f"PAIR {pair_id}")

            print("Metadata:")
            if meta:
                for k, v in meta.items():
                    print(f"  - {k!r}: type={type(v).__name__}")
            else:
                print("  (empty)")
            print(f"  + internal '_pair_id' = {pair_id}")

            print("\nDataFrames:")
            if not df_dict:
                print("  (none)")
            else:
                for df_name, df in df_dict.items():
                    print(f"\n  '{df_name}': {df.shape[0]} rows × {df.shape[1]} cols")
                    for col in df.columns:
                        if check_dtypes:
                            print(f"    - {col}: {df[col].dtype}")
                        else:
                            print(f"    - {col}")
                    print("    + internal columns: _pair_id, _df_name")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"Total pairs           : {n_pairs}")
    print(f"Total DataFrames      : {total_dfs}")
    print(f"Total rows            : {total_rows}")
    print("Single Parquet table  : YES")
    print("Metadata stored once  : YES (file-level JSON)")

    # ------------------------------------------------------------------
    # Structural results
    # ------------------------------------------------------------------
    if check_structure:
        print("\nSTRUCTURAL CONSISTENCY CHECKS")
        print("- Metadata structure shared     :", not meta_structure_mismatch)
        print("- DataFrame name set shared     :", not df_name_mismatch)
        print("- DataFrame schemas consistent  :", not df_schema_mismatch)

        if meta_structure_mismatch and meta_mismatch_example:
            pair_id, meta_struct = meta_mismatch_example
            print("\nMETADATA STRUCTURE MISMATCH EXAMPLE")
            print(f"  Reference (pair 0): {ref_meta_structure}")
            print(f"  Pair {pair_id}     : {meta_struct}")

        if df_name_mismatch and df_name_mismatch_example:
            pair_id, df_names = df_name_mismatch_example
            print("\nDATAFRAME NAME SET MISMATCH EXAMPLE")
            print(f"  Reference (pair 0): {ref_df_names}")
            print(f"  Pair {pair_id}     : {df_names}")

        if df_schema_mismatch and df_schema_mismatch_example:
            pair_id, df_name, cols, dtypes = df_schema_mismatch_example
            print("\nDATAFRAME SCHEMA MISMATCH EXAMPLE")
            print(f"  DataFrame '{df_name}'")
            print(f"  Reference columns: {ref_df_columns[df_name]}")
            print(f"  Pair {pair_id} columns: {cols}")
            if check_dtypes and dtypes is not None:
                print(f"  Reference dtypes: {ref_df_dtypes[df_name]}")
                print(f"  Pair {pair_id} dtypes: {dtypes}")

    elif not verbose:
        print("\n(Structural checks disabled.)")

def write_multiple_to_parquet(
    pairs: List[Tuple[Dict[Any, Any], Dict[str, pd.DataFrame]]], path: Path
):
    """
    Write a list of (metadata_dict, dict_of_dataframes) pairs to a single Parquet file.
    Adds an internal 'pair_id' to link metadata to rows.
    Each dataframe in the dict gets an additional 'df_name' column to identify its source.
    """
    df_list = []
    metadata_list = []

    for i, (meta, df_dict) in enumerate(pairs):
        meta_copy = dict(meta)
        meta_copy["_pair_id"] = i
        metadata_list.append(meta_copy)

        for df_name, df in df_dict.items():
            df_copy = df.copy()
            df_copy["_pair_id"] = i
            df_copy["_df_name"] = df_name
            df_list.append(df_copy)

    # Concatenate all dataframes
    full_df = pd.concat(df_list, ignore_index=True)
    table = pa.Table.from_pandas(full_df)

    # Encode metadata_list as JSON → bytes
    json_bytes = json.dumps(metadata_list).encode("utf-8")
    existing_meta = table.schema.metadata or {}
    merged_meta = {**existing_meta, b"user_metadata_list": json_bytes}
    table = table.replace_schema_metadata(merged_meta)

    pq.write_table(table, path)
    print(
        f'Written {len(pairs)} metadata/dict_of_dataframes pairs to "{path.resolve().absolute()}"'
    )


def read_multiple_from_parquet(
    paths: List[Path] | Path,
) -> Tuple[List[Tuple[Any, Dict[str, pd.DataFrame]]], pd.DataFrame | None]:
    """
    Read one or multiple Parquet files written by write_multiple_to_parquet().
    Returns:
        mapping: list of (metadata_dict, dict_of_dataframes) tuples
        metadata_df: optional DataFrame with metadata values if all metadata dicts
                     share the same keys and contain only scalar values
    """
    if isinstance(paths, Path):
        paths = [paths]
    combined_result = []
    seen_keys = set()
    metadata_records = []

    for path in paths:
        table = pq.read_table(path)
        df = table.to_pandas()
        schema_meta = table.schema.metadata or {}
        json_bytes = schema_meta.get(b"user_metadata_list")
        if json_bytes is None:
            continue
        metadata_list = json.loads(json_bytes.decode("utf-8"))

        for meta in metadata_list:
            pair_id = meta.pop("_pair_id")
            subset_df = df[df["_pair_id"] == pair_id].drop(columns=["_pair_id"])

            # Split subset_df by _df_name into a dictionary
            df_dict = {}
            if "_df_name" in subset_df.columns:
                for df_name in subset_df["_df_name"].unique():
                    df_subset = subset_df[subset_df["_df_name"] == df_name].drop(
                        columns=["_df_name"]
                    )
                    # Drop columns that are all NaN (artifacts from concatenation)
                    df_subset = df_subset.dropna(axis=1, how="all")
                    df_dict[df_name] = df_subset.reset_index(drop=True)
            else:
                # Fallback if no _df_name column exists (shouldn't happen with new format)
                df_dict["default"] = subset_df

            # Use hashable string for duplicate checking
            key_str = json.dumps(meta, sort_keys=True)
            if key_str in seen_keys:
                warnings.warn(
                    f"Duplicate metadata found in {path}: {meta}. Keeping first occurrence."
                )
                continue
            seen_keys.add(key_str)
            combined_result.append((meta, df_dict))
            metadata_records.append(meta)

    # Try building metadata DataFrame if all dicts share keys & scalar values
    metadata_df = None
    if metadata_records:
        keys = set(metadata_records[0].keys())
        same_keys = all(set(d.keys()) == keys for d in metadata_records)

        if same_keys:
            serialized_records = []
            for d in metadata_records:
                serialized_records.append({
                    k: json.dumps(v) if isinstance(v, (dict, list)) else v
                    for k, v in d.items()
                })

            metadata_df = pd.DataFrame(serialized_records)

    return combined_result, metadata_df


def read_multiple_from_csv(
    paths: List[Path | str] | Path | str,
) -> pd.DataFrame | None:
    """
    Read one or multiple CSV paths and merge them into a single DataFrame.

    Parameters
    ----------
    paths : List[Path] | Path
        A single path or list of paths. Each path can be:
        - a CSV file
        - a directory (all *.csv files inside will be read)

    Returns
    -------
    pd.DataFrame | None
        Concatenated DataFrame or None if no valid CSV files were found.
    """

    if isinstance(paths, Path):
        paths = [paths]

    dataframes = []

    for p in paths:
        path = Path(p)
            
        if not path.exists():
            continue

        if path.is_file() and path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            dataframes.append(df)

        elif path.is_dir():
            for csv_file in sorted(path.glob("*.csv")):
                df = pd.read_csv(csv_file)
                dataframes.append(df)

    if not dataframes:
        return None

    return pd.concat(dataframes, ignore_index=True)

def query_meta_df_dict_pairs(
    meta_df_dict_pairs: List[Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]],
    meta_filters: List[Tuple[str, Any]],
) -> list[Tuple[Any, Dict[str, pd.DataFrame]]]:
    res = []
    for meta, df_dict in meta_df_dict_pairs:
        ok = True
        for filter_key, filter_val in meta_filters:
            if meta[filter_key] != filter_val:
                ok = False
                break
        if ok:
            res.append((meta, df_dict))
    return res