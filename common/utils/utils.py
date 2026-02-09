from typing import Any, Dict, List, Tuple
import pandas as pd


def dict_get(d, key):
    r = d.get(key)
    if r is None:
        raise KeyError(f"{key} not found")
    return r


def raise_none(v, msg):
    if v is None:
        raise ValueError(f"{msg} not found")
    return v


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
