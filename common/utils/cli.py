from pathlib import Path
from typing import List, Union
import pandas as pd

def load_csv_files(filepaths: List[Union[Path, str]]) -> pd.DataFrame:
  dfs = []
  for file in filepaths:
    cluster = Path(file).stem.split("_")[0]
    df = pd.read_csv(file)
    df["cluster"] = cluster
    dfs.append(df)
  return pd.concat(dfs, ignore_index=True)
