# -*- coding: utf-8 -*-
import pandas as pd
from ..config import PARQ_DIR_SPOT
from ..utils.io import read_parquet

def load_universe(limit:int=0) -> pd.Series:
    spot = read_parquet(PARQ_DIR_SPOT / "a_spot.parquet")
    if spot.empty:
        return pd.Series([], dtype=str)
    codes = spot["code"].astype(str).str.zfill(6).unique()
    if limit and limit > 0:
        codes = codes[:limit]
    return pd.Series(codes, name="symbol")

