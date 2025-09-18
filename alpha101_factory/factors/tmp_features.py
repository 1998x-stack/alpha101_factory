# -*- coding: utf-8 -*-
import pandas as pd
from loguru import logger
from tqdm import tqdm
from pathlib import Path

from ..config import PARQ_DIR_KLINES, PARQ_DIR_TMP, LIMIT_STOCKS
from ..utils.io import read_parquet, write_parquet
from ..utils import ops

ADV_WINDOWS = [5, 10, 20, 30, 40, 60, 120, 150, 180]

def build_tmp_for_symbol(sym: str) -> bool:
    path_k = PARQ_DIR_KLINES / f"{sym}.parquet"
    df = read_parquet(path_k)
    if df.empty:
        return False
    # 基础列
    for c in ["open","high","low","close","volume","amount","datetime","symbol"]:
        if c not in df.columns:
            return False
    df = df.sort_values("datetime").reset_index(drop=True)

    # 中间变量
    df["returns"] = ops.returns(df["close"])
    df["vwap"] = ops.vwap_from_amount(df["close"], df["high"], df["low"], df["volume"], df["amount"])
    for n in ADV_WINDOWS:
        df[f"adv{n}"] = ops.adv(df["volume"], n)

    # 保存
    out = PARQ_DIR_TMP / f"{sym}.parquet"
    write_parquet(df[["symbol","datetime","open","high","low","close","volume","amount","returns","vwap", *[f"adv{n}" for n in ADV_WINDOWS]]], out)
    return True

def build_tmp_all(symbols: list[str]) -> int:
    cnt = 0
    for sym in tqdm(symbols, desc="build tmp"):
        ok = build_tmp_for_symbol(sym)
        if ok: cnt += 1
    logger.info(f"tmp features saved: {cnt}")
    return cnt

def load_panel(symbols: list[str]) -> pd.DataFrame:
    """合并所需股票的 tmp 表（长表）"""
    dfs = []
    for sym in symbols:
        p = PARQ_DIR_TMP / f"{sym}.parquet"
        tmp = read_parquet(p)
        if tmp.empty: 
            continue
        dfs.append(tmp)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    return df.sort_values(["datetime","symbol"]).reset_index(drop=True)
