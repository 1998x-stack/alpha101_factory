# -*- coding: utf-8 -*-
"""股票池加载模块.

从本地存储的 A 股快照文件中加载股票代码，用于构建投资组合或回测。
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))


import pandas as pd
from alpha101_factory.config import PARQ_DIR_SPOT
from alpha101_factory.utils.io import read_parquet



def load_universe(limit: int = 0) -> pd.Series:
    """加载股票池（Universe）。

    从本地快照文件 `a_spot.parquet` 中读取全部股票代码，
    并支持可选的截取数量限制。

    Args:
        limit (int, optional): 限制返回的股票数量。
            - 若为 0 或负数，则返回全部股票（默认）。
            - 若为正数，则仅返回前 limit 个股票代码。

    Returns:
        pd.Series: 股票代码序列（6 位数字字符串），Series 名为 "symbol"。
                   若文件不存在或为空，则返回空的 Series。
    """
    try:
        # 读取本地快照文件
        spot = read_parquet(PARQ_DIR_SPOT / "a_spot.parquet")
    except Exception as e:
        # 出错时返回空 Series
        return pd.Series([], dtype=str, name="symbol")

    if spot.empty:
        return pd.Series([], dtype=str, name="symbol")

    # 确保代码为 6 位字符串
    codes = spot["code"].astype(str).str.zfill(6).unique()

    # 可选截取前 limit 个股票
    if limit and limit > 0:
        codes = codes[:limit]

    return pd.Series(codes, name="symbol")


if __name__ == "__main__":
    from pprint import pprint
    codes = load_universe(100)
    pprint(codes)