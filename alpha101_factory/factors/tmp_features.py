# -*- coding: utf-8 -*-
"""
构建与加载中间特征 (tmp features) 模块

本模块主要功能：
1. 为单只股票构建中间特征文件 (tmp parquet)，包括收益率、VWAP、ADV 等；
2. 批量构建多个股票的中间特征文件；
3. 统一加载多个股票的 tmp 表，合并成长表 DataFrame。

该模块是因子计算的前置步骤，确保数据格式与必需字段齐全，
并为后续因子计算提供高效的输入。
"""

import sys
from pathlib import Path
import pandas as pd
from loguru import logger
from tqdm import tqdm

# 确保可以从项目根目录导入模块
try:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
except Exception as e:
    raise RuntimeError("无法设置 sys.path，请检查项目目录结构") from e

# 项目内部依赖
from alpha101_factory.config import PARQ_DIR_KLINES, PARQ_DIR_TMP, START_DATE, END_DATE, ADJUST
from alpha101_factory.utils.io import read_parquet, write_parquet
from alpha101_factory.utils import ops

# 定义常用 ADV 窗口
ADV_WINDOWS = [5, 10, 20, 30, 40, 60, 120, 150, 180]


def build_tmp_for_symbol(sym: str) -> bool:
    """为单只股票构建中间特征文件 (tmp parquet)。

    Args:
        sym (str): 股票代码。

    Returns:
        bool: 若成功生成并保存 tmp 文件返回 True，否则返回 False。

    Notes:
        - 必需字段: ["open","high","low","close","volume","amount","datetime","symbol"]
        - 生成特征: 收益率、VWAP、不同窗口的 ADV
    """
    path_k = PARQ_DIR_KLINES / f"{sym}_{START_DATE}_{END_DATE}_{ADJUST}.parquet" if START_DATE and END_DATE else PARQ_DIR_KLINES / f"{sym}.parquet"
    try:
        df = read_parquet(path_k)
    except Exception as e:
        logger.error(f"读取 K线数据失败: {sym}, 错误: {e}")
        return False

    if df.empty:
        logger.warning(f"K线数据为空: {sym}")
        return False

    # 检查必需列是否齐全
    required_cols = ["open", "high", "low", "close", "volume", "amount", "datetime", "symbol"]
    for c in required_cols:
        if c not in df.columns:
            logger.error(f"缺少必要列 {c}: {sym}")
            return False

    # 按时间排序
    df = df.sort_values("datetime").reset_index(drop=True)

    # ===== 构造中间特征 =====
    try:
        # 收益率
        df["returns"] = ops.returns(df["close"])

        # 成交量加权平均价 (VWAP)
        df["vwap"] = ops.vwap_from_amount(
            df["close"], df["high"], df["low"], df["volume"], df["amount"]
        )

        # 平均成交量 (ADV)
        for n in ADV_WINDOWS:
            df[f"adv{n}"] = ops.adv(df["volume"], n)
    except Exception as e:
        logger.error(f"计算特征失败: {sym}, 错误: {e}")
        return False

    # ===== 保存结果 =====
    try:
        out = PARQ_DIR_TMP / f"{sym}_{START_DATE}_{END_DATE}_{ADJUST}.parquet" if START_DATE and END_DATE else PARQ_DIR_TMP / f"{sym}.parquet"
        cols_to_save = [
            "symbol", "datetime", "open", "high", "low", "close",
            "volume", "amount", "returns", "vwap",
            *[f"adv{n}" for n in ADV_WINDOWS]
        ]
        write_parquet(df[cols_to_save], out)
        return True
    except Exception as e:
        logger.error(f"保存 tmp 文件失败: {sym}, 错误: {e}")
        return False


def build_tmp_all(symbols: list[str]) -> int:
    """批量构建多个股票的中间特征文件。

    Args:
        symbols (list[str]): 股票代码列表。

    Returns:
        int: 成功生成 tmp 文件的数量。
    """
    cnt = 0
    for sym in tqdm(symbols, desc="构建 tmp"):
        try:
            ok = build_tmp_for_symbol(sym)
            if ok:
                cnt += 1
        except Exception as e:
            logger.error(f"处理股票 {sym} 失败: {e}")
            continue
    logger.info(f"成功保存 tmp 特征文件数量: {cnt}")
    return cnt


def load_panel(symbols: list[str]) -> pd.DataFrame:
    """合并多个股票的 tmp 文件，生成长表形式的 DataFrame。

    Args:
        symbols (list[str]): 股票代码列表。

    Returns:
        pd.DataFrame: 合并后的长表数据，按 datetime 和 symbol 排序。
                      若所有股票均无数据，则返回空 DataFrame。
    """
    dfs = []
    for sym in symbols:
        p = PARQ_DIR_TMP / f"{sym}_{START_DATE}_{END_DATE}_{ADJUST}.parquet" if START_DATE and END_DATE else PARQ_DIR_TMP / f"{sym}.parquet"
        try:
            tmp = read_parquet(p)
        except Exception as e:
            logger.error(f"读取 tmp 文件失败: {sym}, 错误: {e}")
            continue
        if tmp.empty:
            continue
        dfs.append(tmp)

    if not dfs:
        logger.warning("未加载到任何 tmp 文件，返回空 DataFrame")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    return df.sort_values(["datetime", "symbol"]).reset_index(drop=True)
