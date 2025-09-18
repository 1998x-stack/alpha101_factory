# -*- coding: utf-8 -*-
"""
因子计算与保存脚本

本模块功能：
1. 加载指定股票的 K线数据与临时特征数据（tmp features），并进行合并；
2. 动态获取因子类，调用其 `compute` 方法计算因子值；
3. 将计算结果保存为 Parquet 文件，存放在 `PARQ_DIR_FACT` 目录下；
4. 提供 main 函数批量计算一组常见 Alpha 因子。

适用于量化回测与因子库管理，确保数据处理与因子生成自动化。
"""

import pandas as pd
from loguru import logger
from pathlib import Path
from typing import List, Optional

from alpha101_factory.utils.log import setup_logger
from alpha101_factory.config import (
    PARQ_DIR_KLINES,
    PARQ_DIR_TMP,
    PARQ_DIR_FACT,
    START_DATE,
    END_DATE,
    ADJUST,
)
from alpha101_factory.utils.io import read_parquet, write_parquet
from alpha101_factory.factors.registry import get_factor


def _load_join(symbols: Optional[List[str]]) -> pd.DataFrame:
    """加载并合并指定股票的 K线数据与临时特征表。

    Args:
        symbols (Optional[List[str]]): 股票代码列表；若为 None，则自动读取 tmp 目录中的全部股票。

    Returns:
        pd.DataFrame: 合并后的长表数据，按 [datetime, symbol] 排序。
                      若无有效数据，返回空 DataFrame。
    """
    if symbols is None:
        symbols = sorted({p.stem for p in (PARQ_DIR_TMP).glob("*.parquet")})

    dfs = []
    for sym in symbols:
        try:
            # 动态选择文件路径（是否加上日期与复权标记）
            if START_DATE and END_DATE:
                kline_path = PARQ_DIR_KLINES / f"{sym}_{START_DATE}_{END_DATE}_{ADJUST}.parquet"
                tmp_path = PARQ_DIR_TMP / f"{sym}_{START_DATE}_{END_DATE}_{ADJUST}.parquet"
            else:
                kline_path = PARQ_DIR_KLINES / f"{sym}.parquet"
                tmp_path = PARQ_DIR_TMP / f"{sym}.parquet"

            k = read_parquet(kline_path)
            t = read_parquet(tmp_path)

            if k.empty or t.empty:
                continue

            # 外连接合并，保留所有信息
            m = pd.merge(
                k,
                t,
                on=["symbol", "datetime", "open", "high", "low", "close", "volume", "amount"],
                how="outer",
                sort=True,
            )
            dfs.append(m)
        except Exception as e:
            logger.error(f"加载或合并数据失败: {sym}, 错误: {e}")
            continue

    if not dfs:
        logger.warning("未加载到任何有效数据。")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    return df.sort_values(["datetime", "symbol"]).reset_index(drop=True)


def compute_and_save(factor_name: str, symbols: Optional[List[str]] = None) -> None:
    """计算并保存指定因子。

    Args:
        factor_name (str): 因子名称（需已注册到 registry）。
        symbols (Optional[List[str]]): 股票代码列表，若为 None 则处理全部股票。

    Side Effects:
        在 `PARQ_DIR_FACT` 目录下生成对应的因子结果文件。
    """
    try:
        FactorCls = get_factor(factor_name)  # 动态获取因子类
    except Exception as e:
        logger.error(f"获取因子类失败: {factor_name}, 错误: {e}")
        return

    logger.info(f"开始计算因子 {factor_name}，股票范围: {('ALL' if symbols is None else len(symbols))}")

    df = _load_join(symbols)
    if df.empty:
        logger.error("数据为空，请先运行 build_tmp 生成中间特征。")
        return

    try:
        fac = FactorCls()
        s = fac.compute(df)  # 结果为 MultiIndex: [datetime, symbol]
    except Exception as e:
        logger.error(f"因子 {factor_name} 计算失败: {e}")
        return

    try:
        out = s.reset_index().rename(columns={0: "value"})
        out_path = PARQ_DIR_FACT / f"{factor_name}.parquet"
        write_parquet(out, out_path)
        logger.info(f"因子 {factor_name} 已保存至 {out_path}, 共 {len(out)} 行。")
    except Exception as e:
        logger.error(f"保存因子 {factor_name} 失败: {e}")


def main() -> None:
    """批量计算一组 Alpha 因子。"""
    setup_logger()

    # 自动扫描 tmp 目录下的股票池
    symbols = sorted({p.stem for p in (PARQ_DIR_TMP).glob("*.parquet")})
    if not symbols:
        logger.error("未发现 tmp 文件，请先运行 build_tmp。")
        return

    factor_list = [
        "Alpha001", "Alpha003", "Alpha004", "Alpha005", "Alpha006", "Alpha009", "Alpha010",
        "Alpha011", "Alpha012", "Alpha013", "Alpha014", "Alpha016", "Alpha018", "Alpha019",
        "Alpha020", "Alpha021", "Alpha022", "Alpha023", "Alpha024", "Alpha025", "Alpha026",
        "Alpha030", "Alpha031", "Alpha032", "Alpha033", "Alpha034", "Alpha035", "Alpha036",
        "Alpha037", "Alpha038", "Alpha039", "Alpha040", "Alpha041", "Alpha042", "Alpha043",
        "Alpha044", "Alpha045", "Alpha046", "Alpha047", "Alpha049", "Alpha050", "Alpha051",
        "Alpha052", "Alpha053", "Alpha054", "Alpha055", "Alpha060", "Alpha061", "Alpha064",
        "Alpha065", "Alpha071", "Alpha083", "Alpha084", "Alpha085", "Alpha086", "Alpha094",
        "Alpha095", "Alpha096", "Alpha098", "Alpha099", "Alpha101",
    ]

    for name in factor_list:
        compute_and_save(name, symbols)


if __name__ == "__main__":
    main()
