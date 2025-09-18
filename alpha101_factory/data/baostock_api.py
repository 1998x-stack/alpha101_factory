# -*- coding: utf-8 -*-
"""Baostock K线数据获取模块.

提供从 Baostock 获取股票历史行情数据的工具函数，包括
代码映射、复权方式映射、K线数据下载与标准化。
"""

from __future__ import annotations
import pandas as pd
import baostock as bs
from loguru import logger


def bs_code(symbol: str) -> str:
    """将 6 位股票代码映射为 Baostock 格式代码.

    Args:
        symbol (str): 股票代码（如 '600000' 或 '000001'）。

    Returns:
        str: Baostock 格式代码，如 'sh.600000' 或 'sz.000001'。
    """
    s = str(symbol).zfill(6)
    if s.startswith("6"):
        return f"sh.{s}"
    return f"sz.{s}"


def _map_adjustflag(adjust: str) -> str:
    """将输入的复权标志映射为 Baostock API 所需的 adjustflag.

    Args:
        adjust (str): 复权方式，支持 'hfq'（后复权）、'qfq'（前复权）、其他（不复权）。

    Returns:
        str: Baostock 对应的复权标志（'1'、'2'、'3'）。
    """
    if adjust == "hfq":
        return "1"
    if adjust == "qfq":
        return "2"
    return "3"


def fetch_kline_bs(
    symbol: str,
    start_date: str | None,
    end_date: str | None,
    period: str = "d",
    adjust: str = "qfq",
) -> pd.DataFrame:
    """从 Baostock 获取股票 K 线数据.

    Args:
        symbol (str): 股票代码（6 位数字，如 '600000'）。
        start_date (str | None): 起始日期，格式 'YYYYMMDD'，若为 None 则不限制。
        end_date (str | None): 结束日期，格式 'YYYYMMDD'，若为 None 则不限制。
        period (str, optional): K线周期，支持 'd'（日）、'w'（周）、'm'（月）、'5'、'15'、'30'、'60'。默认 'd'。
        adjust (str, optional): 复权方式，'hfq'=后复权，'qfq'=前复权，其他=不复权。默认 'qfq'。

    Returns:
        pd.DataFrame: 包含 ['datetime','open','high','low','close','volume','amount'] 的行情数据，
                      若无数据则返回空 DataFrame。
    """
    try:
        lg = bs.login()
        if lg.error_code != "0":
            logger.warning(f"Baostock 登录失败: {lg.error_msg}")
            return pd.DataFrame()

        code = bs_code(symbol)
        fields = ",".join(["date", "open", "high", "low", "close", "volume", "amount"])

        # 处理日期格式，将 YYYYMMDD 转换为 YYYY-MM-DD
        start_fmt = None if not start_date else f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
        end_fmt = None if not end_date else f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

        rs = bs.query_history_k_data_plus(
            code=code,
            fields=fields,
            start_date=start_fmt,
            end_date=end_fmt,
            frequency=period,
            adjustflag=_map_adjustflag(adjust),
        )

        rows: list[list[str]] = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
    except Exception as e:
        logger.exception(f"获取 Baostock 数据失败: {e}")
        return pd.DataFrame()
    finally:
        try:
            bs.logout()
        except Exception:
            pass

    if not rows:
        return pd.DataFrame()

    try:
        # 构建 DataFrame 并格式化字段
        df = pd.DataFrame(rows, columns=fields.split(","))
        df.rename(columns={"date": "datetime"}, inplace=True)
        df["datetime"] = pd.to_datetime(df["datetime"])

        # 数值型字段转换为 float
        num_cols = ["open", "high", "low", "close", "volume", "amount"]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df.sort_values("datetime", inplace=True)
        return df
    except Exception as e:
        logger.exception(f"Baostock 数据解析失败: {e}")
        return pd.DataFrame()
