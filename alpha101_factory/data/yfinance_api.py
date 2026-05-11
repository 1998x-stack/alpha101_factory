# -*- coding: utf-8 -*-
"""Yahoo Finance 数据获取模块.

提供从 Yahoo Finance (yfinance) 获取全球股票历史行情数据的能力。
支持 A 股 (通过 139/163 后缀)、美股、港股等全球市场。

Usage:
    from alpha101_factory.data.yfinance_api import fetch_kline_yf

    # A 股 (使用 139/163 后缀)
    df = fetch_kline_yf("600519.SS", start="2024-01-01", end="2025-01-01")

    # 美股
    df = fetch_kline_yf("AAPL", start="2024-01-01", end="2025-01-01")
"""

import pandas as pd
import yfinance as yf
from loguru import logger


# ─── 市场代码映射 ───────────────────────────────────────────

# A 股后缀: .SS = 上海, .SZ = 深圳
_MARKET_SUFFIX = {
    "6": ".SS",  # 60xxxx 上海主板
    "9": ".SS",  # 90xxxx 上海 B 股
    "0": ".SZ",  # 00xxxx 深圳主板/中小板
    "2": ".SZ",  # 20xxxx 深圳 B 股
    "3": ".SZ",  # 30xxxx 创业板
    "4": ".SZ",  # 40xxxx 新三板
    "8": ".BJ",  # 8xxxxx 北交所
}


def to_yf_ticker(symbol: str) -> str:
    """将股票代码转换为 Yahoo Finance ticker.

    Args:
        symbol: 股票代码 (如 '600519', 'AAPL', '000001')

    Returns:
        Yahoo Finance ticker (如 '600519.SS', 'AAPL', '000001.SZ')
    """
    s = str(symbol).strip()
    # 纯 6 位数字 → A 股
    if s.isdigit() and len(s) == 6:
        suffix = _MARKET_SUFFIX.get(s[0], ".SS")
        return f"{s}{suffix}"
    # 已包含后缀 → 直接使用
    if "." in s:
        return s
    # 美股/港股代码 → 直接使用 (如 AAPL, 0700.HK)
    return s


def _map_adjustflag(adjust: str) -> str:
    """映射复权方式到 yfinance 参数.

    Args:
        adjust: 'qfq' (前复权), 'hfq' (后复权), 其他 (不复权)

    Returns:
        yfinance adjust 参数
    """
    if adjust == "hfq":
        return "adjclose"
    if adjust == "qfq":
        return "adjclose"
    return "close"


def fetch_kline_yf(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    period: str = "d",
    adjust: str = "qfq",
) -> pd.DataFrame:
    """从 Yahoo Finance 获取股票 K 线数据.

    Args:
        symbol: 股票代码 (6 位数字 A 股代码 或 Yahoo ticker)
        start_date: 起始日期 (YYYY-MM-DD 或 YYYYMMDD)
        end_date: 结束日期 (YYYY-MM-DD 或 YYYYMMDD)
        period: K 线周期 ('d'=日, 'wk'=周, 'mo'=月)
        adjust: 复权方式 ('qfq'/'hfq'=复权, 其他=不复权)

    Returns:
        DataFrame with columns:
        [datetime, open, high, low, close, volume, amount]
        失败时返回空 DataFrame
    """
    try:
        ticker = to_yf_ticker(symbol)
        logger.info(f"Yahoo Finance 获取 {symbol} → {ticker}")

        # 日期格式标准化
        if start_date:
            s = str(start_date)
            if len(s) == 8 and s.isdigit():
                start_date = f"{s[:4]}-{s[4:6]}-{s[6:]}"
        if end_date:
            e = str(end_date)
            if len(e) == 8 and e.isdigit():
                end_date = f"{e[:4]}-{e[4:6]}-{e[6:]}"

        # 获取数据
        stock = yf.Ticker(ticker)
        hist = stock.history(
            start=start_date,
            end=end_date,
            interval="1d" if period == "d" else "1wk" if period == "wk" else "1mo",
        )

        if hist.empty:
            logger.warning(f"Yahoo Finance 无数据: {symbol} ({ticker})")
            return pd.DataFrame()

        # 标准化
        df = hist.reset_index()
        df.rename(columns={"Date": "datetime"}, inplace=True)

        # 确保 datetime 列存在 (有时 Date 是索引)
        if "datetime" not in df.columns and "Date" in df.columns:
            df.rename(columns={"Date": "datetime"}, inplace=True)
        elif "datetime" not in df.columns:
            # 索引作为日期
            df.index.name = "datetime"
            df = df.reset_index()

        # 选择需要的列
        col_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
        }

        if adjust in ("qfq", "hfq"):
            col_map["Adj Close"] = "close"
        else:
            col_map["Close"] = "close"

        col_map["Volume"] = "volume"

        df = df.rename(columns=col_map)
        keep_cols = ["datetime", "open", "high", "low", "close", "volume"]
        df = df[[c for c in keep_cols if c in df.columns]]

        # 数值转换
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # 日期转换
        df["datetime"] = pd.to_datetime(df["datetime"])

        # 添加 symbol 列
        df.insert(0, "symbol", symbol)

        # 计算成交额 (volume * close)
        if "amount" not in df.columns and "volume" in df.columns and "close" in df.columns:
            df["amount"] = df["volume"] * df["close"]

        df = df.sort_values("datetime").reset_index(drop=True)

        logger.info(f"Yahoo Finance 成功: {symbol} ({ticker}), {len(df)} 行")
        return df

    except Exception as e:
        logger.error(f"Yahoo Finance 获取失败 {symbol}: {e}")
        return pd.DataFrame()
