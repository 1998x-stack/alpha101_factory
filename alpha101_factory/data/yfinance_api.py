# -*- coding: utf-8 -*-
"""Yahoo Finance 数据获取模块.

提供从 Yahoo Finance (yfinance) 获取全球股票历史行情数据的能力。
支持 A 股 (通过 .SS/.SZ 后缀)、美股、港股等全球市场。

Features:
- 自动 ticker 转换 (600519 → 600519.SS)
- 重试机制 + 指数退避 (应对 Rate Limit)
- 代理感知 (自动检测并跳过代理)
- 标准化输出格式 (与 Baostock/AkShare 一致)

Usage:
    from alpha101_factory.data.yfinance_api import fetch_kline_yf

    # A 股
    df = fetch_kline_yf("600519", start_date="2024-01-01")

    # 美股
    df = fetch_kline_yf("AAPL", start_date="2024-01-01")

    # 港股
    df = fetch_kline_yf("0700.HK", start_date="2024-01-01")
"""

import os
import time
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

# 重试配置
_MAX_RETRIES = 3
_RETRY_DELAY = 2  # 秒


def _strip_proxy_env():
    """临时移除代理环境变量 (Yahoo Finance 不需要代理)."""
    saved = {}
    for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
                'ALL_PROXY', 'all_proxy', 'no_proxy', 'NO_PROXY']:
        if key in os.environ:
            saved[key] = os.environ.pop(key)
    return saved


def _restore_proxy_env(saved: dict):
    """恢复代理环境变量."""
    os.environ.update(saved)


def to_yf_ticker(symbol: str) -> str:
    """将股票代码转换为 Yahoo Finance ticker.

    Args:
        symbol: 股票代码 (如 '600519', 'AAPL', '000001', '0700.HK')

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
    # 美股/港股代码 → 直接使用 (如 AAPL, TSLA)
    return s


def _normalize_date(date_str: str) -> str:
    """标准化日期格式: YYYYMMDD → YYYY-MM-DD."""
    s = str(date_str).strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s


def _normalize_df(df: pd.DataFrame, symbol: str, adjust: str) -> pd.DataFrame:
    """标准化 yfinance 输出为项目统一格式.

    Returns:
        DataFrame with columns: [symbol, datetime, open, high, low, close, volume, amount]
    """
    if df.empty:
        return df

    # 确保 DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Date' in df.columns:
            df = df.set_index('Date')
        elif 'datetime' in df.columns:
            df = df.set_index('datetime')
        else:
            return pd.DataFrame()

    df.index.name = "datetime"

    # 列映射
    col_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Volume": "volume",
    }
    if adjust in ("qfq", "hfq") and "Adj Close" in df.columns:
        col_map["Adj Close"] = "close"
    elif "Close" in df.columns:
        col_map["Close"] = "close"

    df = df.rename(columns=col_map)
    keep_cols = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # 数值转换
    for c in keep_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 重置索引
    df = df.reset_index()

    # 添加 symbol 列
    df.insert(0, "symbol", symbol)

    # 计算成交额
    if "volume" in df.columns and "close" in df.columns:
        df["amount"] = df["volume"] * df["close"]

    return df.sort_values("datetime").reset_index(drop=True)


def fetch_kline_yf(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    period: str = "d",
    adjust: str = "qfq",
) -> pd.DataFrame:
    """从 Yahoo Finance 获取股票 K 线数据 (带重试机制).

    Args:
        symbol: 股票代码 (6 位数字 A 股代码 或 Yahoo ticker)
        start_date: 起始日期 (YYYY-MM-DD 或 YYYYMMDD)
        end_date: 结束日期 (YYYY-MM-DD 或 YYYYMMDD)
        period: K 线周期 ('d'=日, 'wk'=周, 'mo'=月)
        adjust: 复权方式 ('qfq'/'hfq'=前复权, 其他=不复权)

    Returns:
        标准化 DataFrame，失败返回空 DataFrame
    """
    ticker = to_yf_ticker(symbol)
    start_date = _normalize_date(start_date) if start_date else None
    end_date = _normalize_date(end_date) if end_date else None

    logger.info(f"Yahoo Finance 获取 {symbol} → {ticker}")

    # 临时移除代理 (YF 直连)
    saved_env = _strip_proxy_env()

    interval = "1d" if period == "d" else "1wk" if period == "wk" else "1mo"

    try:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                )

                if hist.empty:
                    logger.warning(f"Yahoo Finance 无数据: {symbol} ({ticker})")
                    return pd.DataFrame()

                df = _normalize_df(hist, symbol, adjust)
                if not df.empty:
                    logger.info(f"Yahoo Finance 成功: {symbol} ({ticker}), {len(df)} 行")
                return df

            except Exception as e:
                error_msg = str(e)
                if "Too Many Requests" in error_msg or "Rate limited" in error_msg:
                    if attempt < _MAX_RETRIES:
                        delay = _RETRY_DELAY * (2 ** (attempt - 1))
                        logger.warning(f"Yahoo Finance 限流，{delay}秒后重试 ({attempt}/{_MAX_RETRIES})")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"Yahoo Finance 限流，已达最大重试次数: {symbol}")
                elif "possibly delisted" in error_msg:
                    logger.warning(f"Yahoo Finance 股票可能已退市: {symbol} ({ticker})")
                    return pd.DataFrame()
                else:
                    logger.error(f"Yahoo Finance 错误 {symbol}: {e}")
                    return pd.DataFrame()

        return pd.DataFrame()

    finally:
        _restore_proxy_env(saved_env)
