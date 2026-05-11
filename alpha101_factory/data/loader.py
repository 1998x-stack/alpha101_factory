from __future__ import annotations
# -*- coding: utf-8 -*-
"""A股行情数据获取与管理模块.

该模块封装了 A 股行情获取、数据归一化、图像保存、本地文件完整性检查等功能。
默认优先使用 AkShare 获取数据，若失败则回退至 Baostock。
"""
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import pandas as pd
import akshare as ak
import time

from alpha101_factory.config import (
    PARQ_DIR_SPOT,
    PARQ_DIR_KLINES,
    ADJUST,
    START_DATE,
    END_DATE,
    LIMIT_STOCKS,
    REQUEST_PAUSE,
    IMG_KLINES_DIR,
)
from alpha101_factory.utils.io import write_parquet, read_parquet
from alpha101_factory.viz.plots import plot_kline, save_fig
from alpha101_factory.data.baostock_api import fetch_kline_bs


# ---------- 辅助工具 ----------
def _resolve_kline_path(symbol: str,
                        start_date: str | None,
                        end_date: str | None,
                        adjust: str | None) -> Path:
    """统一生成 K 线 parquet 文件路径。

    Args:
        symbol (str): 股票代码，允许带交易所前缀（如 ``sh600000``）。
        start_date (str | None): 起始日期 ``YYYYMMDD``，若为空则使用全量文件名。
        end_date (str | None): 结束日期 ``YYYYMMDD``，若为空则使用全量文件名。
        adjust (str | None): 复权方式，默认回退到全局 ``ADJUST``。

    Returns:
        Path: 生成的 parquet 路径，若未指定日期范围则返回 ``<symbol>.parquet``。
    """

    sym = ''.join(filter(str.isdigit, symbol)) or symbol
    start = (start_date or "").strip()
    end = (end_date or "").strip()
    adj = (adjust if adjust is not None else ADJUST)
    adj = adj.strip() if isinstance(adj, str) else adj

    if start and end:
        return PARQ_DIR_KLINES / f"{sym}_{start}_{end}_{adj}.parquet"
    return PARQ_DIR_KLINES / f"{sym}.parquet"


# ---------- 数据标准化 ----------
def normalize_k(df: pd.DataFrame) -> pd.DataFrame:
    """标准化 K 线数据字段名称和格式.

    Args:
        df (pd.DataFrame): 原始 K 线数据。

    Returns:
        pd.DataFrame: 标准化后的 DataFrame。
    """
    if df is None or df.empty:
        return df

    mapping = {
        "日期": "datetime",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "涨跌幅": "pct_change",
        "涨跌额": "change",
        "振幅": "amplitude",
        "换手率": "turnover",
    }

    # 字段重命名
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})

    # 移除冗余列 (如果 symbol 已存在，删除'股票代码')
    if 'symbol' in df.columns and '股票代码' in df.columns:
        df = df.drop(columns=['股票代码'])

    # 时间字段转换
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")

    # 数值型字段转换
    num_cols = ["open", "high", "low", "close", "volume", "amount",
                "pct_change", "change", "amplitude", "turnover"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------- 数据获取 ----------
def _fetch_kline_ak(symbol: str, start_date: str | None,
                    end_date: str | None, adjust: str) -> pd.DataFrame:
    """优先使用 AkShare 获取 K 线数据."""
    # 提取数字的部分
    symbol = ''.join(filter(str.isdigit, symbol))
    logger.info(f"正在获取 {symbol} {start_date or 'start'}~{end_date or 'end'} {adjust} K 线数据 …")
    if start_date and end_date:
        k = ak.stock_zh_a_hist(
            symbol=symbol,
            start_date=start_date or None,
            end_date=end_date or None,
            period="daily",
            adjust=adjust,
        )
    else:
        k = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            adjust=adjust,
        )
    return normalize_k(k)


def _fetch_kline_fallback(symbol: str, start_date: str | None,
                          end_date: str | None, adjust: str) -> pd.DataFrame:
    """获取 K 线数据，三级回退: AkShare → Baostock → Yahoo Finance."""
    # 1. AkShare (主源)
    try:
        k = _fetch_kline_ak(symbol, start_date, end_date, adjust)
        if k is not None and not k.empty:
            return k
    except Exception as e:
        logger.warning(f"AkShare 获取 {symbol} 数据失败: {e}")

    # 2. Baostock (备源)
    try:
        logger.info(f"尝试 Baostock 获取 {symbol} …")
        k = fetch_kline_bs(symbol, start_date, end_date, period="d", adjust=adjust)
        if k is not None and not k.empty:
            return k
    except Exception as e:
        logger.warning(f"Baostock 获取 {symbol} 数据失败: {e}")

    # 3. Yahoo Finance (全球数据)
    try:
        logger.info(f"尝试 Yahoo Finance 获取 {symbol} …")
        from alpha101_factory.data.yfinance_api import fetch_kline_yf
        k = fetch_kline_yf(symbol, start_date, end_date, adjust=adjust)
        if k is not None and not k.empty:
            return k
    except Exception as e:
        logger.warning(f"Yahoo Finance 获取 {symbol} 数据失败: {e}")

    logger.error(f"所有数据源均获取失败: {symbol}")
    return pd.DataFrame()


# ---------- 图片保存 ----------
def _save_kline_png(sym: str, df: pd.DataFrame,
                    start_date: str | None,
                    end_date: str | None,
                    adjust: str) -> Path:
    """保存单只股票的 K 线图."""
    start_tag = start_date or "start"
    end_tag = end_date or "end"
    out_png = IMG_KLINES_DIR / f"{sym}_{start_tag}_{end_tag}_{adjust}.png"

    try:
        fig = plot_kline(
            df,
            f"{sym} {adjust} {start_tag}-{end_tag}",
            tickformat="%Y-%m-%d",
            tickangle=-45,
        )
        save_fig(fig, out_png)
    except Exception as e:
        logger.warning(f"{sym} 绘制/保存 K 线图失败: {e}")

    return out_png


# ---------- 公共 API ----------
def fetch_spot(save: bool = True) -> pd.DataFrame:
    """获取或读取 A 股实时行情快照。

    若本地已存在缓存文件 ``a_spot.parquet``，优先读取缓存以避免重复请求。
    否则将通过 AkShare 获取实时行情，并根据 ``save`` 参数决定是否缓存。
    """
    logger.info("正在获取 A 股实时行情 …")
    spot_path = PARQ_DIR_SPOT / "a_spot.parquet"
    if spot_path.exists():
        spot = read_parquet(spot_path)
        logger.info(f"读取本地快照文件 {spot_path}")
        return spot
    
    try:
        spot = ak.stock_zh_a_spot()
    except Exception as e:
        logger.exception(f"获取实时行情失败: {e}")
        return pd.DataFrame()

    spot = spot.rename(columns={"代码": "code", "名称": "name"})
    if save:
        write_parquet(spot, PARQ_DIR_SPOT / "a_spot.parquet")

    logger.info(f"实时行情共 {len(spot)} 行 | 保存={save}")
    return spot


def fetch_klines_from_spot(spot: pd.DataFrame) -> int:
    """从实时快照获取股票代码并批量下载 K 线数据."""
    codes = spot["code"].astype(str).str.zfill(6).unique().tolist()
    if LIMIT_STOCKS and LIMIT_STOCKS > 0:
        codes = codes[:LIMIT_STOCKS]

    logger.info(
        f"准备下载 {len(codes)} 只股票 | adjust={ADJUST} "
        f"start={START_DATE} end={END_DATE or 'latest'}"
    )

    n_new = 0
    for sym in tqdm(codes, desc="下载日线"):
        sym = ''.join(filter(str.isdigit, sym))
        out = _resolve_kline_path(sym, START_DATE if START_DATE else None,
                                  END_DATE if END_DATE else None, ADJUST)

        if out.exists():
            # 已有本地文件，直接绘图
            df_local = read_parquet(out)
            if not df_local.empty:
                _save_kline_png(sym, df_local, START_DATE or "all", END_DATE or "all", ADJUST)
            continue

        try:
            k = _fetch_kline_fallback(sym, START_DATE or None, END_DATE or None, ADJUST)
            if k is not None and not k.empty:
                k.insert(0, "symbol", sym)
                write_parquet(k, out)
                _save_kline_png(sym, k, START_DATE or "all", END_DATE or "all", ADJUST)
                n_new += 1
        except Exception as e:
            logger.warning(f"{sym} 下载失败: {e}")

        time.sleep(REQUEST_PAUSE)

    logger.info(f"新保存 {n_new} 个文件")
    return n_new


def check_klines_integrity() -> pd.DataFrame:
    """检查本地 K 线文件完整性，输出报告."""
    spot = read_parquet(PARQ_DIR_SPOT / "a_spot.parquet")
    if spot.empty:
        logger.warning("未找到本地快照文件。")
        return pd.DataFrame()

    codes = spot["code"].astype(str).str.zfill(6).unique()
    rows = []

    for code in codes:
        code = ''.join(filter(str.isdigit, code))
        path = _resolve_kline_path(code, START_DATE if START_DATE else None,
                                   END_DATE if END_DATE else None, ADJUST)
        if not path.exists():
            rows.append([code, False, 0, None, None, str(path)])
            continue

        try:
            df = read_parquet(path)
            if df.empty:
                rows.append([code, True, 0, None, None, str(path)])
            else:
                dmin, dmax = pd.to_datetime(df["datetime"]).min(), pd.to_datetime(df["datetime"]).max()
                rows.append([code, True, len(df), dmin, dmax, str(path)])
        except Exception as e:
            rows.append([code, False, -1, None, None, f"{path} ERROR: {e}"])

    report = pd.DataFrame(rows, columns=["symbol", "exists", "rows", "date_min", "date_max", "path"])
    logger.info(
        f"K线文件检查: 存在 {report['exists'].sum()}/{len(report)}; "
        f"空文件 {(report['rows'] == 0).sum()}"
    )
    return report


def load_or_fetch_symbol(symbol: str, start_date: str | None,
                         end_date: str | None,
                         adjust: str = ADJUST,
                         save_image: bool = True) -> pd.DataFrame:
    """加载或下载单只股票的 K 线数据.

    优先读取本地 parquet 文件；若不存在则调用数据接口获取。
    无论数据来源，都会尝试绘制并保存 K 线图。

    Args:
        symbol (str): 股票代码。
        start_date (str | None): 起始日期。
        end_date (str | None): 结束日期。
        adjust (str, optional): 复权方式。默认 ADJUST。
        save_image (bool, optional): 是否保存图片。默认 True。

    Returns:
        pd.DataFrame: 股票 K 线数据。
    """
    sym = ''.join(filter(str.isdigit, symbol)) or symbol
    out = _resolve_kline_path(sym, start_date if start_date else None,
                              end_date if end_date else None, adjust)

    if out.exists():
        df = read_parquet(out)
        if not df.empty:
            d = df.copy()
            if start_date:
                d = d[d["datetime"] >= pd.to_datetime(start_date)]
            if end_date:
                d = d[d["datetime"] <= pd.to_datetime(end_date)]
            if save_image and not d.empty:
                _save_kline_png(sym, d, start_date or "all", end_date or "all", adjust)
            return d

    # 无本地数据则获取
    k = _fetch_kline_fallback(sym, start_date, end_date, adjust)
    if k is not None and not k.empty:
        k.insert(0, "symbol", sym)
        write_parquet(k, out)
        if save_image:
            _save_kline_png(sym, k, start_date or "all", end_date or "all", adjust)

    return k

if __name__ == '__main__':
    spot = fetch_spot()
    fetch_klines_from_spot(spot)
    check_klines_integrity()


# ---------- 增量更新 ----------
def update_kline_incremental(symbol: str, adjust: str = ADJUST) -> pd.DataFrame:
    """增量更新单只股票的 K 线数据。

    读取本地最新日期，只获取更新的数据并追加。

    Args:
        symbol (str): 股票代码。
        adjust (str): 复权方式。

    Returns:
        pd.DataFrame: 更新后的完整 K 线数据，若无更新则返回本地数据。
    """
    sym = ''.join(filter(str.isdigit, symbol)) or symbol
    out = _resolve_kline_path(sym, START_DATE if START_DATE else None,
                              END_DATE if END_DATE else None, adjust)

    if not out.exists():
        # 无本地数据，全量获取
        logger.info(f"无本地数据，全量获取 {sym}")
        return load_or_fetch_symbol(sym, START_DATE, END_DATE, adjust)

    # 读取本地数据
    df_local = read_parquet(out)
    if df_local.empty:
        return load_or_fetch_symbol(sym, START_DATE, END_DATE, adjust)

    # 获取本地最新日期
    last_date = pd.to_datetime(df_local["datetime"]).max()
    # 从下一天开始增量获取
    next_date = last_date + pd.Timedelta(days=1)
    start_str = next_date.strftime("%Y%m%d")
    end_str = pd.Timestamp.today().strftime("%Y%m%d")

    logger.info(f"增量更新 {sym}: {start_str} ~ {end_str}")

    try:
        df_new = _fetch_kline_fallback(sym, start_str, end_str, adjust)
        if df_new is not None and not df_new.empty:
            if "symbol" not in df_new.columns:
                df_new.insert(0, "symbol", sym)
            # 合并并去重
            df_combined = pd.concat([df_local, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=["datetime"], keep="last")
            df_combined = df_combined.sort_values("datetime").reset_index(drop=True)
            write_parquet(df_combined, out)
            logger.info(f"增量更新完成 {sym}: 新增 {len(df_new)} 行，共 {len(df_combined)} 行")
            return df_combined
        else:
            logger.info(f"无新数据 {sym}")
            return df_local
    except Exception as e:
        logger.error(f"增量更新失败 {sym}: {e}")
        return df_local


# ---------- 数据质量检查 ----------
def check_data_quality(df: pd.DataFrame, symbol: str = "") -> dict:
    """检查 K 线数据质量。

    Args:
        df (pd.DataFrame): K 线数据。
        symbol (str): 股票代码（用于日志）。

    Returns:
        dict: 质量检查报告。
    """
    report = {
        "symbol": symbol,
        "rows": len(df),
        "date_range": None,
        "missing": {},
        "gaps": 0,
        "anomalies": [],
        "quality_score": 100,
    }

    if df.empty:
        report["quality_score"] = 0
        return report

    # 日期范围
    report["date_range"] = (str(df["datetime"].min()), str(df["datetime"].max()))

    # 缺失值检查
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                report["missing"][col] = int(missing)

    # 连续性检查（交易日间隙）
    if len(df) > 1:
        dates = pd.to_datetime(df["datetime"]).sort_values()
        diffs = dates.diff().dt.days
        # 正常交易间隔应 <= 3 天（跨周末）
        gaps = (diffs > 3).sum()
        report["gaps"] = int(gaps)

    # 异常值检查
    if "close" in df.columns and "volume" in df.columns:
        # 价格为 0 或负
        zero_price = (df["close"] <= 0).sum()
        if zero_price > 0:
            report["anomalies"].append(f"{zero_price} 条记录价格<=0")

        # 成交量为负
        neg_vol = (df["volume"] < 0).sum()
        if neg_vol > 0:
            report["anomalies"].append(f"{neg_vol} 条记录成交量<0")

    # 质量评分
    penalty = sum(report["missing"].values()) * 2 + report["gaps"] * 5 + len(report["anomalies"]) * 10
    report["quality_score"] = max(0, 100 - penalty)

    return report