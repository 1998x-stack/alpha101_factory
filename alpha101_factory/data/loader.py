# -*- coding: utf-8 -*-
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import pandas as pd
import akshare as ak
import time

from ..config import (PARQ_DIR_SPOT, PARQ_DIR_KLINES,
                      ADJUST, START_DATE, END_DATE,
                      LIMIT_STOCKS, REQUEST_PAUSE,
                      IMG_KLINES_DIR)
from ..utils.io import write_parquet, read_parquet
from ..viz.plots import plot_kline, save_fig
from .baostock_api import fetch_kline_bs

# ---------- normalize ----------
def normalize_k(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    m = {
        "日期":"datetime","开盘":"open","最高":"high","最低":"low",
        "收盘":"close","成交量":"volume","成交额":"amount",
        "涨跌幅":"pct_change","涨跌额":"change","振幅":"amplitude",
        "换手率":"turnover"
    }
    df = df.rename(columns={k:v for k,v in m.items() if k in df.columns})
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")
    num_cols = ["open","high","low","close","volume","amount","pct_change","change","amplitude","turnover"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ---------- akshare then baostock fallback ----------
def _fetch_kline_ak(symbol: str, start_date: str|None, end_date: str|None, adjust: str) -> pd.DataFrame:
    k = ak.stock_zh_a_hist(symbol=symbol, start_date=start_date or None,
                           end_date=end_date or None, period="daily", adjust=adjust)
    return normalize_k(k)

def _fetch_kline_fallback(symbol: str, start_date: str|None, end_date: str|None, adjust: str) -> pd.DataFrame:
    try:
        k = _fetch_kline_ak(symbol, start_date, end_date, adjust)
        if k is not None and not k.empty:
            return k
    except Exception as e:
        logger.warning(f"AkShare failed for {symbol}: {e}")
    logger.info(f"Trying Baostock fallback for {symbol} …")
    k = fetch_kline_bs(symbol, start_date, end_date, period="d", adjust=adjust)
    return k

# ---------- save image ----------
def _save_kline_png(sym: str, df: pd.DataFrame, start_date: str|None, end_date: str|None, adjust: str):
    start_tag = start_date or "start"
    end_tag = end_date or "end"
    out_png = IMG_KLINES_DIR / f"{sym}_{start_tag}_{end_tag}_{adjust}.png"
    fig = plot_kline(df, f"{sym} {adjust} {start_tag}-{end_tag}",
                     tickformat="%Y-%m-%d", tickangle=-45)
    save_fig(fig, out_png)
    return out_png

# ---------- public APIs ----------
def fetch_spot(save: bool = True) -> pd.DataFrame:
    logger.info("Fetching A-share spot …")
    spot = ak.stock_zh_a_spot()
    spot = spot.rename(columns={"代码":"code","名称":"name"})
    if save:
        write_parquet(spot, PARQ_DIR_SPOT / "a_spot.parquet")
    logger.info(f"Spot rows={len(spot)} saved: {save}")
    return spot

def fetch_klines_from_spot(spot: pd.DataFrame) -> int:
    codes = spot["code"].astype(str).str.zfill(6).unique().tolist()
    if LIMIT_STOCKS and LIMIT_STOCKS > 0:
        codes = codes[:LIMIT_STOCKS]
    logger.info(f"Total symbols to fetch: {len(codes)} | adjust={ADJUST} start={START_DATE} end={END_DATE or 'latest'}")

    n_new = 0
    for sym in tqdm(codes, desc="download daily"):
        out = PARQ_DIR_KLINES / f"{sym}.parquet"
        if out.exists():
            # already have historical file; still save kline image (full)
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
            logger.warning(f"{sym} failed: {e}")
        time.sleep(REQUEST_PAUSE)
    logger.info(f"Newly saved: {n_new}")
    return n_new

def check_klines_integrity() -> pd.DataFrame:
    spot = read_parquet(PARQ_DIR_SPOT / "a_spot.parquet")
    if spot.empty:
        logger.warning("Spot parquet not found.")
        return pd.DataFrame()

    codes = spot["code"].astype(str).str.zfill(6).unique()
    rows = []
    for c in codes:
        p = PARQ_DIR_KLINES / f"{c}.parquet"
        if not p.exists():
            rows.append([c, False, 0, None, None, str(p)])
            continue
        try:
            df = read_parquet(p)
            if df.empty:
                rows.append([c, True, 0, None, None, str(p)])
            else:
                dmin = pd.to_datetime(df["datetime"]).min()
                dmax = pd.to_datetime(df["datetime"]).max()
                rows.append([c, True, len(df), dmin, dmax, str(p)])
        except Exception as e:
            rows.append([c, False, -1, None, None, f"{p} ERROR: {e}"])
    report = pd.DataFrame(rows, columns=["symbol","exists","rows","date_min","date_max","path"])
    logger.info(f"Klines files: exists={report['exists'].sum()}/{len(report)}; zero_rows={(report['rows']==0).sum()}")
    return report

def load_or_fetch_symbol(symbol: str, start_date: str|None, end_date: str|None,
                         adjust: str = ADJUST, save_image: bool = True) -> pd.DataFrame:
    """Load local parquet if exists, slice [start,end]; otherwise fetch (akshare->baostock fallback).
       Always save kline image when loaded/downloaded.
    """
    out = PARQ_DIR_KLINES / f"{symbol}.parquet"
    if out.exists():
        df = read_parquet(out)
        if not df.empty:
            d = df.copy()
            if start_date:
                d = d[d["datetime"] >= pd.to_datetime(start_date)]
            if end_date:
                d = d[d["datetime"] <= pd.to_datetime(end_date)]
            if save_image and not d.empty:
                _save_kline_png(symbol, d, start_date or "all", end_date or "all", adjust)
            return d

    # fetch
    k = _fetch_kline_fallback(symbol, start_date, end_date, adjust)
    if k is not None and not k.empty:
        k.insert(0, "symbol", symbol)
        write_parquet(k, out)  # save full df we got
        if save_image:
            _save_kline_png(symbol, k, start_date or "all", end_date or "all", adjust)
    return k
