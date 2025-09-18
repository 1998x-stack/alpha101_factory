# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# -------- optional accelerators --------
try:
    import bottleneck as bn
    _BN = True
except Exception:
    _BN = False

try:
    from numba import njit
    _NUMBA = True
except Exception:
    _NUMBA = False

# ===== helpers to wrap array <-> Series =====
def _as_series(x: np.ndarray, idx) -> pd.Series:
    return pd.Series(x, index=idx)

# ===== rolling primitives (nan-safe) =====
def rolling_sum(s: pd.Series, n: int) -> pd.Series:
    if _BN:
        out = bn.move_sum(s.to_numpy(dtype=float), window=n, min_count=n)
        return _as_series(out, s.index)
    return s.rolling(n, min_periods=n).sum()

def rolling_min(s: pd.Series, n: int) -> pd.Series:
    if _BN:
        out = bn.move_min(s.to_numpy(dtype=float), window=n, min_count=n)
        return _as_series(out, s.index)
    return s.rolling(n, min_periods=n).min()

def rolling_max(s: pd.Series, n: int) -> pd.Series:
    if _BN:
        out = bn.move_max(s.to_numpy(dtype=float), window=n, min_count=n)
        return _as_series(out, s.index)
    return s.rolling(n, min_periods=n).max()

def rolling_std(s: pd.Series, n: int) -> pd.Series:
    if _BN:
        out = bn.move_std(s.to_numpy(dtype=float), window=n, min_count=n, ddof=0)
        return _as_series(out, s.index)
    return s.rolling(n, min_periods=n).std(ddof=0)

def rolling_cov(s1: pd.Series, s2: pd.Series, n: int) -> pd.Series:
    # covariance: use pandas (stable & good enough)
    return s1.rolling(n, min_periods=n).cov(s2)

def rolling_corr(s1: pd.Series, s2: pd.Series, n: int) -> pd.Series:
    return s1.rolling(n, min_periods=n).corr(s2)

# ===== ts_rank (percentile of the last item in a rolling window) =====
if _NUMBA:
    @njit(cache=True)
    def _ts_rank_last(arr: np.ndarray, n: int) -> np.ndarray:
        m = arr.size
        out = np.empty(m, dtype=np.float64)
        out[:] = np.nan
        for i in range(n - 1, m):
            cnt = 0.0
            valid = 0.0
            last = arr[i]
            for j in range(i - n + 1, i + 1):
                v = arr[j]
                if not np.isnan(v):
                    valid += 1.0
                    if v <= last:
                        cnt += 1.0
            if valid > 0:
                out[i] = cnt / valid
        return out

def ts_rank(s: pd.Series, n: int) -> pd.Series:
    arr = s.to_numpy(dtype=float)
    if _NUMBA:
        return _as_series(_ts_rank_last(arr, n), s.index)
    # pandas fallback
    def _last_rank(x):
        r = pd.Series(x).rank(pct=True)
        return r.iloc[-1]
    return s.rolling(n, min_periods=n).apply(_last_rank, raw=False)

# ===== decay_linear (weighted moving avg, newest weight largest) =====
if _NUMBA:
    @njit(cache=True)
    def _decay_linear(arr: np.ndarray, n: int) -> np.ndarray:
        w = np.arange(1, n + 1, dtype=np.float64)
        w = w / w.sum()
        m = arr.size
        out = np.empty(m, dtype=np.float64)
        out[:] = np.nan
        for i in range(n - 1, m):
            acc = 0.0
            nanhit = False
            k = 0
            for j in range(i - n + 1, i + 1):
                v = arr[j]
                if np.isnan(v):
                    nanhit = True
                    break
                acc += v * w[k]
                k += 1
            out[i] = np.nan if nanhit else acc
        return out

def decay_linear(s: pd.Series, n: int) -> pd.Series:
    arr = s.to_numpy(dtype=float)
    if _NUMBA:
        return _as_series(_decay_linear(arr, n), s.index)
    w = np.arange(1, n + 1, dtype=float); w /= w.sum()
    return s.rolling(n, min_periods=n).apply(lambda x: np.dot(x, w), raw=True)

# ===== basic transforms =====
def delay(s: pd.Series, n: int = 1) -> pd.Series:
    return s.shift(n)

def delta(s: pd.Series, n: int = 1) -> pd.Series:
    return s - s.shift(n)

def returns(close: pd.Series) -> pd.Series:
    return close.pct_change()

def vwap_from_amount(close: pd.Series, high: pd.Series, low: pd.Series,
                     volume: pd.Series, amount: pd.Series) -> pd.Series:
    v = volume.replace(0, np.nan)
    return amount / v

def adv(volume: pd.Series, n: int) -> pd.Series:
    if _BN:
        out = bn.move_mean(volume.to_numpy(dtype=float), window=n, min_count=n)
        return _as_series(out, volume.index)
    return volume.rolling(n, min_periods=n).mean()

# ===== cross-sectional utilities =====
def cs_rank(s: pd.Series) -> pd.Series:
    return s.groupby(level=0).rank(pct=True)

def cs_zscore(s: pd.Series) -> pd.Series:
    g = s.groupby(level=0)
    return (s - g.transform("mean")) / g.transform("std")

# ===== group-wise apply shortcut =====
def by_symbol(df: pd.DataFrame, col: str, func, *args, **kwargs) -> pd.Series:
    return df.groupby("symbol", group_keys=False)[col].apply(lambda x: func(x, *args, **kwargs))
