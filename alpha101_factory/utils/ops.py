# -*- coding: utf-8 -*-
"""
金融量化计算工具模块 (Financial Quantitative Utility Functions)

本模块实现了常见的时间序列滚动计算、截面计算和金融因子研究所需的基础函数。
支持可选的性能加速库：
    - bottleneck: 用于加速滚动窗口的求和、最值、均值、标准差等操作。
    - numba: 用于加速循环逻辑，如 ts_rank 与线性衰减加权平均。

即便上述库不可用，本模块也会回退至 pandas 实现，保证工业环境下的稳定性。
"""

import numpy as np
import pandas as pd

# -------- 尝试导入可选加速库 --------
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


# ============================================================================
# 内部辅助函数
# ============================================================================
def _as_series(x: np.ndarray, idx) -> pd.Series:
    """将 NumPy 数组转换为 Pandas Series 并保留索引。

    Args:
        x (np.ndarray): 输入的一维数组。
        idx (pd.Index): 与数组对齐的索引。

    Returns:
        pd.Series: 带索引的 Pandas Series。
    """
    return pd.Series(x, index=idx)


# ============================================================================
# 时间序列滚动窗口函数
# ============================================================================
def rolling_sum(s: pd.Series, n: int) -> pd.Series:
    """计算滚动窗口的和。

    Args:
        s (pd.Series): 输入序列。
        n (int): 窗口大小。

    Returns:
        pd.Series: 滚动窗口求和结果。

    Notes:
        - 优先使用 bottleneck.move_sum 加速。
        - 回退至 pandas.rolling 实现。
    """
    try:
        if _BN:
            out = bn.move_sum(s.to_numpy(dtype=float), window=n, min_count=n)
            return _as_series(out, s.index)
    except Exception:
        pass
    return s.rolling(n, min_periods=n).sum()


def rolling_min(s: pd.Series, n: int) -> pd.Series:
    """计算滚动窗口的最小值。"""
    try:
        if _BN:
            out = bn.move_min(s.to_numpy(dtype=float), window=n, min_count=n)
            return _as_series(out, s.index)
    except Exception:
        pass
    return s.rolling(n, min_periods=n).min()


def rolling_max(s: pd.Series, n: int) -> pd.Series:
    """计算滚动窗口的最大值。"""
    try:
        if _BN:
            out = bn.move_max(s.to_numpy(dtype=float), window=n, min_count=n)
            return _as_series(out, s.index)
    except Exception:
        pass
    return s.rolling(n, min_periods=n).max()


def rolling_std(s: pd.Series, n: int) -> pd.Series:
    """计算滚动窗口的标准差（无偏差调整）。"""
    try:
        if _BN:
            out = bn.move_std(s.to_numpy(dtype=float), window=n, min_count=n, ddof=0)
            return _as_series(out, s.index)
    except Exception:
        pass
    return s.rolling(n, min_periods=n).std(ddof=0)


def rolling_cov(s1: pd.Series, s2: pd.Series, n: int) -> pd.Series:
    """计算滚动窗口的协方差。"""
    return s1.rolling(n, min_periods=n).cov(s2)


def rolling_corr(s1: pd.Series, s2: pd.Series, n: int) -> pd.Series:
    """计算滚动窗口的相关系数。"""
    return s1.rolling(n, min_periods=n).corr(s2)


# ============================================================================
# 时间序列排名 (ts_rank)
# ============================================================================
if _NUMBA:
    @njit(cache=True)
    def _ts_rank_last(arr: np.ndarray, n: int) -> np.ndarray:
        """Numba 加速版 ts_rank，返回窗口最后一个元素的分位排名。"""
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
    """计算时间序列 ts_rank（窗口最后值的百分位排名）。

    Args:
        s (pd.Series): 输入序列。
        n (int): 滚动窗口大小。

    Returns:
        pd.Series: 每个位置对应的 ts_rank 值。
    """
    arr = s.to_numpy(dtype=float)
    try:
        if _NUMBA:
            return _as_series(_ts_rank_last(arr, n), s.index)
    except Exception:
        pass

    # pandas 回退实现
    def _last_rank(x: np.ndarray) -> float:
        r = pd.Series(x).rank(pct=True)
        return r.iloc[-1]

    return s.rolling(n, min_periods=n).apply(_last_rank, raw=False)


# ============================================================================
# 线性衰减加权平均 (decay_linear)
# ============================================================================
if _NUMBA:
    @njit(cache=True)
    def _decay_linear(arr: np.ndarray, n: int) -> np.ndarray:
        """Numba 加速版线性衰减加权平均。"""
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
    """计算线性衰减加权平均，越新的值权重越大。"""
    arr = s.to_numpy(dtype=float)
    try:
        if _NUMBA:
            return _as_series(_decay_linear(arr, n), s.index)
    except Exception:
        pass
    w = np.arange(1, n + 1, dtype=float)
    w /= w.sum()
    return s.rolling(n, min_periods=n).apply(lambda x: np.dot(x, w), raw=True)


# ============================================================================
# 基础变换函数
# ============================================================================
def delay(s: pd.Series, n: int = 1) -> pd.Series:
    """计算滞后 n 期。"""
    return s.shift(n)


def delta(s: pd.Series, n: int = 1) -> pd.Series:
    """计算差分：当前值减去 n 期前的值。"""
    return s - s.shift(n)


def returns(close: pd.Series) -> pd.Series:
    """计算收益率：相邻价格的百分比变化。"""
    return close.pct_change()


def vwap_from_amount(close: pd.Series, high: pd.Series, low: pd.Series,
                     volume: pd.Series, amount: pd.Series) -> pd.Series:
    """计算成交均价 (VWAP)。

    Notes:
        - 使用成交额 / 成交量。
        - 避免成交量为 0 的除零问题。
    """
    v = volume.replace(0, np.nan)
    return amount / v


def adv(volume: pd.Series, n: int) -> pd.Series:
    """计算平均成交量 (n 日均量)。"""
    try:
        if _BN:
            out = bn.move_mean(volume.to_numpy(dtype=float), window=n, min_count=n)
            return _as_series(out, volume.index)
    except Exception:
        pass
    return volume.rolling(n, min_periods=n).mean()


# ============================================================================
# 截面计算工具（同一时间点跨股票）
# ============================================================================
def cs_rank(s: pd.Series) -> pd.Series:
    """截面分位数排名：对每个时间点上的股票进行排序。"""
    return s.groupby(level=0).rank(pct=True)


def cs_zscore(s: pd.Series) -> pd.Series:
    """截面标准化 (Z-score)。"""
    g = s.groupby(level=0)
    return (s - g.transform("mean")) / g.transform("std")


# ============================================================================
# 按股票分组计算
# ============================================================================
def by_symbol(df: pd.DataFrame, col: str, func, *args, **kwargs) -> pd.Series:
    """对 DataFrame 按 symbol 分组后在指定列上应用函数。

    Args:
        df (pd.DataFrame): 输入数据框，需包含列 "symbol"。
        col (str): 需要处理的列名。
        func (Callable): 待应用的函数。
        *args: 函数的其他参数。
        **kwargs: 函数的关键字参数。

    Returns:
        pd.Series: 分组计算结果。
    """
    return df.groupby("symbol", group_keys=False)[col].apply(lambda x: func(x, *args, **kwargs))
