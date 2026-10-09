# -*- coding: utf-8 -*-
"""
金融量化计算工具模块 (Financial Quantitative Utility Functions)

本模块实现了常见的时间序列滚动计算、截面计算和金融因子研究所需的基础函数。
支持可选的性能加速库：
    - bottleneck: 用于加速滚动窗口的求和、最值、均值、标准差等操作。
    - numba: 用于加速循环逻辑，如 ts_rank 与线性衰减加权平均。

即便上述库不可用，本模块也会回退至 pandas 实现，保证工业环境下的稳定性。

Usage:
    >>> from alpha101_factory.utils.ops import rolling_sum, ts_rank, cs_rank
    >>> result = rolling_sum(series, 20)
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 日志配置 (Logger Configuration)
# ---------------------------------------------------------------------------
_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 尝试导入可选加速库 (Optional Acceleration Libraries)
# ---------------------------------------------------------------------------
try:
    import bottleneck as bn

    _BN_AVAILABLE = True
    _logger.info("bottleneck 加速库已加载")
except Exception:
    _BN_AVAILABLE = False

try:
    from numba import njit

    _NUMBA_AVAILABLE = True
    _logger.info("numba 加速库已加载")
except Exception:
    _NUMBA_AVAILABLE = False


# ============================================================================
# 内部辅助函数 (Internal Helper Functions)
# ============================================================================
def _as_series(values: np.ndarray, index: pd.Index) -> pd.Series:
    """将 NumPy 数组转换为 Pandas Series 并保留原始索引。

    Args:
        values: 输入的一维 NumPy 数组。
        index: 与数组对齐的 Pandas 索引。

    Returns:
        带索引的 Pandas Series。
    """
    return pd.Series(values, index=index)


def _validate_window(window: int, func_name: str) -> None:
    """校验滚动窗口参数的合法性。

    Args:
        window: 窗口大小。
        func_name: 调用此校验的函数名称，用于错误提示。

    Raises:
        ValueError: 当 window <= 0 时抛出。
    """
    if window <= 0:
        raise ValueError(f"{func_name}: 窗口大小 (window) 必须为正整数，当前值为 {window}")


def _handle_empty_series(series: pd.Series, func_name: str) -> bool:
    """处理空序列的边界情况。

    Args:
        series: 待检查的 Pandas Series。
        func_name: 调用此校验的函数名称，用于日志输出。

    Returns:
        True 表示序列为空（已记录警告），调用方应直接返回；False 表示序列非空可继续计算。
    """
    if series.empty:
        _logger.warning("%s: 输入序列为空，返回空序列", func_name)
        return True
    return False


# ============================================================================
# 时间序列滚动窗口函数 (Time-Series Rolling Window Functions)
# ============================================================================
def rolling_sum(series: pd.Series, window: int) -> pd.Series:
    """计算滚动窗口的和。

    优先使用 bottleneck.move_sum 加速；若不可用则回退至 pandas.rolling。

    Args:
        series: 输入序列。
        window: 滚动窗口大小，必须为正整数。

    Returns:
        滚动窗口求和结果。窗口内数据不足时返回 NaN。

    Raises:
        ValueError: 当 window <= 0 时抛出。
    """
    _validate_window(window, "rolling_sum")
    if _handle_empty_series(series, "rolling_sum"):
        return series

    try:
        if _BN_AVAILABLE:
            result = bn.move_sum(series.to_numpy(dtype=float), window=window, min_count=window)
            return _as_series(result, series.index)
    except Exception as exc:
        _logger.warning("rolling_sum: bottleneck 加速失败，回退至 pandas 实现 (%s)", exc)

    return series.rolling(window, min_periods=window).sum()


def rolling_min(series: pd.Series, window: int) -> pd.Series:
    """计算滚动窗口的最小值。

    优先使用 bottleneck.move_min 加速；若不可用则回退至 pandas.rolling。

    Args:
        series: 输入序列。
        window: 滚动窗口大小，必须为正整数。

    Returns:
        滚动窗口最小值。窗口内数据不足时返回 NaN。

    Raises:
        ValueError: 当 window <= 0 时抛出。
    """
    _validate_window(window, "rolling_min")
    if _handle_empty_series(series, "rolling_min"):
        return series

    try:
        if _BN_AVAILABLE:
            result = bn.move_min(series.to_numpy(dtype=float), window=window, min_count=window)
            return _as_series(result, series.index)
    except Exception as exc:
        _logger.warning("rolling_min: bottleneck 加速失败，回退至 pandas 实现 (%s)", exc)

    return series.rolling(window, min_periods=window).min()


def rolling_max(series: pd.Series, window: int) -> pd.Series:
    """计算滚动窗口的最大值。

    优先使用 bottleneck.move_max 加速；若不可用则回退至 pandas.rolling。

    Args:
        series: 输入序列。
        window: 滚动窗口大小，必须为正整数。

    Returns:
        滚动窗口最大值。窗口内数据不足时返回 NaN。

    Raises:
        ValueError: 当 window <= 0 时抛出。
    """
    _validate_window(window, "rolling_max")
    if _handle_empty_series(series, "rolling_max"):
        return series

    try:
        if _BN_AVAILABLE:
            result = bn.move_max(series.to_numpy(dtype=float), window=window, min_count=window)
            return _as_series(result, series.index)
    except Exception as exc:
        _logger.warning("rolling_max: bottleneck 加速失败，回退至 pandas 实现 (%s)", exc)

    return series.rolling(window, min_periods=window).max()


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    """计算滚动窗口的标准差（无偏估计，ddof=0）。

    优先使用 bottleneck.move_std 加速；若不可用则回退至 pandas.rolling。

    Args:
        series: 输入序列。
        window: 滚动窗口大小，必须为正整数。

    Returns:
        滚动窗口标准差。窗口内数据不足时返回 NaN。

    Raises:
        ValueError: 当 window <= 0 时抛出。
    """
    _validate_window(window, "rolling_std")
    if _handle_empty_series(series, "rolling_std"):
        return series

    try:
        if _BN_AVAILABLE:
            result = bn.move_std(
                series.to_numpy(dtype=float), window=window, min_count=window, ddof=0
            )
            return _as_series(result, series.index)
    except Exception as exc:
        _logger.warning("rolling_std: bottleneck 加速失败，回退至 pandas 实现 (%s)", exc)

    return series.rolling(window, min_periods=window).std(ddof=0)


def rolling_cov(series_a: pd.Series, series_b: pd.Series, window: int) -> pd.Series:
    """计算滚动窗口的协方差。

    Args:
        series_a: 第一个输入序列。
        series_b: 第二个输入序列。
        window: 滚动窗口大小，必须为正整数。

    Returns:
        滚动窗口协方差。窗口内数据不足时返回 NaN。

    Raises:
        ValueError: 当 window <= 0 时抛出。
    """
    _validate_window(window, "rolling_cov")
    if _handle_empty_series(series_a, "rolling_cov"):
        return series_a
    if _handle_empty_series(series_b, "rolling_cov"):
        return series_b

    if len(series_a) != len(series_b):
        raise ValueError(
            f"rolling_cov: 两个序列长度必须一致 (series_a={len(series_a)}, series_b={len(series_b)})"
        )

    return series_a.rolling(window, min_periods=window).cov(series_b)


def rolling_corr(series_a: pd.Series, series_b: pd.Series, window: int) -> pd.Series:
    """计算滚动窗口的相关系数 (Pearson)。

    Args:
        series_a: 第一个输入序列。
        series_b: 第二个输入序列。
        window: 滚动窗口大小，必须为正整数。

    Returns:
        滚动窗口相关系数。窗口内数据不足时返回 NaN。

    Raises:
        ValueError: 当 window <= 0 或两个序列长度不一致时抛出。
    """
    _validate_window(window, "rolling_corr")
    if _handle_empty_series(series_a, "rolling_corr"):
        return series_a
    if _handle_empty_series(series_b, "rolling_corr"):
        return series_b

    if len(series_a) != len(series_b):
        raise ValueError(
            f"rolling_corr: 两个序列长度必须一致 (series_a={len(series_a)}, series_b={len(series_b)})"
        )

    return series_a.rolling(window, min_periods=window).corr(series_b)


# ============================================================================
# 时间序列排名 (Time-Series Rank)
# ============================================================================
if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _ts_rank_last(values: np.ndarray, window: int) -> np.ndarray:
        """Numba 加速版 ts_rank，返回窗口最后一个元素的分位排名。

        Args:
            values: 输入的一维 NumPy 数组。
            window: 滚动窗口大小。

        Returns:
            与输入等长的数组，每个位置为窗口内最后一个值的百分位排名。
            窗口内有效数据不足时对应位置为 NaN。
        """
        length = values.size
        result = np.empty(length, dtype=np.float64)
        result[:] = np.nan

        for i in range(window - 1, length):
            count_le = 0.0
            valid_count = 0.0
            last_value = values[i]

            for j in range(i - window + 1, i + 1):
                val = values[j]
                if not np.isnan(val):
                    valid_count += 1.0
                    if val <= last_value:
                        count_le += 1.0

            if valid_count > 0:
                result[i] = count_le / valid_count

        return result


def ts_rank(series: pd.Series, window: int) -> pd.Series:
    """计算时间序列 ts_rank（窗口最后值的百分位排名）。

    排名定义为窗口中小于等于最后一个值的元素占比，取值范围 [0, 1]。
    优先使用 numba 加速；若不可用则回退至 pandas.rolling.apply。

    Args:
        series: 输入序列。
        window: 滚动窗口大小，必须为正整数。

    Returns:
        每个位置对应的 ts_rank 值。窗口内数据不足时返回 NaN。

    Raises:
        ValueError: 当 window <= 0 时抛出。
    """
    _validate_window(window, "ts_rank")
    if _handle_empty_series(series, "ts_rank"):
        return series

    values = series.to_numpy(dtype=float)

    try:
        if _NUMBA_AVAILABLE:
            return _as_series(_ts_rank_last(values, window), series.index)
    except Exception as exc:
        _logger.warning("ts_rank: numba 加速失败，回退至 pandas 实现 (%s)", exc)

    # pandas 回退实现
    def _last_percentile_rank(window_values: np.ndarray) -> float:
        """计算窗口内最后一个值的百分位排名。"""
        ranked = pd.Series(window_values).rank(pct=True)
        return ranked.iloc[-1]

    return series.rolling(window, min_periods=window).apply(_last_percentile_rank, raw=False)


# ============================================================================
# 线性衰减加权平均 (Decay Linear Weighted Average)
# ============================================================================
if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _decay_linear_numba(values: np.ndarray, window: int) -> np.ndarray:
        """Numba 加速版线性衰减加权平均。

        权重从 1 到 window 线性递增，即越新的值权重越大。
        窗口内存在 NaN 时对应输出为 NaN。

        Args:
            values: 输入的一维 NumPy 数组。
            window: 滚动窗口大小。

        Returns:
            与输入等长的数组，每个位置为线性衰减加权平均值。
        """
        weights = np.arange(1, window + 1, dtype=np.float64)
        weights = weights / weights.sum()

        length = values.size
        result = np.empty(length, dtype=np.float64)
        result[:] = np.nan

        for i in range(window - 1, length):
            weighted_sum = 0.0
            has_nan = False
            weight_idx = 0

            for j in range(i - window + 1, i + 1):
                val = values[j]
                if np.isnan(val):
                    has_nan = True
                    break
                weighted_sum += val * weights[weight_idx]
                weight_idx += 1

            result[i] = np.nan if has_nan else weighted_sum

        return result


def decay_linear(series: pd.Series, window: int) -> pd.Series:
    """计算线性衰减加权平均，越新的值权重越大。

    权重序列为 [1, 2, ..., window]，归一化后与窗口内数据做点积。
    窗口内存在 NaN 时对应输出为 NaN。
    优先使用 numba 加速；若不可用则回退至 pandas.rolling.apply。

    Args:
        series: 输入序列。
        window: 滚动窗口大小，必须为正整数。

    Returns:
        线性衰减加权平均结果。窗口内数据不足或含 NaN 时返回 NaN。

    Raises:
        ValueError: 当 window <= 0 时抛出。
    """
    _validate_window(window, "decay_linear")
    if _handle_empty_series(series, "decay_linear"):
        return series

    values = series.to_numpy(dtype=float)

    try:
        if _NUMBA_AVAILABLE:
            return _as_series(_decay_linear_numba(values, window), series.index)
    except Exception as exc:
        _logger.warning("decay_linear: numba 加速失败，回退至 pandas 实现 (%s)", exc)

    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()
    return series.rolling(window, min_periods=window).apply(
        lambda window_values: np.dot(window_values, weights), raw=True
    )


# ============================================================================
# 基础变换函数 (Basic Transformation Functions)
# ============================================================================
def delay(series: pd.Series, lag: int = 1) -> pd.Series:
    """计算滞后 lag 期的值。

    Args:
        series: 输入序列。
        lag: 滞后期数，默认为 1。正值表示向过去偏移。

    Returns:
        滞后后的序列，前 lag 个位置为 NaN。
    """
    return series.shift(lag)


def delta(series: pd.Series, lag: int = 1) -> pd.Series:
    """计算差分：当前值减去 lag 期前的值。

    Args:
        series: 输入序列。
        lag: 差分跨度，默认为 1。

    Returns:
        差分结果，前 lag 个位置为 NaN。
    """
    return series - series.shift(lag)


def returns(close_prices: pd.Series) -> pd.Series:
    """计算收益率：相邻价格的百分比变化。

    Args:
        close_prices: 收盘价序列。

    Returns:
        收益率序列，第一个位置为 NaN。
    """
    return close_prices.pct_change()


def vwap_from_amount(
    close_prices: pd.Series,
    high_prices: pd.Series,
    low_prices: pd.Series,
    volume: pd.Series,
    amount: pd.Series,
) -> pd.Series:
    """计算成交均价 (VWAP = Volume Weighted Average Price)。

    计算公式: VWAP = amount / volume
    当成交量为 0 时，对应位置返回 NaN 以避免除零错误。

    Args:
        close_prices: 收盘价序列（保留参数兼容性，当前未使用）。
        high_prices: 最高价序列（保留参数兼容性，当前未使用）。
        low_prices: 最低价序列（保留参数兼容性，当前未使用）。
        volume: 成交量序列。
        amount: 成交额序列。

    Returns:
        VWAP 序列。成交量为 0 或 NaN 时对应位置为 NaN。
    """
    # 将零成交量替换为 NaN，避免除零错误
    safe_volume = volume.replace(0, np.nan)
    return amount / safe_volume


def adv(volume: pd.Series, window: int) -> pd.Series:
    """计算平均成交量 (Average Daily Volume)。

    优先使用 bottleneck.move_mean 加速；若不可用则回退至 pandas.rolling。

    Args:
        volume: 成交量序列。
        window: 计算均值的窗口大小，必须为正整数。

    Returns:
        滚动窗口内的平均成交量。窗口内数据不足时返回 NaN。

    Raises:
        ValueError: 当 window <= 0 时抛出。
    """
    _validate_window(window, "adv")
    if _handle_empty_series(volume, "adv"):
        return volume

    try:
        if _BN_AVAILABLE:
            result = bn.move_mean(volume.to_numpy(dtype=float), window=window, min_count=window)
            return _as_series(result, volume.index)
    except Exception as exc:
        _logger.warning("adv: bottleneck 加速失败，回退至 pandas 实现 (%s)", exc)

    return volume.rolling(window, min_periods=window).mean()


# ============================================================================
# 截面计算工具（同一时间点跨股票）(Cross-Sectional Functions)
# ============================================================================
def cs_rank(series: pd.Series) -> pd.Series:
    """截面分位数排名：对每个时间点（datetime level=0）上的所有股票进行排序。

    返回每个股票在其所属时间截面中的百分位排名，取值范围 [0, 1]。

    Args:
        series: 输入序列，索引需包含 MultiIndex 且 level=0 为 datetime。

    Returns:
        截面百分位排名序列。
    """
    return series.groupby(level=0).rank(pct=True)


def cs_zscore(series: pd.Series) -> pd.Series:
    """截面标准化 (Z-score)：对每个时间点上的股票进行去均值除以标准差。

    计算公式: z = (x - mean) / std
    当截面标准差为 0 时，对应位置返回 NaN。

    Args:
        series: 输入序列，索引需包含 MultiIndex 且 level=0 为 datetime。

    Returns:
        截面标准化后的 Z-score 序列。
    """
    grouped = series.groupby(level=0)
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    return (series - mean) / std


# ============================================================================
# 按股票分组计算 (Per-Symbol Grouped Computation)
# ============================================================================
def by_symbol(
    dataframe: pd.DataFrame,
    column: str,
    func: Callable[..., pd.Series],
    *args,
    **kwargs,
) -> pd.Series:
    """对 DataFrame 按 symbol 分组后在指定列上应用函数。

    该函数将 DataFrame 按股票代码分组，对每个股票的指定列独立应用目标函数，
    最后将结果拼接为单一 Series。

    Args:
        dataframe: 输入数据框，需包含列 "symbol"。
        column: 需要处理的列名。
        func: 待应用的函数，签名为 func(series, *args, **kwargs) -> pd.Series。
        *args: 传递给 func 的位置参数。
        **kwargs: 传递给 func 的关键字参数。

    Returns:
        分组计算结果拼接后的 Series。
    """
    return dataframe.groupby("symbol", group_keys=False)[column].apply(
        lambda series: func(series, *args, **kwargs)
    )
