# -*- coding: utf-8 -*-
"""
可视化工具模块

本模块提供常见的量化研究可视化函数，包括：
1. K线图绘制；
2. 因子时间序列绘制；
3. 因子截面分布绘制；
4. 因子热力图绘制；
5. 通用图像保存函数。

内部辅助函数 `_ensure_datetime_series` 与 `_datetime_array_for_plot`
用于时间数据的标准化，确保绘图时兼容各种输入格式。
"""

from pathlib import Path
from typing import Sequence, Union
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from loguru import logger
from plotly.subplots import make_subplots
from typing import Optional


def _ensure_datetime_series(s: pd.Series) -> pd.Series:
    """确保序列转换为 pandas datetime 类型。

    支持的输入类型：
        - datetime64
        - 整数/浮点（时间戳，单位可为 ns/ms/s）
        - 字符串

    Args:
        s (pd.Series): 输入序列。

    Returns:
        pd.Series: 转换后的 datetime 序列（无时区）。
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, utc=False, errors="coerce")

    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        v = pd.to_numeric(s, errors="coerce")
        m = v.dropna().abs().median()
        # ===== 基于数量级的启发式判断时间戳单位 =====
        # ~1e18: ns, ~1e12: ms, ~1e9: s
        if m > 1e14:
            unit = "ns"
        elif m > 1e11:
            unit = "ms"
        else:
            unit = "s"
        return pd.to_datetime(v, unit=unit, errors="coerce")

    # 默认：字符串或 object 类型
    return pd.to_datetime(s, errors="coerce")


def _datetime_array_for_plot(s: Union[pd.Series, Sequence]) -> np.ndarray:
    """将输入序列转换为无时区的 Python datetime 数组。

    Args:
        s (Union[pd.Series, Sequence]): 输入序列，可为 pandas.Series 或 list。

    Returns:
        np.ndarray: datetime 对象数组，dtype=object。
    """
    if not isinstance(s, pd.Series):
        s = pd.Series(s)

    dt = pd.to_datetime(s, errors="coerce")
    dt = dt.dt.tz_localize(None)  # 去除时区，避免绘图库报错
    return np.array(dt.dt.to_pydatetime())


def plot_kline(df: pd.DataFrame, title: str = "Kline",
               tickformat: str = "%Y-%m-%d", tickangle: int = -45):
    """绘制 K 线图。

    Args:
        df (pd.DataFrame): 包含 ["datetime","open","high","low","close"] 的行情数据。
        title (str): 图表标题。
        tickformat (str): x 轴日期格式。
        tickangle (int): x 轴刻度角度。

    Returns:
        go.Figure: Plotly K线图对象。
    """
    d = df.dropna().copy().sort_values("datetime")
    x = _datetime_array_for_plot(d["datetime"])
    fig = go.Figure(data=[go.Candlestick(
        x=x, open=d["open"], high=d["high"], low=d["low"], close=d["close"]
    )])
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=520)
    fig.update_xaxes(type="date", tickformat=tickformat,
                     tickangle=tickangle, ticks="outside")
    return fig


def plot_factor_timeseries(fdf: pd.DataFrame, symbol: str, title: str,
                           tickformat: str = "%Y-%m-%d", tickangle: int = -45):
    """绘制单只股票的因子时间序列。

    Args:
        fdf (pd.DataFrame): 因子结果表，需包含 ["datetime","symbol","value"]。
        symbol (str): 股票代码。
        title (str): 图表标题。
        tickformat (str): x 轴日期格式。
        tickangle (int): x 轴刻度角度。

    Returns:
        go.Figure: Plotly 折线图对象。
    """
    d = fdf[fdf["symbol"] == symbol].copy().sort_values("datetime")
    x = _datetime_array_for_plot(d["datetime"])
    fig = px.line(x=x, y=d["value"], title=f"{title} | {symbol}")
    fig.update_layout(height=420)
    fig.update_xaxes(type="date", tickformat=tickformat,
                     tickangle=tickangle, ticks="outside")
    return fig


def plot_factor_cross_section(fdf: pd.DataFrame, dt=None, topn: int = 100,
                              title: str = "Factor cross-section",
                              tickformat: str = "%Y-%m-%d", tickangle: int = -45):
    """绘制某日因子的截面分布（柱状图，按绝对值排序取前 topn）。

    Args:
        fdf (pd.DataFrame): 因子结果表。
        dt (可选): 截面日期，若为 None，则取最新日期。
        topn (int): 选取的前 N 个股票。
        title (str): 图表标题。
        tickformat (str): x 轴日期格式。
        tickangle (int): x 轴刻度角度。

    Returns:
        go.Figure: Plotly 柱状图对象。
    """
    if dt is None:
        dt = pd.to_datetime(fdf["datetime"]).max()

    d = fdf.copy()
    d["datetime"] = _ensure_datetime_series(d["datetime"])
    d = d[d["datetime"] == pd.to_datetime(dt)].copy()

    d["abs"] = d["value"].abs()
    d = d.sort_values("abs", ascending=False).head(topn)

    fig = px.bar(d, x="symbol", y="value",
                 title=f"{title} | {pd.to_datetime(dt).date()}")
    fig.update_layout(height=420, xaxis={'categoryorder': 'total descending'})
    return fig


def plot_heatmap(fdf: pd.DataFrame, symbols: list[str], title: str = "Factor heatmap",
                 tickformat: str = "%Y-%m-%d", tickangle: int = -45):
    """绘制因子热力图（时间 × 股票）。

    Args:
        fdf (pd.DataFrame): 因子结果表。
        symbols (list[str]): 股票代码列表。
        title (str): 图表标题。
        tickformat (str): x 轴日期格式。
        tickangle (int): x 轴刻度角度。

    Returns:
        go.Figure: Plotly 热力图对象。
    """
    d = fdf[fdf["symbol"].isin(symbols)].copy()
    d["datetime"] = _ensure_datetime_series(d["datetime"])
    pvt = d.pivot_table(index="datetime", columns="symbol", values="value")

    fig = px.imshow(pvt.T, aspect="auto", origin="lower", title=title)
    fig.update_layout(height=500)
    fig.update_xaxes(type="date", tickformat=tickformat,
                     tickangle=tickangle, ticks="outside")
    return fig


def save_fig(fig, path: Path) -> Path:
    """保存图像为文件。

    Args:
        fig: Plotly 图表对象。
        path (Path): 输出文件路径。

    Returns:
        Path: 实际保存的文件路径。
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(path))  # 依赖 `kaleido`
        logger.info(f"图像已保存至: {path}")
    except Exception as e:
        logger.error(f"保存图像失败: {path}, 错误: {e}")
    return path



def plot_kline_with_factor(
    kline_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    symbol: str,
    title: str = "Kline + Factor",
    tickformat: str = "%Y-%m-%d",
    tickangle: int = -45,
    factor_label: Optional[str] = None,
):
    """绘制 K线 + 因子时序的组合图（上下两个子图）。

    Args:
        kline_df (pd.DataFrame): 行情数据，需包含 ["datetime","open","high","low","close"]。
        factor_df (pd.DataFrame): 因子数据，需包含 ["datetime","symbol","value"]。
        symbol (str): 股票代码。
        title (str): 图表标题。
        tickformat (str): x 轴日期格式。
        tickangle (int): x 轴刻度角度。
        factor_label (Optional[str]): 因子名称，用于 y 轴标题。

    Returns:
        go.Figure: Plotly 子图对象。
    """
    # 筛选并排序行情数据
    d_price = kline_df.dropna().copy()
    d_price = d_price.sort_values("datetime")
    x_price = _datetime_array_for_plot(d_price["datetime"])

    # 筛选并排序因子数据
    d_fac = factor_df[factor_df["symbol"] == symbol].copy().sort_values("datetime")
    x_fac = _datetime_array_for_plot(d_fac["datetime"])

    # 子图布局：2 行 1 列，X 轴共享
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4],
        subplot_titles=("Kline", factor_label or "Factor")
    )

    # K线图
    fig.add_trace(
        go.Candlestick(
            x=x_price,
            open=d_price["open"],
            high=d_price["high"],
            low=d_price["low"],
            close=d_price["close"],
            name="Kline"
        ),
        row=1, col=1
    )

    # 因子时间序列
    fig.add_trace(
        go.Scatter(
            x=x_fac,
            y=d_fac["value"],
            mode="lines",
            name=factor_label or "Factor",
            line=dict(color="royalblue", width=2)
        ),
        row=2, col=1
    )

    # 全局布局
    fig.update_layout(
        title=title,
        height=720,
        xaxis_rangeslider_visible=False
    )
    fig.update_xaxes(type="date", tickformat=tickformat,
                     tickangle=tickangle, ticks="outside")

    return fig