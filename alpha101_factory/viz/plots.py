# -*- coding: utf-8 -*-
"""
工业级可视化绘图模块

本模块提供 A 股量化研究场景下的标准可视化函数，涵盖：
1. K 线图绘制（支持单图及与因子时序组合图）；
2. 因子时间序列折线图；
3. 因子截面分布柱状图；
4. 因子热力图（时间 × 股票）；
5. 通用图像持久化函数（Plotly Figure → PNG）。

所有公开函数均遵循以下规范：
- Google-style 文档字符串（PEP 257）；
- 完整类型注解（typing 模块）；
- 输入数据边界校验（空 DataFrame、缺失列等）；
- 异常安全（绘图/保存失败时返回 None 并记录日志）。

内部辅助函数：
- ``_ensure_datetime_series``：将任意格式序列标准化为 datetime64；
- ``_datetime_array_for_plot``：转换为 Python datetime 对象数组供 Plotly 使用；
- ``_validate_dataframe``：校验 DataFrame 非空且包含必需列。

Usage:
    >>> from alpha101_factory.viz.plots import plot_kline, save_fig
    >>> fig = plot_kline(df, title="600000 Kline")
    >>> save_fig(fig, Path("output.png"))
"""

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------

# 绘图默认样式常量
_DEFAULT_FIGURE_HEIGHT_KLINE: int = 520
_DEFAULT_FIGURE_HEIGHT_TIMESERIES: int = 420
_DEFAULT_FIGURE_HEIGHT_HEATMAP: int = 500
_DEFAULT_FIGURE_HEIGHT_COMBINED: int = 720
_DEFAULT_TICK_FORMAT: str = "%Y-%m-%d"
_DEFAULT_TICK_ANGLE: int = -45


def _validate_dataframe(
    dataframe: pd.DataFrame, required_columns: Sequence[str], label: str = "DataFrame"
) -> bool:
    """校验 DataFrame 非空且包含全部必需列。

    Args:
        dataframe: 待校验的 DataFrame 对象。
        required_columns: 必须存在的列名序列。
        label: 日志中使用的数据标识符，便于定位问题来源。

    Returns:
        bool: 校验通过返回 True；数据为空或缺失列时返回 False 并记录警告日志。
    """
    if dataframe is None or dataframe.empty:
        logger.warning(f"{label} 为空，跳过绘图")
        return False

    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    if missing_columns:
        logger.warning(
            f"{label} 缺失必需列: {missing_columns}，"
            f"当前列: {list(dataframe.columns)}"
        )
        return False

    return True


def _ensure_datetime_series(series: pd.Series) -> pd.Series:
    """将任意格式序列标准化为无时区 datetime64 类型。

    支持的输入类型及推断策略：
        - datetime64：直接转换，去除时区信息；
        - 整数/浮点数：基于中位数数量级推断时间戳单位
          （>1e14 → ns，>1e11 → ms，其余 → s）；
        - 字符串/object：按字符串解析。

    Args:
        series: 输入序列，可为 datetime、数值时间戳或日期字符串。

    Returns:
        pd.Series: 转换后的 datetime 序列（无时区，解析失败的值转为 NaT）。
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, utc=False, errors="coerce")

    if pd.api.types.is_numeric_dtype(series):
        numeric_values = pd.to_numeric(series, errors="coerce")
        median_value = numeric_values.dropna().abs().median()
        # 基于数量级启发式推断时间戳单位
        if median_value > 1e14:
            timestamp_unit = "ns"
        elif median_value > 1e11:
            timestamp_unit = "ms"
        else:
            timestamp_unit = "s"
        return pd.to_datetime(numeric_values, unit=timestamp_unit, errors="coerce")

    # 默认按字符串解析
    return pd.to_datetime(series, errors="coerce")


def _datetime_array_for_plot(series: Union[pd.Series, Sequence]) -> np.ndarray:
    """将输入序列转换为 Plotly 兼容的 Python datetime 对象数组。

    该函数确保输出为无时区的 datetime 对象数组，避免 Plotly
    在处理带时区或非标准日期类型时出现渲染异常。

    Args:
        series: 输入序列，可为 pandas.Series 或任意可迭代序列。

    Returns:
        np.ndarray: dtype=object 的 datetime 对象数组。
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    datetime_series = pd.to_datetime(series, errors="coerce").dt.tz_localize(None)
    return np.array(datetime_series.dt.to_pydatetime())


def _apply_xaxis_style(
    figure: go.Figure,
    tick_format: str = _DEFAULT_TICK_FORMAT,
    tick_angle: int = _DEFAULT_TICK_ANGLE,
) -> go.Figure:
    """为 Plotly Figure 应用统一的 x 轴日期样式。

    Args:
        figure: 待样式化的 Plotly Figure 对象。
        tick_format: x 轴日期显示格式，默认 "%Y-%m-%d"。
        tick_angle: x 轴刻度标签旋转角度，默认 -45 度。

    Returns:
        go.Figure: 样式化后的 Figure 对象（原地修改并返回）。
    """
    figure.update_xaxes(
        type="date",
        tickformat=tick_format,
        tickangle=tick_angle,
        ticks="outside",
    )
    return figure


# ---------------------------------------------------------------------------
# 公开绘图函数
# ---------------------------------------------------------------------------


def plot_kline(
    dataframe: pd.DataFrame,
    title: str = "Kline",
    tickformat: str = _DEFAULT_TICK_FORMAT,
    tickangle: int = _DEFAULT_TICK_ANGLE,
) -> Optional[go.Figure]:
    """绘制标准 K 线图（Candlestick）。

    从包含 OHLC 数据的 DataFrame 生成 Plotly 交互式 K 线图，
    自动按时间排序并过滤无效行。

    Args:
        dataframe: 行情数据，必须包含列
            ``["datetime", "open", "high", "low", "close"]``。
        title: 图表标题，默认 "Kline"。
        tickformat: x 轴日期格式字符串，默认 "%Y-%m-%d"。
        tickangle: x 轴刻度标签旋转角度（度），默认 -45。

    Returns:
        Optional[go.Figure]: 成功时返回 Plotly Figure 对象；
            数据校验失败时返回 None。

    Examples:
        >>> df = pd.DataFrame({
        ...     "datetime": ["2024-01-02", "2024-01-03"],
        ...     "open": [10.5, 10.6], "high": [10.8, 10.9],
        ...     "low": [10.3, 10.4], "close": [10.6, 10.7],
        ... })
        >>> fig = plot_kline(df, title="600000")
    """
    required_cols = ["datetime", "open", "high", "low", "close"]
    if not _validate_dataframe(dataframe, required_cols, label="K线数据"):
        return None

    # 过滤无效行并按时间排序
    cleaned_data = dataframe.dropna(subset=required_cols).copy()
    cleaned_data = cleaned_data.sort_values("datetime")

    if cleaned_data.empty:
        logger.warning("K线数据经清洗后为空，跳过绘图")
        return None

    datetime_array = _datetime_array_for_plot(cleaned_data["datetime"])

    figure = go.Figure(
        data=[
            go.Candlestick(
                x=datetime_array,
                open=cleaned_data["open"],
                high=cleaned_data["high"],
                low=cleaned_data["low"],
                close=cleaned_data["close"],
            )
        ]
    )

    figure.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=_DEFAULT_FIGURE_HEIGHT_KLINE,
    )
    _apply_xaxis_style(figure, tickformat, tickangle)
    return figure


def plot_factor_timeseries(
    factor_dataframe: pd.DataFrame,
    symbol: str,
    title: str,
    tickformat: str = _DEFAULT_TICK_FORMAT,
    tickangle: int = _DEFAULT_TICK_ANGLE,
) -> Optional[go.Figure]:
    """绘制单只股票的因子值时间序列折线图。

    从因子结果集中筛选指定股票，按时间升序绘制因子值变化趋势。

    Args:
        factor_dataframe: 因子结果表，必须包含列
            ``["datetime", "symbol", "value"]``。
        symbol: 目标股票代码（如 "600000"）。
        title: 图表标题前缀，最终标题格式为 "{title} | {symbol}"。
        tickformat: x 轴日期格式字符串，默认 "%Y-%m-%d"。
        tickangle: x 轴刻度标签旋转角度（度），默认 -45。

    Returns:
        Optional[go.Figure]: 成功时返回 Plotly Figure 对象；
            数据校验失败或该股票无数据时返回 None。

    Examples:
        >>> fig = plot_factor_timeseries(fdf, symbol="600000", title="Alpha101")
    """
    required_cols = ["datetime", "symbol", "value"]
    if not _validate_dataframe(
        factor_dataframe, required_cols, label="因子时序数据"
    ):
        return None

    # 筛选目标股票并排序
    symbol_data = factor_dataframe[factor_dataframe["symbol"] == symbol].copy()
    symbol_data = symbol_data.sort_values("datetime")

    if symbol_data.empty:
        logger.warning(f"股票 {symbol} 无因子数据，跳过绘图")
        return None

    datetime_array = _datetime_array_for_plot(symbol_data["datetime"])
    chart_title = f"{title} | {symbol}"

    figure = px.line(
        x=datetime_array,
        y=symbol_data["value"],
        title=chart_title,
    )
    figure.update_layout(height=_DEFAULT_FIGURE_HEIGHT_TIMESERIES)
    _apply_xaxis_style(figure, tickformat, tickangle)
    return figure


def plot_factor_cross_section(
    factor_dataframe: pd.DataFrame,
    dt: Optional[Union[str, pd.Timestamp]] = None,
    topn: int = 100,
    title: str = "Factor cross-section",
    tickformat: str = _DEFAULT_TICK_FORMAT,
    tickangle: int = _DEFAULT_TICK_ANGLE,
) -> Optional[go.Figure]:
    """绘制某日因子值的截面分布柱状图。

    按因子绝对值降序排列，取前 ``topn`` 只股票绘制柱状图，
    直观展示因子在横截面上的分布特征。

    Args:
        factor_dataframe: 因子结果表，必须包含列
            ``["datetime", "symbol", "value"]``。
        dt: 目标截面日期。若为 None，则自动取数据中最新日期。
        topn: 选取的前 N 只股票（按因子绝对值排序），默认 100。
        title: 图表标题前缀。
        tickformat: x 轴日期格式字符串，默认 "%Y-%m-%d"。
        tickangle: x 轴刻度标签旋转角度（度），默认 -45。

    Returns:
        Optional[go.Figure]: 成功时返回 Plotly Figure 对象；
            数据校验失败或目标日期无数据时返回 None。

    Examples:
        >>> fig = plot_factor_cross_section(fdf, topn=50, title="Alpha101")
    """
    required_cols = ["datetime", "symbol", "value"]
    if not _validate_dataframe(
        factor_dataframe, required_cols, label="因子截面数据"
    ):
        return None

    working_data = factor_dataframe.copy()
    working_data["datetime"] = _ensure_datetime_series(working_data["datetime"])

    # 确定目标日期
    if dt is None:
        target_date = working_data["datetime"].max()
        if pd.isna(target_date):
            logger.warning("因子截面数据中无有效日期，跳过绘图")
            return None
    else:
        target_date = pd.to_datetime(dt)

    # 筛选目标日期
    date_filtered = working_data[working_data["datetime"] == target_date].copy()

    if date_filtered.empty:
        logger.warning(f"日期 {target_date.date()} 无因子截面数据，跳过绘图")
        return None

    # 按绝对值排序取前 N
    date_filtered["abs_value"] = date_filtered["value"].abs()
    top_stocks = date_filtered.sort_values("abs_value", ascending=False).head(topn)

    figure = px.bar(
        top_stocks,
        x="symbol",
        y="value",
        title=f"{title} | {target_date.date()}",
    )
    figure.update_layout(
        height=_DEFAULT_FIGURE_HEIGHT_TIMESERIES,
        xaxis={"categoryorder": "total descending"},
    )
    return figure


def plot_heatmap(
    factor_dataframe: pd.DataFrame,
    symbols: list[str],
    title: str = "Factor heatmap",
    tickformat: str = _DEFAULT_TICK_FORMAT,
    tickangle: int = _DEFAULT_TICK_ANGLE,
) -> Optional[go.Figure]:
    """绘制因子热力图（时间 × 股票矩阵）。

    将指定股票列表的因子值透视成二维矩阵，以颜色深浅表示因子值大小，
    适用于观察因子在时间和截面两个维度上的变化模式。

    Args:
        factor_dataframe: 因子结果表，必须包含列
            ``["datetime", "symbol", "value"]``。
        symbols: 目标股票代码列表（如 ["600000", "000001"]）。
        title: 图表标题，默认 "Factor heatmap"。
        tickformat: x 轴日期格式字符串，默认 "%Y-%m-%d"。
        tickangle: x 轴刻度标签旋转角度（度），默认 -45。

    Returns:
        Optional[go.Figure]: 成功时返回 Plotly Figure 对象；
            数据校验失败或透视后无有效数据时返回 None。

    Examples:
        >>> fig = plot_heatmap(fdf, symbols=["600000", "000001"], title="Alpha101")
    """
    required_cols = ["datetime", "symbol", "value"]
    if not _validate_dataframe(
        factor_dataframe, required_cols, label="因子热力图数据"
    ):
        return None

    if not symbols:
        logger.warning("股票代码列表为空，跳过热力图绘制")
        return None

    # 筛选目标股票并标准化日期
    filtered_data = factor_dataframe[
        factor_dataframe["symbol"].isin(symbols)
    ].copy()
    filtered_data["datetime"] = _ensure_datetime_series(filtered_data["datetime"])

    # 透视成时间 × 股票矩阵
    pivot_matrix = filtered_data.pivot_table(
        index="datetime", columns="symbol", values="value"
    )

    if pivot_matrix.empty:
        logger.warning("因子热力图透视后无有效数据，跳过绘图")
        return None

    # px.imshow 需要转置（股票 × 时间）以匹配热力图常规布局
    figure = px.imshow(
        pivot_matrix.T,
        aspect="auto",
        origin="lower",
        title=title,
    )
    figure.update_layout(height=_DEFAULT_FIGURE_HEIGHT_HEATMAP)
    _apply_xaxis_style(figure, tickformat, tickangle)
    return figure


def save_fig(
    figure: go.Figure,
    output_path: Path,
) -> Optional[Path]:
    """将 Plotly Figure 保存为 PNG 图像文件。

    使用 Kaleido 引擎将交互式图表导出为静态 PNG 图片。
    若文件已存在则跳过保存（避免重复渲染开销）。

    Args:
        figure: Plotly Figure 对象，必须为有效的 go.Figure 实例。
        output_path: 目标文件路径（支持任意嵌套目录，自动创建）。

    Returns:
        Optional[Path]: 成功时返回已保存文件的 Path 对象；
            文件已存在时返回原路径；保存失败时返回 None。

    Note:
        依赖 ``kaleido`` 包。若未安装，保存将失败并记录错误日志。

    Examples:
        >>> fig = plot_kline(df)
        >>> save_fig(fig, Path("output/kline.png"))
    """
    if figure is None:
        logger.error("save_fig: 传入的 Figure 对象为 None")
        return None

    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    # 文件已存在则跳过，避免重复渲染
    if output_path.exists():
        logger.info(f"文件已存在，跳过保存: {output_path}")
        return output_path

    try:
        # 确保父目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.write_image(str(output_path))
        logger.info(f"图像已保存至: {output_path}")
        return output_path

    except Exception as save_error:
        logger.error(f"保存图像失败: {output_path}, 错误: {save_error}")
        return None


def plot_kline_with_factor(
    kline_dataframe: pd.DataFrame,
    factor_dataframe: pd.DataFrame,
    symbol: str,
    title: str = "Kline + Factor",
    tickformat: str = _DEFAULT_TICK_FORMAT,
    tickangle: int = _DEFAULT_TICK_ANGLE,
    factor_label: Optional[str] = None,
) -> Optional[go.Figure]:
    """绘制 K 线与因子时序的组合图（上下双面板布局）。

    上方子图展示股票 K 线走势，下方子图展示对应因子值的时间序列，
    两图共享 x 轴以便直观对比价格与因子的关联关系。

    Args:
        kline_dataframe: 行情数据，必须包含列
            ``["datetime", "open", "high", "low", "close"]``。
        factor_dataframe: 因子数据，必须包含列
            ``["datetime", "symbol", "value"]``。
        symbol: 目标股票代码（如 "600000"）。
        title: 图表总标题。
        tickformat: x 轴日期格式字符串，默认 "%Y-%m-%d"。
        tickangle: x 轴刻度标签旋转角度（度），默认 -45。
        factor_label: 因子名称标签，用于下方子图标题和图例；
            若为 None 则使用默认标签 "Factor"。

    Returns:
        Optional[go.Figure]: 成功时返回 Plotly Figure 对象；
            任一数据源校验失败时返回 None。

    Examples:
        >>> fig = plot_kline_with_factor(
        ...     kline_df, factor_df, symbol="600000",
        ...     factor_label="Alpha101"
        ... )
    """
    kline_required = ["datetime", "open", "high", "low", "close"]
    factor_required = ["datetime", "symbol", "value"]

    # 校验行情数据
    if not _validate_dataframe(kline_dataframe, kline_required, label="K线数据"):
        return None

    # 校验因子数据
    if not _validate_dataframe(
        factor_dataframe, factor_required, label="因子数据"
    ):
        return None

    # 清洗并排序行情数据
    price_data = kline_dataframe.dropna(subset=kline_required).copy()
    price_data = price_data.sort_values("datetime")

    if price_data.empty:
        logger.warning("K线数据经清洗后为空，跳过组合图绘制")
        return None

    # 筛选目标股票的因子数据并排序
    factor_data = factor_dataframe[
        factor_dataframe["symbol"] == symbol
    ].copy().sort_values("datetime")

    if factor_data.empty:
        logger.warning(f"股票 {symbol} 无因子数据，跳过组合图绘制")
        return None

    price_datetime = _datetime_array_for_plot(price_data["datetime"])
    factor_datetime = _datetime_array_for_plot(factor_data["datetime"])

    # 构建双面板布局：上行 K 线（60% 高度），下行因子（40% 高度）
    subplot_labels = ("Kline", factor_label or "Factor")
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4],
        subplot_titles=subplot_labels,
    )

    # 添加 K 线轨迹
    figure.add_trace(
        go.Candlestick(
            x=price_datetime,
            open=price_data["open"],
            high=price_data["high"],
            low=price_data["low"],
            close=price_data["close"],
            name="Kline",
        ),
        row=1,
        col=1,
    )

    # 添加因子时序轨迹
    figure.add_trace(
        go.Scatter(
            x=factor_datetime,
            y=factor_data["value"],
            mode="lines",
            name=factor_label or "Factor",
            line=dict(color="royalblue", width=2),
        ),
        row=2,
        col=1,
    )

    # 全局布局
    figure.update_layout(
        title=title,
        height=_DEFAULT_FIGURE_HEIGHT_COMBINED,
        xaxis_rangeslider_visible=False,
    )
    _apply_xaxis_style(figure, tickformat, tickangle)
    return figure
