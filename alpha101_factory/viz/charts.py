# -*- coding: utf-8 -*-
"""工业级图表渲染模块 — 可插拔 Plotly 图表系统。

提供 K 线图、因子时间序列、截面柱状图、热力图以及 K 线+因子叠加图
的渲染与 PNG 导出功能。所有图表通过 ``ChartFactory`` 统一创建，
支持通过 ``register_chart`` 装饰器注册自定义图表类型。

典型用法::

    from alpha101_factory.viz.charts import ChartFactory, register_chart

    # 创建内置图表
    chart = ChartFactory.create("kline")
    chart.save(Path("output/kline.png"), df=kline_df)

    # 注册自定义图表
    @register_chart
    class MyChart(Chart):
        name = "my_chart"
        def render(self, **kwargs) -> go.Figure:
            ...
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

_DT_FMT: str = "%Y-%m-%d"
"""日期时间轴默认格式化字符串。"""

_DT_ANGLE: int = -45
"""日期时间轴标签默认旋转角度（度）。"""

_KLINE_REQUIRED_COLS: Sequence[str] = ("datetime", "open", "high", "low", "close")
"""K 线图所需的最小列集合。"""

_FACTOR_REQUIRED_COLS: Sequence[str] = ("datetime", "symbol", "value")
"""因子图表所需的最小列集合。"""


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------

def _to_datetime_array(series: pd.Series) -> np.ndarray:
    """将字符串/日期序列转换为 ``datetime`` numpy 数组。

    Args:
        series: 包含日期信息的 pandas Series。

    Returns:
        一维 ``datetime64`` numpy 数组；若输入为空则返回空数组。
    """
    if series.empty:
        return np.array([], dtype="datetime64[ns]")
    dt = pd.to_datetime(series, errors="coerce").dt.tz_localize(None)
    return np.array(dt.dt.to_pydatetime())


def _apply_date_axis_style(
    figure: go.Figure,
    tickformat: str = _DT_FMT,
    tickangle: int = _DT_ANGLE,
) -> None:
    """为 Plotly Figure 的 X 轴应用统一的日期样式。

    Args:
        figure: 待修改的 Plotly Figure 对象。
        tickformat: 日期格式化模板，如 ``"%Y-%m-%d"``。
        tickangle: 标签旋转角度，负值表示逆时针。
    """
    figure.update_xaxes(
        type="date",
        tickformat=tickformat,
        tickangle=tickangle,
        ticks="outside",
    )


def _validate_dataframe(df: pd.DataFrame, required_cols: Sequence[str], chart_name: str) -> None:
    """校验 DataFrame 是否包含图表所需的最小列集合。

    Args:
        df: 待校验的 DataFrame。
        required_cols: 必须存在的列名序列。
        chart_name: 图表名称，用于错误信息。

    Raises:
        ValueError: 当缺少必需列或 DataFrame 为空时。
    """
    if df is None or df.empty:
        raise ValueError(f"[{chart_name}] DataFrame 为空，无法渲染图表")
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"[{chart_name}] DataFrame 缺少必需列: {missing}，"
            f"当前列: {list(df.columns)}"
        )


def _safe_write_image(figure: go.Figure, file_path: Path) -> None:
    """安全地将 Figure 导出为 PNG，捕获 Kaleido 相关异常。

    Args:
        figure: Plotly Figure 对象。
        file_path: 目标文件路径。

    Raises:
        RuntimeError: 当图像写入失败时。
    """
    try:
        figure.write_image(str(file_path))
    except Exception as exc:  # noqa: BLE001 — Kaleido 异常类型不稳定，统一捕获
        raise RuntimeError(f"图像写入失败 [{file_path}]: {exc}") from exc


# ---------------------------------------------------------------------------
# 图表基类
# ---------------------------------------------------------------------------

class Chart(ABC):
    """图表抽象基类，定义渲染与保存接口。

    所有具体图表类型必须继承此类并实现 :meth:`render` 方法。
    子类应设置 ``name`` 类属性作为唯一标识符。

    Attributes:
        name: 图表类型的唯一标识名称，用于工厂注册与查找。
    """

    name: str = "base"

    @abstractmethod
    def render(self, **kwargs: Any) -> go.Figure:
        """渲染图表并返回 Plotly Figure 对象。

        Args:
            **kwargs: 图表渲染所需的参数，由具体子类定义。

        Returns:
            渲染完成的 Plotly Figure 对象。

        Raises:
            NotImplementedError: 子类未实现此方法。
        """
        ...  # pragma: no cover

    def save(
        self,
        path: Path,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> Optional[Path]:
        """渲染图表并保存为 PNG 文件。

        Args:
            path: 目标 PNG 文件路径。
            overwrite: 若为 ``False`` 且文件已存在则跳过保存。
            **kwargs: 传递给 :meth:`render` 的参数。

        Returns:
            成功时返回保存的文件路径；若因已存在而跳过则仍返回该路径。

        Raises:
            ValueError: ``path`` 为空路径。
            RuntimeError: 图像写入失败。
        """
        if not path or str(path).strip() == "":
            raise ValueError("保存路径不能为空")

        if path.exists() and not overwrite:
            logger.info(f"文件已存在，跳过: {path}")
            return path

        # 渲染图表
        figure = self.render(**kwargs)
        if figure is None:
            raise RuntimeError("图表渲染返回 None，无法保存")

        # 确保父目录存在
        path.parent.mkdir(parents=True, exist_ok=True)

        # 写入图像文件
        _safe_write_image(figure, path)
        logger.info(f"图像已保存: {path}")
        return path


# ---------------------------------------------------------------------------
# 具体图表实现
# ---------------------------------------------------------------------------

class KlineChart(Chart):
    """K 线图（蜡烛图）。

    绘制标准 OHLC 蜡烛图，自动过滤缺失值并按日期排序。
    """

    name: str = "kline"

    def render(
        self,
        df: pd.DataFrame,
        title: str = "Kline",
        tickformat: str = _DT_FMT,
        tickangle: int = _DT_ANGLE,
        **kwargs: Any,
    ) -> go.Figure:
        """渲染 K 线蜡烛图。

        Args:
            df: 包含 K 线数据的 DataFrame，必需列:
                ``datetime``, ``open``, ``high``, ``low``, ``close``。
            title: 图表标题。
            tickformat: X 轴日期格式化字符串。
            tickangle: X 轴标签旋转角度。
            **kwargs: 保留参数，暂不使用。

        Returns:
            渲染完成的 K 线图 Figure 对象。

        Raises:
            ValueError: DataFrame 缺少必需列或为空。
        """
        _validate_dataframe(df, _KLINE_REQUIRED_COLS, self.name)

        # 过滤缺失值并按日期升序排列
        cleaned_df = df.dropna(subset=list(_KLINE_REQUIRED_COLS)).copy()
        cleaned_df = cleaned_df.sort_values("datetime")

        if cleaned_df.empty:
            raise ValueError(f"[{self.name}] 过滤后无有效数据，无法渲染 K 线图")

        figure = go.Figure(
            data=[
                go.Candlestick(
                    x=_to_datetime_array(cleaned_df["datetime"]),
                    open=cleaned_df["open"],
                    high=cleaned_df["high"],
                    low=cleaned_df["low"],
                    close=cleaned_df["close"],
                )
            ]
        )
        figure.update_layout(
            title=title,
            xaxis_rangeslider_visible=False,
            height=520,
        )
        _apply_date_axis_style(figure, tickformat, tickangle)
        return figure


class FactorTimeseriesChart(Chart):
    """因子时间序列折线图。

    绘制单只股票因子值随时间变化的趋势线。
    """

    name: str = "factor_ts"

    def render(
        self,
        fdf: pd.DataFrame,
        symbol: str,
        title: str = "Factor",
        tickformat: str = _DT_FMT,
        tickangle: int = _DT_ANGLE,
        **kwargs: Any,
    ) -> go.Figure:
        """渲染因子时间序列折线图。

        Args:
            fdf: 包含因子数据的 DataFrame，必需列:
                ``datetime``, ``symbol``, ``value``。
            symbol: 目标股票代码。
            title: 图表标题前缀。
            tickformat: X 轴日期格式化字符串。
            tickangle: X 轴标签旋转角度。
            **kwargs: 保留参数，暂不使用。

        Returns:
            渲染完成的因子时间序列 Figure 对象。

        Raises:
            ValueError: DataFrame 缺少必需列、为空或无目标股票数据。
        """
        _validate_dataframe(fdf, _FACTOR_REQUIRED_COLS, self.name)

        # 筛选目标股票并按日期排序
        stock_df = fdf[fdf["symbol"] == symbol].copy()
        if stock_df.empty:
            raise ValueError(
                f"[{self.name}] 未找到股票 {symbol} 的因子数据"
            )
        stock_df = stock_df.sort_values("datetime")

        figure = px.line(
            x=_to_datetime_array(stock_df["datetime"]),
            y=stock_df["value"],
            title=f"{title} | {symbol}",
        )
        figure.update_layout(height=420)
        _apply_date_axis_style(figure, tickformat, tickangle)
        return figure


class FactorCrossSectionChart(Chart):
    """因子截面柱状图。

    展示某一截面上因子值绝对值最大的 N 只股票的因子值分布。
    """

    name: str = "factor_cs"

    def render(
        self,
        fdf: pd.DataFrame,
        dt: Optional[Any] = None,
        topn: int = 100,
        title: str = "Factor cross-section",
        **kwargs: Any,
    ) -> go.Figure:
        """渲染因子截面柱状图。

        Args:
            fdf: 包含因子数据的 DataFrame，必需列:
                ``datetime``, ``symbol``, ``value``。
            dt: 目标截面日期。若为 ``None`` 则取数据中最新日期。
            topn: 展示绝对值最大的股票数量。
            title: 图表标题前缀。
            **kwargs: 保留参数，暂不使用。

        Returns:
            渲染完成的因子截面柱状图 Figure 对象。

        Raises:
            ValueError: DataFrame 为空或缺少必需列。
        """
        _validate_dataframe(fdf, _FACTOR_REQUIRED_COLS, self.name)

        working_df = fdf.copy()
        working_df["datetime"] = pd.to_datetime(working_df["datetime"], errors="coerce")

        # 确定截面日期
        if dt is None:
            dt = working_df["datetime"].max()
            if pd.isna(dt):
                raise ValueError(f"[{self.name}] 无法确定截面日期，数据中无有效日期")

        target_dt = pd.to_datetime(dt)
        cs_df = working_df[working_df["datetime"] == target_dt].copy()

        if cs_df.empty:
            raise ValueError(
                f"[{self.name}] 日期 {target_dt.date()} 无因子数据"
            )

        # 按绝对值排序，取 top N
        cs_df["abs_value"] = cs_df["value"].abs()
        cs_df = cs_df.sort_values("abs_value", ascending=False).head(topn)

        figure = px.bar(
            cs_df,
            x="symbol",
            y="value",
            title=f"{title} | {target_dt.date()}",
        )
        figure.update_layout(
            height=420,
            xaxis={"categoryorder": "total descending"},
        )
        return figure


class FactorHeatmapChart(Chart):
    """因子热力图。

    以热力图形式展示多只股票因子值在时间维度上的变化。
    """

    name: str = "factor_heatmap"

    def render(
        self,
        fdf: pd.DataFrame,
        symbols: List[str],
        title: str = "Factor heatmap",
        tickformat: str = _DT_FMT,
        tickangle: int = _DT_ANGLE,
        **kwargs: Any,
    ) -> go.Figure:
        """渲染因子热力图。

        Args:
            fdf: 包含因子数据的 DataFrame，必需列:
                ``datetime``, ``symbol``, ``value``。
            symbols: 目标股票代码列表。
            title: 图表标题。
            tickformat: X 轴日期格式化字符串。
            tickangle: X 轴标签旋转角度。
            **kwargs: 保留参数，暂不使用。

        Returns:
            渲染完成的因子热力图 Figure 对象。

        Raises:
            ValueError: DataFrame 为空、缺少必需列或 symbols 为空列表。
        """
        _validate_dataframe(fdf, _FACTOR_REQUIRED_COLS, self.name)

        if not symbols:
            raise ValueError(f"[{self.name}] symbols 列表不能为空")

        filtered_df = fdf[fdf["symbol"].isin(symbols)].copy()
        if filtered_df.empty:
            raise ValueError(
                f"[{self.name}] 未找到指定股票的因子数据: {symbols}"
            )

        filtered_df["datetime"] = pd.to_datetime(filtered_df["datetime"], errors="coerce")

        # 构建透视表：行为日期，列为股票
        pivot_table = filtered_df.pivot_table(
            index="datetime",
            columns="symbol",
            values="value",
        )

        if pivot_table.empty:
            raise ValueError(f"[{self.name}] 透视表为空，无法渲染热力图")

        # 转置使行为股票、列为日期，符合热力图阅读习惯
        figure = px.imshow(
            pivot_table.T,
            aspect="auto",
            origin="lower",
            title=title,
        )
        figure.update_layout(height=500)
        _apply_date_axis_style(figure, tickformat, tickangle)
        return figure


class KlineWithFactorChart(Chart):
    """K 线与因子叠加图。

    上下双面板布局：上方为 K 线蜡烛图，下方为因子时间序列折线图，
    共享 X 轴以便对比分析。
    """

    name: str = "kline_factor"

    def render(
        self,
        kline_df: pd.DataFrame,
        factor_df: pd.DataFrame,
        symbol: str,
        title: str = "Kline + Factor",
        tickformat: str = _DT_FMT,
        tickangle: int = _DT_ANGLE,
        factor_label: Optional[str] = None,
        **kwargs: Any,
    ) -> go.Figure:
        """渲染 K 线与因子叠加双面板图。

        Args:
            kline_df: K 线数据 DataFrame，必需列:
                ``datetime``, ``open``, ``high``, ``low``, ``close``。
            factor_df: 因子数据 DataFrame，必需列:
                ``datetime``, ``symbol``, ``value``。
            symbol: 目标股票代码。
            title: 图表总标题。
            tickformat: X 轴日期格式化字符串。
            tickangle: X 轴标签旋转角度。
            factor_label: 因子图例子标签，默认使用 "Factor"。
            **kwargs: 保留参数，暂不使用。

        Returns:
            渲染完成的双面板 Figure 对象。

        Raises:
            ValueError: 任一 DataFrame 为空或缺少必需列。
        """
        _validate_dataframe(kline_df, _KLINE_REQUIRED_COLS, self.name)
        _validate_dataframe(factor_df, _FACTOR_REQUIRED_COLS, self.name)

        # 准备 K 线数据
        cleaned_kline = kline_df.dropna(subset=list(_KLINE_REQUIRED_COLS)).copy()
        cleaned_kline = cleaned_kline.sort_values("datetime")

        if cleaned_kline.empty:
            raise ValueError(f"[{self.name}] K 线数据过滤后为空")

        # 准备因子数据
        stock_factor = factor_df[factor_df["symbol"] == symbol].copy()
        if stock_factor.empty:
            raise ValueError(
                f"[{self.name}] 未找到股票 {symbol} 的因子数据"
            )
        stock_factor = stock_factor.sort_values("datetime")

        label = factor_label or "Factor"

        # 创建双面板布局
        figure = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4],
            subplot_titles=("Kline", label),
        )

        # 上方面板：K 线蜡烛图
        figure.add_trace(
            go.Candlestick(
                x=_to_datetime_array(cleaned_kline["datetime"]),
                open=cleaned_kline["open"],
                high=cleaned_kline["high"],
                low=cleaned_kline["low"],
                close=cleaned_kline["close"],
                name="Kline",
            ),
            row=1,
            col=1,
        )

        # 下方面板：因子折线图
        figure.add_trace(
            go.Scatter(
                x=_to_datetime_array(stock_factor["datetime"]),
                y=stock_factor["value"],
                mode="lines",
                name=label,
                line=dict(color="royalblue", width=2),
            ),
            row=2,
            col=1,
        )

        figure.update_layout(
            title=title,
            height=720,
            xaxis_rangeslider_visible=False,
        )
        _apply_date_axis_style(figure, tickformat, tickangle)
        return figure


# ---------------------------------------------------------------------------
# 图表注册表与工厂
# ---------------------------------------------------------------------------

_CHART_REGISTRY: Dict[str, Type[Chart]] = {}
"""已注册图表类型的映射表，键为图表名称，值为图表类。"""


def register_chart(cls: Type[Chart]) -> Type[Chart]:
    """将图表类注册到全局注册表中。

    可作为装饰器使用，注册后该类可通过 :class:`ChartFactory` 创建。

    Args:
        cls: 待注册的图表类，必须继承 :class:`Chart` 并设置 ``name`` 属性。

    Returns:
        原始图表类（便于链式使用）。

    Raises:
        TypeError: 当 ``cls`` 不是 :class:`Chart` 的子类时。
    """
    if not isinstance(cls, type) or not issubclass(cls, Chart):
        raise TypeError(f"register_chart 要求 Chart 的子类，收到: {type(cls)}")

    if not hasattr(cls, "name") or not cls.name:
        raise ValueError(f"图表类 {cls.__name__} 必须设置非空 name 属性")

    _CHART_REGISTRY[cls.name] = cls
    logger.debug(f"已注册图表: {cls.name}")
    return cls


# 自动注册内置图表类型
for _chart_cls in [
    KlineChart,
    FactorTimeseriesChart,
    FactorCrossSectionChart,
    FactorHeatmapChart,
    KlineWithFactorChart,
]:
    register_chart(_chart_cls)


class ChartFactory:
    """图表创建工厂类。

    通过名称从注册表中查找并实例化对应的图表类。
    提供 ``create`` 和 ``list_all`` 两个类方法。
    """

    @classmethod
    def create(cls, name: str) -> Chart:
        """根据名称创建图表实例。

        Args:
            name: 图表类型名称，如 ``"kline"``、``"factor_ts"`` 等。

        Returns:
            新创建的图表实例。

        Raises:
            KeyError: 当名称不在注册表中时，错误信息包含可用图表列表。
        """
        if name not in _CHART_REGISTRY:
            available = sorted(_CHART_REGISTRY.keys())
            raise KeyError(f"未知图表: {name}. 可用图表: {available}")
        return _CHART_REGISTRY[name]()

    @classmethod
    def list_all(cls) -> List[str]:
        """列出所有已注册的图表类型名称。

        Returns:
            按字母顺序排序的图表名称列表。
        """
        return sorted(_CHART_REGISTRY.keys())
