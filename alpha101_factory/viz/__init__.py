# -*- coding: utf-8 -*-
"""Alpha101 Factory 可视化模块。

提供 A 股量化研究场景下的标准可视化功能，包括：

- **图表基类**: ``Chart`` 抽象基类定义 ``render()`` 和 ``save()`` 接口。
- **内置图表**:
    - ``KlineChart``: 标准 OHLC 蜡烛图。
    - ``FactorTimeseriesChart``: 因子时间序列折线图。
    - ``FactorCrossSectionChart``: 因子截面柱状图（Top N）。
    - ``FactorHeatmapChart``: 因子热力图（时间 × 股票矩阵）。
    - ``KlineWithFactorChart``: K 线与因子叠加双面板图。
- **图表工厂**: ``ChartFactory`` 按名称创建图表实例，支持自定义注册。
- **便捷绘图函数**: 提供 ``plot_kline``、``plot_factor_timeseries`` 等
  快捷函数，无需实例化图表类即可生成图表。

典型用法::

    from alpha101_factory.viz import (
        ChartFactory, plot_kline, plot_factor_timeseries,
        plot_factor_cross_section, plot_heatmap, plot_kline_with_factor,
        save_fig,
    )

    # 使用便捷函数绘图
    fig = plot_kline(kline_df, title="600000 Kline")
    save_fig(fig, Path("output/kline.png"))

    # 使用图表工厂
    chart = ChartFactory.create("factor_ts")
    chart.save(Path("output/factor.png"), fdf=factor_df, symbol="600000")

注意:
    PNG 导出依赖 ``kaleido`` 包，请确保已安装。
"""
from __future__ import annotations

from alpha101_factory.viz.charts import (
    Chart,
    KlineChart,
    FactorTimeseriesChart,
    FactorCrossSectionChart,
    FactorHeatmapChart,
    KlineWithFactorChart,
    ChartFactory,
    register_chart,
)
from alpha101_factory.viz.plots import (
    plot_kline,
    plot_factor_timeseries,
    plot_factor_cross_section,
    plot_heatmap,
    plot_kline_with_factor,
    save_fig,
)

__all__ = [
    # 图表基类与实现
    "Chart",
    "KlineChart",
    "FactorTimeseriesChart",
    "FactorCrossSectionChart",
    "FactorHeatmapChart",
    "KlineWithFactorChart",
    # 图表工厂与注册
    "ChartFactory",
    "register_chart",
    # 便捷绘图函数
    "plot_kline",
    "plot_factor_timeseries",
    "plot_factor_cross_section",
    "plot_heatmap",
    "plot_kline_with_factor",
    "save_fig",
]
