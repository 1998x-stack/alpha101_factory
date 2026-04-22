# -*- coding: utf-8 -*-
"""Alpha101 Factory — A 股日线 Alpha 因子工厂。

可插拔的 Alpha 因子计算与回测框架，支持 AkShare/BaoStock 数据源、
JSONL 存储、IC/RankIC 评估与分位组合分析。

Pipeline 流程::

    fetch → tmp features → factor compute → backtest (IC/RankIC, quantile portfolios) → viz

典型用法::

    from alpha101_factory import config
    from alpha101_factory.pipeline.engine import PipelineEngine

    # 查看当前配置
    print(config.DATA_ROOT)

    # 一键运行完整流水线
    engine = PipelineEngine()
    result = engine.run_full(alpha="Alpha101", horizon=1, quantiles=5)

子包概览:
    - ``data``: 数据源、加载器、股票池管理
    - ``factors``: 因子基类、注册表、Alpha 因子实现
    - ``pipeline``: Pipeline 阶段定义与编排引擎
    - ``backtest``: 回测引擎、IC/RankIC 与分位组合评估器
    - ``viz``: Plotly 图表渲染与 PNG 导出
    - ``utils``: 金融量化算子（滚动窗口、截面排名等）与 IO 工具

环境变量:
    通过 ``config`` 模块解析，详见该模块文档。
"""
from __future__ import annotations

from alpha101_factory import config

__all__ = [
    "config",
]
