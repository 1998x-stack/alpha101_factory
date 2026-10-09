# -*- coding: utf-8 -*-
"""Alpha101 Factory Pipeline 模块。

提供因子工厂流水线的阶段定义与编排引擎，包括：

- **阶段基类**: ``Stage`` 抽象基类定义统一的 ``run(ctx)`` 接口，
  各阶段通过上下文字典传递数据，实现阶段间解耦。
- **内置阶段**:
    - ``FetchStage``: 从 AkShare/BaoStock 抓取行情快照与 K 线数据。
    - ``TmpStage``: 构建中间特征缓存（收益率、VWAP、ADV 等）。
    - ``CheckStage``: 校验 K 线 JSONL 文件的完整性。
    - ``FactorStage``: 加载数据并计算已注册的 Alpha 因子。
    - ``BacktestStage``: 执行 IC/RankIC 评估与分位组合回测。
- **阶段工厂**: ``StageFactory`` 按名称创建阶段实例，支持自定义阶段注册。
- **编排引擎**: ``PipelineEngine`` 按顺序执行阶段，支持错误处理策略配置。

Pipeline 流程::

    fetch → tmp → factor → backtest

典型用法::

    from alpha101_factory.pipeline import (
        PipelineEngine, StageFactory, register_stage, Stage,
    )

    # 一键运行完整流水线
    engine = PipelineEngine()
    result = engine.run_full(alpha="Alpha101", horizon=1, quantiles=5)

    # 自定义阶段
    @register_stage
    class DataQualityStage(Stage):
        name = "quality"

        def run(self, ctx):
            ctx["quality_report"] = {"status": "ok"}
            return ctx
"""
from __future__ import annotations

from alpha101_factory.pipeline.stages import (
    Stage,
    FetchStage,
    TmpStage,
    CheckStage,
    FactorStage,
    BacktestStage,
    StageFactory,
    register_stage,
)
from alpha101_factory.pipeline.engine import PipelineEngine

__all__ = [
    # 阶段基类与内置阶段
    "Stage",
    "FetchStage",
    "TmpStage",
    "CheckStage",
    "FactorStage",
    "BacktestStage",
    # 阶段工厂与注册
    "StageFactory",
    "register_stage",
    # 编排引擎
    "PipelineEngine",
]
