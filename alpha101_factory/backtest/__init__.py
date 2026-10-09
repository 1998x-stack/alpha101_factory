# -*- coding: utf-8 -*-
"""Alpha101 Factory 回测模块。

提供因子质量评估与回测分析功能，包括：

- **评估器基类**: ``Evaluator`` 抽象基类定义统一的 ``evaluate()`` 和 ``save()`` 接口。
- **IC 评估器**: ``ICEvaluator`` 计算横截面 IC/RankIC、时间序列 IC 统计。
- **分位组合评估器**: ``QuantileEvaluator`` 将股票按因子值分位，
  计算各组平均前瞻收益率及多空组合收益。
- **评估器工厂**: ``EvaluatorFactory`` 按名称创建评估器实例，支持自定义注册。
- **回测引擎**: ``BacktestEngine`` 封装完整的回测流程：
  数据加载 → 评估器执行 → 结果保存 → 图表绘制。

核心指标::

    - IC: 横截面 Pearson 相关系数（因子值 vs 前瞻收益）
    - RankIC: 横截面 Spearman 秩相关系数
    - IC.t: IC 的 t 统计量，衡量显著性
    - TS.IC: 单只股票时间序列 IC
    - Avg.N: 日均样本股票数

典型用法::

    from alpha101_factory.backtest import (
        BacktestEngine, EvaluatorFactory, make_forward_return,
        ICEvaluator, QuantileEvaluator,
    )

    # 一键运行回测
    engine = BacktestEngine("Alpha101", horizon=1, quantiles=5)
    engine.run()

    # 手动执行评估器
    ic_eval = EvaluatorFactory.create("ic")
    results = ic_eval.evaluate(factor_df, price_df, horizon=1)

注意:
    - 横截面 IC 需要同一天至少两只股票，单股票场景请查看 TS-IC。
    - 分位组合在样本不足时自动跳过，不会抛出异常。
"""
from __future__ import annotations

from alpha101_factory.backtest.evaluators import (
    Evaluator,
    ICEvaluator,
    QuantileEvaluator,
    EvaluatorFactory,
    register_evaluator,
    make_forward_return,
)
from alpha101_factory.backtest.engine import BacktestEngine

__all__ = [
    # 评估器基类与实现
    "Evaluator",
    "ICEvaluator",
    "QuantileEvaluator",
    # 评估器工厂与注册
    "EvaluatorFactory",
    "register_evaluator",
    # 前瞻收益计算工具
    "make_forward_return",
    # 回测引擎
    "BacktestEngine",
]
