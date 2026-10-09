# -*- coding: utf-8 -*-
"""Alpha101 Factory 工具模块。

提供金融量化计算所需的通用工具函数，包括：

- **时间序列算子**: 滚动窗口求和/最值/均值/标准差、时间序列排名、
  线性衰减加权平均、滞后、差分、收益率计算等。
- **截面算子**: 截面分位数排名、截面 Z-score 标准化。
- **分组算子**: 按股票分组应用计算函数。
- **IO 工具**: JSONL 文件的读写与元数据管理。

所有算子均支持可选的性能加速库（bottleneck/numba），
并在不可用时自动回退至纯 pandas 实现。

典型用法::

    from alpha101_factory.utils.ops import (
        rolling_sum, ts_rank, decay_linear,
        cs_rank, cs_zscore, returns, adv,
    )
    from alpha101_factory.utils.io import read_jsonl, write_jsonl

    # 滚动窗口计算
    result = rolling_sum(series, window=20)

    # 截面排名
    ranked = cs_rank(factor_series)

    # JSONL 读写
    df = read_jsonl("data/factors/Alpha101.jsonl")
    write_jsonl(df, "output.jsonl", meta={"factor": "Alpha101"})
"""
from __future__ import annotations

from alpha101_factory.utils.ops import (
    rolling_sum,
    rolling_min,
    rolling_max,
    rolling_std,
    rolling_cov,
    rolling_corr,
    ts_rank,
    decay_linear,
    delay,
    delta,
    returns,
    vwap_from_amount,
    adv,
    cs_rank,
    cs_zscore,
    by_symbol,
)
from alpha101_factory.utils.io import read_jsonl, write_jsonl

__all__ = [
    # 滚动窗口算子
    "rolling_sum",
    "rolling_min",
    "rolling_max",
    "rolling_std",
    "rolling_cov",
    "rolling_corr",
    # 时间序列算子
    "ts_rank",
    "decay_linear",
    "delay",
    "delta",
    "returns",
    "vwap_from_amount",
    "adv",
    # 截面算子
    "cs_rank",
    "cs_zscore",
    # 分组算子
    "by_symbol",
    # IO 工具
    "read_jsonl",
    "write_jsonl",
]
