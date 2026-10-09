# -*- coding: utf-8 -*-
"""Alpha101 Factory 数据模块。

提供 A 股日线行情数据的获取、存储与管理功能，包括：

- **数据源抽象**: ``DataSource`` 基类定义统一的数据获取接口。
- **多数据源支持**: ``AkShareSource``（主数据源）与 ``BaoStockSource``（降级方案）。
- **数据源工厂**: ``DataSourceFactory`` 管理多数据源注册与自动降级链。
- **数据加载器**: 市场快照获取、批量 K 线下载、完整性校验、单股票加载。
- **股票池管理**: 从 JSONL 文件加载股票代码列表。

数据流::

    DataSourceFactory → fetch_spot() → fetch_klines_from_spot() → JSONL 持久化

典型用法::

    from alpha101_factory.data import (
        DataSourceFactory, fetch_spot, fetch_klines_from_spot,
        load_universe, check_klines_integrity, load_or_fetch_symbol,
    )

    # 获取市场快照
    spot_df = fetch_spot()

    # 批量下载 K 线
    fetch_klines_from_spot(spot_df)

    # 加载股票池
    symbols = load_universe()

    # 加载或抓取单只股票
    df = load_or_fetch_symbol("600000", "20240101", "20241231")
"""
from __future__ import annotations

from alpha101_factory.data.sources import (
    DataSource,
    AkShareSource,
    BaoStockSource,
)
from alpha101_factory.data.factory import DataSourceFactory
from alpha101_factory.data.loader import (
    fetch_spot,
    fetch_klines_from_spot,
    check_klines_integrity,
    load_or_fetch_symbol,
    kline_path,
)
from alpha101_factory.data.universe import load_universe

__all__ = [
    # 数据源基类与实现
    "DataSource",
    "AkShareSource",
    "BaoStockSource",
    # 数据源工厂（含降级链管理）
    "DataSourceFactory",
    # 数据加载与抓取
    "fetch_spot",
    "fetch_klines_from_spot",
    "check_klines_integrity",
    "load_or_fetch_symbol",
    "kline_path",
    # 股票池管理
    "load_universe",
]
