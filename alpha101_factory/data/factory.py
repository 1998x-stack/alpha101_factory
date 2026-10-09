# -*- coding: utf-8 -*-
"""数据源工厂模块 — 创建与管理多数据源及降级链。

本模块提供 ``DataSourceFactory`` 类，用于统一创建数据源实例并在多个
数据源之间实现自动降级（fallback）。默认降级链为 AkShare → BaoStock，
当主数据源不可用时自动切换至备用数据源。

典型用法::

    # 使用默认降级链获取 K 线数据
    df = DataSourceFactory.fetch_kline_fallback("600000", "20240101", "20241231", "qfq")

    # 注册自定义数据源
    DataSourceFactory.register("custom", MyCustomSource)

    # 指定自定义降级顺序
    df = DataSourceFactory.fetch_kline_fallback(
        "600000", None, None, "qfq", order=["custom", "akshare"]
    )

"""
from __future__ import annotations

from typing import Type, Optional

import pandas as pd
from loguru import logger

from alpha101_factory.data.sources import (
    DataSource,
    AkShareSource,
    BaoStockSource,
)


class DataSourceFactory:
    """数据源工厂类，负责创建数据源实例并管理降级链。

    该类采用类方法（classmethod）实现，所有操作无需实例化即可调用。
    内部维护两个类级别属性：

    - ``_sources``: 注册表，映射数据源名称到其类对象
    - ``_fallback_order``: 降级链顺序，按优先级排列的数据源名称列表

    默认注册的数据源：
        - ``akshare``: AkShare 数据源（主数据源）
        - ``baostock``: BaoStock 数据源（备用数据源）

    默认降级顺序：
        ``["akshare", "baostock"]``

    属性:
        _sources: 数据源注册表，键为数据源名称，值为数据源类。
        _fallback_order: 降级链中的数据源名称列表，按优先级排序。

    """

    # 数据源注册表：名称 → 数据源类
    _sources: dict[str, Type[DataSource]] = {
        "akshare": AkShareSource,
        "baostock": BaoStockSource,
    }
    # 默认降级链：按优先级排列的数据源名称
    _fallback_order: list[str] = ["akshare", "baostock"]

    @classmethod
    def register(cls, name: str, source_cls: Type[DataSource]) -> None:
        """注册新的数据源类到工厂中。

        将数据源类添加到内部注册表，并将其追加到降级链末尾（如果尚未存在）。
        注册后可通过 ``create()`` 方法创建该数据源实例，或在降级链中使用。

        Args:
            name: 数据源的唯一标识名称，用于后续创建和降级链引用。
            source_cls: 数据源类，必须继承自 ``DataSource`` 基类。

        Raises:
            TypeError: 当 ``source_cls`` 不是 ``DataSource`` 的子类时抛出。

        Example:
            >>> DataSourceFactory.register("custom", MyCustomSource)

        """
        # 验证数据源类是否继承自 DataSource 基类
        if not isinstance(source_cls, type) or not issubclass(source_cls, DataSource):
            raise TypeError(
                f"数据源类 '{source_cls}' 必须继承自 DataSource 基类"
            )

        # 验证名称非空
        if not name or not name.strip():
            raise ValueError("数据源名称不能为空")

        # 注册到注册表
        cls._sources[name] = source_cls

        # 若不在降级链中，则追加到末尾
        if name not in cls._fallback_order:
            cls._fallback_order.append(name)

        logger.info(f"已注册数据源: '{name}' ({source_cls.__name__})")

    @classmethod
    def create(cls, name: str) -> DataSource:
        """根据名称创建并返回数据源实例。

        从内部注册表中查找对应的数据源类并实例化。如果名称不存在，
        抛出 ``KeyError`` 并列出所有可用的数据源。

        Args:
            name: 要创建的数据源名称。

        Returns:
            新创建的数据源实例。

        Raises:
            KeyError: 当指定的数据源名称未在注册表中找到时抛出，
                错误信息包含所有可用数据源列表。

        Example:
            >>> source = DataSourceFactory.create("akshare")
            >>> df = source.fetch_kline("600000", "20240101", "20241231", "qfq")

        """
        # 验证数据源名称是否存在于注册表中
        if name not in cls._sources:
            available_sources = list(cls._sources.keys())
            raise KeyError(
                f"未知数据源: '{name}'。可用数据源: {available_sources}"
            )

        # 实例化并返回
        source_instance = cls._sources[name]()
        logger.debug(f"已创建数据源实例: '{name}'")
        return source_instance

    @classmethod
    def fetch_kline_fallback(
        cls,
        symbol: str,
        start_date: Optional[str],
        end_date: Optional[str],
        adjust: str,
        order: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """按降级链顺序获取 K 线数据，直到成功或所有数据源耗尽。

        依次尝试降级链中的每个数据源，获取指定股票的日线 K 线数据。
        一旦某个数据源返回非空 DataFrame 即立即返回；若所有数据源均失败，
        返回空 DataFrame 并记录错误日志。

        每个数据源的尝试过程和结果都会被详细记录到日志中。

        Args:
            symbol: 股票代码，如 ``"600000"``。
            start_date: 起始日期，格式 ``"YYYYMMDD"``，``None`` 表示不限制。
            end_date: 结束日期，格式 ``"YYYYMMDD"``，``None`` 表示不限制。
            adjust: 复权方式，可选值 ``"qfq"``（前复权）、``"hfq"``（后复权）、
                ``""``（不复权）。
            order: 自定义降级顺序的数据源名称列表。若为 ``None``，
                使用默认的 ``_fallback_order``。

        Returns:
            包含 K 线数据的 DataFrame，列包括 ``datetime``, ``open``, ``high``,
            ``low``, ``close``, ``volume``, ``amount`` 等。
            若所有数据源均失败，返回空 DataFrame。

        Example:
            >>> df = DataSourceFactory.fetch_kline_fallback(
            ...     "600000", "20240101", "20241231", "qfq"
            ... )
            >>> len(df)
            242

        """
        # 确定降级链顺序：使用自定义顺序或默认顺序
        fallback_chain = order if order else cls._fallback_order

        # 验证降级链非空
        if not fallback_chain:
            logger.error("降级链为空，无法获取数据")
            return pd.DataFrame()

        logger.info(
            f"开始获取 K 线数据: symbol={symbol}, "
            f"start_date={start_date or '无'}, end_date={end_date or '无'}, "
            f"adjust={adjust or '无'}, 降级链={fallback_chain}"
        )

        # 依次尝试降级链中的每个数据源
        for source_name in fallback_chain:
            # 验证数据源名称是否在注册表中
            if source_name not in cls._sources:
                logger.warning(
                    f"降级链中的数据源 '{source_name}' 未注册，跳过"
                )
                continue

            try:
                # 创建数据源实例并获取 K 线数据
                source_instance = cls.create(source_name)
                kline_dataframe = source_instance.fetch_kline(
                    symbol, start_date, end_date, adjust
                )

                # 验证返回结果是否有效（非 None 且非空）
                if kline_dataframe is not None and not kline_dataframe.empty:
                    row_count = len(kline_dataframe)
                    logger.info(
                        f"数据源 '{source_name}' 成功获取 K 线数据: "
                        f"{row_count} 行"
                    )
                    return kline_dataframe
                else:
                    # 数据源正常返回但无数据（如股票退市、日期范围无数据）
                    # 注意：这种情况不应触发降级，因为可能是真实无数据
                    logger.warning(
                        f"数据源 '{source_name}' 返回空数据（API 无记录或日期范围内无数据），"
                        f"降级到下一个数据源"
                    )

            except Exception as exc:
                # 网络错误、API 异常等 — 触发降级到下一个数据源
                logger.warning(
                    f"数据源 '{source_name}' 异常: {type(exc).__name__}: {exc}，"
                    f"降级到下一个数据源"
                )

        # 所有数据源均失败
        logger.error(f"所有数据源均无法获取 K 线数据: symbol={symbol}")
        return pd.DataFrame()

    @classmethod
    def fetch_spot_fallback(
        cls,
        order: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """按降级链顺序获取市场快照数据，直到成功或所有数据源耗尽。

        依次尝试降级链中的每个数据源，获取全市场股票的最新快照数据
       （包含代码、名称、当前价格等信息）。一旦某个数据源返回非空
        DataFrame 即立即返回；若所有数据源均失败，返回空 DataFrame。

        每个数据源的尝试过程和结果都会被详细记录到日志中。

        Args:
            order: 自定义降级顺序的数据源名称列表。若为 ``None``，
                使用默认的 ``_fallback_order``。

        Returns:
            包含市场快照数据的 DataFrame，列至少包括 ``code``（股票代码）
            和 ``name``（股票名称）。若所有数据源均失败，返回空 DataFrame。

        Example:
            >>> df = DataSourceFactory.fetch_spot_fallback()
            >>> len(df)
            5000
            >>> df.columns
            Index(['code', 'name', ...], dtype='object')

        """
        # 确定降级链顺序：使用自定义顺序或默认顺序
        fallback_chain = order if order else cls._fallback_order

        # 验证降级链非空
        if not fallback_chain:
            logger.error("降级链为空，无法获取快照数据")
            return pd.DataFrame()

        logger.info(
            f"开始获取市场快照数据，降级链={fallback_chain}"
        )

        # 依次尝试降级链中的每个数据源
        for source_name in fallback_chain:
            # 验证数据源名称是否在注册表中
            if source_name not in cls._sources:
                logger.warning(
                    f"降级链中的数据源 '{source_name}' 未注册，跳过"
                )
                continue

            try:
                # 创建数据源实例并获取快照数据
                source_instance = cls.create(source_name)
                spot_dataframe = source_instance.fetch_spot()

                # 验证返回结果是否有效（非空）
                if not spot_dataframe.empty:
                    row_count = len(spot_dataframe)
                    logger.info(
                        f"数据源 '{source_name}' 成功获取快照数据: "
                        f"{row_count} 行"
                    )
                    return spot_dataframe
                else:
                    logger.warning(
                        f"数据源 '{source_name}' 返回空快照数据，尝试下一个"
                    )

            except Exception as exc:
                # 记录异常并继续尝试下一个数据源
                logger.warning(
                    f"数据源 '{source_name}' 获取快照失败: {exc}"
                )

        # 所有数据源均失败
        logger.error("所有数据源均无法获取市场快照数据")
        return pd.DataFrame()
