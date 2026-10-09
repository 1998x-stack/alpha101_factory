# -*- coding: utf-8 -*-
"""
因子基类模块 (Factor Base Class Module)

本模块定义了所有 Alpha 因子的抽象基类 ``Factor``，提供：
    - 因子元数据声明（名称、所需列）
    - 输入列校验（validate_requires）
    - 截面排名工具（_cs_rank）
    - 按股票分组计算工具（_g）
    - 截面 Series 构建工具（as_cs_series）

所有自定义因子必须继承此类并实现 ``compute()`` 方法。

Usage:
    >>> from alpha101_factory.factors.base import Factor
    >>> class MyAlpha(Factor):
    ...     name = "MyAlpha"
    ...     requires = ["close", "volume"]
    ...     def compute(self, df):
    ...         return self.as_cs_series(df, df["close"] - df["volume"])
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List

import pandas as pd


class Factor(ABC):
    """Alpha 因子抽象基类 (Abstract Base Class for Alpha Factors)。

    所有因子实现必须继承此类，设置 ``name`` 和 ``requires`` 类属性，
    并实现 ``compute(df)`` 方法。

    类属性:
        name: 因子唯一标识名称，用于注册和回测引用。
        requires: 因子计算所需的 DataFrame 列名列表。

    示例:
        >>> @register
        ... class MyAlpha(Factor):
        ...     name = "MyAlpha"
        ...     requires = ["close", "volume"]
        ...
        ...     def compute(self, df):
        ...         close = df["close"]
        ...         volume = df["volume"]
        ...         return self.as_cs_series(df, close / volume)
    """

    # 因子唯一标识名称 (Unique factor identifier)
    name: str = "BaseFactor"

    # 因子计算所需的 DataFrame 列名列表 (Required column names for computation)
    requires: List[str] = []

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算因子值 (Compute factor values)。

        子类必须实现此方法，接收包含行情数据的长表 DataFrame，
        返回以 MultiIndex[datetime, symbol] 为索引的因子值 Series。

        Args:
            df: 长表格式的行情数据，必须包含 ``datetime`` 和 ``symbol`` 列，
                以及 ``requires`` 中声明的所有列。

        Returns:
            因子值 Series，索引为 MultiIndex[datetime, symbol]，
            name 属性设为 ``"value"``。

        Raises:
            NotImplementedError: 子类未实现此方法时抛出。
        """
        raise NotImplementedError(f"因子 {self.name} 未实现 compute() 方法")

    def validate_requires(self, df: pd.DataFrame) -> None:
        """校验输入 DataFrame 是否包含因子所需的全部列 (Validate required columns)。

        逐列检查 ``requires`` 中声明的列是否存在于输入 DataFrame 中。
        若存在缺失列，抛出 ``KeyError`` 并打印详细的缺失列信息。

        Args:
            df: 待校验的行情数据 DataFrame。

        Raises:
            KeyError: 当 ``requires`` 中存在列不在 ``df.columns`` 中时抛出，
                错误信息包含缺失列列表和实际列列表。

        示例:
            >>> factor = MyAlpha()
            >>> factor.validate_requires(df)  # 若缺少列则抛出 KeyError
        """
        # 收集缺失的列名 (Collect missing column names)
        missing_columns: List[str] = [
            column_name for column_name in self.requires if column_name not in df.columns
        ]

        if missing_columns:
            error_message = (
                f"[{self.name}] 缺少必需的列: {missing_columns}，"
                f"实际可用的列: {list(df.columns)}"
            )
            print(f"[ERROR] {error_message}")
            raise KeyError(error_message)

        print(f"[INFO] [{self.name}] 列校验通过: 所需列 {self.requires} 均已存在")

    def _cs_rank(self, df: pd.DataFrame, values: pd.Series) -> pd.Series:
        """计算截面分位数排名 (Cross-Sectional Percentile Rank)。

        对每个交易日（datetime level=0）上的所有股票，计算 ``values``
        的百分位排名。排名值域为 [0, 1]，表示该股票在当日截面中的相对位置。

        Args:
            df: 长表格式的行情数据，必须包含 ``datetime`` 和 ``symbol`` 列，
                用于构建 MultiIndex。
            values: 待排名的数值 Series，长度需与 ``df`` 行数一致。

        Returns:
            截面百分位排名 Series，索引为 MultiIndex[datetime, symbol]。

        Raises:
            ValueError: 当 ``df`` 缺少 ``datetime`` 或 ``symbol`` 列时抛出。
            ValueError: 当 ``values`` 长度与 ``df`` 行数不一致时抛出。
        """
        # 校验 DataFrame 包含必需的索引列 (Validate required index columns)
        required_index_columns = ["datetime", "symbol"]
        missing_index_columns = [
            col for col in required_index_columns if col not in df.columns
        ]
        if missing_index_columns:
            raise ValueError(
                f"[{self.name}] _cs_rank: DataFrame 缺少必需的列 {missing_index_columns}，"
                f"无法构建截面排名索引"
            )

        # 校验 values 长度与 DataFrame 行数一致 (Validate values length matches df rows)
        if len(values) != len(df):
            raise ValueError(
                f"[{self.name}] _cs_rank: values 长度 ({len(values)}) 与 "
                f"DataFrame 行数 ({len(df)}) 不一致"
            )

        # 构建 MultiIndex 并执行截面排名 (Build MultiIndex and perform cross-sectional rank)
        multi_index = pd.MultiIndex.from_frame(
            df[required_index_columns], names=required_index_columns
        )
        values_with_index = pd.Series(values.values, index=multi_index)
        return values_with_index.groupby(level=0).rank(pct=True)

    def _g(
        self,
        df: pd.DataFrame,
        column_name: str | None,
        compute_fn: Callable[..., pd.Series],
        *compute_args,
    ) -> pd.Series:
        """按股票分组应用计算函数 (Group-by-Symbol Apply Function)。

        将 DataFrame 按 ``symbol`` 列分组，对每只股票的数据独立应用
        ``compute_fn``。支持两种模式：
            - ``column_name`` 为 ``None``：将整个分组 DataFrame 传入 ``compute_fn``
            - ``column_name`` 为具体列名：仅将该列 Series 传入 ``compute_fn``

        Args:
            df: 长表格式的行情数据，必须包含 ``symbol`` 列。
            column_name: 待处理的列名。若为 ``None``，则将整个分组 DataFrame
                传入 ``compute_fn``；否则仅传入该列的 Series。
            compute_fn: 分组计算函数，签名为 ``fn(series_or_df, *compute_args) -> pd.Series``。
            *compute_args: 传递给 ``compute_fn`` 的额外位置参数。

        Returns:
            分组计算结果拼接后的 Series，索引与原始 ``df`` 对齐。

        Raises:
            ValueError: 当 ``df`` 缺少 ``symbol`` 列时抛出。
            ValueError: 当 ``df`` 为空 DataFrame 时返回空 Series。

        示例:
            >>> # 对每只股票的 close 列计算 20 日滚动标准差
            >>> result = factor._g(df, "close", ops.rolling_std, 20)
            >>>
            >>> # 将整个分组 DataFrame 传入自定义函数
            >>> result = factor._g(df, None, lambda g: g["close"] - g["open"])
        """
        # 校验 DataFrame 包含 symbol 列 (Validate symbol column exists)
        if "symbol" not in df.columns:
            raise ValueError(
                f"[{self.name}] _g: DataFrame 缺少必需的 'symbol' 列，无法执行分组计算"
            )

        # 处理空 DataFrame 边界情况 (Handle empty DataFrame edge case)
        if df.empty:
            return pd.Series(dtype=float)

        if column_name is None:
            # 模式一：将整个分组 DataFrame 传入计算函数 (Pass entire group DataFrame)
            return df.groupby("symbol", group_keys=False).apply(
                lambda group_data: compute_fn(group_data, *compute_args)
            )
        else:
            # 模式二：仅将指定列的 Series 传入计算函数 (Pass specific column Series)
            return df.groupby("symbol", group_keys=False)[column_name].apply(
                lambda group_series: compute_fn(group_series, *compute_args)
            )

    @staticmethod
    def as_cs_series(df: pd.DataFrame, values: pd.Series) -> pd.Series:
        """将因子值构建为标准截面 Series (Build Standard Cross-Sectional Series)。

        以 ``df`` 的 ``datetime`` 和 ``symbol`` 列构建 MultiIndex，
        将 ``values`` 包装为带索引的 Series，name 设为 ``"value"``。
        这是因子 ``compute()`` 方法的标准返回格式。

        Args:
            df: 长表格式的行情数据，必须包含 ``datetime`` 和 ``symbol`` 列，
                用于构建 MultiIndex。
            values: 因子值 Series 或数组，长度需与 ``df`` 行数一致。

        Returns:
            标准格式的因子值 Series，索引为 MultiIndex[datetime, symbol]，
            name 为 ``"value"``。

        Raises:
            ValueError: 当 ``df`` 缺少 ``datetime`` 或 ``symbol`` 列时抛出。
            ValueError: 当 ``values`` 长度与 ``df`` 行数不一致时抛出。

        示例:
            >>> class MyAlpha(Factor):
            ...     def compute(self, df):
            ...         factor_values = df["close"] - df["open"]
            ...         return self.as_cs_series(df, factor_values)
        """
        # 校验 DataFrame 包含必需的索引列 (Validate required index columns)
        required_index_columns = ["datetime", "symbol"]
        missing_index_columns = [
            col for col in required_index_columns if col not in df.columns
        ]
        if missing_index_columns:
            raise ValueError(
                f"as_cs_series: DataFrame 缺少必需的列 {missing_index_columns}，"
                f"无法构建因子输出索引"
            )

        # 校验 values 长度与 DataFrame 行数一致 (Validate values length matches df rows)
        if len(values) != len(df):
            raise ValueError(
                f"as_cs_series: values 长度 ({len(values)}) 与 "
                f"DataFrame 行数 ({len(df)}) 不一致"
            )

        # 构建 MultiIndex 并返回标准格式 Series (Build MultiIndex and return standard Series)
        multi_index = pd.MultiIndex.from_frame(
            df[required_index_columns], names=required_index_columns
        )
        return pd.Series(values.values, index=multi_index, name="value")
