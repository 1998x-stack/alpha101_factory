# -*- coding: utf-8 -*-
"""数据源模块：A 股日线行情数据的获取与标准化。

本模块定义了数据源的抽象基类 ``DataSource``，并提供两个具体实现：
- ``AkShareSource``：基于 AkShare 库，优先使用的数据源。
- ``BaoStockSource``：基于 BaoStock 库，作为 AkShare 失败时的降级方案。

所有数据源返回的 DataFrame 均经过 ``_normalize()`` 函数统一处理，
确保列名、数据类型和排序的一致性。

典型用法::

    from alpha101_factory.data.sources import AkShareSource, BaoStockSource

    source = AkShareSource()
    df = source.fetch_kline("600000", "20240101", "20241231", "qfq")
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Optional

import akshare as ak
import pandas as pd
from loguru import logger

from alpha101_factory.data.baostock_api import login, logout, fetch_stock_data
from alpha101_factory.utils.validation import (
    is_valid_stock_code,
    is_valid_adjust,
    normalize_stock_code,
    convert_to_baostock_code,
    convert_adjust_mode,
    format_date_for_baostock
)

# 代理配置：默认禁用代理避免 EastMoney 代理拦截
# 设置 ALPHA101_NO_PROXY=0 可保留系统代理（适用于企业内网）
if os.environ.get("ALPHA101_NO_PROXY", "1") != "0":
    for _proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(_proxy_var, None)
    # 强制所有请求绕过代理（requests 库会读取此变量）
    os.environ["NO_PROXY"] = "*"

# 重试配置
_MAX_RETRIES: int = 2
_BASE_DELAY: float = 1.0
_REQUEST_TIMEOUT: int = 15

# 中文列名到英文列名的映射表
_COLUMN_NAME_MAPPING: dict[str, str] = {
    "日期": "datetime",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
    "振幅": "amplitude",
    "换手率": "turnover",
}

# 需要转换为数值类型的列名列表
_NUMERIC_COLUMNS: list[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "pct_change",
    "change",
    "amplitude",
    "turnover",
]


def _normalize(dataframe: Optional[pd.DataFrame]) -> pd.DataFrame:
    """将原始行情 DataFrame 标准化为统一的列名和数据类型。

    该函数执行以下操作：
    1. 将中文列名映射为英文列名。
    2. 将 ``datetime`` 列转换为 ``datetime64`` 类型并按时间升序排序。
    3. 将数值列转换为 ``float64`` 类型，无法转换的值设为 ``NaN``。

    Args:
        dataframe: 原始行情 DataFrame，可能为 ``None`` 或空。

    Returns:
        标准化后的 DataFrame。若输入为 ``None`` 或空，则原样返回。
    """
    if dataframe is None or dataframe.empty:
        return dataframe

    # 步骤 1：映射中文列名到英文列名（仅映射存在的列）
    existing_mapping = {
        chinese_name: english_name
        for chinese_name, english_name in _COLUMN_NAME_MAPPING.items()
        if chinese_name in dataframe.columns
    }
    if existing_mapping:
        dataframe = dataframe.rename(columns=existing_mapping)

    # 步骤 2：处理 datetime 列
    if "datetime" in dataframe.columns:
        dataframe["datetime"] = pd.to_datetime(dataframe["datetime"])
        dataframe = dataframe.sort_values("datetime").reset_index(drop=True)

    # 步骤 3：转换数值列
    for column_name in _NUMERIC_COLUMNS:
        if column_name in dataframe.columns:
            dataframe[column_name] = pd.to_numeric(
                dataframe[column_name], errors="coerce"
            )

    return dataframe


class DataSource(ABC):
    """数据源抽象基类。

    所有具体数据源必须继承此类并实现两个抽象方法：
    - ``fetch_kline()``：获取单只股票的日线 K 线数据。
    - ``fetch_spot()``：获取全市场当日行情快照。

    子类应设置 ``name`` 类属性作为数据源的唯一标识。
    """

    name: str = "base"

    @abstractmethod
    def fetch_kline(
        self,
        symbol: str,
        start_date: Optional[str],
        end_date: Optional[str],
        adjust: str,
    ) -> pd.DataFrame:
        """获取单只股票的日线 K 线数据。

        Args:
            symbol: 股票代码，如 ``"600000"`` 或 ``"sh.600000"``。
            start_date: 起始日期，格式为 ``"YYYYMMDD"``。为 ``None`` 时不限制。
            end_date: 结束日期，格式为 ``"YYYYMMDD"``。为 ``None`` 时不限制。
            adjust: 复权方式，可选值为 ``"qfq"``（前复权）、``"hfq"``（后复权）
                或 ``""``（不复权）。

        Returns:
            标准化后的 K 线 DataFrame，包含列：
            ``datetime, open, high, low, close, volume, amount`` 等。
            若获取失败或无数据，返回空 DataFrame。
        """
        ...

    @abstractmethod
    def fetch_spot(self) -> pd.DataFrame:
        """获取全市场当日行情快照。

        Returns:
            包含 ``code`` 和 ``name`` 列的 DataFrame。
            若获取失败，返回空 DataFrame。
        """
        ...


class AkShareSource(DataSource):
    """基于 AkShare 库的 A 股数据源。

    AkShare 是优先使用的数据源，提供丰富的 A 股行情数据。
    该实现调用 ``ak.stock_zh_a_hist()`` 获取历史 K 线，
    调用 ``ak.stock_zh_a_spot_em()`` 获取实时行情快照（东方财富版，支持分页）。
    """

    name: str = "akshare"

    def fetch_kline(
        self,
        symbol: str,
        start_date: Optional[str],
        end_date: Optional[str],
        adjust: str,
    ) -> pd.DataFrame:
        # 提取纯数字股票代码
        numeric_symbol: str = "".join(filter(str.isdigit, symbol))

        # 参数校验：股票代码不能为空
        if not numeric_symbol or not is_valid_stock_code(numeric_symbol):
            logger.error(f"AkShare 股票代码为空或格式无效: {symbol}")
            return pd.DataFrame()

        # 参数校验：复权方式必须合法
        if not is_valid_adjust(adjust):
            logger.error(
                f"AkShare 非法复权方式: '{adjust}'，"
                f"可选值: {sorted(['qfq', 'hfq', ''])}"
            )
            return pd.DataFrame()

        logger.info(
            f"AkShare 获取 {numeric_symbol} "
            f"{start_date or '起始'}~{end_date or '结束'} {adjust or '不复权'}"
        )

        # 构建 AkShare API 调用参数
        request_parameters: dict[str, str] = {
            "symbol": numeric_symbol,
            "period": "daily",
            "adjust": adjust,
        }
        if start_date and end_date:
            request_parameters["start_date"] = start_date
            request_parameters["end_date"] = end_date
        elif start_date or end_date:
            logger.warning(
                f"AkShare 日期参数不完整: start_date={start_date}, "
                f"end_date={end_date}，将忽略日期过滤"
            )

        # 带重试的请求逻辑
        last_exception: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES + 1):  # Include original attempt
            try:
                request_parameters["timeout"] = _REQUEST_TIMEOUT
                raw_dataframe: pd.DataFrame = ak.stock_zh_a_hist(**request_parameters)

                # Check if the returned dataframe is valid
                if raw_dataframe is None or raw_dataframe.empty:
                    logger.warning(f"AkShare {numeric_symbol} 返回空数据")
                    return pd.DataFrame()

                break  # Success, break out of retry loop
            except Exception as exception:
                last_exception = exception
                if attempt < _MAX_RETRIES:  # Still have retries left
                    import random as _random
                    jitter = _random.uniform(0.5, 1.5)
                    delay = _BASE_DELAY * (2 ** attempt) * jitter
                    logger.warning(
                        f"AkShare 第 {attempt + 1} 次尝试失败 ({type(exception).__name__}: {str(exception)[:100]}...), "
                        f"{delay:.1f}s 后重试"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"AkShare K 线获取失败 (已重试 {_MAX_RETRIES} 次): "
                        f"{symbol}，错误: {exception}"
                    )
                    # Don't raise the exception here, return empty DataFrame to allow fallback
                    return pd.DataFrame()
        else:
            # This should theoretically never be reached, but as backup:
            logger.error(f"AkShare K 线获取失败: {symbol}")
            return pd.DataFrame()

        normalized_dataframe: pd.DataFrame = _normalize(raw_dataframe)

        # 打印获取结果统计信息
        if normalized_dataframe.empty:
            logger.warning(f"AkShare {numeric_symbol} 返回空数据")
        else:
            record_count: int = len(normalized_dataframe)
            logger.info(
                f"AkShare {numeric_symbol} 获取 {record_count} 条记录，"
                f"时间范围: {normalized_dataframe['datetime'].iloc[0].date()} ~ "
                f"{normalized_dataframe['datetime'].iloc[-1].date()}"
            )

        return normalized_dataframe

    def fetch_spot(self) -> pd.DataFrame:
        """获取全市场当日行情快照。

        Returns:
            包含 ``code`` 和 ``name`` 列的行情快照 DataFrame。
        """
        logger.info("AkShare 获取全市场行情快照")
        try:
            # 使用 stock_zh_a_spot_em()（东方财富版），支持分页且更稳定
            raw_dataframe: pd.DataFrame = ak.stock_zh_a_spot_em()
            # 注意：stock_zh_a_spot_em 内部使用 fetch_paginated_data，
            # 该函数自带超时和重试机制，无需额外设置 timeout
        except Exception as exception:
            logger.warning(f"AkShare 行情快照获取失败: {exception}")
            return pd.DataFrame()

        if raw_dataframe.empty:
            logger.warning("AkShare 行情快照返回空数据")
            return raw_dataframe

        # 映射中文列名到英文列名
        column_renaming: dict[str, str] = {"代码": "code", "名称": "name"}
        existing_renaming = {
            chinese: english
            for chinese, english in column_renaming.items()
            if chinese in raw_dataframe.columns
        }

        if existing_renaming:
            result_dataframe: pd.DataFrame = raw_dataframe.rename(
                columns=existing_renaming
            )
            stock_count: int = len(result_dataframe)
            logger.info(f"AkShare 行情快照获取 {stock_count} 只股票")
        else:
            logger.error("AkShare 行情快照缺少 '代码'/'名称' 列，列名可能已变更")
            return pd.DataFrame()

        return result_dataframe


class BaoStockSource(DataSource):
    """基于 BaoStock 库的 A 股数据源（降级方案）。

    BaoStock 作为 AkShare 失败时的备用数据源。
    该实现通过 ``bs.query_history_k_data_plus()`` 获取历史 K 线数据。

    注意：
    - BaoStock 需要在每次请求前调用 ``bs.login()``，请求后调用 ``bs.logout()``。
    - BaoStock 不支持实时行情快照，``fetch_spot()`` 始终返回空 DataFrame。
    - BaoStock 股票代码格式为 ``sh.XXXXXX``（上海）或 ``sz.XXXXXX``（深圳）。
    """

    name: str = "baostock"

    @staticmethod
    def _convert_to_baostock_code(symbol: str) -> str:
        """将股票代码转换为 BaoStock 格式。

        Args:
            symbol: 原始股票代码，如 ``"600000"``。

        Returns:
            BaoStock 格式的代码，如 ``"sh.600000"`` 或 ``"sz.000001"``。
        """
        return convert_to_baostock_code(symbol)

    @staticmethod
    def _convert_adjust_mode(adjust: str) -> str:
        """将复权方式字符串转换为 BaoStock 的 adjustflag 参数值。

        Args:
            adjust: 复权方式，``"qfq"``（前复权）、``"hfq"``（后复权）或 ``""``（不复权）。

        Returns:
            BaoStock 的 adjustflag 值：``"1"``（后复权）、``"2"``（前复权）、``"3"``（不复权）。
        """
        return convert_adjust_mode(adjust)

    @staticmethod
    def _format_date_for_baostock(date_string: Optional[str]) -> Optional[str]:
        """将 ``YYYYMMDD`` 格式日期转换为 BaoStock 所需的 ``YYYY-MM-DD`` 格式。

        Args:
            date_string: ``YYYYMMDD`` 格式的日期字符串，或 ``None``。

        Returns:
            ``YYYY-MM-DD`` 格式的日期字符串，或 ``None``。
            若输入格式不正确，返回 ``None`` 并记录警告日志。
        """
        return format_date_for_baostock(date_string)

    def fetch_kline(
        self,
        symbol: str,
        start_date: Optional[str],
        end_date: Optional[str],
        adjust: str,
    ) -> pd.DataFrame:
        # 提取纯数字股票代码
        numeric_symbol: str = "".join(filter(str.isdigit, symbol))

        # 参数校验：股票代码不能为空
        if not numeric_symbol or not is_valid_stock_code(numeric_symbol):
            logger.error(f"BaoStock 股票代码为空或格式无效: {symbol}")
            return pd.DataFrame()

        # 参数校验：复权方式必须合法
        if not is_valid_adjust(adjust):
            logger.error(
                f"BaoStock 非法复权方式: '{adjust}'，"
                f"可选值: {sorted(['qfq', 'hfq', ''])}"
            )
            return pd.DataFrame()

        logger.info(f"BaoStock 获取 {numeric_symbol}")

        # 转换日期格式
        formatted_start_date: Optional[str] = self._format_date_for_baostock(
            start_date
        )
        formatted_end_date: Optional[str] = self._format_date_for_baostock(end_date)

        # 转换 BaoStock 代码格式
        baostock_code: str = self._convert_to_baostock_code(numeric_symbol)

        # 转换复权标志
        adjust_flag: str = self._convert_adjust_mode(adjust)

        try:
            # 使用 baostock_api 模块获取数据
            data_frame: Optional[pd.DataFrame] = fetch_stock_data(
                code=baostock_code,
                start_date=formatted_start_date,
                end_date=formatted_end_date,
                adjustflag=adjust_flag
            )

            if data_frame is None or data_frame.empty:
                logger.warning(f"BaoStock {numeric_symbol} 无数据返回")
                return pd.DataFrame()

            # 打印获取结果统计信息
            record_count: int = len(data_frame)
            if 'datetime' in data_frame.columns:
                logger.info(
                    f"BaoStock {numeric_symbol} 获取 {record_count} 条记录，"
                    f"时间范围: {data_frame['datetime'].iloc[0].date()} ~ "
                    f"{data_frame['datetime'].iloc[-1].date()}"
                )

            return data_frame

        except Exception as exception:
            logger.exception(f"BaoStock 查询异常: {baostock_code}，错误: {exception}")
            return pd.DataFrame()

    def fetch_spot(self) -> pd.DataFrame:
        """获取全市场当日行情快照。

        BaoStock 不支持实时行情快照功能。

        Returns:
            空 DataFrame。
        """
        logger.info("BaoStock 不支持行情快照功能，返回空数据")
        return pd.DataFrame()
