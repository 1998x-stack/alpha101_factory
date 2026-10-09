# -*- coding: utf-8 -*-
"""股票池（Universe）加载模块。

从 ``universe/stocks.jsonl`` 文件中读取股票池数据，返回股票代码序列。
该模块为 Pipeline 阶段和批量操作提供股票列表，是整个数据流水线的入口之一。

核心特性:
    - 从 JSONL 格式的股票池文件加载股票代码和名称
    - 支持 ``limit`` 参数限制返回的股票数量（0 表示不限制）
    - 股票代码自动补零至 6 位（如 ``"600000"``）
    - 完善的边界条件检查：文件不存在、空文件、畸形数据、权限错误等
    - 详细的日志输出：加载数量、文件路径、警告信息

典型用法::

    from alpha101_factory.data.universe import load_universe

    # 加载全部股票
    all_symbols = load_universe()

    # 仅加载前 100 只股票（用于调试）
    debug_symbols = load_universe(limit=100)

注意:
    股票池文件由 ``fetch`` 命令自动生成，格式为每行一个 JSON 对象：
    ``{"code": "600000", "name": "浦发银行"}``
"""

from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd

from alpha101_factory.config import DIR_UNIVERSE
from alpha101_factory.utils.io import read_jsonl

# 模块级 logger
_logger: logging.Logger = logging.getLogger(__name__)

# 股票池文件路径
_STOCKS_FILE_PATH: Path = DIR_UNIVERSE / "stocks.jsonl"

# 必需的列名
_REQUIRED_COLUMNS: frozenset[str] = frozenset({"code", "name"})

# 股票代码标准长度
_SYMBOL_LENGTH: int = 6


def load_universe(limit: int = 0) -> pd.Series:
    """加载股票池中的股票代码列表。

    从 ``universe/stocks.jsonl`` 文件中读取股票数据，提取 ``code`` 列，
    补零至 6 位后去重，返回为 ``pd.Series``。

    当股票池文件不存在、为空或数据格式异常时，返回空序列而非抛出异常，
    以保证 Pipeline 的健壮性。

    Args:
        limit: 限制返回的股票数量。
            - ``0``（默认值）：返回全部股票
            - ``> 0``：仅返回前 ``limit`` 只股票
            - ``< 0``：视为无效参数，记录警告并返回空序列

    Returns:
        pd.Series: 股票代码序列，dtype 为 ``str``，name 为 ``"symbol"``。
            如果文件不存在、为空或解析失败，返回空序列。

    Examples:
        加载全部股票::

            >>> symbols = load_universe()
            >>> print(len(symbols))
            5000

        仅加载前 100 只股票用于调试::

            >>> debug_symbols = load_universe(limit=100)
            >>> print(len(debug_symbols))
            100
    """
    # 校验 limit 参数：负数视为无效
    if limit < 0:
        _logger.warning(
            "load_universe: limit 参数为负数 (%d)，返回空股票池",
            limit,
        )
        return _create_empty_series()

    # 检查股票池文件是否存在
    if not _STOCKS_FILE_PATH.exists():
        _logger.warning(
            "股票池文件不存在: %s，返回空股票池",
            _STOCKS_FILE_PATH,
        )
        return _create_empty_series()

    # 检查是否为文件（而非目录）
    if not _STOCKS_FILE_PATH.is_file():
        _logger.warning(
            "股票池路径不是文件: %s，返回空股票池",
            _STOCKS_FILE_PATH,
        )
        return _create_empty_series()

    # 读取 JSONL 文件（read_jsonl 内部已处理权限错误、畸形 JSON 等）
    stock_dataframe: pd.DataFrame = read_jsonl(_STOCKS_FILE_PATH)

    # 处理空文件情况
    if stock_dataframe.empty:
        _logger.info("股票池文件为空: %s", _STOCKS_FILE_PATH)
        return _create_empty_series()

    # 校验必需的列是否存在
    missing_columns: set[str] = _REQUIRED_COLUMNS - set(stock_dataframe.columns)
    if missing_columns:
        _logger.warning(
            "股票池文件缺少必需列 %s，文件: %s，返回空股票池",
            sorted(missing_columns),
            _STOCKS_FILE_PATH,
        )
        return _create_empty_series()

    # 提取股票代码，补零至 6 位，去重
    symbol_codes: pd.Series = (
        stock_dataframe["code"]
        .astype(str)
        .str.zfill(_SYMBOL_LENGTH)
        .unique()
    )

    # 应用 limit 限制
    if limit > 0:
        symbol_codes = symbol_codes[:limit]

    # 构建结果 Series
    result_series: pd.Series = pd.Series(symbol_codes, name="symbol")

    # 打印加载信息
    _logger.info(
        "成功加载 %d 只股票 ← %s%s",
        len(result_series),
        _STOCKS_FILE_PATH,
        f" (limit={limit})" if limit > 0 else "",
    )

    return result_series


def _create_empty_series() -> pd.Series:
    """创建空的股票代码序列。

    Returns:
        pd.Series: 空的 Series，dtype 为 ``str``，name 为 ``"symbol"``。
    """
    return pd.Series([], dtype=str, name="symbol")


if __name__ == "__main__":
    from pprint import pprint

    # 测试加载前 100 只股票
    test_codes: pd.Series = load_universe(limit=100)
    pprint(test_codes)
