# -*- coding: utf-8 -*-
"""通用验证和公共工具模块 (Common Validation and Utility Functions)

本模块集中了在多个地方使用的验证函数和公共工具，以减少代码重复。
主要包括日期验证、股票代码验证等功能。
"""

from __future__ import annotations

import re
from typing import Union
import argparse
import pandas as pd
from loguru import logger


# 股票代码校验正则：恰好 6 位数字
STOCK_CODE_PATTERN: re.Pattern[str] = re.compile(r"^\d{6}$")

# 日期格式正则：YYYYMMDD
DATE_PATTERN: re.Pattern[str] = re.compile(r"^\d{8}$")

# 合法的复权方式集合
VALID_ADJUST_VALUES: frozenset[str] = frozenset({"qfq", "hfq", ""})


def is_valid_stock_code(code: str) -> bool:
    """校验股票代码格式是否为 6 位纯数字。

    Args:
        code: 待校验的股票代码字符串。

    Returns:
        bool: 符合 6 位数字格式返回 True，否则返回 False。
    """
    return bool(STOCK_CODE_PATTERN.match(code))


def is_valid_adjust(adjust: str) -> bool:
    """校验复权方式是否为允许值。

    Args:
        adjust: 待校验的复权方式字符串。

    Returns:
        bool: 为 "qfq"、"hfq" 或空字符串时返回 True，否则返回 False。
    """
    return adjust in VALID_ADJUST_VALUES


def normalize_stock_code(code: str) -> str:
    """提取股票代码中的纯数字部分并补齐至 6 位。

    过滤掉非数字字符后，使用零填充至 6 位长度。
    若提取后无有效数字，返回空字符串。

    Args:
        code: 原始股票代码（可能包含非数字字符）。

    Returns:
        str: 规范化后的 6 位数字股票代码；若无有效数字则返回空字符串。
    """
    digits_only: str = "".join(filter(str.isdigit, code))
    return digits_only.zfill(6) if digits_only else ""


def validate_stock_code_arg(code: str) -> str:
    """校验股票代码格式用于命令行参数验证。

    Args:
        code: 待校验的股票代码字符串。

    Returns:
        校验通过的原始股票代码。

    Raises:
        argparse.ArgumentTypeError: 当代码非 6 位纯数字时抛出。
    """
    if not STOCK_CODE_PATTERN.match(code):
        raise argparse.ArgumentTypeError(
            f"无效的股票代码 '{code}'，必须为 6 位纯数字（如 600000）"
        )
    return code


def validate_date_format(date_str: str) -> str:
    """校验日期格式是否为 YYYYMMDD。

    Args:
        date_str: 待校验的日期字符串。

    Returns:
        校验通过的原始日期字符串。

    Raises:
        argparse.ArgumentTypeError: 当格式不匹配时抛出。
    """
    if not DATE_PATTERN.match(date_str):
        raise argparse.ArgumentTypeError(
            f"无效的日期格式 '{date_str}'，必须为 YYYYMMDD 格式"
        )
    return date_str


def validate_adjust_mode(adjust: str) -> str:
    """校验复权方式参数。

    Args:
        adjust: 待校验的复权方式字符串。

    Returns:
        校验通过的原始复权方式字符串。

    Raises:
        argparse.ArgumentTypeError: 当值不在合法集合中时抛出。
    """
    if adjust not in VALID_ADJUST_VALUES:
        raise argparse.ArgumentTypeError(
            f"无效的复权方式 '{adjust}'，可选值: qfq（前复权）/ hfq（后复权）/ ''（不复权）"
        )
    return adjust


def is_valid_date_format(date_str: str) -> bool:
    """校验日期字符串是否为 YYYYMMDD 格式。

    Args:
        date_str: 待校验的日期字符串。

    Returns:
        bool: 格式正确返回 True，否则返回 False。
    """
    return bool(DATE_PATTERN.match(date_str))


def format_date_for_baostock(date_string: Union[str, None]) -> Union[str, None]:
    """将 ``YYYYMMDD`` 格式日期转换为 BaoStock 所需的 ``YYYY-MM-DD`` 格式。

    Args:
        date_string: ``YYYYMMDD`` 格式的日期字符串，或 ``None``。

    Returns:
        ``YYYY-MM-DD`` 格式的日期字符串，或 ``None``。
        若输入格式不正确，返回 ``None`` 并记录警告日志。
    """
    if date_string is None:
        return None

    # 校验日期格式长度
    if len(date_string) != 8:
        logger.warning(
            f"BaoStock 日期格式不正确: '{date_string}'，期望 8 位数字"
        )
        return None

    try:
        year: str = date_string[:4]
        month: str = date_string[4:6]
        day: str = date_string[6:]
        return f"{year}-{month}-{day}"
    except (ValueError, IndexError) as exception:
        logger.warning(f"BaoStock 日期解析失败: '{date_string}'，错误: {exception}")
        return None


def convert_to_baostock_code(symbol: str) -> str:
    """将股票代码转换为 BaoStock 格式。

    Args:
        symbol: 原始股票代码，如 ``"600000"``。

    Returns:
        BaoStock 格式的代码，如 ``"sh.600000"`` 或 ``"sz.000001"``。
    """
    numeric_code: str = str(symbol).zfill(6)
    if numeric_code.startswith("6"):
        return f"sh.{numeric_code}"
    return f"sz.{numeric_code}"


def convert_adjust_mode(adjust: str) -> str:
    """将复权方式字符串转换为 BaoStock 的 adjustflag 参数值。

    Args:
        adjust: 复权方式，``"qfq"``（前复权）、``"hfq"``（后复权）或 ``""``（不复权）。

    Returns:
        BaoStock 的 adjustflag 值：``"1"``（后复权）、``"2"``（前复权）、``"3"``（不复权）。
    """
    adjust_mode_mapping: dict[str, str] = {"hfq": "1", "qfq": "2"}
    return adjust_mode_mapping.get(adjust, "3")