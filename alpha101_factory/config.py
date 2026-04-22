# -*- coding: utf-8 -*-
"""Alpha101 Factory 全局配置模块。

本模块在导入时解析环境变量、校验配置值、创建数据目录，
并导出所有路径与参数常量供项目其他模块使用。

模块级别常量（全部带类型标注）:
    DATA_ROOT: 数据根目录的绝对路径。
    DIR_UNIVERSE: 股票池目录。
    DIR_QUOTES: 日线行情数据目录。
    DIR_SPOT: 行情快照目录。
    DIR_FEATURES: 中间特征缓存目录。
    DIR_FACTORS: 因子输出目录。
    DIR_BACKTEST: 回测结果目录。
    LOG_DIR: 日志目录。
    IMG_DIR: 图片根目录。
    IMG_KLINES_DIR: K 线图目录。
    IMG_BT_DIR: 回测图表目录。
    ADJUST: 复权方式（"qfq" / "hfq" / ""）。
    START_DATE: 全局抓取起始日期（YYYYMMDD）。
    END_DATE: 全局抓取结束日期（YYYYMMDD）。
    MAX_WORKERS: 并行 worker 数量。
    REQUEST_PAUSE: 请求节流间隔（秒）。
    LIMIT_STOCKS: 调试用股票数量上限（0 = 全部）。

环境变量（保持向后兼容，不可修改名称）:
    ALPHA101_DATA_ROOT, ALPHA101_ADJUST, ALPHA101_START,
    ALPHA101_END, ALPHA101_LIMIT, ALPHA101_PAUSE,
    ALPHA101_MAX_WORKERS
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# 日志配置 — 模块级 logger，用于打印配置解析信息
# ---------------------------------------------------------------------------
_logger: Final[logging.Logger] = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 辅助函数：安全解析环境变量
# ---------------------------------------------------------------------------

def _get_env_str(name: str, default: str) -> str:
    """读取字符串型环境变量，去除首尾空白。

    Args:
        name: 环境变量名称。
        default: 默认值。

    Returns:
        解析后的字符串（已 strip）。
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip()


def _get_env_int(name: str, default: int, *, minimum: int = 0) -> int:
    """读取整型环境变量，校验非负（或指定最小值）。

    Args:
        name: 环境变量名称。
        default: 默认值。
        minimum: 允许的最小值，默认 0。

    Returns:
        解析后的整数；若解析失败或低于最小值则使用默认值并记录警告。
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    try:
        value = int(raw)
    except ValueError:
        _logger.warning(
            "环境变量 %s 值 '%s' 非有效整数，使用默认值 %d",
            name, raw, default,
        )
        return default
    if value < minimum:
        _logger.warning(
            "环境变量 %s 值 %d 低于最小值 %d，使用默认值 %d",
            name, value, minimum, default,
        )
        return default
    return value


def _get_env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    """读取浮点型环境变量，校验非负（或指定最小值）。

    Args:
        name: 环境变量名称。
        default: 默认值。
        minimum: 允许的最小值，默认 0.0。

    Returns:
        解析后的浮点数；若解析失败或低于最小值则使用默认值并记录警告。
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    try:
        value = float(raw)
    except ValueError:
        _logger.warning(
            "环境变量 %s 值 '%s' 非有效浮点数，使用默认值 %.2f",
            name, raw, default,
        )
        return default
    if value < minimum:
        _logger.warning(
            "环境变量 %s 值 %.4f 低于最小值 %.2f，使用默认值 %.2f",
            name, value, minimum, default,
        )
        return default
    return value


# ---------------------------------------------------------------------------
# 日期格式校验
# ---------------------------------------------------------------------------
_DATE_PATTERN: Final = re.compile(r"^\d{8}$")


def _validate_date(value: str, env_name: str, default: str) -> str:
    """校验日期字符串是否为 YYYYMMDD 格式。

    Args:
        value: 待校验的日期字符串。
        env_name: 环境变量名称（用于日志）。
        default: 校验失败时的默认值。

    Returns:
        有效的日期字符串；否则返回默认值。
    """
    if not _DATE_PATTERN.match(value):
        _logger.warning(
            "环境变量 %s 值 '%s' 不符合 YYYYMMDD 格式，使用默认值 %s",
            env_name, value, default,
        )
        return default
    # 进一步校验日期合法性（如 20201301 不是有效日期）
    _, month, day = int(value[:4]), int(value[4:6]), int(value[6:8])
    if not (1 <= month <= 12 and 1 <= day <= 31):
        _logger.warning(
            "环境变量 %s 值 '%s' 不是有效日期，使用默认值 %s",
            env_name, value, default,
        )
        return default
    return value


# ---------------------------------------------------------------------------
# ADJUST 校验
# ---------------------------------------------------------------------------
_VALID_ADJUST_VALUES: Final[frozenset[str]] = frozenset({"qfq", "hfq", ""})


def _validate_adjust(value: str) -> str:
    """校验复权方式是否为允许值。

    Args:
        value: 待校验的复权方式字符串。

    Returns:
        有效的复权方式；否则返回默认值 "qfq" 并记录警告。
    """
    if value not in _VALID_ADJUST_VALUES:
        _logger.warning(
            "环境变量 ALPHA101_ADJUST 值 '%s' 无效，允许值: %s，使用默认值 'qfq'",
            value, ", ".join(repr(v) for v in _VALID_ADJUST_VALUES),
        )
        return "qfq"
    return value


# ---------------------------------------------------------------------------
# 路径解析与校验
# ---------------------------------------------------------------------------

def _resolve_data_root(raw: str) -> Path:
    """解析并校验数据根目录。

    尝试创建目录（含父目录），若失败则回退到当前工作目录下的 ./data。

    Args:
        raw: 环境变量提供的原始路径字符串。

    Returns:
        解析后的绝对路径 Path 对象。
    """
    try:
        root = Path(raw).resolve()
        root.mkdir(parents=True, exist_ok=True)
        # 验证可写性：尝试在目录下创建临时文件
        test_file = root / ".write_test"
        test_file.touch()
        test_file.unlink()
        return root
    except OSError as exc:
        fallback = Path("./data").resolve()
        _logger.warning(
            "无法访问或写入数据目录 '%s'（%s），回退到 '%s'",
            raw, exc, fallback,
        )
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


# ---------------------------------------------------------------------------
# 配置解析
# ---------------------------------------------------------------------------

# 数据根目录
DATA_ROOT: Final[Path] = _resolve_data_root(
    _get_env_str("ALPHA101_DATA_ROOT", "./data")
)

# --- 数据目录 ---
# 股票池目录
DIR_UNIVERSE: Final[Path] = DATA_ROOT / "universe"
# 日线行情数据目录
DIR_QUOTES: Final[Path] = DATA_ROOT / "quotes" / "daily"
# 行情快照目录
DIR_SPOT: Final[Path] = DATA_ROOT / "quotes" / "spot"
# 中间特征缓存目录
DIR_FEATURES: Final[Path] = DATA_ROOT / "features"
# 因子输出目录
DIR_FACTORS: Final[Path] = DATA_ROOT / "factors"
# 回测结果目录
DIR_BACKTEST: Final[Path] = DATA_ROOT / "backtest"
# 日志目录
LOG_DIR: Final[Path] = DATA_ROOT / "logs"

# --- 图片目录 ---
# 图片根目录
IMG_DIR: Final[Path] = DATA_ROOT / "images"
# K 线图目录
IMG_KLINES_DIR: Final[Path] = IMG_DIR / "klines"
# 回测图表目录
IMG_BT_DIR: Final[Path] = IMG_DIR / "backtest"

# 创建所有数据目录
for _dir in [
    DIR_UNIVERSE, DIR_QUOTES, DIR_SPOT, DIR_FEATURES,
    DIR_FACTORS, DIR_BACKTEST, LOG_DIR,
    IMG_DIR, IMG_KLINES_DIR, IMG_BT_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)

# --- 抓取配置 ---
# 复权方式：前复权(qfq) / 后复权(hfq) / 不复权("")
ADJUST: Final[str] = _validate_adjust(
    _get_env_str("ALPHA101_ADJUST", "qfq")
)
# 全局抓取起始日期（YYYYMMDD）
START_DATE: Final[str] = _validate_date(
    _get_env_str("ALPHA101_START", "20200101"),
    "ALPHA101_START",
    "20200101",
)
# 全局抓取结束日期（YYYYMMDD）
END_DATE: Final[str] = _validate_date(
    _get_env_str("ALPHA101_END", "20250917"),
    "ALPHA101_END",
    "20250917",
)

# --- 并发 / 限流 ---
# 并行 worker 数量
MAX_WORKERS: Final[int] = _get_env_int("ALPHA101_MAX_WORKERS", 1, minimum=1)
# 请求节流间隔（秒）
REQUEST_PAUSE: Final[float] = _get_env_float("ALPHA101_PAUSE", 0.6, minimum=0.0)
# 调试用股票数量上限（0 = 全部）
LIMIT_STOCKS: Final[int] = _get_env_int("ALPHA101_LIMIT", 0, minimum=0)

# ---------------------------------------------------------------------------
# 打印活跃配置信息（便于调试与审计）
# ---------------------------------------------------------------------------
_CONFIG_SUMMARY: Final[str] = (
    f"Alpha101 Factory 配置已加载:\n"
    f"  DATA_ROOT      = {DATA_ROOT}\n"
    f"  ADJUST         = {ADJUST!r}\n"
    f"  START_DATE     = {START_DATE}\n"
    f"  END_DATE       = {END_DATE}\n"
    f"  MAX_WORKERS    = {MAX_WORKERS}\n"
    f"  REQUEST_PAUSE  = {REQUEST_PAUSE}s\n"
    f"  LIMIT_STOCKS   = {LIMIT_STOCKS}"
)
_logger.info(_CONFIG_SUMMARY)
