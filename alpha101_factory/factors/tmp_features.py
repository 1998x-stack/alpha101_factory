# -*- coding: utf-8 -*-
"""构建与加载中间特征 (Temporary Features) 模块。

本模块负责从 K 线行情数据中预计算中间特征（如收益率、VWAP、不同窗口的
平均成交量 ADV），并以 JSONL 格式缓存到 ``features/`` 目录。这些中间特征
作为因子计算的前置输入，避免在每次因子计算时重复执行滚动窗口运算。

主要功能：
    1. 为单只股票构建中间特征文件 (``build_tmp_for_symbol``)；
    2. 批量构建多只股票的中间特征文件 (``build_tmp_all``)；
    3. 加载并合并多只股票的特征表为面板数据 (``load_panel``)。

数据流向::

    quotes/daily/{symbol}.jsonl  ──▶  features/{symbol}.jsonl
    (K 线原始数据)                    (中间特征缓存)

特征列表：
    - ``returns``: 收盘价收益率 (pct_change)
    - ``vwap``: 成交量加权平均价 (amount / volume)
    - ``adv{N}``: N 日平均成交量 (N ∈ ADV_WINDOWS)

注意事项：
    - 本模块不抛出异常，所有错误通过 loguru 记录并返回 False/空值；
    - 单只股票构建失败不影响批量流程中的其他股票；
    - ``load_panel`` 在无任何有效数据时返回空 DataFrame。

典型用法::

    from alpha101_factory.factors.tmp_features import (
        build_tmp_for_symbol,
        build_tmp_all,
        load_panel,
        ADV_WINDOWS,
    )

    # 构建单只股票
    success = build_tmp_for_symbol("600000")

    # 批量构建
    n_success = build_tmp_all(["600000", "000001", "600519"])

    # 加载面板数据
    panel_df = load_panel()  # 自动扫描 features/ 目录
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final, Sequence

import pandas as pd
from loguru import logger
from tqdm import tqdm

# 确保项目根目录位于 sys.path 中，以便从任意入口点导入
try:
    _PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
except Exception as _exc:
    raise RuntimeError("无法设置 sys.path，请检查项目目录结构") from _exc

# 项目内部依赖
from alpha101_factory.config import DIR_FEATURES, DIR_QUOTES
from alpha101_factory.utils.io import read_jsonl, write_jsonl
from alpha101_factory.utils import ops

# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

# K 线数据中必须存在的列名集合
_REQUIRED_KLINE_COLUMNS: Final[frozenset[str]] = frozenset(
    {"open", "high", "low", "close", "volume", "amount", "datetime", "symbol"}
)

# 需要保存到特征文件中的列名顺序
_FEATURE_OUTPUT_COLUMNS: Final[list[str]] = [
    "symbol",
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "returns",
    "vwap",
]

# ADV（平均成交量）计算所用的滚动窗口列表（单位：交易日）
ADV_WINDOWS: Final[list[int]] = [5, 10, 20, 30, 40, 60, 120, 150, 180]

# 在模块加载时校验 ADV_WINDOWS 中的窗口值均为正整数
for _adv_window in ADV_WINDOWS:
    if _adv_window <= 0:
        raise ValueError(
            f"ADV_WINDOWS 包含非正整数值 {_adv_window}，窗口大小必须为正整数"
        )

# 将 ADV 列名追加到输出列列表中
_FEATURE_OUTPUT_COLUMNS.extend(f"adv{n}" for n in ADV_WINDOWS)

# 计算型特征列名（不含 OHLCV 基础列）
_COMPUTED_FEATURE_NAMES: Final[list[str]] = [
    "returns",
    "vwap",
    *[f"adv{n}" for n in ADV_WINDOWS],
]


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------


def _get_kline_file_path(symbol: str) -> Path:
    """获取指定股票的 K 线数据文件路径。

    Args:
        symbol: 股票代码（如 "600000"）。

    Returns:
        指向 ``quotes/daily/{symbol}.jsonl`` 的 Path 对象。
    """
    return DIR_QUOTES / f"{symbol}.jsonl"


def _get_feature_file_path(symbol: str) -> Path:
    """获取指定股票的中间特征文件路径。

    Args:
        symbol: 股票代码（如 "600000"）。

    Returns:
        指向 ``features/{symbol}.jsonl`` 的 Path 对象。
    """
    return DIR_FEATURES / f"{symbol}.jsonl"


def _validate_kline_dataframe(data: pd.DataFrame, symbol: str) -> bool:
    """校验 K 线 DataFrame 是否包含所有必需列。

    逐列检查 ``_REQUIRED_KLINE_COLUMNS`` 中定义的字段是否存在于
    DataFrame 的列索引中。若缺少任一列，记录错误日志并返回 False。

    Args:
        data: 待校验的 K 线数据 DataFrame。
        symbol: 股票代码（用于日志输出）。

    Returns:
        True 表示所有必需列均存在；False 表示存在缺失列。
    """
    missing_columns = _REQUIRED_KLINE_COLUMNS - set(data.columns)
    if missing_columns:
        logger.error(
            "股票 {} 的 K 线数据缺少必要列: {}",
            symbol,
            ", ".join(sorted(missing_columns)),
        )
        return False
    return True


def _compute_features_for_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """在 DataFrame 上计算所有中间特征（原地修改）。

    计算的特征包括：
        - ``returns``: 收盘价收益率
        - ``vwap``: 成交量加权平均价
        - ``adv{N}``: 各窗口下的平均成交量

    Args:
        data: 已按时间排序的 K 线数据 DataFrame。

    Returns:
        添加了特征列的 DataFrame（与输入为同一对象）。

    Raises:
        Exception: 特征计算过程中出现的任何异常均向上抛出，
            由调用方决定是否捕获。
    """
    # 计算收益率：相邻收盘价的百分比变化
    data["returns"] = ops.returns(data["close"])

    # 计算 VWAP：成交额 / 成交量（零成交量自动处理为 NaN）
    data["vwap"] = ops.vwap_from_amount(
        data["close"], data["high"], data["low"], data["volume"], data["amount"]
    )

    # 计算各窗口下的平均成交量
    for window_size in ADV_WINDOWS:
        data[f"adv{window_size}"] = ops.adv(data["volume"], window_size)

    return data


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------


def build_tmp_for_symbol(symbol: str) -> bool:
    """为单只股票构建中间特征文件。

    从 ``quotes/daily/{symbol}.jsonl`` 读取 K 线数据，计算收益率、
    VWAP 及各窗口 ADV，将结果写入 ``features/{symbol}.jsonl``。

    处理流程：
        1. 检查 K 线文件是否存在；
        2. 读取并解析 JSONL 数据；
        3. 校验必需列是否齐全；
        4. 按时间排序；
        5. 计算中间特征；
        6. 写入特征文件（含元数据行）。

    Args:
        symbol: 股票代码（如 "600000"）。

    Returns:
        True 表示特征文件成功生成；False 表示任一环节失败
        （文件不存在、数据为空、缺少列、计算异常、写入异常）。

    Notes:
        - 本函数不抛出异常，所有错误通过 loguru 记录；
        - 若特征文件已存在，将被覆盖写入。
    """
    # 步骤 1：检查 K 线文件是否存在
    kline_path = _get_kline_file_path(symbol)
    if not kline_path.exists():
        logger.warning("K 线文件不存在: {}", kline_path)
        return False

    if not kline_path.is_file():
        logger.warning("K 线路径不是文件: {}", kline_path)
        return False

    # 步骤 2：读取 K 线数据
    try:
        kline_data = read_jsonl(kline_path, parse_dates=["datetime"], skip_meta=True)
    except Exception as exc:
        logger.error("读取 K 线数据失败 (股票 {}): {}", symbol, exc)
        return False

    # 步骤 3：检查数据是否为空
    if kline_data.empty:
        logger.warning("K 线数据为空 (股票 {})", symbol)
        return False

    # 步骤 4：校验必需列
    if not _validate_kline_dataframe(kline_data, symbol):
        return False

    # 步骤 5：按时间排序
    kline_data = kline_data.sort_values("datetime").reset_index(drop=True)

    # 步骤 6：计算中间特征
    try:
        kline_data = _compute_features_for_dataframe(kline_data)
    except Exception as exc:
        logger.error("计算中间特征失败 (股票 {}): {}", symbol, exc)
        return False

    # 步骤 7：写入特征文件
    feature_path = _get_feature_file_path(symbol)
    try:
        # 构建元数据信息，便于后续审计与调试
        feature_metadata = {
            "symbol": symbol,
            "rows": len(kline_data),
            "features": _COMPUTED_FEATURE_NAMES,
        }
        write_jsonl(
            kline_data[_FEATURE_OUTPUT_COLUMNS],
            feature_path,
            meta=feature_metadata,
        )
        logger.debug(
            "成功生成特征文件: {} ({} 行, {} 个特征)",
            symbol,
            len(kline_data),
            len(_COMPUTED_FEATURE_NAMES),
        )
        return True
    except Exception as exc:
        logger.error("保存特征文件失败 (股票 {}): {}", symbol, exc)
        return False


def build_tmp_all(symbols: Sequence[str]) -> int:
    """批量构建多只股票的中间特征文件。

    遍历股票代码列表，逐只调用 ``build_tmp_for_symbol``。单只股票构建
    失败不会中断整体流程，仅记录错误日志并继续处理下一只。

    Args:
        symbols: 股票代码序列（如 ``["600000", "000001"]``）。

    Returns:
        成功生成特征文件的股票数量。

    Notes:
        - 使用 tqdm 显示处理进度条；
        - 若 ``symbols`` 为空序列，直接返回 0；
        - 处理完成后通过 logger.info 输出汇总统计。
    """
    if not symbols:
        logger.info("未提供股票代码列表，跳过批量构建")
        return 0

    total_count = len(symbols)
    success_count = 0

    logger.info("开始批量构建中间特征，共 {} 只股票", total_count)

    for symbol in tqdm(symbols, desc="构建中间特征", unit="只"):
        try:
            if build_tmp_for_symbol(symbol):
                success_count += 1
        except Exception as exc:
            logger.error("处理股票 {} 时发生未预期异常: {}", symbol, exc)
            continue

    failure_count = total_count - success_count
    logger.info(
        "批量构建完成: 成功 {} 只, 失败 {} 只, 总计 {} 只",
        success_count,
        failure_count,
        total_count,
    )
    return success_count


def load_panel(symbols: list[str] | None = None) -> pd.DataFrame:
    """加载并合并多只股票的中间特征文件为面板数据。

    从 ``features/`` 目录读取指定股票（或全部）的特征 JSONL 文件，
    拼接为长表形式的 DataFrame，并按 ``(datetime, symbol)`` 排序。

    Args:
        symbols: 股票代码列表。若为 ``None``，则自动扫描
            ``DIR_FEATURES`` 目录下所有 ``*.jsonl`` 文件，
            以文件 stem 作为股票代码。

    Returns:
        合并后的面板数据 DataFrame，包含所有股票的特征列。
        按 ``(datetime, symbol)`` 升序排列，索引已重置。
        若未加载到任何有效数据，返回空 DataFrame。

    Notes:
        - 单个文件读取失败不影响其他文件的加载；
        - 空文件自动跳过；
        - 返回的 DataFrame 列集合取决于各文件中实际存在的列，
          不强制要求所有文件列一致。
    """
    # 自动发现股票代码（扫描 features/ 目录）
    if symbols is None:
        symbols = [path.stem for path in DIR_FEATURES.glob("*.jsonl")]
        if not symbols:
            logger.warning("特征目录 {} 中未找到任何 JSONL 文件", DIR_FEATURES)
            return pd.DataFrame()

    data_frames: list[pd.DataFrame] = []

    for symbol in symbols:
        feature_path = _get_feature_file_path(symbol)

        # 检查文件是否存在
        if not feature_path.exists():
            logger.warning("特征文件不存在，跳过: {}", feature_path)
            continue

        # 读取特征数据
        try:
            feature_data = read_jsonl(
                feature_path, parse_dates=["datetime"], skip_meta=True
            )
        except Exception as exc:
            logger.error("读取特征文件失败 (股票 {}): {}", symbol, exc)
            continue

        # 跳过空数据
        if feature_data.empty:
            continue

        data_frames.append(feature_data)

    # 无任何有效数据时返回空 DataFrame
    if not data_frames:
        logger.warning("未加载到任何有效的特征数据，返回空 DataFrame")
        return pd.DataFrame()

    # 合并所有 DataFrame 并排序
    logger.info("正在合并 {} 只股票的特征数据...", len(data_frames))
    combined_panel = pd.concat(data_frames, ignore_index=True)
    combined_panel = combined_panel.sort_values(
        ["datetime", "symbol"]
    ).reset_index(drop=True)

    logger.info(
        "面板数据加载完成: {} 行, {} 列, 股票数 {}",
        len(combined_panel),
        len(combined_panel.columns),
        combined_panel["symbol"].nunique(),
    )
    return combined_panel
