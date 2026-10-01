# -*- coding: utf-8 -*-
"""数据加载与抓取模块。

本模块是 alpha101_factory 的核心数据入口，负责：

- 获取 A 股市场快照（实时行情）
- 批量下载个股日线 K 线数据
- 保存 K 线可视化图表（PNG）
- 校验已保存 K 线文件的完整性
- 按需加载或抓取单只股票数据

数据源通过 ``DataSourceFactory`` 统一管理，支持 AkShare 主数据源
与 BaoStock 备用数据源的自动降级。所有数据以 JSONL 格式持久化。

典型用法::

    # 获取市场快照
    spot_df = fetch_spot()

    # 批量下载 K 线
    fetch_klines_from_spot(spot_df)

    # 校验数据完整性
    report = check_klines_integrity()

    # 加载或抓取单只股票
    df = load_or_fetch_symbol("600000", "20240101", "20241231")

注意:
    本模块依赖 ``alpha101_factory.config`` 中的全局配置常量
    （如 ``ADJUST``, ``START_DATE``, ``END_DATE`` 等）。
"""
from __future__ import annotations

import re
import time
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from tqdm import tqdm

from alpha101_factory.config import (
    DIR_QUOTES,
    DIR_SPOT,
    DIR_UNIVERSE,
    ADJUST,
    START_DATE,
    END_DATE,
    LIMIT_STOCKS,
    REQUEST_PAUSE,
    IMG_KLINES_DIR,
)
from alpha101_factory.utils.io import write_jsonl, read_jsonl
from alpha101_factory.viz.plots import plot_kline, save_fig
from alpha101_factory.data.factory import DataSourceFactory
from alpha101_factory.utils.validation import (
    is_valid_stock_code,
    is_valid_adjust,
    normalize_stock_code
)

# ---------------------------------------------------------------------------
# 常量与正则表达式
# ---------------------------------------------------------------------------

# 股票代码校验正则：恰好 6 位数字
_STOCK_CODE_PATTERN: re.Pattern[str] = re.compile(r"^\d{6}$")

# 合法的复权方式集合
_VALID_ADJUST_VALUES: frozenset[str] = frozenset({"qfq", "hfq", ""})


def _is_valid_stock_code(code: str) -> bool:
    """校验股票代码格式是否为 6 位纯数字。

    Args:
        code: 待校验的股票代码字符串。

    Returns:
        bool: 符合 6 位数字格式返回 True，否则返回 False。

    Examples:
        >>> _is_valid_stock_code("600000")
        True
        >>> _is_valid_stock_code("abc")
        False
    """
    return is_valid_stock_code(code)


def _is_valid_adjust(adjust: str) -> bool:
    """校验复权方式是否为允许值。

    Args:
        adjust: 待校验的复权方式字符串。

    Returns:
        bool: 为 "qfq"、"hfq" 或空字符串时返回 True，否则返回 False。
    """
    return is_valid_adjust(adjust)


def _normalize_stock_code(code: str) -> str:
    """提取股票代码中的纯数字部分并补齐至 6 位。

    过滤掉非数字字符后，使用零填充至 6 位长度。
    若提取后无有效数字，返回空字符串。

    Args:
        code: 原始股票代码（可能包含非数字字符）。

    Returns:
        str: 规范化后的 6 位数字股票代码；若无有效数字则返回空字符串。

    Examples:
        >>> _normalize_stock_code("sh600000")
        '600000'
        >>> _normalize_stock_code("600000")
        '600000'
    """
    return normalize_stock_code(code)


def kline_path(symbol: str) -> Path:
    """获取指定股票日线 K 线 JSONL 文件的完整路径。

    Args:
        symbol: 股票代码（6 位数字字符串）。

    Returns:
        Path: 指向 ``quotes/daily/{symbol}.jsonl`` 的绝对路径对象。

    Examples:
        >>> kline_path("600000")
        Path('/path/to/data/quotes/daily/600000.jsonl')
    """
    return DIR_QUOTES / f"{symbol}.jsonl"


def _save_kline_png(
    symbol: str,
    data_frame: pd.DataFrame,
    start_date: Optional[str],
    end_date: Optional[str],
    adjust: str,
) -> Optional[Path]:
    """绘制并保存单只股票的 K 线图（PNG 格式）。

    根据股票代码、日期范围和复权方式生成唯一的 PNG 文件名。
    若绘图或保存过程中发生异常，记录警告日志但不中断调用方流程。

    Args:
        symbol: 股票代码（6 位数字字符串）。
        data_frame: 包含 OHLCV 列的行情数据 DataFrame。
            必须包含 ``datetime``, ``open``, ``high``, ``low``, ``close`` 列。
        start_date: 起始日期标签（用于文件名），格式 ``YYYYMMDD`` 或任意描述字符串。
            为 None 时使用 "start" 作为标签。
        end_date: 结束日期标签（用于文件名），格式 ``YYYYMMDD`` 或任意描述字符串。
            为 None 时使用 "end" 作为标签。
        adjust: 复权方式（"qfq" / "hfq" / ""）。

    Returns:
        Optional[Path]: 成功时返回 PNG 文件的绝对路径；
            若 data_frame 为空或保存失败则返回 None。
    """
    # 空数据无需绘图
    if data_frame is None or data_frame.empty:
        logger.debug(f"{symbol} 数据为空，跳过 K 线图绘制")
        return None

    # 生成文件名标签
    start_tag: str = start_date if start_date else "start"
    end_tag: str = end_date if end_date else "end"
    output_png_path: Path = IMG_KLINES_DIR / f"{symbol}_{start_tag}_{end_tag}_{adjust}.png"

    # 文件已存在则跳过
    if output_png_path.exists():
        logger.debug(f"{symbol} K 线图已存在: {output_png_path}")
        return output_png_path

    try:
        # 生成图表标题
        chart_title: str = f"{symbol} {adjust} {start_tag}-{end_tag}"
        figure = plot_kline(
            data_frame,
            chart_title,
            tickformat="%Y-%m-%d",
            tickangle=-45,
        )
        save_fig(figure, output_png_path)
        logger.info(f"{symbol} K 线图已保存: {output_png_path}")
        return output_png_path
    except Exception as exc:
        # 绘图/保存失败不中断批量流程
        logger.warning(f"{symbol} 绘制/保存 K 线图失败: {exc}")
        return None


def fetch_spot(save: bool = True) -> pd.DataFrame:
    """获取 A 股市场快照（实时行情）。

    优先从本地 ``universe/stocks.jsonl`` 读取已缓存的股票池数据。
    若本地不存在，则通过 ``DataSourceFactory`` 从远程数据源获取最新快照，
    并可选择性地保存到本地。

    Args:
        save: 是否将获取的快照持久化到本地文件。
            默认为 True，保存到 ``universe/stocks.jsonl`` 和
            ``quotes/spot/spot_YYYYMMDD.jsonl``。

    Returns:
        pd.DataFrame: 包含 ``code`` 和 ``name`` 列的股票池数据。
            若获取失败或数据为空，返回空 DataFrame。

    Note:
        本地快照存在时直接返回，不发起网络请求。
        若需强制刷新，请手动删除 ``universe/stocks.jsonl`` 文件。
    """
    logger.info("正在获取 A 股实时行情 …")
    universe_path: Path = DIR_UNIVERSE / "stocks.jsonl"

    # 优先读取本地缓存
    if universe_path.exists():
        try:
            spot_dataframe: pd.DataFrame = read_jsonl(universe_path)
            if not spot_dataframe.empty:
                logger.info(f"读取本地股票池快照: {universe_path} ({len(spot_dataframe)} 只股票)")
                return spot_dataframe
            logger.warning(f"本地股票池文件为空: {universe_path}")
        except Exception as e:
            logger.error(f"读取本地股票池文件失败: {e}")
            # 继续尝试从远程获取

    # 从远程数据源获取
    try:
        spot_dataframe = DataSourceFactory.fetch_spot_fallback()
        if spot_dataframe.empty:
            logger.error("未能从任何数据源获取市场快照数据")
            return pd.DataFrame()

        # 持久化到本地
        if save:
            try:
                write_jsonl(spot_dataframe[["code", "name"]], universe_path)
                today_tag: str = date.today().strftime("%Y%m%d")
                spot_dated_path: Path = DIR_SPOT / f"spot_{today_tag}.jsonl"
                write_jsonl(spot_dataframe[["code", "name"]], spot_dated_path)
                logger.info(
                    f"股票池快照已保存: {universe_path}, {spot_dated_path} "
                    f"(共 {len(spot_dataframe)} 只股票)"
                )
            except Exception as e:
                logger.error(f"保存股票池快照失败: {e}")

        logger.info(f"实时行情共 {len(spot_dataframe)} 行 | 保存={save}")
        return spot_dataframe
    except Exception as e:
        logger.error(f"获取市场快照过程中发生异常: {e}")
        return pd.DataFrame()


def fetch_klines_from_spot(spot_dataframe: pd.DataFrame) -> int:
    """根据市场快照批量下载个股日线 K 线数据。

    遍历快照中的每只股票，检查本地是否已存在对应的 K 线文件。
    若不存在，则从远程数据源下载并保存为 JSONL 格式，同时生成 K 线 PNG 图表。
    若已存在，则直接生成 K 线图表（用于可视化更新）。

    下载过程中自动应用请求节流（``REQUEST_PAUSE``），避免触发 API 限流。

    Args:
        spot_dataframe: 包含 ``code`` 列的市场快照 DataFrame。
            通常由 ``fetch_spot()`` 返回。

    Returns:
        int: 本次新下载并保存的 K 线文件数量。

    Note:
        - 已存在的文件不会重新下载，但会重新生成 K 线 PNG。
        - 单只股票下载失败时记录警告日志，继续处理下一只。
        - 受 ``LIMIT_STOCKS`` 环境变量限制，可能仅处理部分股票。
    """
    # 空快照直接返回
    if spot_dataframe.empty:
        logger.warning("快照数据为空，跳过 K 线批量下载")
        return 0

    # 验证必需列是否存在
    if "code" not in spot_dataframe.columns:
        logger.error(
            f"快照数据缺少 'code' 列，可用列: {list(spot_dataframe.columns)}"
        )
        return 0

    # 提取并规范化股票代码
    try:
        raw_codes: list[str] = (
            spot_dataframe["code"]
            .astype(str)
            .apply(_normalize_stock_code)
            .unique()
            .tolist()
        )
        # 过滤无效代码
        valid_codes: list[str] = [c for c in raw_codes if _is_valid_stock_code(c)]

        if len(raw_codes) != len(valid_codes):
            invalid_count: int = len(raw_codes) - len(valid_codes)
            logger.warning(f"过滤掉 {invalid_count} 个无效股票代码")

        # 应用股票数量限制
        codes_to_fetch: list[str] = valid_codes
        if LIMIT_STOCKS and LIMIT_STOCKS > 0:
            codes_to_fetch = valid_codes[:LIMIT_STOCKS]
            logger.info(f"受 LIMIT_STOCKS 限制，仅处理前 {LIMIT_STOCKS} 只股票")

        logger.info(
            f"准备下载 {len(codes_to_fetch)} 只股票 | "
            f"adjust={ADJUST} start={START_DATE} end={END_DATE or 'latest'}"
        )

        newly_saved_count: int = 0
        failed_count: int = 0

        for symbol in tqdm(codes_to_fetch, desc="下载日线"):
            target_path: Path = kline_path(symbol)

            # 本地文件已存在：仅生成 K 线图，不重新下载
            if target_path.exists():
                try:
                    local_dataframe: pd.DataFrame = read_jsonl(target_path)
                    if not local_dataframe.empty:
                        _save_kline_png(
                            symbol,
                            local_dataframe,
                            START_DATE or "all",
                            END_DATE or "all",
                            ADJUST,
                        )
                except Exception as e:
                    logger.warning(f"{symbol} 读取本地文件失败: {e}")
                continue

            # 从远程数据源下载
            try:
                kline_dataframe = DataSourceFactory.fetch_kline_fallback(
                    symbol,
                    START_DATE or None,
                    END_DATE or None,
                    ADJUST,
                )

                if kline_dataframe is not None and not kline_dataframe.empty:
                    # 插入 symbol 列
                    kline_dataframe.insert(0, "symbol", symbol)
                    write_jsonl(
                        kline_dataframe,
                        target_path,
                        meta={
                            "symbol": symbol,
                            "adjust": ADJUST,
                            "start": START_DATE,
                            "end": END_DATE,
                            "rows": len(kline_dataframe),
                        },
                    )
                    _save_kline_png(
                        symbol,
                        kline_dataframe,
                        START_DATE or "all",
                        END_DATE or "all",
                        ADJUST,
                    )
                    newly_saved_count += 1
                    logger.debug(f"{symbol} 已保存: {target_path} ({len(kline_dataframe)} 行)")
                else:
                    failed_count += 1
                    logger.warning(f"{symbol} 数据源返回空数据")

            except Exception as exc:
                failed_count += 1
                logger.warning(f"{symbol} 下载失败: {exc}")

            # 请求节流，避免触发 API 限流
            time.sleep(REQUEST_PAUSE)

        # 打印汇总信息
        logger.info(
            f"K 线下载完成: 新保存 {newly_saved_count} 个文件, "
            f"失败/空数据 {failed_count} 个, "
            f"跳过（已存在）{len(codes_to_fetch) - newly_saved_count - failed_count} 个"
        )
        return newly_saved_count

    except Exception as e:
        logger.error(f"批量下载 K 线数据过程中发生异常: {e}")
        return 0


def check_klines_integrity() -> pd.DataFrame:
    """校验已保存 K 线文件的完整性。

    遍历股票池中的所有股票，检查对应的 K 线 JSONL 文件是否存在、
    是否可读、是否包含有效数据。生成包含存在性、行数、日期范围
    等信息的完整性报告。

    Returns:
        pd.DataFrame: 完整性报告，包含以下列：
            - ``symbol``: 股票代码
            - ``exists``: 文件是否存在且可读（bool）
            - ``rows``: 有效数据行数（0 表示空文件，-1 表示读取错误）
            - ``date_min``: 最早日期（datetime 或 None）
            - ``date_max``: 最晚日期（datetime 或 None）
            - ``path``: 文件路径（读取失败时包含错误信息）

    Note:
        若本地股票池文件不存在，返回空 DataFrame 并记录警告日志。
    """
    universe_path: Path = DIR_UNIVERSE / "stocks.jsonl"
    spot_dataframe: pd.DataFrame = read_jsonl(universe_path)

    if spot_dataframe.empty:
        logger.warning(f"未找到股票池文件，无法校验 K 线完整性: {universe_path}")
        return pd.DataFrame()

    # 提取并规范化股票代码
    codes: list[str] = (
        spot_dataframe["code"]
        .astype(str)
        .apply(_normalize_stock_code)
        .unique()
        .tolist()
    )

    integrity_rows: list[list] = []

    for code in codes:
        file_path: Path = kline_path(code)

        if not file_path.exists():
            # 文件不存在
            integrity_rows.append([code, False, 0, None, None, str(file_path)])
            continue

        try:
            data_frame: pd.DataFrame = read_jsonl(file_path)

            if data_frame.empty:
                # 文件存在但无有效数据
                integrity_rows.append([code, True, 0, None, None, str(file_path)])
            else:
                # 解析日期范围
                datetime_series: pd.Series = pd.to_datetime(data_frame["datetime"])
                date_minimum: pd.Timestamp = datetime_series.min()
                date_maximum: pd.Timestamp = datetime_series.max()
                integrity_rows.append(
                    [code, True, len(data_frame), date_minimum, date_maximum, str(file_path)]
                )

        except Exception as exc:
            # 文件读取异常
            error_path_info: str = f"{file_path} ERROR: {exc}"
            integrity_rows.append([code, False, -1, None, None, error_path_info])

    # 构建报告 DataFrame
    report_dataframe: pd.DataFrame = pd.DataFrame(
        integrity_rows,
        columns=["symbol", "exists", "rows", "date_min", "date_max", "path"],
    )

    # 打印汇总统计
    total_count: int = len(report_dataframe)
    existing_count: int = int(report_dataframe["exists"].sum())
    empty_count: int = int((report_dataframe["rows"] == 0).sum())
    error_count: int = int((report_dataframe["rows"] == -1).sum())

    logger.info(
        f"K 线文件完整性检查: "
        f"总计 {total_count} 只股票, "
        f"存在 {existing_count}/{total_count}, "
        f"空文件 {empty_count}, "
        f"读取错误 {error_count}, "
        f"缺失 {total_count - existing_count}"
    )

    return report_dataframe


def load_or_fetch_symbol(
    symbol: str,
    start_date: Optional[str],
    end_date: Optional[str],
    adjust: str = ADJUST,
    save_image: bool = True,
) -> pd.DataFrame:
    """加载或抓取单只股票的日线 K 线数据。

    优先从本地 JSONL 文件加载指定股票的 K 线数据，并按日期范围过滤。
    若本地文件不存在或为空，则从远程数据源下载并保存到本地。

    Args:
        symbol: 股票代码（6 位数字字符串，如 "600000"）。
        start_date: 起始日期，格式 ``YYYYMMDD``。为 None 时不限制起始日期。
        end_date: 结束日期，格式 ``YYYYMMDD``。为 None 时不限制结束日期。
        adjust: 复权方式，可选值 ``"qfq"``（前复权）、``"hfq"``（后复权）、
            ``""``（不复权）。默认为全局配置 ``ADJUST``。
        save_image: 是否生成并保存 K 线 PNG 图表。默认为 True。

    Returns:
        pd.DataFrame: 过滤后的 K 线数据 DataFrame，包含 ``symbol`` 列。
            若本地不存在且远程获取失败，返回空 DataFrame。

    Raises:
        ValueError: 当股票代码格式无效或复权方式不合法时抛出。

    Note:
        - 本地文件存在时，仅按日期范围过滤返回，不重新下载。
        - 远程下载成功后会自动保存到本地 JSONL 文件。
    """
    # 参数校验
    normalized_symbol: str = _normalize_stock_code(symbol)
    if not _is_valid_stock_code(normalized_symbol):
        raise ValueError(
            f"无效的股票代码: '{symbol}'，必须为 6 位数字（如 '600000'）"
        )

    if not _is_valid_adjust(adjust):
        raise ValueError(
            f"无效的复权方式: '{adjust}'，允许值: qfq, hfq, ''（空字符串）"
        )

    target_path: Path = kline_path(normalized_symbol)

    # 尝试从本地加载
    if target_path.exists():
        local_dataframe: pd.DataFrame = read_jsonl(target_path)
        if not local_dataframe.empty:
            filtered_dataframe: pd.DataFrame = local_dataframe.copy()

            # 按日期范围过滤
            if start_date:
                start_datetime: pd.Timestamp = pd.to_datetime(start_date)
                filtered_dataframe = filtered_dataframe[
                    filtered_dataframe["datetime"] >= start_datetime
                ]
            if end_date:
                end_datetime: pd.Timestamp = pd.to_datetime(end_date)
                filtered_dataframe = filtered_dataframe[
                    filtered_dataframe["datetime"] <= end_datetime
                ]

            # 生成 K 线图
            if save_image and not filtered_dataframe.empty:
                _save_kline_png(
                    normalized_symbol,
                    filtered_dataframe,
                    start_date or "all",
                    end_date or "all",
                    adjust,
                )

            logger.info(
                f"{normalized_symbol} 从本地加载 {len(filtered_dataframe)} 行 "
                f"(路径: {target_path})"
            )
            return filtered_dataframe

        logger.warning(f"{normalized_symbol} 本地文件为空: {target_path}")

    # 从远程数据源下载
    logger.info(f"{normalized_symbol} 本地不存在，从远程数据源下载 …")
    kline_dataframe = DataSourceFactory.fetch_kline_fallback(
        normalized_symbol,
        start_date,
        end_date,
        adjust,
    )

    if kline_dataframe is not None and not kline_dataframe.empty:
        kline_dataframe.insert(0, "symbol", normalized_symbol)
        write_jsonl(
            kline_dataframe,
            target_path,
            meta={
                "symbol": normalized_symbol,
                "adjust": adjust,
                "start": start_date,
                "end": end_date,
                "rows": len(kline_dataframe),
            },
        )

        if save_image:
            _save_kline_png(
                normalized_symbol,
                kline_dataframe,
                start_date or "all",
                end_date or "all",
                adjust,
            )

        logger.info(
            f"{normalized_symbol} 下载并保存 {len(kline_dataframe)} 行 "
            f"(路径: {target_path})"
        )
    else:
        logger.error(f"{normalized_symbol} 远程数据源返回空数据")

    return kline_dataframe


if __name__ == "__main__":
    # 完整的数据抓取与校验流程
    spot_df: pd.DataFrame = fetch_spot()
    if not spot_df.empty:
        fetch_klines_from_spot(spot_df)
    check_klines_integrity()
