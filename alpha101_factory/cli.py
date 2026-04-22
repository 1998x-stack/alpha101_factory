# -*- coding: utf-8 -*-
"""Alpha101 Factory 命令行入口模块 (CLI Entry Point).

本模块提供完整的命令行接口，用于执行数据抓取、特征构建、因子计算
和数据完整性校验等核心功能。

典型用法::

    # 抓取全量股票 K 线数据
    python -m alpha101_factory.cli fetch

    # 抓取单只股票数据并生成 K 线图
    python -m alpha101_factory.cli fetch-one --stock 600000 --start 20200101 --end 20240101 --adjust qfq

    # 构建中间特征缓存
    python -m alpha101_factory.cli tmp --stock 600000

    # 计算指定因子
    python -m alpha101_factory.cli factor --factors Alpha101

    # 计算所有已注册因子
    python -m alpha101_factory.cli factor --all

    # 校验数据完整性
    python -m alpha101_factory.cli check
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

from loguru import logger

# 确保项目根目录加入 sys.path，便于模块导入
try:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.append(str(_PROJECT_ROOT))
except Exception as exc:
    print(f"[警告] 无法设置 sys.path: {exc}")

from alpha101_factory.config import ADJUST
from alpha101_factory.data.loader import (
    check_klines_integrity,
    fetch_klines_from_spot,
    fetch_spot,
    load_or_fetch_symbol,
)
from alpha101_factory.data.universe import load_universe
from alpha101_factory.factors.registry import list_factors
from alpha101_factory.factors.tmp_features import build_tmp_all
from alpha101_factory.pipeline.compute_factor import compute_and_save
from alpha101_factory.utils.log import setup_logger

# ============================================================
# 常量定义
# ============================================================

# 股票代码正则：严格匹配 6 位纯数字
_STOCK_CODE_PATTERN = re.compile(r"^\d{6}$")

# 日期格式正则：YYYYMMDD 或空字符串
_DATE_PATTERN = re.compile(r"^(\d{8})?$")

# 合法的复权方式集合
_VALID_ADJUST_MODES = frozenset({"qfq", "hfq", ""})


# ============================================================
# 参数校验辅助函数
# ============================================================


def _validate_stock_code(code: str) -> str:
    """校验股票代码格式。

    Args:
        code: 待校验的股票代码字符串。

    Returns:
        校验通过的原始股票代码。

    Raises:
        argparse.ArgumentTypeError: 当代码非 6 位纯数字时抛出。
    """
    if not _STOCK_CODE_PATTERN.match(code):
        raise argparse.ArgumentTypeError(
            f"无效的股票代码 '{code}'，必须为 6 位纯数字（如 600000）"
        )
    return code


def _validate_date(value: str) -> str:
    """校验日期格式。

    允许空字符串或 YYYYMMDD 格式。

    Args:
        value: 待校验的日期字符串。

    Returns:
        校验通过的原始日期字符串。

    Raises:
        argparse.ArgumentTypeError: 当格式不匹配时抛出。
    """
    if not _DATE_PATTERN.match(value):
        raise argparse.ArgumentTypeError(
            f"无效的日期格式 '{value}'，必须为 YYYYMMDD 或留空"
        )
    return value


def _validate_adjust(value: str) -> str:
    """校验复权方式参数。

    Args:
        value: 待校验的复权方式字符串。

    Returns:
        校验通过的原始复权方式字符串。

    Raises:
        argparse.ArgumentTypeError: 当值不在合法集合中时抛出。
    """
    if value not in _VALID_ADJUST_MODES:
        raise argparse.ArgumentTypeError(
            f"无效的复权方式 '{value}'，可选值: qfq（前复权）/ hfq（后复权）/ ''（不复权）"
        )
    return value


# ============================================================
# 命令处理函数
# ============================================================


def _handle_fetch(args: argparse.Namespace) -> None:
    """执行全量股票行情快照与 K 线数据抓取。

    首先获取当前市场快照并持久化，随后基于快照中的股票列表
    逐一抓取日线 K 线数据。

    Args:
        args: 解析后的命令行参数（本命令无需额外参数）。
    """
    logger.info("开始执行全量数据抓取...")
    snapshot_df = fetch_spot(save=True)
    logger.info(f"行情快照获取完成，共 {len(snapshot_df)} 只股票")
    fetched_count = fetch_klines_from_spot(snapshot_df)
    logger.info(f"K 线数据抓取完成，成功 {fetched_count} 只股票")


def _handle_fetch_one(args: argparse.Namespace) -> None:
    """执行单只股票数据加载或抓取，并生成 K 线 PNG 图表。

    优先从本地缓存加载，若不存在则通过数据源 API 抓取。

    Args:
        args: 解析后的命令行参数，包含：
            - stock: 6 位股票代码
            - start: 起始日期（YYYYMMDD 或空）
            - end: 结束日期（YYYYMMDD 或空）
            - adjust: 复权方式（qfq/hfq/空）
    """
    symbol: str = args.stock
    start_date: str = args.start
    end_date: str = args.end
    adjust_mode: str = args.adjust or ADJUST

    logger.info(
        f"开始处理单只股票: {symbol}, "
        f"日期范围=[{start_date or '默认起始'} .. {end_date or '默认结束'}], "
        f"复权方式={adjust_mode or '不复权'}"
    )

    kline_df = load_or_fetch_symbol(
        symbol,
        start_date,
        end_date,
        adjust=adjust_mode,
        save_image=True,
    )

    if kline_df is None or kline_df.empty:
        logger.warning(f"股票 {symbol} 无可用数据")
    else:
        date_min = kline_df["datetime"].min()
        date_max = kline_df["datetime"].max()
        row_count = len(kline_df)
        logger.info(
            f"股票 {symbol} 处理完成: 共 {row_count} 行数据, "
            f"日期范围={date_min} .. {date_max}"
        )


def _handle_tmp(args: argparse.Namespace) -> None:
    """构建中间特征缓存（收益率、VWAP、ADV 等）。

    若指定了单只股票，则仅构建该股票的特征；
    否则从股票池中加载全部股票并逐一构建。

    Args:
        args: 解析后的命令行参数，包含：
            - stock: 可选，单只股票代码。
    """
    if args.stock:
        symbols_to_build: list[str] = [args.stock]
        logger.info(f"开始为单只股票 {args.stock} 构建中间特征...")
    else:
        symbols_to_build = load_universe().tolist()
        logger.info(f"开始为全量 {len(symbols_to_build)} 只股票构建中间特征...")

    built_count = build_tmp_all(symbols_to_build)
    logger.info(f"中间特征构建完成，成功 {built_count} 只股票")


def _handle_factor(args: argparse.Namespace) -> None:
    """计算指定因子值并持久化。

    支持两种模式：
    1. 通过 --all 计算所有已注册因子
    2. 通过 --factors 指定因子名称列表

    若指定了 --stock 参数，则仅计算该股票的因子值。

    Args:
        args: 解析后的命令行参数，包含：
            - all: 是否计算所有因子
            - factors: 因子名称列表
            - stock: 可选，单只股票代码
    """
    # 确定待计算的因子列表
    if args.all:
        factor_names: list[str] = list_factors()
        logger.info(f"已注册因子共 {len(factor_names)} 个: {', '.join(factor_names)}")
    else:
        factor_names = args.factors

    if not factor_names:
        logger.warning("未指定任何因子，请通过 --factors 或 --all 参数指定")
        return

    # 确定待计算的股票范围
    target_symbols: Optional[list[str]] = [args.stock] if args.stock else None
    if target_symbols:
        logger.info(f"限制计算范围: 股票 {args.stock}")

    # 逐一计算因子
    success_count = 0
    failure_count = 0
    for factor_name in factor_names:
        try:
            compute_and_save(factor_name, symbols=target_symbols)
            logger.info(f"因子 {factor_name} 计算完成")
            success_count += 1
        except Exception as exc:
            logger.error(f"因子 {factor_name} 计算失败: {exc}")
            failure_count += 1

    # 输出汇总信息
    logger.info(
        f"因子计算汇总: 成功 {success_count} 个, 失败 {failure_count} 个, "
        f"共计 {len(factor_names)} 个"
    )


def _handle_check(args: argparse.Namespace) -> None:
    """校验已保存的 K 线数据完整性。

    检查每只股票的 JSONL 文件是否存在、行数是否为零等，
    并输出汇总统计报告。

    Args:
        args: 解析后的命令行参数（本命令无需额外参数）。
    """
    logger.info("开始校验 K 线数据完整性...")
    report_df = check_klines_integrity()

    if report_df.empty:
        logger.info("未找到任何数据，请先运行 `fetch` 命令抓取数据")
        return

    # 输出前 10 行明细
    logger.info(f"数据完整性报告（前 10 行）:\n{report_df.head(10).to_string()}")

    # 输出汇总统计
    total_count = len(report_df)
    exists_count = int(report_df["exists"].sum())
    empty_count = int((report_df["rows"] == 0).sum())
    logger.info(
        f"数据完整性汇总: 文件存在 {exists_count}/{total_count}, "
        f"空文件 {empty_count} 个"
    )


# ============================================================
# CLI 入口
# ============================================================


def _build_argument_parser() -> argparse.ArgumentParser:
    """构建并配置命令行参数解析器。

    Returns:
        配置完成的 ArgumentParser 实例。
    """
    parser = argparse.ArgumentParser(
        prog="alpha101-factory",
        description="Alpha101 Factory — A 股日线 Alpha 因子工厂命令行工具",
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True, help="可用命令")

    # --- fetch: 全量数据抓取 ---
    fetch_parser = subparsers.add_parser(
        "fetch",
        help="抓取全量股票的行情快照与日线 K 线数据（前复权/后复权）",
    )
    fetch_parser.set_defaults(func=_handle_fetch)

    # --- fetch-one: 单只股票数据抓取 ---
    fetch_one_parser = subparsers.add_parser(
        "fetch-one",
        help="加载本地或抓取单只股票数据，并保存 K 线 PNG 图表",
    )
    fetch_one_parser.add_argument(
        "--stock",
        required=True,
        type=_validate_stock_code,
        help="6 位股票代码，如 600000",
    )
    fetch_one_parser.add_argument(
        "--start",
        default="",
        type=_validate_date,
        help="起始日期，格式 YYYYMMDD 或留空使用默认值",
    )
    fetch_one_parser.add_argument(
        "--end",
        default="",
        type=_validate_date,
        help="结束日期，格式 YYYYMMDD 或留空使用默认值",
    )
    fetch_one_parser.add_argument(
        "--adjust",
        default="",
        type=_validate_adjust,
        help="复权方式: qfq（前复权）/ hfq（后复权）/ ''（不复权）",
    )
    fetch_one_parser.set_defaults(func=_handle_fetch_one)

    # --- tmp: 中间特征构建 ---
    tmp_parser = subparsers.add_parser(
        "tmp",
        help="构建中间特征缓存（收益率、VWAP、ADV 等）",
    )
    tmp_parser.add_argument(
        "--stock",
        default="",
        type=_validate_stock_code,
        help="若指定，则仅构建该股票的中间特征",
    )
    tmp_parser.set_defaults(func=_handle_tmp)

    # --- factor: 因子计算 ---
    factor_parser = subparsers.add_parser(
        "factor",
        help="计算指定因子值并保存",
    )
    factor_parser.add_argument(
        "--all",
        action="store_true",
        help="计算所有已注册的因子",
    )
    factor_parser.add_argument(
        "--factors",
        nargs="*",
        default=["Alpha101"],
        help="因子名称列表，默认为 Alpha101",
    )
    factor_parser.add_argument(
        "--stock",
        default="",
        type=_validate_stock_code,
        help="若指定，则仅计算该股票的因子值",
    )
    factor_parser.set_defaults(func=_handle_factor)

    # --- check: 数据完整性校验 ---
    check_parser = subparsers.add_parser(
        "check",
        help="校验已保存的 K 线数据文件完整性",
    )
    check_parser.set_defaults(func=_handle_check)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI 主入口函数。

    初始化日志系统，解析命令行参数，并分发到对应的命令处理函数。
    优雅处理 KeyboardInterrupt（Ctrl+C）和参数解析错误。

    Args:
        argv: 可选的命令行参数列表，默认为 sys.argv[1:]。
    """
    # 初始化日志系统
    setup_logger()

    # 构建参数解析器
    parser = _build_argument_parser()

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse 在参数错误时会调用 sys.exit()，此处优雅处理
        # code=0 表示 --help，正常退出；code!=0 表示参数错误
        if exc.code == 0:
            sys.exit(0)
        logger.error(f"参数解析失败: {exc}")
        sys.exit(1)

    try:
        # 分发到对应的命令处理函数
        args.func(args)
    except KeyboardInterrupt:
        # 优雅处理 Ctrl+C 中断
        logger.warning("操作被用户中断 (Ctrl+C)")
        sys.exit(130)
    except Exception as exc:
        # 捕获未预期的异常，输出错误信息
        logger.error(f"命令执行失败: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
