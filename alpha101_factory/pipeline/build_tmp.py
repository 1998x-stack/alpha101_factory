# -*- coding: utf-8 -*-
"""中间特征构建 CLI 入口 (Intermediate Feature Build CLI Entry Point)。

本模块提供命令行接口，用于为单只或全量股票构建中间特征缓存
（如收益率、VWAP、ADV 等）。构建结果保存至 ``features/{symbol}.jsonl``，
供后续因子计算阶段复用。

典型用法::

    # 为全量股票构建中间特征
    python -m alpha101_factory.pipeline.build_tmp

    # 仅为单只股票构建中间特征
    python -m alpha101_factory.pipeline.build_tmp --stock 600000

注意:
    - 本模块亦可作为独立脚本直接执行。
    - 通过 PipelineEngine 调用 ``tmp`` 阶段，与 ``cli.py`` 中的
      ``_handle_tmp`` 功能等价但入口不同。
"""
from __future__ import annotations

import argparse
import re
import sys
from typing import List, Optional, Sequence

from loguru import logger

from alpha101_factory.pipeline.engine import PipelineEngine

# ============================================================
# 常量定义 (Constants)
# ============================================================

# 股票代码正则：严格匹配 6 位纯数字（如 600000、000001）
_STOCK_CODE_PATTERN = re.compile(r"^\d{6}$")


# ============================================================
# 参数校验 (Argument Validation)
# ============================================================


def _validate_stock_code(stock_code: str) -> str:
    """校验股票代码格式是否为 6 位纯数字。

    Args:
        stock_code: 待校验的股票代码字符串。

    Returns:
        校验通过的原始股票代码。

    Raises:
        argparse.ArgumentTypeError: 当代码非 6 位纯数字时抛出。
    """
    if not _STOCK_CODE_PATTERN.match(stock_code):
        raise argparse.ArgumentTypeError(
            f"无效的股票代码 '{stock_code}'，必须为 6 位纯数字（如 600000）"
        )
    return stock_code


# ============================================================
# 核心逻辑 (Core Logic)
# ============================================================


def _resolve_target_symbols(
    single_stock_code: Optional[str],
) -> Optional[List[str]]:
    """根据输入解析待处理的股票代码列表。

    若指定了单只股票代码，则返回仅包含该代码的列表；
    若未指定（即全量模式），则返回 ``None``，交由 PipelineEngine
    从全市场股票池中自动加载。

    Args:
        single_stock_code: 可选的单只股票代码。若为 ``None`` 或空字符串，
            表示全量模式。

    Returns:
        单只股票代码列表（如 ``["600000"]``），或 ``None`` 表示全量模式。
    """
    if single_stock_code:
        return [single_stock_code]
    return None


def _build_argument_parser() -> argparse.ArgumentParser:
    """构建并配置命令行参数解析器。

    Returns:
        配置完成的 ArgumentParser 实例。
    """
    parser = argparse.ArgumentParser(
        prog="build_tmp",
        description="构建中间特征缓存（收益率、VWAP、ADV 等），"
        "供后续因子计算阶段复用。",
    )
    parser.add_argument(
        "--stock",
        default="",
        type=_validate_stock_code,
        help="若指定，则仅构建该股票的中间特征（6 位代码，如 600000）",
    )
    return parser


def run_build_tmp(
    stock_code: Optional[str] = None,
) -> int:
    """执行中间特征构建任务。

    通过 PipelineEngine 调用 ``tmp`` 阶段，为指定股票或全量股票
    构建中间特征缓存。

    Args:
        stock_code: 可选的单只股票代码。若为 ``None``，则为全量模式。

    Returns:
        成功构建特征的股票数量。若股票池为空或构建过程出错，返回 0。
    """
    target_symbols: Optional[List[str]] = _resolve_target_symbols(stock_code)

    if target_symbols is not None:
        logger.info(f"开始为单只股票 {target_symbols[0]} 构建中间特征...")
    else:
        logger.info("开始为全量股票构建中间特征...")

    engine = PipelineEngine()
    engine.add("symbols", target_symbols)
    context = engine.run(["tmp"])

    success_count: int = context.get("tmp_count", 0)
    return success_count


# ============================================================
# CLI 入口 (CLI Entry Point)
# ============================================================


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI 主入口函数。

    解析命令行参数，执行中间特征构建任务，并输出构建结果汇总。
    优雅处理 KeyboardInterrupt（Ctrl+C）和参数解析错误。

    Args:
        argv: 可选的命令行参数列表，默认为 ``sys.argv[1:]``。
    """
    parser = _build_argument_parser()

    try:
        parsed_args = parser.parse_args(argv)
    except SystemExit as exit_exception:
        # argparse 在参数错误时调用 sys.exit()：
        # code=0 表示 --help，正常退出；code!=0 表示参数错误。
        if exit_exception.code == 0:
            sys.exit(0)
        logger.error(f"参数解析失败: {exit_exception}")
        sys.exit(1)

    try:
        # 执行中间特征构建
        built_count: int = run_build_tmp(stock_code=parsed_args.stock or None)

        # 输出构建结果汇总
        if parsed_args.stock:
            logger.info(
                f"中间特征构建完成: 股票 {parsed_args.stock}, "
                f"成功 {built_count}/1 只"
            )
        else:
            logger.info(f"中间特征构建完成: 成功 {built_count} 只股票")

    except KeyboardInterrupt:
        # 优雅处理 Ctrl+C 中断
        logger.warning("操作被用户中断 (Ctrl+C)")
        sys.exit(130)
    except Exception as runtime_error:
        # 捕获未预期的异常，输出错误信息
        logger.error(f"中间特征构建失败: {runtime_error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
