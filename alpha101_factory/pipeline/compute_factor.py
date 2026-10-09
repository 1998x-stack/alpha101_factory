# -*- coding: utf-8 -*-
"""因子计算 CLI 入口模块 (Factor Computation CLI Entry Point).

本模块提供 Alpha 因子的命令行计算入口，支持单因子计算、批量因子计算
以及单股票限定计算。核心功能包括:

1. **compute_and_save()**: 计算指定因子并持久化结果至 ``factors/{name}.jsonl``。
2. **main()**: CLI 主入口，解析 ``--all``、``--factors``、``--stock`` 参数。

典型用法::

    # 通过 PipelineEngine 计算单个因子
    from alpha101_factory.pipeline.compute_factor import compute_and_save
    compute_and_save("Alpha101", symbols=["600000"])

    # 命令行直接运行
    python -m alpha101_factory.pipeline.compute_factor --factors Alpha101
    python -m alpha101_factory.pipeline.compute_factor --all
    python -m alpha101_factory.pipeline.compute_factor --factors Alpha101 --stock 600000

注意事项:
    - 因子名称必须在注册表中存在，否则计算前会报错并提示可用因子列表。
    - 股票代码必须为 6 位纯数字格式（如 ``600000``）。
    - 单股票场景下横截面 IC 为 NaN，请使用 TS-IC 指标评估。
"""
from __future__ import annotations

import argparse
import re
import sys
from typing import List, Optional

from loguru import logger

from alpha101_factory.factors.registry import list_factors
from alpha101_factory.pipeline.engine import PipelineEngine
from alpha101_factory.utils.log import setup_logger

# 股票代码正则表达式：严格匹配 6 位纯数字
_STOCK_CODE_PATTERN: re.Pattern[str] = re.compile(r"^\d{6}$")


def _validate_stock_code(stock_code: str) -> str:
    """校验股票代码格式是否为 6 位纯数字。

    Args:
        stock_code: 待校验的股票代码字符串。

    Returns:
        校验通过的原始股票代码。

    Raises:
        argparse.ArgumentTypeError: 当代码非 6 位纯数字时抛出。

    示例::

        >>> _validate_stock_code("600000")
        '600000'
        >>> _validate_stock_code("abc")
        argparse.ArgumentTypeError: 无效的股票代码 'abc'...
    """
    if not _STOCK_CODE_PATTERN.match(stock_code):
        raise argparse.ArgumentTypeError(
            f"无效的股票代码 '{stock_code}'，必须为 6 位纯数字（如 600000）"
        )
    return stock_code


def compute_and_save(
    factor_name: str,
    symbols: Optional[List[str]] = None,
) -> None:
    """计算指定因子的因子值并持久化至 JSONL 文件。

    通过 ``PipelineEngine`` 构建因子计算流水线，加载行情数据与中间特征，
    执行因子计算后将结果保存至 ``factors/{factor_name}.jsonl``。

    计算前会校验因子名称是否存在于注册表中，若不存在则抛出 ``KeyError``
    并提示所有可用因子名称。

    Args:
        factor_name: 待计算的因子名称，必须已在注册表中登记。
        symbols: 可选的股票代码列表，限定计算范围。
            若为 ``None`` 或空列表，则从 ``features/`` 目录自动发现全部股票。

    Raises:
        KeyError: 当因子名称不存在于注册表中时抛出，错误信息包含可用因子列表。
        Exception: 因子计算或保存过程中发生的其他异常将向上传播。

    示例::

        # 计算全市场 Alpha101 因子
        compute_and_save("Alpha101")

        # 仅计算单只股票
        compute_and_save("Alpha101", symbols=["600000"])
    """
    registered_factors: List[str] = list_factors()
    if factor_name not in registered_factors:
        raise KeyError(
            f"未找到因子 '{factor_name}'。"
            f"当前已注册 {len(registered_factors)} 个因子: {registered_factors}"
        )

    symbol_count: int = len(symbols) if symbols else "全部"
    logger.info(
        f"开始计算因子 '{factor_name}'，"
        f"股票范围: {symbol_count} 只"
    )

    pipeline: PipelineEngine = (
        PipelineEngine()
        .add("factors", [factor_name])
        .add("symbols", symbols)
    )
    pipeline.run(["factor"])

    logger.info(f"因子 '{factor_name}' 计算流程已结束")


def _build_argument_parser() -> argparse.ArgumentParser:
    """构建因子计算的命令行参数解析器。

    Returns:
        配置完成的 ArgumentParser 实例，支持以下参数:
            - ``--all``: 计算所有已注册因子。
            - ``--factors``: 指定因子名称列表，默认为 ``["Alpha101"]``。
            - ``--stock``: 限定单只股票计算，格式为 6 位纯数字。
    """
    parser = argparse.ArgumentParser(
        prog="compute-factor",
        description="Alpha101 因子计算工具 — 计算并持久化 Alpha 因子值",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="计算所有已注册的因子",
    )
    parser.add_argument(
        "--factors",
        nargs="*",
        default=["Alpha101"],
        metavar="FACTOR",
        help="因子名称列表（默认: Alpha101）",
    )
    parser.add_argument(
        "--stock",
        default="",
        type=_validate_stock_code,
        metavar="CODE",
        help="限定单只股票计算，股票代码为 6 位纯数字（如 600000）",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """CLI 主入口函数，解析参数并执行因子计算。

    支持三种运行模式:
        1. ``--all``: 计算所有已注册因子。
        2. ``--factors A B``: 计算指定的因子列表。
        3. ``--stock 600000``: 限定仅计算单只股票。

    优雅处理以下异常场景:
        - 参数解析错误（argparse 自动退出）。
        - 用户中断（KeyboardInterrupt / Ctrl+C）。
        - 因子计算失败（记录错误日志并继续下一个因子）。

    Args:
        argv: 可选的命令行参数列表，默认为 ``sys.argv[1:]``。
            主要用于单元测试时传入模拟参数。
    """
    setup_logger()

    argument_parser: argparse.ArgumentParser = _build_argument_parser()

    try:
        parsed_args: argparse.Namespace = argument_parser.parse_args(argv)
    except SystemExit as system_exit:
        # argparse 在参数错误或 --help 时调用 sys.exit()。
        # code=0 表示 --help，正常退出；code!=0 表示参数错误。
        if system_exit.code == 0:
            sys.exit(0)
        logger.error(f"参数解析失败，退出码: {system_exit.code}")
        sys.exit(1)

    if parsed_args.all:
        factor_names: List[str] = list_factors()
        logger.info(f"已注册因子共 {len(factor_names)} 个: {', '.join(factor_names)}")
    else:
        factor_names = parsed_args.factors

    if not factor_names:
        logger.warning("未指定任何因子，请通过 --factors 或 --all 参数指定")
        sys.exit(0)

    symbols: Optional[List[str]] = (
        [parsed_args.stock] if parsed_args.stock else None
    )
    if symbols:
        logger.info(f"限定计算范围: 股票 {parsed_args.stock}")

    success_count: int = 0
    failure_count: int = 0

    for current_factor_name in factor_names:
        try:
            compute_and_save(current_factor_name, symbols=symbols)
            logger.info(f"因子 '{current_factor_name}' 计算成功")
            success_count += 1
        except KeyboardInterrupt:
            logger.warning("操作被用户中断 (Ctrl+C)")
            sys.exit(130)
        except Exception as computation_error:
            logger.error(
                f"因子 '{current_factor_name}' 计算失败: {computation_error}"
            )
            failure_count += 1

    total_count: int = len(factor_names)
    logger.info(
        f"因子计算汇总: "
        f"成功 {success_count}/{total_count} 个, "
        f"失败 {failure_count} 个"
    )


if __name__ == "__main__":
    main()
