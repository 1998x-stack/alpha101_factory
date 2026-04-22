# -*- coding: utf-8 -*-
"""Alpha101 因子回测 CLI 入口模块。

本模块提供命令行接口，用于执行单因子回测流程。
支持横截面 IC/RankIC 评估与分位组合分析。

使用方式:
    python -m alpha101_factory.backtest.run_bt --alpha Alpha101 --horizon 1 --quantiles 5

输出结果保存在 data/backtest/{alpha}_h{horizon}_q{quantiles}/ 目录下。
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

from loguru import logger

from alpha101_factory.backtest.engine import BacktestEngine


def _build_argument_parser() -> argparse.ArgumentParser:
    """构建并返回命令行参数解析器。

    Returns:
        配置好所有 CLI 参数的 ArgumentParser 实例。
    """
    parser = argparse.ArgumentParser(
        prog="alpha101_backtest",
        description="Alpha101 因子回测工具 — 计算 IC/RankIC 与分位组合收益",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python -m alpha101_factory.backtest.run_bt --alpha Alpha101\n"
            "  python -m alpha101_factory.backtest.run_bt --alpha Alpha101 --horizon 5 --quantiles 10\n"
        ),
    )
    parser.add_argument(
        "--alpha",
        required=True,
        type=str,
        help="因子名称（对应 factors/{alpha}.jsonl 文件）",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="前瞻收益期数，必须为正整数（默认: 1）",
    )
    parser.add_argument(
        "--quantiles",
        type=int,
        default=5,
        help="分位组合数量，必须 >= 2（默认: 5）",
    )
    return parser


def _validate_backtest_args(alpha_name: str, horizon: int, quantile_count: int) -> None:
    """校验回测参数的合法性。

    Args:
        alpha_name: 因子名称，不可为空字符串。
        horizon: 前瞻收益期数，必须为正整数。
        quantile_count: 分位组合数量，必须 >= 2。

    Raises:
        ValueError: 当任一参数不满足约束条件时抛出。
    """
    if not alpha_name or not alpha_name.strip():
        raise ValueError("因子名称不可为空")

    if horizon <= 0:
        raise ValueError(f"前瞻收益期数必须为正整数，当前值: {horizon}")

    if quantile_count < 2:
        raise ValueError(f"分位组合数量必须 >= 2，当前值: {quantile_count}")


def parse_and_validate_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数并执行合法性校验。

    本函数封装参数解析与校验的完整流程：
    1. 构建 ArgumentParser 并解析参数。
    2. 对解析后的参数进行业务逻辑校验。
    3. 校验失败时打印错误信息并退出进程。

    Args:
        argv: 命令行参数列表。为 None 时从 sys.argv 自动读取。

    Returns:
        包含解析后参数的 Namespace 对象。

    SystemExit:
        当参数解析或校验失败时，打印错误信息并以退出码 2 终止进程。
    """
    parser = _build_argument_parser()

    # 捕获 argparse 自身的解析错误（如缺少必需参数）
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse 在 --help 时退出码为 0，应正常放行
        if exc.code == 0:
            raise
        # 其他情况（如缺少 --alpha）以非零码退出
        sys.exit(exc.code)

    # 执行业务逻辑校验
    try:
        _validate_backtest_args(
            alpha_name=args.alpha,
            horizon=args.horizon,
            quantile_count=args.quantiles,
        )
    except ValueError as exc:
        parser.error(str(exc))
        # parser.error 会调用 sys.exit(2)，此处为类型标注需要
        raise  # pragma: no cover

    return args


def run_backtest(alpha_name: str, horizon: int, quantile_count: int) -> None:
    """执行完整的因子回测流程。

    本函数为回测的核心执行入口，依次完成：
    1. 初始化 BacktestEngine。
    2. 加载因子数据与行情价格数据。
    3. 运行 IC 评估器与分位组合评估器。
    4. 生成 IC/RankIC 与分位组合收益图表。
    5. 保存所有回测结果至磁盘。

    Args:
        alpha_name: 因子名称，对应 factors 目录下的 JSONL 文件名。
        horizon: 前瞻收益期数，用于计算未来 N 期收益率。
        quantile_count: 分位组合数量，用于构建多空投资组合。
    """
    # 打印回测启动信息
    logger.info("=" * 60)
    logger.info("Alpha101 因子回测启动")
    logger.info(f"  因子名称   : {alpha_name}")
    logger.info(f"  前瞻期数   : {horizon}")
    logger.info(f"  分位数量   : {quantile_count}")
    logger.info("=" * 60)

    # 初始化回测引擎并执行
    engine = BacktestEngine(
        alpha=alpha_name,
        horizon=horizon,
        quantiles=quantile_count,
    )
    engine.run()

    # 打印回测完成信息与输出路径
    output_directory = engine.run_dir
    logger.info("=" * 60)
    logger.info("回测执行完成")
    logger.info(f"  结果目录   : {output_directory}")
    logger.info(f"  IC 图表    : data/images/backtest/{alpha_name}_IC_RankIC_h{horizon}.png")
    logger.info(
        f"  组合图表   : data/images/backtest/{alpha_name}_ports_h{horizon}_q{quantile_count}.png"
    )
    logger.info("=" * 60)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI 入口函数 — 解析参数、校验合法性、执行回测。

    本函数为 `python -m alpha101_factory.backtest.run_bt` 的入口点。
    完整流程：
    1. 解析并校验命令行参数。
    2. 调用 run_backtest 执行回测。
    3. 捕获 KeyboardInterrupt 实现优雅退出。

    Args:
        argv: 命令行参数列表。为 None 时从 sys.argv 自动读取。
              主要用于单元测试时传入模拟参数。
    """
    args = parse_and_validate_args(argv)

    try:
        run_backtest(
            alpha_name=args.alpha,
            horizon=args.horizon,
            quantile_count=args.quantiles,
        )
    except KeyboardInterrupt:
        logger.warning("\n回测被用户中断 (Ctrl+C)")
        sys.exit(130)


if __name__ == "__main__":
    main()
