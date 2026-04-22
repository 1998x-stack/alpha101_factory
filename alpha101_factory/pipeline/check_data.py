# -*- coding: utf-8 -*-
"""数据完整性校验 CLI 入口模块 (Data Integrity Check CLI Entry Point)。

本模块提供命令行接口，用于校验 ``quotes/daily/`` 目录下所有股票
K 线 JSONL 文件的完整性。通过 PipelineEngine 执行 ``check`` 阶段，
生成并展示完整性报告。

典型用法::

    # 通过模块方式运行
    python -m alpha101_factory.cli check

    # 或直接导入 main 函数
    from alpha101_factory.pipeline.check_data import main
    main()

报告输出:
    - 终端日志: 正常文件数、缺失/空文件数、总股票数
    - CSV 文件: ``logs/klines_integrity.csv``（由 CheckStage 自动保存）

注意:
    - 本模块仅负责 CLI 入口与结果展示，实际校验逻辑由
      ``pipeline.stages.CheckStage`` 和 ``data.loader.check_klines_integrity``
      完成。
    - 若数据目录中无任何 K 线文件，程序会输出警告并正常退出（返回码 0）。
"""

from __future__ import annotations

import sys
from typing import Optional

import pandas as pd
from loguru import logger

from alpha101_factory.config import LOG_DIR
from alpha101_factory.pipeline.engine import PipelineEngine


# CSV 报告文件的默认路径 (Default CSV report file path)
_INTEGRITY_REPORT_FILENAME: str = "klines_integrity.csv"


def _format_report_summary(integrity_report: pd.DataFrame) -> str:
    """格式化完整性报告的摘要信息 (Format Integrity Report Summary)。

    统计正常文件（存在且行数大于零）与异常文件（不存在或行数为零）的数量，
    返回人类可读的摘要字符串。

    Args:
        integrity_report: 完整性报告 DataFrame，应包含 ``exists``（bool）
            和 ``rows``（int）列。

    Returns:
        格式化的摘要字符串，包含总股票数、正常文件数和异常文件数。
    """
    total_stock_count: int = len(integrity_report)
    valid_file_mask: pd.Series = (
        integrity_report["exists"] & (integrity_report["rows"] > 0)
    )
    valid_file_count: int = int(valid_file_mask.sum())
    invalid_file_count: int = total_stock_count - valid_file_count

    return (
        f"数据完整性校验结果:\n"
        f"  总股票数: {total_stock_count}\n"
        f"  正常文件: {valid_file_count}\n"
        f"  缺失/空文件: {invalid_file_count}"
    )


def _get_report_path() -> str:
    """获取 CSV 报告文件的绝对路径字符串 (Get CSV Report File Path)。

    Returns:
        CSV 报告文件的绝对路径字符串。
    """
    return str(LOG_DIR / _INTEGRITY_REPORT_FILENAME)


def _print_check_results(integrity_report: Optional[pd.DataFrame]) -> None:
    """打印数据完整性校验结果到终端 (Print Check Results to Terminal)。

    根据完整性报告的状态输出不同的信息：
        - 报告为 None: 提示校验流程未生成报告
        - 报告为空 DataFrame: 提示无 K 线文件可校验
        - 报告有效: 输出摘要统计和 CSV 报告路径

    Args:
        integrity_report: 完整性报告 DataFrame，可能为 None 或空 DataFrame。
    """
    # 边界条件 1: 报告对象为 None (Boundary: report is None)
    if integrity_report is None:
        logger.warning("校验流程未生成完整性报告，请检查日志排查原因")
        return

    # 边界条件 2: 报告为空 DataFrame (Boundary: report is empty)
    if integrity_report.empty:
        logger.warning("完整性报告为空，数据目录中无 K 线文件可校验")
        return

    # 正常情况: 输出摘要统计 (Normal case: print summary)
    report_summary: str = _format_report_summary(integrity_report)
    logger.info(report_summary)

    # 输出 CSV 报告路径 (Print CSV report path)
    csv_report_path: str = _get_report_path()
    logger.info(f"详细报告已保存至: {csv_report_path}")


def main() -> None:
    """CLI 入口函数：执行数据完整性校验并展示结果 (CLI Entry Point)。

    通过 PipelineEngine 运行 ``check`` 阶段，该阶段会：
        1. 扫描 ``quotes/daily/`` 目录下所有股票的 JSONL 文件
        2. 检查文件是否存在、记录数是否大于零
        3. 生成完整性报告并保存至 ``logs/klines_integrity.csv``

    执行完成后，本函数会从上下文中提取报告并打印摘要信息到终端。

    异常处理:
        - ``KeyboardInterrupt``: 捕获用户中断信号，输出提示后正常退出
        - 其他异常: 记录错误日志后以返回码 1 退出

    示例::

        $ python -m alpha101_factory.cli check
        [INFO] 开始校验 K 线数据完整性...
        [INFO] 数据完整性校验完成: 正常文件 4500 个, 缺失/空文件 120 个
        [INFO] 完整性报告已保存: /path/to/data/logs/klines_integrity.csv
        [INFO] 数据完整性校验结果:
          总股票数: 4620
          正常文件: 4500
          缺失/空文件: 120
        [INFO] 详细报告已保存至: /path/to/data/logs/klines_integrity.csv
    """
    try:
        # 步骤 1: 通过 PipelineEngine 执行 check 阶段
        # (Execute check stage via PipelineEngine)
        pipeline_context = PipelineEngine().run(["check"])

        # 步骤 2: 从上下文中提取完整性报告
        # (Extract integrity report from context)
        integrity_report: Optional[pd.DataFrame] = pipeline_context.get(
            "integrity_report"
        )

        # 步骤 3: 打印校验结果摘要到终端
        # (Print check results summary to terminal)
        _print_check_results(integrity_report)

    except KeyboardInterrupt:
        # 用户主动中断程序 (User-initiated interrupt)
        logger.warning("\n数据完整性校验已被用户中断")
        sys.exit(0)

    except Exception as unexpected_error:
        # 捕获未预期的异常并记录详细日志 (Catch unexpected exceptions)
        logger.error(f"数据完整性校验过程中发生未预期错误: {unexpected_error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
