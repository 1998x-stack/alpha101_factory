# -*- coding: utf-8 -*-
"""Pipeline 引擎模块 (Pipeline Engine Module)。

本模块提供 ``PipelineEngine`` 类，用于编排和执行 Alpha101 因子工厂的
完整流水线。引擎按顺序执行各个阶段（Stage），通过上下文字典（context）
在阶段间传递数据和配置参数。

典型用法::

    from alpha101_factory.pipeline.engine import PipelineEngine

    # 方式一：逐步添加上下文并运行指定阶段
    engine = PipelineEngine()
    engine.add("symbols", ["600000", "000001"])
    result = engine.run(["fetch", "tmp", "factor"])

    # 方式二：一键运行完整流水线（支持参数覆盖）
    engine = PipelineEngine()
    result = engine.run_full(alpha="Alpha101", horizon=1, quantiles=5)

注意:
    - 阶段执行失败时，引擎默认中断流水线并返回当前上下文。
    - 可通过 ``on_error`` 参数配置失败时的行为（"break" 或 "continue"）。
    - 所有阶段通过 ``StageFactory`` 动态创建，支持自定义阶段扩展。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Literal

from loguru import logger

from alpha101_factory.pipeline.stages import StageFactory


# 支持的错误处理策略 (Supported error handling strategies)
_ERROR_STRATEGIES: tuple[str, ...] = ("break", "continue")


class PipelineEngine:
    """Pipeline 编排引擎 (Pipeline Orchestration Engine)。

    按预定义顺序执行流水线阶段，通过上下文字典在阶段间传递数据。
    支持灵活配置错误处理策略、阶段列表和初始上下文。

    类属性:
        _DEFAULT_STAGES: 默认完整流水线阶段列表。

    示例::

        # 基本用法
        engine = PipelineEngine()
        result = engine.run_full()

        # 自定义阶段和错误策略
        engine = PipelineEngine(on_error="continue")
        result = engine.run(["fetch", "tmp", "factor", "backtest"])

        # 链式调用添加上下文
        engine = PipelineEngine()
        engine.add("symbols", ["600000"]).add("alpha", "Alpha101")
        result = engine.run(["factor", "backtest"])
    """

    # 默认完整流水线阶段顺序 (Default full pipeline stage sequence)
    _DEFAULT_STAGES: List[str] = ["fetch", "tmp", "factor", "backtest"]

    def __init__(
        self,
        on_error: Literal["break", "continue"] = "break",
    ) -> None:
        """初始化 Pipeline 引擎 (Initialize Pipeline Engine)。

        Args:
            on_error: 阶段执行失败时的处理策略。
                - ``"break"``: 立即中断流水线（默认行为）。
                - ``"continue"``: 记录错误日志后继续执行后续阶段。

        Raises:
            ValueError: 当 ``on_error`` 不是支持的值时抛出。
        """
        # 校验错误处理策略 (Validate error handling strategy)
        if on_error not in _ERROR_STRATEGIES:
            raise ValueError(
                f"无效的错误处理策略: {on_error!r}。"
                f"支持的值: {list(_ERROR_STRATEGIES)}"
            )

        # 初始化上下文字典 (Initialize context dictionary)
        self.ctx: Dict[str, Any] = {}
        self._on_error: Literal["break", "continue"] = on_error

    def add(self, key: str, value: Any) -> "PipelineEngine":
        """向上下文中添加键值对（支持链式调用）(Add Key-Value to Context)。

        此方法用于在运行流水线之前配置上下文参数，例如指定股票代码列表、
        因子名称、回测参数等。

        Args:
            key: 上下文键名，应为非空字符串。
            value: 上下文值，可以是任意类型。

        Returns:
            引擎自身实例，支持链式调用。

        示例::

            engine = PipelineEngine()
            engine.add("symbols", ["600000"]).add("alpha", "Alpha101")
        """
        # 校验键名有效性 (Validate key name)
        if not key or not isinstance(key, str):
            logger.warning(f"尝试添加无效键名: {key!r}，已跳过")
            return self

        self.ctx[key] = value
        return self

    def run(self, stages: List[str]) -> Dict[str, Any]:
        """按顺序执行指定的流水线阶段 (Execute Pipeline Stages Sequentially)。

        遍历阶段名称列表，通过 ``StageFactory`` 创建阶段实例并依次执行。
        每个阶段接收当前上下文字典，返回更新后的字典作为下一阶段的输入。

        阶段执行过程中：
            - 记录每个阶段的开始和结束时间。
            - 根据 ``on_error`` 配置处理阶段失败情况。
            - 打印流水线整体进度和最终状态。

        Args:
            stages: 待执行的阶段名称列表。列表为空时直接返回当前上下文。

        Returns:
            执行完成后的上下文字典，包含所有阶段产生的结果数据。

        示例::

            engine = PipelineEngine()
            result = engine.run(["fetch", "tmp"])
        """
        # 处理空阶段列表 (Handle empty stage list)
        if not stages:
            logger.warning("阶段列表为空，跳过流水线执行")
            return self.ctx

        # 校验所有阶段名称是否存在 (Validate all stage names exist)
        self._validate_stage_names(stages)

        total_stages: int = len(stages)
        logger.info(f"{'=' * 60}")
        logger.info(f"Pipeline 引擎启动: 共 {total_stages} 个阶段")
        logger.info(f"错误处理策略: {self._on_error}")
        logger.info(f"{'=' * 60}")

        pipeline_start_time: float = time.monotonic()
        completed_count: int = 0
        failed_count: int = 0
        skipped_count: int = 0

        for stage_index, stage_name in enumerate(stages, start=1):
            # 打印阶段进度 (Print stage progress)
            logger.info(
                f"\n▶ [{stage_index}/{total_stages}] 阶段: {stage_name}"
            )

            stage_start_time: float = time.monotonic()

            try:
                # 创建并执行阶段 (Create and execute stage)
                stage_instance = StageFactory.create(stage_name)
                self.ctx = stage_instance.run(self.ctx)

                # 计算阶段耗时 (Calculate stage duration)
                stage_duration: float = time.monotonic() - stage_start_time
                logger.info(
                    f"✔ [{stage_index}/{total_stages}] 阶段 '{stage_name}' 完成 "
                    f"(耗时: {stage_duration:.2f}s)"
                )
                completed_count += 1

            except KeyError as key_error:
                # 阶段名称不存在 (Stage name not found)
                logger.error(f"✘ [{stage_index}/{total_stages}] 阶段 '{stage_name}' 不存在: {key_error}")
                failed_count += 1
                if self._should_break():
                    logger.error(f"流水线中断: 已执行 {completed_count}/{total_stages} 个阶段")
                    break

            except Exception as stage_error:
                # 阶段执行异常 (Stage execution error)
                logger.error(
                    f"✘ [{stage_index}/{total_stages}] 阶段 '{stage_name}' 执行失败: {stage_error}"
                )
                failed_count += 1
                if self._should_break():
                    logger.error(f"流水线中断: 已执行 {completed_count}/{total_stages} 个阶段")
                    break

        # 打印流水线总结 (Print pipeline summary)
        total_duration: float = time.monotonic() - pipeline_start_time
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Pipeline 执行完成 (总耗时: {total_duration:.2f}s)")
        logger.info(
            f"结果: 成功 {completed_count} | 失败 {failed_count} | 跳过 {skipped_count}"
        )
        logger.info(f"{'=' * 60}")

        return self.ctx

    def run_full(self, **overrides: Any) -> Dict[str, Any]:
        """运行完整流水线（支持参数覆盖）(Run Full Pipeline with Overrides)。

        使用默认阶段列表 ``["fetch", "tmp", "factor", "backtest"]`` 执行
        完整流水线。通过关键字参数传入的覆盖值会合并到上下文中，
        可用于指定因子名称、回测参数等。

        Args:
            **overrides: 要合并到上下文中的键值对。常见参数包括：
                - ``alpha`` (str): 因子名称，默认 ``"Alpha101"``。
                - ``horizon`` (int): 前瞻收益期，默认 ``1``。
                - ``quantiles`` (int): 分位数数量，默认 ``5``。
                - ``symbols`` (List[str]): 股票代码列表。

        Returns:
            执行完成后的上下文字典。

        示例::

            engine = PipelineEngine()
            result = engine.run_full(
                alpha="Alpha101",
                horizon=1,
                quantiles=5,
            )
        """
        # 合并覆盖参数到上下文 (Merge overrides into context)
        if overrides:
            self.ctx.update(overrides)
            logger.info(f"已应用 {len(overrides)} 个参数覆盖")

        return self.run(self._DEFAULT_STAGES)

    def _validate_stage_names(self, stage_names: List[str]) -> None:
        """校验所有阶段名称是否已注册 (Validate All Stage Names Are Registered)。

        在流水线执行前检查所有阶段名称是否存在于注册表中，
        若存在无效名称则提前抛出异常，避免运行到一半才发现阶段不存在。

        Args:
            stage_names: 待校验的阶段名称列表。

        Raises:
            KeyError: 当存在未注册的阶段名称时抛出，
                错误信息包含无效名称和可用阶段列表。
        """
        registered_stages: List[str] = StageFactory.list_all()
        invalid_names: List[str] = [
            name for name in stage_names if name not in registered_stages
        ]

        if invalid_names:
            raise KeyError(
                f"存在未注册的阶段: {invalid_names}。"
                f"可用阶段: {registered_stages}"
            )

    def _should_break(self) -> bool:
        """判断是否应中断流水线 (Determine Whether to Break Pipeline)。

        根据 ``on_error`` 配置决定阶段失败后是否继续执行。

        Returns:
            True 表示应中断流水线，False 表示应继续执行。
        """
        return self._on_error == "break"
