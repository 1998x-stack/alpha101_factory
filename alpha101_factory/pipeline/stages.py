# -*- coding: utf-8 -*-
"""Pipeline 阶段定义模块 (Pipeline Stage Definitions)。

本模块定义了 Alpha101 因子工厂的完整流水线阶段系统，包括：
    - ``Stage``: 所有阶段的抽象基类，定义统一的 ``run(ctx)`` 接口。
    - ``FetchStage``: 从 AkShare/BaoStock 抓取行情快照与 K 线数据。
    - ``TmpStage``: 构建中间特征缓存（收益率、VWAP、ADV 等）。
    - ``CheckStage``: 校验 K 线 JSONL 文件的完整性。
    - ``FactorStage``: 加载数据并计算已注册的 Alpha 因子。
    - ``BacktestStage``: 执行 IC/RankIC 评估与分位组合回测。
    - ``register_stage``: 阶段注册装饰器，支持自定义阶段插入。
    - ``StageFactory``: 阶段工厂类，按名称创建阶段实例。

典型用法::

    from alpha101_factory.pipeline.stages import StageFactory, register_stage

    # 创建并运行阶段
    stage = StageFactory.create("fetch")
    ctx = stage.run({"symbols": ["600000"]})

    # 注册自定义阶段
    @register_stage
    class MyStage(Stage):
        name = "my_stage"
        def run(self, ctx):
            ctx["my_result"] = True
            return ctx

注意:
    - 所有阶段通过 ``ctx`` 上下文字典传递数据，阶段间解耦。
    - 阶段执行失败时记录错误日志并优雅降级，不会中断整个流水线。
    - 单股票场景下横截面 IC 为 NaN，请使用 TS-IC 指标。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import pandas as pd
from loguru import logger

from alpha101_factory.config import DIR_QUOTES, DIR_FEATURES, DIR_FACTORS, LOG_DIR
from alpha101_factory.factors.registry import FactorFactory
from alpha101_factory.factors.tmp_features import build_tmp_all
from alpha101_factory.data.loader import check_klines_integrity
from alpha101_factory.data.universe import load_universe
from alpha101_factory.utils.io import read_jsonl, write_jsonl


# ===================================================================
# Stage 抽象基类 (Stage Abstract Base Class)
# ===================================================================
class Stage(ABC):
    """Pipeline 阶段抽象基类 (Abstract Base Class for Pipeline Stages)。

    所有 Pipeline 阶段必须继承此类，设置 ``name`` 类属性作为阶段唯一标识，
    并实现 ``run(ctx)`` 方法。每个阶段接收上下文字典 ``ctx``，执行特定任务后
    返回更新后的上下文字典。

    类属性:
        name: 阶段唯一标识名称，用于注册和通过 ``StageFactory`` 创建实例。

    示例::

        class DataQualityStage(Stage):
            name = "quality"

            def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
                ctx["quality_report"] = {"status": "ok"}
                return ctx
    """

    # 阶段唯一标识名称 (Unique stage identifier)
    name: str = "base"

    @abstractmethod
    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """执行阶段任务并返回更新后的上下文 (Execute Stage Task)。

        子类必须实现此方法，接收上下文字典，执行阶段逻辑后返回更新后的字典。
        阶段执行过程中发生的异常应通过日志记录并优雅降级，不应向上传播。

        Args:
            ctx: 上下文字典，包含前序阶段产生的中间结果和配置参数。

        Returns:
            更新后的上下文字典，包含本阶段产生的结果数据。
        """
        ...


# ===================================================================
# FetchStage — 数据抓取阶段 (Data Fetching Stage)
# ===================================================================
class FetchStage(Stage):
    """行情数据抓取阶段 (Market Data Fetching Stage)。

    从 AkShare 数据源（BaoStock 作为降级备选）抓取全市场行情快照，
    并基于快照中的股票列表批量抓取 K 线日线数据。抓取结果以 JSONL 格式
    保存至 ``quotes/daily/{symbol}.jsonl``。

    上下文输入 (ctx):
        无必需输入参数。

    上下文输出 (ctx):
        - ``fetched`` (int): 成功抓取的 K 线文件数量。
    """

    name = "fetch"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """执行行情数据抓取 (Execute Market Data Fetching)。

        1. 调用 ``fetch_spot`` 获取全市场行情快照并保存。
        2. 若快照为空，记录警告并跳过 K 线抓取。
        3. 基于快照中的股票列表，批量抓取 K 线日线数据。

        Args:
            ctx: 上下文字典。

        Returns:
            更新后的上下文字典，包含 ``fetched`` 键表示抓取的文件数。
        """
        from alpha101_factory.data.loader import fetch_spot, fetch_klines_from_spot

        # 步骤 1: 获取全市场行情快照 (Fetch market snapshot)
        logger.info("开始获取全市场行情快照...")
        spot_snapshot: pd.DataFrame = fetch_spot(save=True)

        if spot_snapshot.empty:
            logger.warning("行情快照为空，跳过 K 线数据抓取")
            return ctx

        # 步骤 2: 基于快照批量抓取 K 线数据 (Fetch K-line data based on snapshot)
        stock_count: int = len(spot_snapshot)
        logger.info(f"行情快照包含 {stock_count} 只股票，开始抓取 K 线数据...")
        fetched_count: int = fetch_klines_from_spot(spot_snapshot)

        ctx["fetched"] = fetched_count
        logger.info(f"数据抓取阶段完成: 成功抓取 {fetched_count} 只股票的 K 线数据")
        return ctx


# ===================================================================
# TmpStage — 中间特征构建阶段 (Intermediate Feature Building Stage)
# ===================================================================
class TmpStage(Stage):
    """中间特征缓存构建阶段 (Intermediate Feature Cache Building Stage)。

    为股票池中的每只股票预计算中间特征（如收益率、VWAP、ADV 等），
    并保存至 ``features/{symbol}.jsonl``。这些特征被因子计算阶段复用，
    避免重复计算开销。

    上下文输入 (ctx):
        - ``symbols`` (List[str], 可选): 待处理的股票代码列表。
          若未提供，则从全市场股票池中自动加载。

    上下文输出 (ctx):
        - ``tmp_count`` (int): 成功构建特征的股票数量。
        - ``symbols`` (List[str]): 实际处理的股票代码列表。
    """

    name = "tmp"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """执行中间特征构建 (Execute Intermediate Feature Building)。

        1. 从上下文或全市场股票池获取股票代码列表。
        2. 若股票池为空，记录错误并跳过。
        3. 调用 ``build_tmp_all`` 批量构建中间特征。

        Args:
            ctx: 上下文字典。

        Returns:
            更新后的上下文字典，包含 ``tmp_count`` 和 ``symbols`` 键。
        """
        # 步骤 1: 获取股票代码列表 (Resolve symbol list)
        symbol_list: Optional[List[str]] = ctx.get("symbols")

        if symbol_list is None:
            symbol_list = load_universe().tolist()
            logger.info(f"未指定股票池，从全市场加载 {len(symbol_list)} 只股票")

        if not symbol_list:
            logger.error("股票池为空，跳过中间特征构建")
            return ctx

        # 步骤 2: 批量构建中间特征 (Build intermediate features)
        logger.info(f"开始为 {len(symbol_list)} 只股票构建中间特征...")
        built_count: int = build_tmp_all(symbol_list)

        ctx["tmp_count"] = built_count
        ctx["symbols"] = symbol_list
        logger.info(f"中间特征构建阶段完成: 成功构建 {built_count}/{len(symbol_list)} 只股票的特征")
        return ctx


# ===================================================================
# CheckStage — 数据完整性校验阶段 (Data Integrity Check Stage)
# ===================================================================
class CheckStage(Stage):
    """K 线数据完整性校验阶段 (K-line Data Integrity Check Stage)。

    扫描 ``quotes/daily/`` 目录下所有股票的 JSONL 文件，检查文件是否存在、
    记录数是否大于零，并生成完整性报告。报告保存至 ``logs/klines_integrity.csv``。

    上下文输入 (ctx):
        无必需输入参数。

    上下文输出 (ctx):
        - ``integrity_report`` (pd.DataFrame): 完整性报告，包含
          ``symbol``、``exists``、``rows`` 等列。
    """

    name = "check"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """执行数据完整性校验 (Execute Data Integrity Check)。

        1. 调用 ``check_klines_integrity`` 生成完整性报告。
        2. 统计正常文件与缺失/空文件的数量。
        3. 将报告保存为 CSV 文件。

        Args:
            ctx: 上下文字典。

        Returns:
            更新后的上下文字典，包含 ``integrity_report`` 键。
        """
        # 步骤 1: 生成完整性报告 (Generate integrity report)
        logger.info("开始校验 K 线数据完整性...")
        integrity_report: pd.DataFrame = check_klines_integrity()

        ctx["integrity_report"] = integrity_report

        if integrity_report.empty:
            logger.warning("完整性报告为空，无 K 线文件可校验")
            return ctx

        # 步骤 2: 统计正常与异常文件 (Count valid and invalid files)
        valid_files: pd.DataFrame = integrity_report[
            integrity_report["exists"] & (integrity_report["rows"] > 0)
        ]
        invalid_files: pd.DataFrame = integrity_report[
            ~integrity_report["exists"] | (integrity_report["rows"] <= 0)
        ]

        logger.info(
            f"数据完整性校验完成: 正常文件 {len(valid_files)} 个, "
            f"缺失/空文件 {len(invalid_files)} 个"
        )

        # 步骤 3: 保存报告至 CSV (Save report to CSV)
        report_path: Path = LOG_DIR / "klines_integrity.csv"
        try:
            integrity_report.to_csv(report_path, index=False, encoding="utf-8-sig")
            logger.info(f"完整性报告已保存: {report_path}")
        except OSError as os_error:
            logger.error(f"保存完整性报告失败: {report_path}, 错误: {os_error}")

        return ctx


# ===================================================================
# FactorStage — 因子计算阶段 (Factor Computation Stage)
# ===================================================================
class FactorStage(Stage):
    """Alpha 因子计算阶段 (Alpha Factor Computation Stage)。

    加载行情数据与中间特征，合并为面板数据后计算指定的 Alpha 因子。
    计算结果以 JSONL 格式保存至 ``factors/{factor_name}.jsonl``。

    上下文输入 (ctx):
        - ``symbols`` (List[str], 可选): 待计算的股票代码列表。
          若未提供，则从 ``features/`` 目录自动发现。
        - ``factors`` (List[str], 可选): 待计算的因子名称列表。
          若未提供，则计算所有已注册的因子。

    上下文输出 (ctx):
        不新增键，但会在 ``factors/`` 目录下生成因子输出文件。
    """

    name = "factor"

    # 合并数据时使用的公共列名 (Common columns for merging data)
    _MERGE_COLUMNS: List[str] = [
        "symbol", "datetime", "open", "high", "low", "close", "volume", "amount"
    ]

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """执行因子计算 (Execute Factor Computation)。

        1. 从上下文获取股票代码列表和因子名称列表。
        2. 若未指定因子，则计算所有已注册因子。
        3. 对每个因子：加载数据 → 校验 → 计算 → 保存结果。
        4. 单个因子计算失败不影响其他因子。

        Args:
            ctx: 上下文字典。

        Returns:
            更新后的上下文字典。
        """
        # 步骤 1: 解析输入参数 (Resolve input parameters)
        symbol_list: Optional[List[str]] = ctx.get("symbols")
        factor_names: Optional[List[str]] = ctx.get("factors")

        factor_factory: FactorFactory = FactorFactory()

        if not factor_names:
            factor_names = list(factor_factory.info_all().keys())
            logger.info(f"未指定因子，将计算所有已注册的 {len(factor_names)} 个因子")

        if not factor_names:
            logger.warning("无可用因子，跳过因子计算阶段")
            return ctx

        logger.info(f"因子计算阶段开始: 待计算 {len(factor_names)} 个因子")

        # 步骤 2: 逐个计算因子 (Compute each factor)
        success_count: int = 0
        failure_count: int = 0

        for factor_name in factor_names:
            # 校验因子是否存在 (Validate factor exists)
            if not self._validate_factor_exists(factor_factory, factor_name):
                failure_count += 1
                continue

            # 加载并合并数据 (Load and merge data)
            merged_data: pd.DataFrame = self._load_join(symbol_list)

            if merged_data.empty:
                logger.warning(f"因子 {factor_name}: 合并数据为空，跳过计算")
                failure_count += 1
                continue

            # 执行因子计算 (Execute factor computation)
            if self._compute_and_save(factor_factory, factor_name, merged_data):
                success_count += 1
            else:
                failure_count += 1

        logger.info(
            f"因子计算阶段完成: 成功 {success_count} 个, 失败 {failure_count} 个"
        )
        return ctx

    @staticmethod
    def _validate_factor_exists(
        factor_factory: FactorFactory, factor_name: str
    ) -> bool:
        """校验因子名称是否存在于注册表中 (Validate Factor Name Exists)。

        Args:
            factor_factory: 因子工厂实例。
            factor_name: 待校验的因子名称。

        Returns:
            True 表示因子存在，False 表示不存在并已记录错误日志。
        """
        try:
            factor_factory.info(factor_name)
            return True
        except KeyError:
            logger.error(f"未知因子: {factor_name}，已跳过")
            return False

    @staticmethod
    def _compute_and_save(
        factor_factory: FactorFactory, factor_name: str, data_frame: pd.DataFrame
    ) -> bool:
        """计算因子值并保存结果 (Compute Factor and Save Results)。

        Args:
            factor_factory: 因子工厂实例。
            factor_name: 因子名称。
            data_frame: 合并后的面板数据。

        Returns:
            True 表示计算和保存成功，False 表示失败并已记录错误日志。
        """
        try:
            factor_series: pd.Series = factor_factory.compute(factor_name, data_frame)
        except Exception as compute_error:
            logger.error(f"因子 {factor_name} 计算失败: {compute_error}")
            return False

        # 构建输出 DataFrame (Build output DataFrame)
        output_frame: pd.DataFrame = (
            factor_series.reset_index().rename(columns={0: "value"})
        )

        # 提取统计元数据 (Extract metadata for output)
        unique_symbols: List[str] = sorted(
            output_frame["symbol"].unique().tolist()
        )
        output_row_count: int = len(output_frame)

        # 保存因子结果 (Save factor results)
        output_path: Path = DIR_FACTORS / f"{factor_name}.jsonl"
        try:
            write_jsonl(
                output_frame,
                output_path,
                meta={
                    "factor": factor_name,
                    "rows": output_row_count,
                    "symbols": unique_symbols,
                },
            )
            logger.info(
                f"因子 {factor_name} 计算完成: {output_row_count} 行, "
                f"{len(unique_symbols)} 只股票"
            )
            return True
        except OSError as os_error:
            logger.error(f"因子 {factor_name} 保存失败: {output_path}, 错误: {os_error}")
            return False

    @staticmethod
    def _load_join(symbols: Optional[List[str]]) -> pd.DataFrame:
        """加载行情数据与中间特征并合并为面板数据 (Load and Merge Data into Panel)。

        逐只股票加载 K 线数据和特征数据，按公共列合并后拼接为长表。
        单只股票加载失败不影响其他股票。

        Args:
            symbols: 股票代码列表。若为 None，则从 ``features/`` 目录
                自动发现所有股票。

        Returns:
            合并后的面板数据 DataFrame，按 ``datetime`` 和 ``symbol`` 排序。
            若无有效数据则返回空 DataFrame。
        """
        # 步骤 1: 解析股票代码列表 (Resolve symbol list)
        if symbols is None:
            symbols = sorted({path.stem for path in DIR_FEATURES.glob("*.jsonl")})
            logger.info(f"未指定股票列表，从特征目录发现 {len(symbols)} 只股票")

        if not symbols:
            logger.warning("股票代码列表为空，返回空 DataFrame")
            return pd.DataFrame()

        # 步骤 2: 逐只股票加载并合并 (Load and merge per symbol)
        merged_frames: List[pd.DataFrame] = []

        for symbol_code in symbols:
            try:
                merged_frame = FactorStage._load_single_symbol(symbol_code)
                if merged_frame is not None:
                    merged_frames.append(merged_frame)
            except Exception as load_error:
                logger.error(f"加载股票 {symbol_code} 时发生异常: {load_error}")

        if not merged_frames:
            logger.warning("所有股票加载均失败，返回空 DataFrame")
            return pd.DataFrame()

        # 步骤 3: 拼接并排序 (Concatenate and sort)
        panel_data: pd.DataFrame = pd.concat(merged_frames, ignore_index=True)
        return (
            panel_data.sort_values(["datetime", "symbol"])
            .reset_index(drop=True)
        )

    @classmethod
    def _load_single_symbol(cls, symbol_code: str) -> Optional[pd.DataFrame]:
        """加载单只股票的行情数据与特征数据并合并 (Load Single Symbol Data)。

        Args:
            symbol_code: 股票代码（如 "600000"）。

        Returns:
            合并后的 DataFrame，若任一数据源为空或加载失败则返回 None。
        """
        kline_path: Path = DIR_QUOTES / f"{symbol_code}.jsonl"
        feature_path: Path = DIR_FEATURES / f"{symbol_code}.jsonl"

        # 加载 K 线数据 (Load K-line data)
        kline_data: pd.DataFrame = read_jsonl(kline_path)
        if kline_data.empty:
            logger.debug(f"股票 {symbol_code}: K 线数据为空，跳过")
            return None

        # 加载特征数据 (Load feature data)
        feature_data: pd.DataFrame = read_jsonl(feature_path)
        if feature_data.empty:
            logger.debug(f"股票 {symbol_code}: 特征数据为空，跳过")
            return None

        # 合并数据 (Merge data on common columns)
        merged_data: pd.DataFrame = pd.merge(
            kline_data,
            feature_data,
            on=cls._MERGE_COLUMNS,
            how="outer",
            sort=True,
        )
        return merged_data


# ===================================================================
# BacktestStage — 回测评估阶段 (Backtest Evaluation Stage)
# ===================================================================
class BacktestStage(Stage):
    """回测评估阶段 (Backtest Evaluation Stage)。

    基于已计算的因子值执行回测评估，包括横截面 IC/RankIC 分析、
    时间序列 IC 分析、分位组合收益分析等。评估结果保存至
    ``backtest/{alpha}_h{horizon}_q{quantiles}/`` 目录。

    上下文输入 (ctx):
        - ``alpha`` (str, 可选): 因子名称，默认 ``"Alpha101"``。
        - ``horizon`` (int, 可选): 前瞻收益期，默认 ``1``。
        - ``quantiles`` (int, 可选): 分位数数量，默认 ``5``。

    上下文输出 (ctx):
        不新增键，但会在 ``backtest/`` 和 ``images/backtest/`` 目录下
        生成回测结果文件和图表。
    """

    name = "backtest"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """执行回测评估 (Execute Backtest Evaluation)。

        1. 从上下文获取回测参数（因子名称、前瞻期、分位数）。
        2. 创建 ``BacktestEngine`` 实例并运行回测。

        Args:
            ctx: 上下文字典。

        Returns:
            更新后的上下文字典。
        """
        from alpha101_factory.backtest.engine import BacktestEngine

        # 步骤 1: 解析回测参数 (Resolve backtest parameters)
        alpha_name: str = ctx.get("alpha", "Alpha101")
        horizon_days: int = ctx.get("horizon", 1)
        quantile_count: int = ctx.get("quantiles", 5)

        logger.info(
            f"回测评估阶段开始: 因子={alpha_name}, "
            f"前瞻期={horizon_days}, 分位数={quantile_count}"
        )

        # 步骤 2: 执行回测 (Execute backtest)
        backtest_engine: BacktestEngine = BacktestEngine(
            alpha_name, horizon=horizon_days, quantiles=quantile_count
        )
        backtest_engine.run()

        logger.info(f"回测评估阶段完成: 因子={alpha_name}")
        return ctx


# ===================================================================
# 阶段注册表 (Stage Registry)
# ===================================================================
_STAGES: Dict[str, Type[Stage]] = {}


def register_stage(stage_class: Type[Stage]) -> Type[Stage]:
    """注册 Pipeline 阶段类的装饰器 (Decorator for Registering Pipeline Stages)。

    将阶段类登记到全局注册表中，注册前执行以下校验：
        1. 阶段类必须是 ``Stage`` 的子类。
        2. 阶段类必须定义 ``name`` 类属性且为非空字符串。
        3. 阶段名称在全局注册表中必须唯一。

    用法::

        @register_stage
        class MyStage(Stage):
            name = "my_stage"

            def run(self, ctx):
                return ctx

    Args:
        stage_class: 待注册的阶段类，必须继承自 ``Stage`` 基类。

    Returns:
        原样返回输入的阶段类，以便装饰器语法正常工作。

    Raises:
        TypeError: 当 ``stage_class`` 不是 ``Stage`` 的子类时抛出。
        ValueError: 当阶段类缺少 ``name`` 属性、``name`` 为空字符串，
            或阶段名称已存在于注册表中时抛出。
    """
    # 校验 1: 必须是 Stage 的子类 (Must be a subclass of Stage)
    if not isinstance(stage_class, type) or not issubclass(stage_class, Stage):
        raise TypeError(
            f"阶段注册失败: 类 '{stage_class.__name__}' 不是 Stage 的子类。"
            f"请确保继承自 alpha101_factory.pipeline.stages.Stage。"
        )

    # 校验 2: 必须定义 name 属性且为非空字符串 (Must have non-empty 'name')
    stage_name: str = getattr(stage_class, "name", None)

    if stage_name is None:
        raise ValueError(
            f"阶段注册失败: 阶段类 '{stage_class.__name__}' 缺少 'name' 类属性。"
            f"请在类定义中设置 name = 'your_stage_name'。"
        )

    if not isinstance(stage_name, str) or not stage_name.strip():
        raise ValueError(
            f"阶段注册失败: 阶段类 '{stage_class.__name__}' 的 'name' 属性必须为非空字符串，"
            f"当前值为 {stage_name!r}。"
        )

    # 校验 3: 阶段名称必须唯一 (Stage name must be globally unique)
    if stage_name in _STAGES:
        existing_class: Type[Stage] = _STAGES[stage_name]
        raise ValueError(
            f"阶段注册失败: 阶段名称 '{stage_name}' 已被占用。"
            f"冲突类: {existing_class.__module__}.{existing_class.__qualname__}, "
            f"当前类: {stage_class.__module__}.{stage_class.__qualname__}。"
            f"请修改 name 属性以确保全局唯一。"
        )

    # 登记到全局注册表 (Register in global registry)
    _STAGES[stage_name] = stage_class
    logger.debug(
        f"阶段 '{stage_name}' 已成功注册: "
        f"{stage_class.__module__}.{stage_class.__qualname__}"
    )
    return stage_class


# 自动注册内置阶段 (Auto-register built-in stages)
for _builtin_stage in [FetchStage, TmpStage, CheckStage, FactorStage, BacktestStage]:
    register_stage(_builtin_stage)


# ===================================================================
# StageFactory — 阶段工厂类 (Stage Factory Class)
# ===================================================================
class StageFactory:
    """Pipeline 阶段工厂，按名称创建阶段实例 (Stage Factory for Creating Instances)。

    封装阶段的实例化逻辑，通过注册表动态查找阶段类并创建实例。
    支持查询所有已注册的阶段名称列表。

    典型用法::

        # 创建阶段实例
        fetch_stage = StageFactory.create("fetch")
        ctx = fetch_stage.run({})

        # 查询所有可用阶段
        print(StageFactory.list_all())  # ['backtest', 'check', 'factor', 'fetch', 'tmp']
    """

    @classmethod
    def create(cls, name: str) -> Stage:
        """创建指定名称的阶段实例 (Create Stage Instance by Name)。

        Args:
            name: 阶段的唯一标识名称。

        Returns:
            新创建的阶段实例。

        Raises:
            KeyError: 当阶段名称不存在于注册表中时抛出，
                错误信息包含所有可用阶段名称。
        """
        if name not in _STAGES:
            available_stages: List[str] = sorted(_STAGES.keys())
            raise KeyError(
                f"未知阶段: {name}。"
                f"可用阶段: {available_stages}"
            )
        return _STAGES[name]()

    @classmethod
    def list_all(cls) -> List[str]:
        """返回所有已注册阶段的名称列表 (Return Names of All Registered Stages)。

        Returns:
            已注册阶段名称的排序列表，每个元素为 str 类型。
        """
        return sorted(_STAGES.keys())
