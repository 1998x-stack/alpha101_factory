# -*- coding: utf-8 -*-
"""因子回测引擎模块。

本模块提供 ``BacktestEngine`` 类，用于执行完整的因子回测流程：
加载因子数据 → 加载价格数据 → 运行评估器 → 保存结果 → 绘制图表。

典型用法::

    from alpha101_factory.backtest.engine import BacktestEngine

    # 创建回测引擎并执行完整流程
    engine = BacktestEngine(alpha="Alpha101", horizon=1, quantiles=5)
    engine.run(evaluators=["ic", "quantile"])

    # 或者分步执行
    engine.load_factor()
    engine.load_prices()
    engine.run_evaluator("ic")
    engine.save_results()
    engine.plot_ic()
    engine.plot_ports()

注意:
    - 回测结果默认保存在 ``data/backtest/{alpha}_h{horizon}_q{quantiles}/`` 目录下。
    - 回测图表默认保存在 ``data/images/backtest/`` 目录下。
    - 所有异常均通过日志记录，不会导致程序崩溃。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger
import plotly.express as px

from alpha101_factory.config import DIR_FACTORS, DIR_QUOTES, DIR_BACKTEST, IMG_BT_DIR
from alpha101_factory.utils.io import read_jsonl
from alpha101_factory.viz.plots import save_fig
from alpha101_factory.backtest.evaluators import EvaluatorFactory


class BacktestEngine:
    """因子回测引擎。

    封装完整的因子回测流程，包括数据加载、评估器执行、结果保存与图表绘制。
    支持多种评估器（IC/RankIC、分位组合等）的组合运行。

    Attributes:
        alpha: 因子名称，对应 ``data/factors/{alpha}.jsonl`` 文件。
        horizon: 前瞻收益期数，表示因子值对应未来第几期的收益。
        quantiles: 分位组合数量，用于将股票按因子值分组。
        run_dir: 回测结果输出目录，格式为 ``data/backtest/{alpha}_h{horizon}_q{quantiles}/``。
        factor_df: 加载后的因子数据 DataFrame。
        prices: 加载后的价格数据 DataFrame。
        results: 各评估器的运行结果，键为评估器名称，值为结果字典。

    Examples:
        执行默认回测流程::

            engine = BacktestEngine("Alpha101")
            engine.run()  # 自动运行 ic 和 quantile 评估器

        自定义评估器::

            engine = BacktestEngine("Alpha101", horizon=5, quantiles=10)
            engine.run(evaluators=["ic"])  # 仅运行 IC 评估器
    """

    def __init__(self, alpha: str, horizon: int = 1, quantiles: int = 5) -> None:
        """初始化回测引擎。

        Args:
            alpha: 因子名称，必须为非空字符串。
            horizon: 前瞻收益期数，必须为正整数，默认值为 1。
            quantiles: 分位组合数量，必须 >= 2，默认值为 5。

        Raises:
            ValueError: 当参数校验失败时抛出。
        """
        # 参数校验
        self._validate_parameters(alpha, horizon, quantiles)

        self.alpha: str = alpha
        self.horizon: int = horizon
        self.quantiles: int = quantiles
        self.run_dir: Path = DIR_BACKTEST / f"{alpha}_h{horizon}_q{quantiles}"
        self.factor_df: Optional[pd.DataFrame] = None
        self.prices: Optional[pd.DataFrame] = None
        self.results: Dict[str, Dict[str, pd.DataFrame]] = {}

        logger.info(
            f"回测引擎已初始化: alpha={alpha}, horizon={horizon}, quantiles={quantiles}, "
            f"输出目录={self.run_dir}"
        )

    @staticmethod
    def _validate_parameters(alpha: str, horizon: int, quantiles: int) -> None:
        """校验回测引擎初始化参数。

        Args:
            alpha: 因子名称。
            horizon: 前瞻收益期数。
            quantiles: 分位组合数量。

        Raises:
            ValueError: 当任一参数不合法时抛出。
        """
        if not isinstance(alpha, str) or not alpha.strip():
            raise ValueError(f"因子名称必须为非空字符串，当前值: {alpha!r}")
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError(f"horizon 必须为正整数，当前值: {horizon!r}")
        if not isinstance(quantiles, int) or quantiles < 2:
            raise ValueError(f"quantiles 必须 >= 2，当前值: {quantiles!r}")

    def load_factor(self) -> bool:
        """从文件系统加载因子数据。

        读取 ``data/factors/{alpha}.jsonl`` 文件，解析为 DataFrame 并
        存储到 ``self.factor_df``。

        Returns:
            bool: 加载成功返回 ``True``，文件不存在或数据为空时返回 ``False``。
        """
        factor_file_path: Path = DIR_FACTORS / f"{self.alpha}.jsonl"

        # 检查因子文件是否存在
        if not factor_file_path.exists():
            logger.error(f"因子文件未找到: {factor_file_path}")
            return False

        # 读取因子数据
        self.factor_df = read_jsonl(
            factor_file_path, parse_dates=["datetime"], skip_meta=True
        )

        # 检查数据是否为空
        if self.factor_df is None or self.factor_df.empty:
            logger.error(f"因子数据为空: {factor_file_path}")
            self.factor_df = None
            return False

        symbol_count: int = self.factor_df["symbol"].nunique()
        date_count: int = self.factor_df["datetime"].nunique()
        row_count: int = len(self.factor_df)
        logger.info(
            f"因子数据加载成功: {self.alpha}, {row_count} 行, "
            f"{symbol_count} 只股票, {date_count} 个交易日"
        )
        return True

    def load_prices(self) -> bool:
        """从文件系统加载价格数据。

        根据因子数据中的股票列表，逐一读取对应的日线行情数据
        ``data/quotes/daily/{symbol}.jsonl``，合并为统一的 DataFrame
        并存储到 ``self.prices``。

        单个股票价格文件缺失不会导致整体失败，仅记录警告日志。

        Returns:
            bool: 至少成功加载一只股票的价格数据时返回 ``True``，
                因子数据未加载或所有股票价格文件均缺失时返回 ``False``。
        """
        # 确保因子数据已加载
        if self.factor_df is None or self.factor_df.empty:
            logger.error("无法加载价格数据：因子数据尚未加载或为空")
            return False

        # 获取因子数据中涉及的所有股票代码
        target_symbols: List[str] = sorted(self.factor_df["symbol"].unique().tolist())
        logger.info(f"开始加载 {len(target_symbols)} 只股票的价格数据...")

        price_dataframes: List[pd.DataFrame] = []
        missing_symbols: List[str] = []

        for stock_symbol in target_symbols:
            price_file_path: Path = DIR_QUOTES / f"{stock_symbol}.jsonl"
            try:
                stock_price_df = read_jsonl(
                    price_file_path, parse_dates=["datetime"], skip_meta=True
                )
                if not stock_price_df.empty:
                    # 仅保留回测所需的列
                    price_dataframes.append(
                        stock_price_df[["datetime", "symbol", "close"]]
                    )
                else:
                    missing_symbols.append(stock_symbol)
            except Exception as read_error:
                logger.warning(f"读取 {stock_symbol} 价格数据失败: {read_error}")
                missing_symbols.append(stock_symbol)

        # 检查是否成功加载任何价格数据
        if not price_dataframes:
            logger.error(
                f"未找到对应股票的行情数据（共 {len(target_symbols)} 只股票）"
            )
            return False

        # 合并所有股票的价格数据
        self.prices = (
            pd.concat(price_dataframes, ignore_index=True)
            .sort_values(["datetime", "symbol"])
            .reset_index(drop=True)
        )

        loaded_count: int = len(target_symbols) - len(missing_symbols)
        logger.info(
            f"价格数据加载完成: {loaded_count}/{len(target_symbols)} 只股票, "
            f"共 {len(self.prices)} 行"
        )

        if missing_symbols:
            logger.warning(
                f"以下 {len(missing_symbols)} 只股票的价格数据缺失: "
                f"{', '.join(missing_symbols[:10])}"
                f"{'...' if len(missing_symbols) > 10 else ''}"
            )

        return True

    def run_evaluator(self, name: str, **kwargs: Any) -> Dict[str, pd.DataFrame]:
        """运行指定的评估器并保存结果。

        通过 ``EvaluatorFactory`` 创建评估器实例，执行因子评估，
        将结果存储到 ``self.results`` 字典中。

        Args:
            name: 评估器名称，必须是已注册的评估器（如 ``"ic"``、``"quantile"``）。
            **kwargs: 传递给评估器的额外参数（如 ``q=5`` 指定分位数数量）。

        Returns:
            Dict[str, pd.DataFrame]: 评估结果字典，键为结果类型名称，
                值为对应的 DataFrame。

        Raises:
            KeyError: 当指定的评估器名称未注册时抛出。
            RuntimeError: 当因子数据或价格数据未加载时抛出。
        """
        # 确保数据已加载
        if self.factor_df is None:
            raise RuntimeError("无法运行评估器：因子数据尚未加载")
        if self.prices is None:
            raise RuntimeError("无法运行评估器：价格数据尚未加载")

        # 验证评估器名称是否已注册
        available_evaluators: List[str] = EvaluatorFactory.list_all()
        if name not in available_evaluators:
            raise KeyError(
                f"未知评估器: {name!r}. 可用评估器: {available_evaluators}"
            )

        logger.info(f"正在运行评估器: {name} (horizon={self.horizon}, kwargs={kwargs})")

        # 创建评估器实例并执行评估
        evaluator = EvaluatorFactory.create(name)
        evaluation_result: Dict[str, pd.DataFrame] = evaluator.evaluate(
            self.factor_df, self.prices, self.horizon, **kwargs
        )

        # 保存结果
        self.results[name] = evaluation_result

        # 打印结果摘要
        for result_key, result_dataframe in evaluation_result.items():
            if isinstance(result_dataframe, pd.DataFrame) and not result_dataframe.empty:
                logger.info(f"  [{name}] {result_key}: {len(result_dataframe)} 行")

        return evaluation_result

    def save_results(self) -> None:
        """将所有评估结果持久化到文件系统。

        遍历 ``self.results`` 中的每个评估器结果，通过对应的评估器
        实例将结果保存到 ``self.run_dir`` 目录下。

        如果 ``self.results`` 为空，则跳过保存并记录警告日志。
        """
        if not self.results:
            logger.warning("无评估结果可保存")
            return

        # 确保输出目录存在
        self.run_dir.mkdir(parents=True, exist_ok=True)

        for evaluator_name, evaluator_results in self.results.items():
            try:
                evaluator = EvaluatorFactory.create(evaluator_name)
                evaluator.save(evaluator_results, self.run_dir)
                logger.info(f"评估结果已保存: {evaluator_name} → {self.run_dir}")
            except Exception as save_error:
                logger.error(f"保存 {evaluator_name} 评估结果失败: {save_error}")

        logger.info(f"回测结果已保存至: {self.run_dir}")

    def plot_ic(self) -> None:
        """绘制 IC/RankIC 时序图。

        从 ``"ic"`` 评估器结果中提取每日 IC/RankIC 数据，
        生成折线图并保存为 PNG 文件到 ``data/images/backtest/`` 目录。

        如果 IC 评估器未运行或无有效数据，则跳过绘图。
        """
        # 检查 IC 评估器是否已运行
        if "ic" not in self.results:
            logger.debug("跳过 IC 绘图：IC 评估器未运行")
            return

        daily_ic_dataframe: pd.DataFrame = self.results["ic"].get(
            "daily_ic", pd.DataFrame()
        )

        # 检查是否有有效数据
        if daily_ic_dataframe.dropna(how="all").empty:
            logger.warning("无有效横截面 IC/RankIC 数据，跳过绘图")
            return

        try:
            # 构建 IC/RankIC 时序图
            plot_dataframe = daily_ic_dataframe.reset_index()
            figure = px.line(
                plot_dataframe,
                x="datetime",
                y=["IC", "RankIC"],
                title=f"{self.alpha} IC/RankIC 时序图 (horizon={self.horizon})",
            )
            figure.update_layout(
                xaxis_title="日期",
                yaxis_title="相关系数",
                legend_title="指标",
            )

            # 保存图表
            output_path: Path = IMG_BT_DIR / f"{self.alpha}_IC_RankIC_h{self.horizon}.png"
            save_fig(figure, output_path)
            logger.info(f"IC/RankIC 图表已保存: {output_path}")

        except Exception as plot_error:
            logger.warning(f"IC/RankIC 绘图失败: {plot_error}")

    def plot_ports(self) -> None:
        """绘制分位组合累积收益时序图。

        从 ``"quantile"`` 评估器结果中提取累积收益数据，
        生成折线图并保存为 PNG 文件到 ``data/images/backtest/`` 目录。

        如果分位组合评估器未运行或无有效数据，则跳过绘图。
        """
        # 检查分位组合评估器是否已运行
        if "quantile" not in self.results:
            logger.debug("跳过分位组合绘图：quantile 评估器未运行")
            return

        cumulative_returns_dataframe: pd.DataFrame = self.results["quantile"].get(
            "cumrets", pd.DataFrame()
        )

        # 检查是否有有效数据
        if cumulative_returns_dataframe.empty:
            logger.warning("无有效分位组合累积收益数据，跳过绘图")
            return

        try:
            # 构建分位组合累积收益图
            plot_dataframe = cumulative_returns_dataframe.reset_index()
            figure = px.line(
                plot_dataframe,
                x="datetime",
                y=cumulative_returns_dataframe.columns.tolist(),
                title=(
                    f"{self.alpha} 分位组合累积收益 "
                    f"(horizon={self.horizon}, quantiles={self.quantiles})"
                ),
            )
            figure.update_layout(
                xaxis_title="日期",
                yaxis_title="累积收益",
                legend_title="分位组",
            )

            # 保存图表
            output_path: Path = (
                IMG_BT_DIR
                / f"{self.alpha}_ports_h{self.horizon}_q{self.quantiles}.png"
            )
            save_fig(figure, output_path)
            logger.info(f"分位组合图表已保存: {output_path}")

        except Exception as plot_error:
            logger.warning(f"分位组合绘图失败: {plot_error}")

    def run(self, evaluators: Optional[List[str]] = None) -> None:
        """执行完整的回测流程。

        按顺序执行以下步骤：
        1. 加载因子数据
        2. 加载价格数据
        3. 运行指定的评估器（默认为 ``["ic", "quantile"]``）
        4. 绘制 IC/RankIC 图表
        5. 绘制分位组合累积收益图表
        6. 保存所有评估结果

        Args:
            evaluators: 要运行的评估器名称列表。
                如果为 ``None`` 或空列表，则使用默认值 ``["ic", "quantile"]``。
        """
        logger.info(f"===== 开始回测: {self.alpha} (h={self.horizon}, q={self.quantiles}) =====")

        # 步骤 1: 加载因子数据
        if not self.load_factor():
            logger.error("回测终止：因子数据加载失败")
            return

        # 步骤 2: 加载价格数据
        if not self.load_prices():
            logger.error("回测终止：价格数据加载失败")
            return

        # 确定要运行的评估器列表
        evaluator_list: List[str] = evaluators or ["ic", "quantile"]
        logger.info(f"将运行以下评估器: {evaluator_list}")

        # 步骤 3: 运行各评估器
        for evaluator_name in evaluator_list:
            try:
                self.run_evaluator(evaluator_name, q=self.quantiles)
            except Exception as evaluator_error:
                logger.exception(f"评估器 {evaluator_name!r} 运行失败: {evaluator_error}")

        # 步骤 4 & 5: 绘制图表
        self.plot_ic()
        self.plot_ports()

        # 步骤 6: 保存结果
        self.save_results()

        logger.info(
            f"===== 回测完成: {self.alpha} (h={self.horizon}, q={self.quantiles}) ====="
        )
