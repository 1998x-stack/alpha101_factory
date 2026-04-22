# -*- coding: utf-8 -*-
"""因子回测评估器模块。

提供因子质量评估的核心功能，包括横截面 IC/RankIC 分析、时间序列 IC 统计
以及分位组合收益分析。所有评估器通过 ``EvaluatorFactory`` 统一创建和管理。

典型用法::

    from alpha101_factory.backtest.evaluators import EvaluatorFactory

    # 创建 IC 评估器并执行评估
    ic_eval = EvaluatorFactory.create("ic")
    ic_results = ic_eval.evaluate(factor_df, price_df, horizon=1)

    # 创建分位组合评估器
    q_eval = EvaluatorFactory.create("quantile")
    q_results = q_eval.evaluate(factor_df, price_df, horizon=1, q=5)

注意:
    - 横截面 IC 需要同一天至少两只股票，单股票场景请查看 TS-IC。
    - 分位组合在样本不足时自动跳过，不会抛出异常。
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd


# ============================================================================
# 统计工具函数
# ============================================================================

def _compute_t_stat(series: pd.Series) -> float:
    """计算 t 统计量，用于评估均值是否显著异于零。

    计算公式: t = mean / (std / sqrt(n))

    Args:
        series: 输入序列，自动丢弃 NaN 值。

    Returns:
        float: t 统计量。样本量不足或标准差为零时返回 ``np.nan``。

    Examples:
        >>> _compute_t_stat(pd.Series([0.1, 0.2, 0.15, 0.05]))
        5.0  # 近似值
    """
    clean_series = series.dropna()
    sample_count = len(clean_series)

    # 样本量不足，无法计算有效的 t 统计量
    if sample_count < 2:
        return np.nan

    mean_value = clean_series.mean()
    std_value = clean_series.std(ddof=1)

    # 标准差为零意味着所有值相同，t 统计量无意义
    if std_value == 0:
        return np.nan

    return mean_value / (std_value / np.sqrt(sample_count))


def make_forward_return(
    price_df: pd.DataFrame,
    horizon: int = 1,
) -> Optional[pd.Series]:
    """基于价格数据计算前瞻收益率。

    对每只股票计算未来 ``horizon`` 期的收益率，并通过 ``shift(-horizon)``
    将其对齐到当前时间点，作为因子评估的前瞻收益基准。

    Args:
        price_df: 价格数据 DataFrame，必须包含 ``symbol``、``datetime``、
            ``close`` 三列。
        horizon: 前瞻期数，必须为正整数。默认值为 1（即下一期收益率）。

    Returns:
        Optional[pd.Series]: 前瞻收益率序列，索引为 ``MultiIndex[datetime, symbol]``，
            名称为 ``fwd_ret``。输入数据无效时返回 ``None``。

    Examples:
        >>> df = pd.DataFrame({
        ...     "symbol": ["A", "A", "A"],
        ...     "datetime": ["2024-01-01", "2024-01-02", "2024-01-03"],
        ...     "close": [10.0, 11.0, 12.0],
        ... })
        >>> ret = make_forward_return(df, horizon=1)
        >>> ret.iloc[0]  # (11 - 10) / 10 = 0.1
        0.1
    """
    # 校验必需列
    required_columns = {"symbol", "datetime", "close"}
    if not required_columns.issubset(price_df.columns):
        warnings.warn(
            f"价格数据缺少必需列: {required_columns - set(price_df.columns)}",
            UserWarning,
        )
        return None

    # 空 DataFrame 直接返回
    if price_df.empty:
        return None

    # 按股票和时间排序，确保 pct_change 计算正确
    sorted_df = price_df.sort_values(["symbol", "datetime"]).copy()

    # 计算每只股票的前瞻收益率：shift(-horizon) 将未来收益对齐到当前行
    forward_returns = sorted_df.groupby("symbol")["close"].pct_change(horizon).shift(-horizon)

    # 构建 MultiIndex 序列
    multi_index = pd.MultiIndex.from_frame(
        sorted_df[["datetime", "symbol"]],
        names=["datetime", "symbol"],
    )

    return pd.Series(forward_returns.values, index=multi_index, name="fwd_ret")


# ============================================================================
# 评估器基类
# ============================================================================

class Evaluator(ABC):
    """评估器抽象基类。

    所有具体评估器必须继承此类并实现 ``evaluate()`` 和 ``save()`` 方法。
    子类需设置 ``name`` 类属性用于工厂注册。

    Attributes:
        name: 评估器唯一标识名称，用于工厂注册和查找。
    """

    name: str = "base"

    @abstractmethod
    def evaluate(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        horizon: int,
        **kwargs: Any,
    ) -> Dict[str, pd.DataFrame]:
        """执行因子评估，返回各项指标结果。

        Args:
            factor_df: 因子数据 DataFrame，必须包含 ``datetime``、``symbol``、
                ``value`` 三列。
            price_df: 价格数据 DataFrame，必须包含 ``datetime``、``symbol``、
                ``close`` 三列。
            horizon: 前瞻收益期数，必须为正整数。
            **kwargs: 评估器特有的额外参数。

        Returns:
            Dict[str, pd.DataFrame]: 评估结果字典，键为结果类型名称，值为对应的
                DataFrame。具体键名由子类决定。
        """
        ...

    @abstractmethod
    def save(
        self,
        results: Dict[str, pd.DataFrame],
        run_dir: Path,
    ) -> None:
        """将评估结果持久化到指定目录。

        Args:
            results: ``evaluate()`` 返回的结果字典。
            run_dir: 结果输出目录路径。
        """
        ...


# ============================================================================
# IC / RankIC 评估器
# ============================================================================

class ICEvaluator(Evaluator):
    """横截面 IC（Information Coefficient）与 RankIC 评估器。

    计算每日因子值与前瞻收益率的 Pearson 相关系数（IC）和 Spearman 秩相关
    系数（RankIC），同时提供时间序列维度的 TS-IC 统计。

    输出结果包含三个 DataFrame：
        - ``daily_ic``: 每日 IC/RankIC 及样本股票数
        - ``summary``: 横截面 IC 汇总统计（均值、t 统计量、天数等）
        - ``ts_summary``: 时间序列 IC 汇总统计（按股票维度）

    注意:
        横截面 IC 需要同一天至少两只股票。单股票场景下 IC/RankIC 为 NaN，
        请查看 TS-IC/TS-RankIC 结果。
    """

    name = "ic"

    @staticmethod
    def _compute_correlation(group_df: pd.DataFrame, method: str) -> float:
        """计算指定组内因子值与前瞻收益率的相关系数。

        Args:
            group_df: 包含 ``value`` 和 ``fwd_ret`` 列的 DataFrame。
            method: 相关系数计算方法，``"pearson"`` 或 ``"spearman"``。

        Returns:
            float: 相关系数值。样本不足或计算异常时返回 ``np.nan``。
        """
        try:
            # 横截面 IC 需要至少两只股票才有意义
            unique_stocks = group_df["symbol"].nunique()
            if unique_stocks < 2:
                return np.nan
            return group_df["value"].corr(group_df["fwd_ret"], method=method)
        except Exception:
            # 捕获所有异常（如全部值为常数导致相关系数无法计算）
            return np.nan

    def evaluate(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        horizon: int,
        **kwargs: Any,
    ) -> Dict[str, pd.DataFrame]:
        """执行 IC/RankIC 评估。

        计算每日横截面 IC/RankIC、汇总统计以及每只股票的时间序列 IC。

        Args:
            factor_df: 因子数据，必须包含 ``datetime``、``symbol``、``value`` 列。
            price_df: 价格数据，必须包含 ``datetime``、``symbol``、``close`` 列。
            horizon: 前瞻收益期数，必须为正整数。
            **kwargs: 额外参数（当前未使用）。

        Returns:
            Dict[str, pd.DataFrame]: 包含 ``daily_ic``、``summary``、
                ``ts_summary`` 三个键的结果字典。输入数据无效时返回空 DataFrame。
        """
        # 空结果模板
        empty_result = {
            "daily_ic": pd.DataFrame(),
            "summary": pd.DataFrame(),
            "ts_summary": pd.DataFrame(),
        }

        # 校验因子数据
        if factor_df is None or factor_df.empty:
            return empty_result

        required_factor_cols = {"datetime", "symbol", "value"}
        if not required_factor_cols.issubset(factor_df.columns):
            missing = required_factor_cols - set(factor_df.columns)
            warnings.warn(f"因子数据缺少必需列: {missing}", UserWarning)
            return empty_result

        # 校验 horizon 参数
        if not isinstance(horizon, int) or horizon < 1:
            warnings.warn(f"horizon 必须为正整数，当前值: {horizon}", UserWarning)
            return empty_result

        # 准备因子数据：设置 MultiIndex
        factor_data = factor_df.copy()
        factor_data["datetime"] = pd.to_datetime(factor_data["datetime"])
        factor_data = factor_data.set_index(["datetime", "symbol"])

        # 计算前瞻收益率
        forward_returns = make_forward_return(
            price_df[["datetime", "symbol", "close"]],
            horizon=horizon,
        )
        if forward_returns is None:
            return empty_result

        # 合并因子值与前瞻收益
        merged_df = factor_data.join(forward_returns, how="inner").reset_index()
        merged_df = merged_df[["datetime", "symbol", "value", "fwd_ret"]].dropna()

        if merged_df.empty:
            return empty_result

        # ---- 计算每日横截面 IC/RankIC ----
        daily_records: List[Dict[str, Any]] = []
        for trade_date, daily_group in merged_df.groupby("datetime", sort=True):
            daily_group = daily_group.dropna()
            daily_records.append({
                "datetime": trade_date,
                "IC": self._compute_correlation(daily_group, "pearson"),
                "RankIC": self._compute_correlation(daily_group, "spearman"),
                "N": daily_group["symbol"].nunique(),
            })

        daily_ic_df = pd.DataFrame(daily_records).set_index("datetime").sort_index()

        # ---- 横截面 IC 汇总统计 ----
        ic_series = daily_ic_df["IC"].dropna()
        rank_ic_series = daily_ic_df["RankIC"].dropna()

        summary_df = pd.DataFrame({
            "IC.mean": [ic_series.mean() if not ic_series.empty else np.nan],
            "IC.t": [_compute_t_stat(ic_series)],
            "RankIC.mean": [rank_ic_series.mean() if not rank_ic_series.empty else np.nan],
            "RankIC.t": [_compute_t_stat(rank_ic_series)],
            "Days": [len(daily_ic_df)],
            "Avg.N": [daily_ic_df["N"].mean() if not daily_ic_df.empty else np.nan],
        })

        # ---- 时间序列 IC 统计（按股票维度） ----
        ts_records: List[Dict[str, Any]] = []
        for stock_symbol, stock_group in merged_df.groupby("symbol", sort=False):
            stock_group = stock_group.sort_values("datetime")
            # 时间序列 IC 需要足够的历史观测点
            if len(stock_group) < 10:
                continue

            ts_records.append({
                "symbol": stock_symbol,
                "TS.IC": stock_group["value"].corr(stock_group["fwd_ret"], method="pearson"),
                "TS.RankIC": stock_group["value"].corr(stock_group["fwd_ret"], method="spearman"),
                "T": len(stock_group),
            })

        ts_ic_df = pd.DataFrame(ts_records)

        if ts_ic_df.empty:
            ts_summary_df = pd.DataFrame({
                "TS.IC.mean": [np.nan],
                "TS.IC.t": [np.nan],
                "TS.RankIC.mean": [np.nan],
                "TS.RankIC.t": [np.nan],
                "Symbols": [0],
                "Avg.T": [np.nan],
            })
        else:
            ts_summary_df = pd.DataFrame({
                "TS.IC.mean": [ts_ic_df["TS.IC"].mean()],
                "TS.IC.t": [_compute_t_stat(ts_ic_df["TS.IC"])],
                "TS.RankIC.mean": [ts_ic_df["TS.RankIC"].mean()],
                "TS.RankIC.t": [_compute_t_stat(ts_ic_df["TS.RankIC"])],
                "Symbols": [len(ts_ic_df)],
                "Avg.T": [ts_ic_df["T"].mean()],
            })

        return {
            "daily_ic": daily_ic_df,
            "summary": summary_df,
            "ts_summary": ts_summary_df,
        }

    def save(
        self,
        results: Dict[str, pd.DataFrame],
        run_dir: Path,
    ) -> None:
        """保存 IC 评估结果到 JSONL/JSON 文件。

        输出文件：
            - ``daily_ic.jsonl``: 每日 IC/RankIC 数据
            - ``summary.json``: 横截面 IC 汇总统计
            - ``ts_summary.json``: 时间序列 IC 汇总统计

        Args:
            results: ``evaluate()`` 返回的结果字典。
            run_dir: 结果输出目录路径。
        """
        from alpha101_factory.utils.io import write_jsonl

        # 确保输出目录存在
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        write_jsonl(results["daily_ic"], run_dir / "daily_ic.jsonl")
        results["summary"].to_json(run_dir / "summary.json", orient="records")
        results["ts_summary"].to_json(run_dir / "ts_summary.json", orient="records")


# ============================================================================
# 分位组合评估器
# ============================================================================

class QuantileEvaluator(Evaluator):
    """分位组合（Quantile Portfolio）评估器。

    将股票按因子值分为 ``q`` 个分位组，计算各组的平均前瞻收益率及
    累积收益，同时输出多空组合（最高分位 - 最低分位）的收益表现。

    输出结果包含三个 DataFrame：
        - ``ports``: 各分位组的每日平均收益率（pivot 格式）
        - ``ls``: 多空组合（Long-Short）每日收益率
        - ``cumrets``: 各分位组及多空组合的累积收益

    注意:
        分位组合需要每天至少两只股票。样本不足的天数自动跳过。
    """

    name = "quantile"

    def evaluate(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        horizon: int,
        q: int = 5,
        **kwargs: Any,
    ) -> Dict[str, pd.DataFrame]:
        """执行分位组合评估。

        按因子值将股票分为 ``q`` 个分位组，计算各组平均前瞻收益率、
        多空组合收益率及累积收益。

        Args:
            factor_df: 因子数据，必须包含 ``datetime``、``symbol``、``value`` 列。
            price_df: 价格数据，必须包含 ``datetime``、``symbol``、``close`` 列。
            horizon: 前瞻收益期数，必须为正整数。
            q: 分位数数量，必须 >= 2。默认值为 5。
            **kwargs: 额外参数（当前未使用）。

        Returns:
            Dict[str, pd.DataFrame]: 包含 ``ports``、``ls``、``cumrets``
                三个键的结果字典。输入数据无效时返回空 DataFrame。
        """
        # 空结果模板
        empty_result = {
            "ports": pd.DataFrame(),
            "ls": pd.DataFrame(),
            "cumrets": pd.DataFrame(),
        }

        # 校验因子数据
        if factor_df is None or factor_df.empty:
            return empty_result

        required_factor_cols = {"datetime", "symbol", "value"}
        if not required_factor_cols.issubset(factor_df.columns):
            missing = required_factor_cols - set(factor_df.columns)
            warnings.warn(f"因子数据缺少必需列: {missing}", UserWarning)
            return empty_result

        # 校验 horizon 参数
        if not isinstance(horizon, int) or horizon < 1:
            warnings.warn(f"horizon 必须为正整数，当前值: {horizon}", UserWarning)
            return empty_result

        # 校验分位数参数
        if not isinstance(q, int) or q < 2:
            warnings.warn(f"分位数 q 必须 >= 2，当前值: {q}", UserWarning)
            return empty_result

        # 准备因子数据
        factor_data = factor_df.copy()
        factor_data["datetime"] = pd.to_datetime(factor_data["datetime"])

        # 计算前瞻收益率
        forward_returns = make_forward_return(
            price_df[["datetime", "symbol", "close"]],
            horizon=horizon,
        )
        if forward_returns is None:
            return empty_result

        # 合并因子值与前瞻收益
        forward_returns_df = forward_returns.reset_index()
        merged_df = factor_data.merge(
            forward_returns_df,
            on=["datetime", "symbol"],
            how="inner",
        )
        merged_df = merged_df[["datetime", "symbol", "value", "fwd_ret"]].dropna()

        if merged_df.empty:
            return empty_result

        def _assign_quantile(group_df: pd.DataFrame) -> pd.DataFrame:
            """为单日数据分配分位组标签。

            Args:
                group_df: 单日的因子数据 DataFrame。

            Returns:
                pd.DataFrame: 添加了 ``q`` 列（分位组标签）的 DataFrame。
                    样本不足时返回空 DataFrame。
            """
            group_df = group_df.dropna().copy()
            stock_count = group_df["symbol"].nunique()

            # 至少需要两只股票才能形成分位
            if stock_count < 2:
                return pd.DataFrame(columns=group_df.columns.tolist() + ["q"])

            # 实际分位数不能超过股票数量
            actual_quantiles = min(q, stock_count)

            # 使用排名值进行分位切割，避免因子值本身有大量重复
            rank_values = group_df["value"].rank(method="first")

            try:
                quantile_labels = list(range(1, actual_quantiles + 1))
                group_df["q"] = pd.qcut(
                    rank_values,
                    q=actual_quantiles,
                    labels=quantile_labels,
                    duplicates="drop",
                )

                # 检查实际产生的分位数是否足够（pd.qcut 可能因重复值减少分位数）
                if group_df["q"].nunique() < 2:
                    return pd.DataFrame(columns=group_df.columns.tolist() + ["q"])

            except Exception:
                # pd.qcut 可能因数据分布问题抛出异常（如所有值相同）
                return pd.DataFrame(columns=group_df.columns.tolist() + ["q"])

            return group_df

        # 按日期分组并分配分位标签
        quantile_dataframes: List[pd.DataFrame] = []
        for _, daily_group in merged_df.groupby("datetime", sort=True):
            assigned_group = _assign_quantile(daily_group)
            if not assigned_group.empty:
                quantile_dataframes.append(assigned_group)

        if not quantile_dataframes:
            return empty_result

        # 合并所有分位数据
        full_quantile_df = pd.concat(quantile_dataframes, ignore_index=True)

        # 计算各分位组的每日平均收益率
        portfolio_returns = (
            full_quantile_df.groupby(["datetime", "q"])["fwd_ret"]
            .mean()
            .reset_index()
        )
        portfolio_returns["q"] = portfolio_returns["q"].astype(int)

        # 转换为 pivot 格式：行为日期，列为分位组
        portfolio_pivot = portfolio_returns.pivot(
            index="datetime",
            columns="q",
            values="fwd_ret",
        ).sort_index()
        portfolio_pivot.columns = [f"Q{col}" for col in portfolio_pivot.columns]

        # 计算多空组合收益（最高分位 - 最低分位）
        long_short_df = pd.DataFrame()
        lowest_quantile_col = "Q1"
        highest_quantile_col = f"Q{q}"

        if (
            not portfolio_pivot.empty
            and lowest_quantile_col in portfolio_pivot.columns
            and highest_quantile_col in portfolio_pivot.columns
        ):
            long_short_series = (
                portfolio_pivot[highest_quantile_col] - portfolio_pivot[lowest_quantile_col]
            )
            long_short_df = long_short_series.rename("LS").to_frame()

        # 计算累积收益（NaN 视为 0 收益，即保持前一天的累积值）
        with np.errstate(invalid="ignore"):
            cumulative_returns = (1 + portfolio_pivot.fillna(0)).cumprod()
            if not long_short_df.empty and "LS" in long_short_df.columns:
                cumulative_returns["LS"] = (1 + long_short_df["LS"].fillna(0)).cumprod()

        return {
            "ports": portfolio_pivot,
            "ls": long_short_df,
            "cumrets": cumulative_returns,
        }

    def save(
        self,
        results: Dict[str, pd.DataFrame],
        run_dir: Path,
    ) -> None:
        """保存分位组合评估结果到 JSONL 文件。

        输出文件：
            - ``cumrets.jsonl``: 各分位组及多空组合的累积收益

        Args:
            results: ``evaluate()`` 返回的结果字典。
            run_dir: 结果输出目录路径。
        """
        from alpha101_factory.utils.io import write_jsonl

        # 确保输出目录存在
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        write_jsonl(results["cumrets"], run_dir / "cumrets.jsonl")


# ============================================================================
# 评估器注册表与工厂
# ============================================================================

# 全局评估器注册表：名称 -> 评估器类
_EVALUATOR_REGISTRY: Dict[str, Type[Evaluator]] = {}


def register_evaluator(cls: Type[Evaluator]) -> Type[Evaluator]:
    """注册评估器类到全局注册表。

    可作为装饰器使用，将评估器类的 ``name`` 属性作为键注册到注册表中。

    Args:
        cls: 要注册的评估器类，必须是 ``Evaluator`` 的子类。

    Returns:
        Type[Evaluator]: 原样返回输入的类，以便作为装饰器使用。

    Examples:
        作为装饰器使用::

            @register_evaluator
            class MyEvaluator(Evaluator):
                name = "my_evaluator"
                ...
    """
    _EVALUATOR_REGISTRY[cls.name] = cls
    return cls


# 自动注册内置评估器
register_evaluator(ICEvaluator)
register_evaluator(QuantileEvaluator)


class EvaluatorFactory:
    """评估器工厂类。

    根据名称创建评估器实例。支持通过 ``register_evaluator()`` 动态注册
    自定义评估器。

    Examples:
        创建内置评估器::

            ic_eval = EvaluatorFactory.create("ic")
            q_eval = EvaluatorFactory.create("quantile")

        列出所有可用评估器::

            print(EvaluatorFactory.list_all())  # ['ic', 'quantile']
    """

    @classmethod
    def create(cls, name: str) -> Evaluator:
        """根据名称创建评估器实例。

        Args:
            name: 评估器名称，必须是已注册的名称。

        Returns:
            Evaluator: 评估器实例。

        Raises:
            KeyError: 当指定的评估器名称未注册时抛出。
        """
        if name not in _EVALUATOR_REGISTRY:
            available = sorted(_EVALUATOR_REGISTRY.keys())
            raise KeyError(f"未知评估器: {name}. 可用评估器: {available}")
        return _EVALUATOR_REGISTRY[name]()

    @classmethod
    def list_all(cls) -> List[str]:
        """列出所有已注册的评估器名称。

        Returns:
            List[str]: 按字母排序的评估器名称列表。
        """
        return sorted(_EVALUATOR_REGISTRY.keys())
