from typing import Dict, Optional
import pandas as pd
import numpy as np


def make_forward_return(price_df: pd.DataFrame, horizon: int = 1) -> Optional[pd.Series]:
    """计算前瞻收益率 (forward return)。

    前瞻收益率定义为未来 horizon 天后的收盘价相对当前收盘价的涨跌幅。

    Args:
        price_df (pd.DataFrame): 包含至少 'symbol'、'datetime'、'close' 三列的行情数据。
        horizon (int, optional): 时间跨度，默认为 1。

    Returns:
        Optional[pd.Series]: MultiIndex (datetime, symbol) 的收益率序列。
            若输入不合法或异常则返回 None。
    """
    try:
        # 检查输入必要字段
        required_cols = {"symbol", "datetime", "close"}
        if not required_cols.issubset(price_df.columns):
            raise ValueError(f"缺少必要字段: {required_cols - set(price_df.columns)}")

        # 若数据为空直接返回 None
        if price_df.empty:
            return None

        # 按 symbol、datetime 排序保证时序正确
        p = price_df.sort_values(["symbol", "datetime"]).copy()

        # 计算 horizon 步长的收益率，并向后移动以对齐到当前时点
        ret = p.groupby("symbol")["close"].pct_change(horizon).shift(-horizon)

        # 构造 MultiIndex 方便后续索引操作
        idx = pd.MultiIndex.from_frame(
            p[["datetime", "symbol"]], names=["datetime", "symbol"]
        )
        return pd.Series(ret.values, index=idx, name="fwd_ret")
    except Exception as e:
        print(f"[make_forward_return] 错误: {e}")
        return None


def _t_stat(x: pd.Series) -> float:
    """计算序列的 t 统计量，用于显著性检验。"""
    try:
        x = pd.Series(x).dropna()
        if len(x) < 2:
            return np.nan
        mu, sd, n = x.mean(), x.std(ddof=1), len(x)
        if sd == 0 or n < 2:
            return np.nan
        return mu / (sd / np.sqrt(n))
    except Exception:
        return np.nan


def _pearson(g: pd.DataFrame) -> float:
    """计算横截面 Pearson 相关系数 (IC)。"""
    try:
        return np.nan if g["symbol"].nunique() < 2 else g["value"].corr(g["fwd_ret"], method="pearson")
    except Exception:
        return np.nan


def _spearman(g: pd.DataFrame) -> float:
    """计算横截面 Spearman 相关系数 (RankIC)。"""
    try:
        return np.nan if g["symbol"].nunique() < 2 else g["value"].corr(g["fwd_ret"], method="spearman")
    except Exception:
        return np.nan


def ic_rankic(factor_df: pd.DataFrame, price_df: pd.DataFrame, horizon: int = 1) -> Dict[str, pd.DataFrame]:
    """计算因子的横截面 IC、RankIC 以及时间序列 TS.IC 指标。

    Args:
        factor_df (pd.DataFrame): 包含 ['datetime', 'symbol', 'value'] 的因子数据。
        price_df (pd.DataFrame): 包含 ['datetime', 'symbol', 'close'] 的行情数据。
        horizon (int, optional): 前瞻收益期，默认 1。

    Returns:
        Dict[str, pd.DataFrame]: 包含 'daily'、'summary'、'ts_summary' 三个结果。
    """
    try:
        f = factor_df.copy()
        f["datetime"] = pd.to_datetime(f["datetime"])
        f.set_index(["datetime", "symbol"], inplace=True)

        # 计算前瞻收益率
        px = price_df[["datetime", "symbol", "close"]].copy()
        fwd = make_forward_return(px, horizon=horizon)
        if fwd is None:
            return {"daily": pd.DataFrame(), "summary": pd.DataFrame(), "ts_summary": pd.DataFrame()}

        # 合并数据
        df = f.join(fwd, how="inner").reset_index()
        df = df[["datetime", "symbol", "value", "fwd_ret"]].dropna()

        # 每日横截面 IC/RankIC
        rows = []
        for dt, g in df.groupby("datetime", sort=True):
            g = g.dropna()
            n = g["symbol"].nunique()
            rows.append({"datetime": dt, "IC": _pearson(g), "RankIC": _spearman(g), "N": n})
        daily = pd.DataFrame(rows).set_index("datetime").sort_index()

        # 汇总统计
        ic_series, ric_series = daily["IC"].dropna(), daily["RankIC"].dropna()
        summary = pd.DataFrame({
            "IC.mean": [ic_series.mean() if not ic_series.empty else np.nan],
            "IC.t": [_t_stat(ic_series)],
            "RankIC.mean": [ric_series.mean() if not ric_series.empty else np.nan],
            "RankIC.t": [_t_stat(ric_series)],
            "Days": [len(daily)],
            "Avg.N": [daily["N"].mean() if not daily.empty else np.nan]
        })

        # 各 symbol 的时间序列相关性
        ts_rows = []
        for sym, g in df.groupby("symbol", sort=False):
            g = g.sort_values("datetime")
            if len(g) < 10:
                continue
            ts_rows.append({
                "symbol": sym,
                "TS.IC": g["value"].corr(g["fwd_ret"], method="pearson"),
                "TS.RankIC": g["value"].corr(g["fwd_ret"], method="spearman"),
                "T": len(g)
            })
        ts = pd.DataFrame(ts_rows)
        if ts.empty:
            ts_summary = pd.DataFrame({"TS.IC.mean": [np.nan], "TS.IC.t": [np.nan],
                                       "TS.RankIC.mean": [np.nan], "TS.RankIC.t": [np.nan],
                                       "Symbols": [0], "Avg.T": [np.nan]})
        else:
            ts_summary = pd.DataFrame({
                "TS.IC.mean": [ts["TS.IC"].mean()],
                "TS.IC.t": [_t_stat(ts["TS.IC"])],
                "TS.RankIC.mean": [ts["TS.RankIC"].mean()],
                "TS.RankIC.t": [_t_stat(ts["TS.RankIC"])],
                "Symbols": [len(ts)], "Avg.T": [ts["T"].mean()]
            })
        return {"daily": daily, "summary": summary, "ts_summary": ts_summary}
    except Exception as e:
        print(f"[ic_rankic] 错误: {e}")
        return {"daily": pd.DataFrame(), "summary": pd.DataFrame(), "ts_summary": pd.DataFrame()}


def quantile_portfolios(factor_df: pd.DataFrame, price_df: pd.DataFrame,
                        horizon: int = 1, q: int = 5) -> Dict[str, pd.DataFrame]:
    """构建分位数组合，并计算多空组合表现。

    Args:
        factor_df (pd.DataFrame): 包含 ['datetime', 'symbol', 'value'] 的因子数据。
        price_df (pd.DataFrame): 包含 ['datetime', 'symbol', 'close'] 的行情数据。
        horizon (int, optional): 前瞻收益期，默认 1。
        q (int, optional): 分组数量，默认 5。

    Returns:
        Dict[str, pd.DataFrame]: 包含 'ports'（分组组合收益）和 'ls'（多空组合）两个结果。
    """
    try:
        f = factor_df.copy()
        f["datetime"] = pd.to_datetime(f["datetime"])

        # 计算前瞻收益率
        px = price_df[["datetime", "symbol", "close"]].copy()
        fwd = make_forward_return(px, horizon=horizon)
        if fwd is None:
            return {"ports": pd.DataFrame(), "ls": pd.DataFrame()}

        fwd = fwd.reset_index()
        f = f.merge(fwd, on=["datetime", "symbol"], how="inner")
        f = f[["datetime", "symbol", "value", "fwd_ret"]].dropna()

        # 内部函数：对单日数据分组
        def _assign_quantiles(g: pd.DataFrame) -> pd.DataFrame:
            g = g.dropna().copy()
            n = g["symbol"].nunique()
            if n < 2:
                return pd.DataFrame(columns=g.columns.tolist() + ["q"])
            k = min(q, n)
            r = g["value"].rank(method="first")
            try:
                labels = list(range(1, k + 1))
                g["q"] = pd.qcut(r, q=k, labels=labels, duplicates="drop")
                if pd.Series(g["q"]).nunique() < 2:
                    return pd.DataFrame(columns=g.columns.tolist() + ["q"])
            except Exception:
                return pd.DataFrame(columns=g.columns.tolist() + ["q"])
            return g

        fq_list = []
        for _, g in f.groupby("datetime", sort=True):
            gg = _assign_quantiles(g)
            if not gg.empty:
                fq_list.append(gg)

        if not fq_list:
            return {"ports": pd.DataFrame(), "ls": pd.DataFrame()}

        fq = pd.concat(fq_list, ignore_index=True)

        # 计算分位数组合收益
        port = fq.groupby(["datetime", "q"])["fwd_ret"].mean().reset_index()
        port["q"] = port["q"].astype(int)
        port_pivot = port.pivot(index="datetime", columns="q", values="fwd_ret").sort_index()
        port_pivot.columns = [f"Q{c}" for c in port_pivot.columns]

        # 多空组合（最高分组 - 最低分组）
        ls = pd.DataFrame()
        if not port_pivot.empty and "Q1" in port_pivot.columns and f"Q{q}" in port_pivot.columns:
            ls = (port_pivot[f"Q{q}"] - port_pivot["Q1"]).rename("LS").to_frame()

        return {"ports": port_pivot, "ls": ls}
    except Exception as e:
        print(f"[quantile_portfolios] 错误: {e}")
        return {"ports": pd.DataFrame(), "ls": pd.DataFrame()}
