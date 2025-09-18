# -*- coding: utf-8 -*-
"""Alpha101 因子回测脚本.

本模块提供对指定因子进行横截面 IC/RankIC 分析、分位数组合回测，
并将结果保存为图表和 CSV 文件。
"""

import argparse
import numpy as np
import pandas as pd
from loguru import logger
import plotly.express as px

from ..config import PARQ_DIR_FACT, PARQ_DIR_KLINES, IMG_BT_DIR
from ..utils.io import read_parquet
from ..viz.plots import save_fig
from .metrics import ic_rankic, quantile_portfolios


def _load_prices(symbols: list[str]) -> pd.DataFrame:
    """加载指定股票的收盘价数据.

    Args:
        symbols (list[str]): 股票代码列表.

    Returns:
        pd.DataFrame: 包含 ['datetime', 'symbol', 'close'] 的行情数据.
    """
    dfs: list[pd.DataFrame] = []
    for sym in symbols:
        try:
            df = read_parquet(PARQ_DIR_KLINES / f"{sym}.parquet")
            if not df.empty:
                dfs.append(df[["datetime", "symbol", "close"]])
        except Exception as e:
            logger.warning(f"读取 {sym} 价格数据失败: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True).sort_values(["datetime", "symbol"])


def main() -> None:
    """命令行入口函数，执行因子回测流程."""
    ap = argparse.ArgumentParser("alpha101 backtest")
    ap.add_argument("--alpha", required=True, help="因子名称 (对应因子 parquet 文件名)")
    ap.add_argument("--horizon", type=int, default=1, help="前瞻收益期 (默认=1)")
    ap.add_argument("--quantiles", type=int, default=5, help="分位数数量 (默认=5)")
    args = ap.parse_args()

    # 读取因子文件
    fpath = PARQ_DIR_FACT / f"{args.alpha}.parquet"
    if not fpath.exists():
        logger.error(f"因子文件未找到: {fpath}")
        return

    try:
        f = pd.read_parquet(fpath)
    except Exception as e:
        logger.exception(f"读取因子文件失败: {e}")
        return

    # 获取涉及的股票代码并加载价格数据
    symbols = sorted(f["symbol"].unique().tolist())
    prices = _load_prices(symbols)
    if prices.empty:
        logger.error("未找到对应股票的行情数据，请先获取行情数据。")
        return

    # 计算 IC / RankIC
    try:
        res = ic_rankic(f, prices, horizon=args.horizon)
        daily, summary, ts_summary = res["daily"], res["summary"], res["ts_summary"]
        logger.info("\n[Cross-Sectional]\n" + str(summary))
        logger.info("\n[Time-Series per symbol]\n" + str(ts_summary))
    except Exception as e:
        logger.exception(f"IC/RankIC 计算失败: {e}")
        return

    # 绘制 IC/RankIC 曲线
    if not daily.dropna(how="all").empty:
        try:
            fig_ic = px.line(
                daily.reset_index(),
                x="datetime",
                y=["IC", "RankIC"],
                title=f"{args.alpha} IC/RankIC (h={args.horizon})"
            )
            save_fig(fig_ic, IMG_BT_DIR / f"{args.alpha}_IC_RankIC_h{args.horizon}.png")
        except Exception as e:
            logger.warning(f"IC/RankIC 绘图失败: {e}")
    else:
        logger.warning("无有效横截面 IC/RankIC，跳过绘图。")

    # 分位数组合分析
    try:
        ports = quantile_portfolios(f, prices, horizon=args.horizon, q=args.quantiles)
    except Exception as e:
        logger.exception(f"分位数组合构建失败: {e}")
        ports = {"ports": pd.DataFrame(), "ls": pd.DataFrame()}

    port_df, ls_df = ports["ports"], ports["ls"]

    # 输出结果并保存
    if not port_df.empty:
        try:
            with np.errstate(invalid="ignore"):
                cum = (1 + port_df.fillna(0)).cumprod()
                if not ls_df.empty and "LS" in ls_df.columns:
                    cum["LS"] = (1 + ls_df["LS"].fillna(0)).cumprod()

            fig_ports = px.line(
                cum.reset_index(),
                x="datetime",
                y=cum.columns,
                title=f"{args.alpha} Quantile cumrets (h={args.horizon}, q={args.quantiles})"
            )
            save_fig(fig_ports, IMG_BT_DIR / f"{args.alpha}_ports_h{args.horizon}_q{args.quantiles}.png")

            # 保存回测结果 CSV
            daily.to_csv(IMG_BT_DIR / f"{args.alpha}_daily_ic.csv", index=True, encoding="utf-8-sig")
            summary.to_csv(IMG_BT_DIR / f"{args.alpha}_summary.csv", index=False, encoding="utf-8-sig")
            ts_summary.to_csv(IMG_BT_DIR / f"{args.alpha}_ts_summary.csv", index=False, encoding="utf-8-sig")
            cum.to_csv(IMG_BT_DIR / f"{args.alpha}_cumrets.csv", index=True, encoding="utf-8-sig")
        except Exception as e:
            logger.exception(f"分位数组合绘图或结果保存失败: {e}")
    else:
        logger.warning("每日股票数不足，无法形成分位数组合。跳过组合绘图和累积收益计算。")
        daily.to_csv(IMG_BT_DIR / f"{args.alpha}_daily_ic.csv", index=True, encoding="utf-8-sig")
        summary.to_csv(IMG_BT_DIR / f"{args.alpha}_summary.csv", index=False, encoding="utf-8-sig")
        ts_summary.to_csv(IMG_BT_DIR / f"{args.alpha}_ts_summary.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
