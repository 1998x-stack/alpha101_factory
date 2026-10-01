# -*- coding: utf-8 -*-
"""高层因子可视化入口。

该模块负责读取 `PARQ_DIR_FACT` 下的因子结果，并生成默认的三类图像：

1. 单只股票的因子时间序列；
2. 最新交易日的截面分布；
3. 热力图（时间 × 股票）。

所有图像会保存至 ``images/factors`` 目录的子文件夹中，也可以在测试
场景中通过 ``save=False`` 仅返回 ``plotly`` 图形对象以便断言。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, MutableMapping, Optional, Sequence

import pandas as pd
from loguru import logger

from alpha101_factory.config import (
    IMG_FACTORS_CS_DIR,
    IMG_FACTORS_HEATMAP_DIR,
    IMG_FACTORS_TS_DIR,
    PARQ_DIR_FACT,
)
from alpha101_factory.utils.io import read_parquet
from alpha101_factory.viz.plots import (
    plot_factor_cross_section,
    plot_factor_timeseries,
    plot_heatmap,
    save_fig,
)


@dataclass
class FactorVisualArtifacts:
    """封装单个因子的可视化产物信息。"""

    factor: str
    ts_symbol: Optional[str]
    cross_section_dt: Optional[pd.Timestamp]
    outputs: MutableMapping[str, Path | object]


def _coerce_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df.dropna(subset=["datetime", "symbol", "value"], how="any")


def _select_symbol(df: pd.DataFrame, preferred: Optional[str]) -> Optional[str]:
    if df.empty:
        return None
    symbols = sorted(df["symbol"].dropna().unique())
    if not symbols:
        return None
    if preferred and preferred in symbols:
        return preferred
    if preferred and preferred not in symbols:
        logger.warning(
            "首选标的 {symbol} 在因子数据中缺失，自动改用 {fallback}",
            symbol=preferred,
            fallback=symbols[0],
        )
    return symbols[0]


def _select_heatmap_symbols(
    df: pd.DataFrame, heatmap_symbols: Optional[Sequence[str]], max_count: int
) -> List[str]:
    if df.empty:
        return []

    all_counts = df["symbol"].value_counts()
    if heatmap_symbols:
        filtered = [s for s in heatmap_symbols if s in all_counts.index]
        if filtered:
            return filtered[:max_count]
        logger.warning(
            "指定的热力图股票全部缺失，将改用覆盖度最高的 {count} 只", count=max_count
        )

    return list(all_counts.head(max_count).index)


def _load_factor_frame(factor_name: str) -> pd.DataFrame:
    path = PARQ_DIR_FACT / f"{factor_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = read_parquet(path)
    if df is None:
        return pd.DataFrame(columns=["datetime", "symbol", "value"])
    return df


def generate_factor_visuals(
    factor_name: str,
    *,
    frame: Optional[pd.DataFrame] = None,
    ts_symbol: Optional[str] = None,
    heatmap_symbols: Optional[Sequence[str]] = None,
    cross_section_dt: Optional[pd.Timestamp] = None,
    heatmap_top: int = 12,
    save: bool = True,
) -> FactorVisualArtifacts:
    """为单个因子生成可视化。"""

    if frame is None:
        try:
            frame = _load_factor_frame(factor_name)
        except FileNotFoundError as exc:
            logger.error("未找到因子 {factor} 的结果文件: {path}", factor=factor_name, path=exc)
            return FactorVisualArtifacts(factor_name, None, None, {})

    df = _coerce_datetime(frame)
    if df.empty:
        logger.error("因子 %s 无有效数据，跳过绘图", factor_name)
        return FactorVisualArtifacts(factor_name, None, None, {})

    symbol = _select_symbol(df, ts_symbol)
    if symbol is None:
        logger.error("因子 %s 没有可用于时间序列的标的", factor_name)
        return FactorVisualArtifacts(factor_name, None, None, {})

    dt = cross_section_dt
    if dt is None:
        dt = pd.to_datetime(df["datetime"]).max()

    heatmap_list = _select_heatmap_symbols(df, heatmap_symbols, heatmap_top)

    outputs: MutableMapping[str, Path | object] = {}

    ts_fig = plot_factor_timeseries(df, symbol=symbol, title=factor_name)
    cs_fig = plot_factor_cross_section(df, dt=dt, title=factor_name)
    heat_fig = None
    if heatmap_list:
        heat_fig = plot_heatmap(df, symbols=heatmap_list, title=factor_name)
    else:
        logger.warning("因子 %s 无法生成热力图，缺少足够的股票样本。", factor_name)

    if save:
        ts_path = IMG_FACTORS_TS_DIR / f"{factor_name}_{symbol}.png"
        cs_suffix = pd.to_datetime(dt).strftime("%Y%m%d") if pd.notna(dt) else "latest"
        cs_path = IMG_FACTORS_CS_DIR / f"{factor_name}_{cs_suffix}.png"
        heat_name = "_".join(heatmap_list[:5]) or "all"
        heat_path = IMG_FACTORS_HEATMAP_DIR / f"{factor_name}_{heat_name}.png"

        outputs["timeseries"] = save_fig(ts_fig, ts_path)
        outputs["cross_section"] = save_fig(cs_fig, cs_path)
        if heat_fig is not None:
            outputs["heatmap"] = save_fig(heat_fig, heat_path)
    else:
        outputs["timeseries"] = ts_fig
        outputs["cross_section"] = cs_fig
        if heat_fig is not None:
            outputs["heatmap"] = heat_fig

    return FactorVisualArtifacts(factor_name, symbol, pd.to_datetime(dt), outputs)


def _discover_factor_names(prefix: str = "Alpha") -> List[str]:
    files = sorted(PARQ_DIR_FACT.glob("*.parquet"))
    names = [p.stem for p in files if not prefix or p.stem.startswith(prefix)]
    return names


def generate_all_factor_visuals(
    *,
    factors: Optional[Sequence[str]] = None,
    prefix: str = "Alpha",
    ts_symbol: Optional[str] = None,
    heatmap_symbols: Optional[Sequence[str]] = None,
    heatmap_top: int = 12,
    limit: Optional[int] = None,
    save: bool = True,
) -> Mapping[str, FactorVisualArtifacts]:
    """批量渲染多个因子的可视化。"""

    if factors is None or not factors:
        factors = _discover_factor_names(prefix)

    if limit is not None and limit > 0:
        factors = list(factors)[:limit]

    results: MutableMapping[str, FactorVisualArtifacts] = {}

    for name in factors:
        art = generate_factor_visuals(
            name,
            ts_symbol=ts_symbol,
            heatmap_symbols=heatmap_symbols,
            heatmap_top=heatmap_top,
            save=save,
        )
        if art.outputs:
            results[name] = art
    return results


__all__ = [
    "FactorVisualArtifacts",
    "generate_factor_visuals",
    "generate_all_factor_visuals",
]

