# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).resolve().parents[1]))

from alpha101_factory.config import PARQ_DIR_FACT
from alpha101_factory.viz.factor_summary import (
    generate_all_factor_visuals,
    generate_factor_visuals,
)


def _sample_frame():
    dates = pd.date_range("2021-01-01", periods=6, freq="D")
    symbols = ["000001", "000002", "000003"]
    rows = []
    for dt in dates:
        for idx, sym in enumerate(symbols):
            rows.append({
                "datetime": dt,
                "symbol": sym,
                "value": (idx + 1) * 0.1 + dt.day / 100,
            })
    return pd.DataFrame(rows)


def test_generate_factor_visuals_returns_figures():
    df = _sample_frame()

    art = generate_factor_visuals("AlphaTest", frame=df, save=False, heatmap_top=2)

    assert art.factor == "AlphaTest"
    assert art.ts_symbol == "000001"
    assert isinstance(art.outputs["timeseries"], go.Figure)
    assert isinstance(art.outputs["cross_section"], go.Figure)
    assert "heatmap" in art.outputs
    assert isinstance(art.outputs["heatmap"], go.Figure)
    assert art.cross_section_dt == pd.Timestamp("2021-01-06")


def test_generate_all_factor_visuals_reads_parquet():
    df = _sample_frame()
    path = PARQ_DIR_FACT / "AlphaVisual.parquet"
    df.to_parquet(path, index=False)

    try:
        res = generate_all_factor_visuals(factors=["AlphaVisual"], save=False)
        assert "AlphaVisual" in res
        assert res["AlphaVisual"].ts_symbol == "000001"
        assert "heatmap" in res["AlphaVisual"].outputs
        assert isinstance(res["AlphaVisual"].outputs["heatmap"], go.Figure)
    finally:
        if path.exists():
            path.unlink()
