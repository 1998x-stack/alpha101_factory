# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from alpha101_factory.viz import factor_summaryx
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


@pytest.fixture
def patched_factor_env(monkeypatch, tmp_path):
    factors_dir = tmp_path / "factors"
    factors_dir.mkdir()
    monkeypatch.setattr(factor_summary, "PARQ_DIR_FACT", factors_dir)

    ts_dir = tmp_path / "ts"
    cs_dir = tmp_path / "cs"
    heat_dir = tmp_path / "heat"
    for attr, path in {
        "IMG_FACTORS_TS_DIR": ts_dir,
        "IMG_FACTORS_CS_DIR": cs_dir,
        "IMG_FACTORS_HEATMAP_DIR": heat_dir,
    }.items():
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(factor_summary, attr, path)

    def _fake_save(fig, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(factor_summary, "save_fig", _fake_save)
    return factors_dir


def test_generate_factor_visuals_returns_figures(patched_factor_env):
    df = _sample_frame()

    art = generate_factor_visuals("AlphaTest", frame=df, save=False, heatmap_top=2)

    assert art.factor == "AlphaTest"
    assert art.ts_symbol == "000001"
    assert isinstance(art.outputs["timeseries"], go.Figure)
    assert isinstance(art.outputs["cross_section"], go.Figure)
    assert "heatmap" in art.outputs
    assert isinstance(art.outputs["heatmap"], go.Figure)
    assert art.cross_section_dt == pd.Timestamp("2021-01-06")


def test_generate_factor_visuals_fallback_symbol(patched_factor_env):
    df = _sample_frame()
    df = df[df["symbol"] != "000003"]

    art = generate_factor_visuals(
        "AlphaFallback",
        frame=df,
        ts_symbol="000003",
        heatmap_top=0,
        save=False,
    )

    assert art.ts_symbol == "000001"
    assert "heatmap" not in art.outputs


def test_generate_factor_visuals_heatmap_top_controls_output(patched_factor_env):
    df = _sample_frame()

    art = generate_factor_visuals(
        "AlphaHeat",
        frame=df,
        save=True,
        heatmap_top=1,
    )

    assert "heatmap" in art.outputs
    heat_path = art.outputs["heatmap"]
    assert heat_path.name.startswith("AlphaHeat_000001")


def test_generate_factor_visuals_skip_heatmap_when_zero(patched_factor_env):
    df = _sample_frame()

    art = generate_factor_visuals(
        "AlphaZero",
        frame=df,
        save=True,
        heatmap_top=0,
    )

    assert "heatmap" not in art.outputs


def test_generate_all_factor_visuals_reads_parquet(patched_factor_env):
    df = _sample_frame()
    path = factor_summary.PARQ_DIR_FACT / "AlphaVisual.parquet"
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

def test_generate_all_factor_visuals_prefix_and_limit(patched_factor_env):
    df = _sample_frame()
    for name in ["AlphaOne", "AlphaTwo", "BetaX"]:
        df.to_parquet(factor_summary.PARQ_DIR_FACT / f"{name}.parquet", index=False)

    res = generate_all_factor_visuals(prefix="Alpha", limit=1, save=False)

    assert len(res) == 1
    assert list(res.keys())[0].startswith("Alpha")
