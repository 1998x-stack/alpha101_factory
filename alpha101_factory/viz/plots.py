# -*- coding: utf-8 -*-
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

def _ensure_datetime_series(s: pd.Series) -> pd.Series:
    """Coerce many possible inputs (datetime64, int ns/ms/s, strings) into pandas datetime (naive)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, utc=False, errors="coerce")
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        v = pd.to_numeric(s, errors="coerce")
        m = v.dropna().abs().median()
        # heuristics for epoch unit
        # ~1e18: ns, ~1e12: ms, ~1e9: s
        if m > 1e14:
            unit = "ns"
        elif m > 1e11:
            unit = "ms"
        else:
            unit = "s"
        return pd.to_datetime(v, unit=unit, errors="coerce")
    # strings or objects
    return pd.to_datetime(s, errors="coerce")

def _datetime_array_for_plot(s: pd.Series):
    dt = _ensure_datetime_series(s).dt.tz_localize(None)
    # plotly + kaleido are happiest with Python datetimes
    return dt.dt.to_pydatetime()

def plot_kline(df: pd.DataFrame, title: str = "Kline",
               tickformat: str = "%Y-%m-%d", tickangle: int = -45):
    d = df.dropna().copy()
    d = d.sort_values("datetime")
    x = _datetime_array_for_plot(d["datetime"])
    fig = go.Figure(data=[go.Candlestick(
        x=x, open=d["open"], high=d["high"], low=d["low"], close=d["close"]
    )])
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=520)
    fig.update_xaxes(type="date", tickformat=tickformat, tickangle=tickangle, ticks="outside")
    return fig

def plot_factor_timeseries(fdf: pd.DataFrame, symbol: str, title: str,
                           tickformat: str = "%Y-%m-%d", tickangle: int = -45):
    d = fdf[fdf["symbol"] == symbol].copy().sort_values("datetime")
    x = _datetime_array_for_plot(d["datetime"])
    fig = px.line(x=x, y=d["value"], title=f"{title} | {symbol}")
    fig.update_layout(height=420)
    fig.update_xaxes(type="date", tickformat=tickformat, tickangle=tickangle, ticks="outside")
    return fig

def plot_factor_cross_section(fdf: pd.DataFrame, dt=None, topn: int = 100,
                              title: str = "Factor x-section",
                              tickformat: str = "%Y-%m-%d", tickangle: int = -45):
    if dt is None:
        dt = pd.to_datetime(fdf["datetime"]).max()
    d = fdf.copy()
    d["datetime"] = _ensure_datetime_series(d["datetime"])
    d = d[d["datetime"] == pd.to_datetime(dt)].copy()
    d["abs"] = d["value"].abs()
    d = d.sort_values("abs", ascending=False).head(topn)
    fig = px.bar(d, x="symbol", y="value", title=f"{title} | {pd.to_datetime(dt).date()}")
    fig.update_layout(height=420, xaxis={'categoryorder':'total descending'})
    # categorical x; keep the date in title; leave axis as category
    return fig

def plot_heatmap(fdf: pd.DataFrame, symbols: list[str], title: str = "Factor heatmap",
                 tickformat: str = "%Y-%m-%d", tickangle: int = -45):
    d = fdf[fdf["symbol"].isin(symbols)].copy()
    d["datetime"] = _ensure_datetime_series(d["datetime"])
    pvt = d.pivot_table(index="datetime", columns="symbol", values="value")
    fig = px.imshow(pvt.T, aspect="auto", origin="lower", title=title)
    fig.update_layout(height=500)
    fig.update_xaxes(type="date", tickformat=tickformat, tickangle=tickangle, ticks="outside")
    return fig

def save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path))  # needs kaleido
    return path
