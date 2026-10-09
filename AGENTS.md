# AGENTS.md — alpha101_factory

Pluggable Alpha factor factory for A-share daily data.
Pipeline: **fetch (AkShare→BaoStock fallback) → tmp features (cached) → factor compute → backtest (IC/RankIC, quantile portfolios) → viz (Plotly+Kaleido PNG)**.

---

## Commands

```bash
pip install -r requirements.txt

python -m alpha101_factory.cli fetch                          # all stocks
python -m alpha101_factory.cli fetch-one --stock 600000       # single stock, saves K-line PNG
python -m alpha101_factory.cli tmp [--stock 600000]           # build cached tmp features
python -m alpha101_factory.cli factor --factors Alpha101      # compute one factor
python -m alpha101_factory.cli factor --all                   # compute all registered factors
python -m alpha101_factory.cli check                          # verify kline JSONL integrity
python -m alpha101_factory.backtest.run_bt --alpha Alpha101 --horizon 1 --quantiles 5
```

Quick smoke test: `bash test.sh`

---

## Config via environment variables (all optional)

| Variable | Default | Purpose |
|---|---|---|
| `ALPHA101_DATA_ROOT` | `./data` | Data directory root |
| `ALPHA101_ADJUST` | `qfq` | Price adjustment: `qfq` / `hfq` / `""` |
| `ALPHA101_START` | `20200101` | Global fetch start date |
| `ALPHA101_END` | `20250917` | Global fetch end date |
| `ALPHA101_LIMIT` | `0` | Debug: limit to N stocks (0 = all) |
| `ALPHA101_PAUSE` | `0.6` | Request throttle (seconds) |
| `ALPHA101_MAX_WORKERS` | `1` | Parallelism |

Override per-command via `fetch-one --start/--end/--adjust`.

---

## Data layout (all JSONL)

```
data/
  universe/stocks.jsonl                # stock pool: {code, name}
  quotes/
    spot/spot_YYYYMMDD.jsonl           # dated market snapshot
    daily/{symbol}.jsonl               # per-stock OHLCV
  features/{symbol}.jsonl              # cached intermediates (returns, vwap, advN…)
  factors/{AlphaName}.jsonl            # factor output: {datetime, symbol, value}
  backtest/
    {alpha}_h{h}_q{q}/                 # per-run directory
      daily_ic.jsonl                   # daily IC/RankIC
      cumrets.jsonl                    # cumulative returns
      summary.json                     # summary stats (JSON array)
      ts_summary.json                  # per-symbol TS-IC stats
  images/klines/                       # K-line PNGs
  images/backtest/                     # backtest charts (PNGs)
```

All data stored as JSONL (one JSON object per line). First line may be `{"_meta": true, ...}` for file-level metadata. Datetime stored as `"YYYY-MM-DD"` strings.

---

## Adding a new factor

1. Create `alpha101_factory/factors/alphas_*.py`
2. Subclass `Factor` from `factors/base.py`, set `name` and `requires`, implement `compute(df) -> pd.Series` with `MultiIndex[datetime, symbol]`
3. Decorate with `@register` from `factors/registry.py`
4. Use operators from `utils/ops.py` (rolling, ts_rank, decay_linear, cs_rank, etc.)
5. Run: `python -m alpha101_factory.cli factor --factors YourAlpha` then backtest

The registry auto-discovers `alphas_*.py` via `pkgutil` — no manual registration needed. `factors/__init__.py` explicitly imports `alphas_basic` and `alphas_more`.

---

## Key gotchas

- **CS-IC/RankIC is NaN for single stocks** — cross-sectional IC needs ≥2 stocks per day. Use `TS-IC`/`TS-RankIC` for single-stock analysis instead.
- **Quantile portfolios skip gracefully** — if a day has insufficient samples, it's skipped (no crash).
- **bottleneck/numba are optional** — `utils/ops.py` falls back to pure pandas if unavailable, but is significantly slower.
- **Kaleido required for PNG export** — Plotly figures save as PNG via kaleido; ensure it's installed.
- **tmp features are per-symbol** — each stock gets its own JSONL in `features/`. Factor computation reads and joins them into a panel.

---

## Python version

Requires Python 3.11+ (based on `.cpython-311` pycache).
