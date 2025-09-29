# Project Overview: alpha101_factory

`alpha101_factory` is a **pluggable alpha-factor research factory** tailored for China A-share daily data. The toolkit streamlines the full research workflow:

> Data acquisition (AkShare with Baostock fallback) → Feature caching (tmp features) → Factor computation (Alpha101 & beyond) → Evaluation & backtesting (IC/RankIC & quantile portfolios) → Visualization & artifact export (Parquet/PNG/CSV)

The guiding principles are “ready to run” for newcomers and “easy to extend” for power users.

---

## Repository Layout & Responsibilities

```
├── __init__.py
├── backtest                  # Backtesting & evaluation helpers
│   ├── metrics.py            # IC/RankIC, forward return, quantile portfolios, robustness helpers
│   └── run_bt.py             # Backtest CLI (`--alpha`, `--horizon`, `--quantiles`) producing plots/CSV
├── cli.py                    # Top-level CLI: fetch / fetch-one / tmp / factor / check / visualize
├── config.py                 # Paths, environment variables, sampling options, rate limits, image dirs
├── data                      # Data acquisition & loading
│   ├── baostock_api.py       # Baostock fallback wrapper (when AkShare fails)
│   ├── loader.py             # Fetch spot & kline data, validation, persistence, K-line image saving
│   └── universe.py           # Stock universe definition (customize as needed)
├── factors                   # Factor catalogue
│   ├── __init__.py           # Auto-discovery/import for factor modules (triggers registration)
│   ├── alphas_basic.py       # Example alpha implementations
│   ├── base.py               # Factor abstraction & IO contract
│   ├── registry.py           # `@register` decorator + discovery + listing helpers
│   └── tmp_features.py       # Intermediate feature builders: returns/vwap/adv etc.
├── pipeline                  # Dataflow orchestration
│   ├── build_tmp.py          # Optional CLI stub for tmp-only workflows
│   ├── check_data.py         # Parquet integrity checker
│   └── compute_factor.py     # Assemble panel data → compute factors → persist `factors/*.parquet`
├── utils                     # Shared utilities
│   ├── io.py                 # Robust parquet IO helpers
│   ├── log.py                # Loguru bootstrapper
│   └── ops.py                # Accelerated ops: bottleneck/numba rolling ops, ts_rank, decay_linear, etc.
└── viz
    └── plots.py              # Plotly helpers (kline/factor timeseries/cross-section/heatmap) + PNG saving
```

---

## Data Flow & Produced Artifacts

1. **fetch / fetch-one**
   * `data/loader.py`
     * Uses **AkShare** first; automatically falls back to **Baostock** on failure.
     * Produces:
       * `data/spot/a_spot.parquet` — spot snapshot
       * `data/klines_daily/{symbol}.parquet` — daily klines (datetime/open/high/low/close/volume/amount…)
       * `images/klines/{symbol}_{start}_{end}_{adjust}.png` — auto-generated kline charts
   * `check_klines_integrity()` validates file presence, row counts, and date ranges.

2. **tmp (intermediate features)**
   * `factors/tmp_features.py`
   * Reads kline data → computes grouped features: returns, vwap, advN, etc.
   * Outputs: `data/tmp_features/{symbol}.parquet`
   * Designed to **avoid recomputation** and share derived features across factors.

3. **factor (factor computation)**
   * `pipeline/compute_factor.py` stitches the panel (klines × tmp features).
   * `factors/base.py` defines each factor class signature `compute(df) -> Series[MultiIndex(datetime, symbol)]`.
   * `factors/registry.py` + `factors/__init__.py`:
     * Register factors via `@register`.
     * Auto-discover `alphas_*.py`; the CLI can target `--factors Alpha101` or `--all`.
   * Outputs: `data/factors/{AlphaName}.parquet` with columns `[datetime, symbol, value]`.

4. **backtest (evaluation)**
   * `backtest/metrics.py`
     * **Forward returns**: compute `t → t+h` forward return on close prices and align back to `t`.
     * **IC / RankIC (cross-sectional)**: align `factor vs fwd_ret` per day; skip days with <2 symbols.
     * **TS-IC / TS-RankIC (per symbol)**: provide longitudinal summaries when coverage is sparse.
     * **Quantile portfolios**: bucket by factor (via `qcut(duplicates='drop')`); gracefully skip low-sample days.
   * `backtest/run_bt.py`
     * CLI options: `--alpha --horizon --quantiles`.
     * Generates charts:
       * `images/backtest/{alpha}_IC_RankIC_h{h}.png`
       * `images/backtest/{alpha}_ports_h{h}_q{q}.png` (cumulative returns incl. long/short)
     * Generates tables:
       * `daily_ic.csv / summary.csv / ts_summary.csv / cumrets.csv`

5. **visualize (factor dashboards)**
   * `viz/factor_summary.py` reads factor parquet outputs and renders default charts:
     * per-symbol time series
     * latest cross-sectional snapshot
     * heatmaps (time × symbol) with configurable symbol selection
   * CLI: `python -m alpha101_factory.cli visualize --all --prefix Alpha`
   * Artifacts saved under `images/factors/{timeseries,cross_section,heatmap}`.

---

## Configuration Highlights (`config.py`)

* `DATA_ROOT` (default `./data`) with sub-folders: `spot`, `klines_daily`, `tmp_features`, `factors`, `images/...`.
* `ADJUST`: `qfq` / `hfq` / `''`.
* `START_DATE` / `END_DATE`: global fetch range (CLI overrides available via `fetch-one`).
* `LIMIT_STOCKS`: restrict the stock universe for debugging.
* `REQUEST_PAUSE`: throttle network requests.
* Image directories: `images/klines`, `images/backtest`, `images/factors/*`.

---

## Typical CLI Workflow

```bash
# 1) Fetch data (full universe or limited by ALPHA101_LIMIT)
python -m alpha101_factory.cli fetch

# 2) Fetch or refresh a single symbol and persist PNG
python -m alpha101_factory.cli fetch-one --stock 600000 --start 20200101 --end 20240101 --adjust qfq

# 3) Build intermediate features (optionally for one stock)
python -m alpha101_factory.cli tmp --stock 600000

# 4) Compute factors (single factor, multiple, or --all)
python -m alpha101_factory.cli factor --factors Alpha101 --stock 600000

# 5) Backtest and generate evaluation artifacts
python -m alpha101_factory.backtest.run_bt --alpha Alpha101 --horizon 1 --quantiles 5

# 6) Visualize factor outputs and save PNG dashboards
python -m alpha101_factory.cli visualize --all --prefix Alpha
```

---

## Testing

Run the automated test suite (includes visualization sanity checks) via:

```bash
pytest
```

---

## Contributing

1. Create a virtual environment and install dependencies from `requirements.txt`.
2. Format & lint prior to submitting PRs (black, isort, flake8, etc. if applicable).
3. Add or update tests to cover new behaviours.
4. Document new CLI options, configuration flags, or data outputs in both README files.

---

## License

This project inherits the license specified in the repository root.
