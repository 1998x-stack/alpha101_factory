# alpha101_factory — Comprehensive Project Analysis

**Date:** 2026-03-30
**Scope:** Full file-by-file reconstruction, bug registry, dependency analysis, test coverage gaps

---

## 1. Project Overview

**Purpose:** Pluggable Alpha101 factor factory for A-share (Chinese stock market) daily kline data. Provides an end-to-end pipeline: data fetching -> feature caching -> factor computation -> backtesting -> visualization.

**Architecture:**
```
CLI (cli.py)
  |-- fetch / fetch-one  -> data/loader.py (AkShare -> Baostock fallback)
  |-- tmp                -> factors/tmp_features.py (returns, vwap, advN)
  |-- factor             -> pipeline/compute_factor.py -> factors/registry.py -> alphas_*.py
  |-- check              -> pipeline/check_data.py
  +-- visualize          -> viz/factor_summary.py -> viz/plots.py

Backtest: backtest/run_bt.py -> backtest/metrics.py
Config:   config.py (env vars, paths)
Utils:    utils/ops.py (bottleneck/numba), utils/io.py, utils/log.py
```

**Data Flow:**
1. `fetch` -> AkShare/Baostock -> `data/klines_daily/{sym}_{start}_{end}_{adj}.parquet` + K-line PNG
2. `tmp` -> klines -> returns/vwap/advN -> `data/tmp_features/{sym}_{start}_{end}_{adj}.parquet`
3. `factor` -> merge(klines, tmp) -> Factor.compute() -> `data/factors/{AlphaName}.parquet`
4. `backtest` -> factors + prices -> IC/RankIC + quantile portfolios -> CSV + PNG
5. `visualize` -> factors -> timeseries/cross-section/heatmap PNGs

**Stats:** ~4,700 lines of Python across 20 modules, 2 test files (76 lines, 4 tests), 55+ Alpha factors.

---

## 2. Category-by-Category File Reconstruction

### 2.1 Data Layer

#### `alpha101_factory/data/loader.py` (320 lines)

**Purpose:** A-share kline data fetching, normalization, local caching, and integrity checking. Default: AkShare with Baostock fallback.

**Imports:** pandas, akshare, tqdm, loguru, config.*, utils.io, viz.plots, data.baostock_api

**Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `_resolve_kline_path` | `(symbol: str, start_date: str\|None, end_date: str\|None, adjust: str\|None) -> Path` | Generates parquet file path, strips exchange prefix (e.g., `sh600000` -> `600000`) |
| `normalize_k` | `(df: DataFrame) -> DataFrame` | Renames Chinese column names to English, converts datetime and numeric types |
| `_fetch_kline_ak` | `(symbol: str, start_date: str\|None, end_date: str\|None, adjust: str) -> DataFrame` | AkShare API call for daily kline data |
| `_fetch_kline_fallback` | `(symbol: str, start_date: str\|None, end_date: str\|None, adjust: str) -> DataFrame` | AkShare first, Baostock on failure |
| `_save_kline_png` | `(sym: str, df: DataFrame, start_date: str\|None, end_date: str\|None, adjust: str) -> Path` | Renders K-line chart via plotly+kaleido |
| `fetch_spot` | `(save: bool = True) -> DataFrame` | Gets/caches A-share real-time snapshot |
| `fetch_klines_from_spot` | `(spot: DataFrame) -> int` | Batch downloads klines for the universe |
| `check_klines_integrity` | `() -> DataFrame` | Validates local parquet files, reports existence/rows/date range |
| `load_or_fetch_symbol` | `(symbol, start_date, end_date, adjust, save_image) -> DataFrame` | Load local parquet or download single stock |

**Issues Found:**
- [MEDIUM] `_fetch_kline_fallback` strips digits from symbol, then calls `_fetch_kline_ak` which strips again and logs again (double-stripping, double-logging)
- [LOW] `fetch_spot` caches without staleness check -- always returns cached file if it exists
- [LOW] `sys.path.append` at module level is fragile

---

#### `alpha101_factory/data/baostock_api.py` (118 lines)

**Purpose:** Baostock K-line data fetching as fallback data source.

**Imports:** pandas, baostock, loguru

**Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `bs_code` | `(symbol: str) -> str` | Maps 6-digit code to Baostock format (`sh.XXXXXX` or `sz.XXXXXX`) |
| `_map_adjustflag` | `(adjust: str) -> str` | Maps qfq/hfq to Baostock adjust flags (1/2/3) |
| `fetch_kline_bs` | `(symbol, start_date, end_date, period, adjust) -> DataFrame` | Full Baostock data fetch workflow with login/logout |

**Issues Found:**
- [MEDIUM] `bs_code` only handles `6xx` as Shanghai -- misses `9xx` (Shanghai B-shares)
- [LOW] Login/logout on every call -- no connection pooling

---

#### `alpha101_factory/data/universe.py` (54 lines)

**Purpose:** Load stock universe from cached spot parquet file.

**Imports:** pandas, config.PARQ_DIR_SPOT, utils.io

**Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_universe` | `(limit: int = 0) -> Series` | Reads `a_spot.parquet`, returns 6-digit stock codes |

**Issues Found:** None -- clean implementation.

---

### 2.2 Factor System

#### `alpha101_factory/factors/base.py` (26 lines)

**Purpose:** Abstract base class for all factor implementations.

**Imports:** abc, pandas, typing

**Classes:**

| Class | Attributes/Methods | Description |
|-------|-------------------|-------------|
| `Factor(ABC)` | `name: str`, `requires: List[str]` | Abstract base class |
| | `compute(df) -> Series` | Abstract method: computes factor from panel DataFrame |
| | `as_cs_series(df, values) -> Series` (static) | Builds MultiIndex(datetime, symbol) Series |

**Issues Found:** None -- clean, minimal design.

---

#### `alpha101_factory/factors/registry.py` (121 lines)

**Purpose:** Factor registration decorator and auto-discovery system.

**Imports:** pathlib, typing, importlib, pkgutil, factors.base.Factor

**Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `register` | `(cls: Type[Factor]) -> Type[Factor]` | Decorator -- stores class in `_REGISTRY` dict |
| `_ensure_loaded` | `() -> None` | Auto-discovers `alphas_*.py` modules via `pkgutil.iter_modules` |
| `get_factor` | `(name: str) -> Type[Factor]` | Lookup factor class by name |
| `list_factors` | `() -> list[str]` | Returns sorted list of registered factor names |

**Issues Found:** None -- clean design with proper error handling.

---

#### `alpha101_factory/factors/__init__.py` (12 lines)

**Purpose:** Triggers `@register` decorators by importing factor modules.

Explicitly imports `alphas_basic` and optionally `alphas_more` (wrapped in try/except).

---

#### `alpha101_factory/factors/alphas_basic.py` (2100 lines)

**Purpose:** 55 Alpha factor implementations following the Alpha101 framework.

**Helper Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `_cs_rank` | `(df: DataFrame, s: Series) -> Series` | Cross-sectional percentile rank with MultiIndex |
| `_g` | `(df: DataFrame, col: str, fn, *args) -> Series` | Groupby symbol, apply function to column |

**Implemented Factors (55):**
Alpha001, 003, 004, 005, 006, 009, 010, 011, 012, 013, 014, 016, 018, 019, 020, 021, 022, 023, 024, 025, 026, 030, 031, 032, 033, 034, 035, 036, 037, 038, 039, 040, 041, 042, 043, 044, 045, 046, 047, 049, 050, 051, 052, 053, 054, 055, 060, 061, 064, 065, 071, 083, 084, 085, 086, 094, 095, 096, 098, 099, 101

**Issues Found:**
- [CRITICAL] `_g(df, None, lambda *_: ...)` pattern in Alpha022, 026, 035, 045, 050, 055, 060, etc. -- the lambda ignores group context, so rolling operations run on the full DataFrame instead of per-symbol. Produces incorrect results with multiple symbols.
- [MEDIUM] Alpha031: `requires = ["close","volume"]` but `compute()` references `df["low"]` -- missing from requires
- [MEDIUM] Alpha099: `requires = ["high","low","volume"]` but `compute()` references `df["close"]` -- missing from requires
- [MEDIUM] Alpha095: Computes variable `b` (ts_rank) but never uses it
- [MEDIUM] Alpha064: Contains `if False else` dead code branch
- [MEDIUM] Alpha086: Contains `.groupby(level=0, group_keys=False) if False else` dead code
- [MEDIUM] Alpha098: References `ops.argmin` which doesn't exist in `utils/ops.py` -- silently falls back to `a*0`
- [LOW] Inconsistent style: first ~20 factors have full docstrings + try/except, later factors have minimal or none

---

#### `alpha101_factory/factors/alphas_more.py` (9 lines)

**Purpose:** Placeholder for additional factor implementations.

Currently empty beyond imports (base, registry, ops). No factors registered.

---

#### `alpha101_factory/factors/tmp_features.py` (150 lines)

**Purpose:** Build intermediate features (returns, vwap, advN) per symbol and cache as parquet.

**Imports:** pandas, loguru, tqdm, config.*, utils.io, utils.ops

**Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `build_tmp_for_symbol` | `(sym: str) -> bool` | Builds tmp parquet for single stock |
| `build_tmp_all` | `(symbols: list[str]) -> int` | Batch builder with tqdm progress bar |
| `load_panel` | `(symbols: list[str]) -> DataFrame` | Merge all tmp files into a single long DataFrame |

**ADV Windows:** 5, 10, 20, 30, 40, 60, 120, 150, 180

**Issues Found:** None -- clean implementation with proper validation.

---

### 2.3 Pipeline

#### `alpha101_factory/pipeline/compute_factor.py` (149 lines)

**Purpose:** Orchestrates factor computation -- loads data, runs Factor.compute(), saves results.

**Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `_load_join` | `(symbols: Optional[List[str]]) -> DataFrame` | Merges kline + tmp data via outer join |
| `compute_and_save` | `(factor_name: str, symbols: Optional[List[str]] = None) -> None` | Computes and saves a single factor |
| `main` | `() -> None` | Batch computation of all factors |

**Issues Found:**
- [CRITICAL] `_load_join(symbols=None)` at line 43: Reads tmp directory stems as symbols -- these are `{sym}_{start}_{end}_{adj}` (full filename), not just the symbol code. Causes complete mismatch when trying to find kline files.
- [CRITICAL] Same issue in `main()` at line 127.

---

#### `alpha101_factory/pipeline/check_data.py` (59 lines)

**Purpose:** K-line data integrity checking script.

Clean wrapper around `loader.check_klines_integrity()`. Saves report to CSV.

**Issues Found:** None.

---

#### `alpha101_factory/pipeline/build_tmp.py` (54 lines)

**Purpose:** Tmp feature building script.

Clean wrapper: loads universe, calls `build_tmp_all()`.

**Issues Found:** None.

---

### 2.4 Backtest

#### `alpha101_factory/backtest/metrics.py` (220 lines)

**Purpose:** IC/RankIC computation and quantile portfolio construction.

**Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `make_forward_return` | `(price_df: DataFrame, horizon: int = 1) -> Optional[Series]` | Computes forward returns with MultiIndex |
| `_t_stat` | `(x: Series) -> float` | T-statistic for significance testing |
| `_pearson` | `(g: DataFrame) -> float` | Cross-sectional Pearson IC |
| `_spearman` | `(g: DataFrame) -> float` | Cross-sectional Spearman RankIC |
| `ic_rankic` | `(factor_df, price_df, horizon=1) -> Dict[str, DataFrame]` | Full IC/RankIC analysis with time-series per-symbol stats |
| `quantile_portfolios` | `(factor_df, price_df, horizon=1, q=5) -> Dict[str, DataFrame]` | Quantile portfolio returns and long-short spread |

**Issues Found:**
- [LOW] Uses `print()` for error messages instead of `logger` -- inconsistent with the rest of the codebase

---

#### `alpha101_factory/backtest/run_bt.py` (139 lines)

**Purpose:** CLI for factor backtesting. Generates IC/RankIC charts and quantile portfolio cumulative returns.

**Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `_load_prices` | `(symbols: list[str]) -> DataFrame` | Loads kline close prices for specified symbols |
| `main` | `() -> None` | Argparse CLI: `--alpha`, `--horizon`, `--quantiles` |

**Outputs:** PNG charts + CSV files in `images/backtest/`

**Issues Found:** None -- clean implementation with good error handling.

---

### 2.5 Visualization

#### `alpha101_factory/viz/plots.py` (276 lines)

**Purpose:** Plotly-based visualization functions for klines, factors, and heatmaps.

**Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `_ensure_datetime_series` | `(s: Series) -> Series` | Handles ns/ms/s timestamp heuristics |
| `_datetime_array_for_plot` | `(s) -> ndarray` | Converts to Python datetime array for Plotly |
| `plot_kline` | `(df, title, tickformat, tickangle) -> Figure` | Candlestick K-line chart |
| `plot_factor_timeseries` | `(fdf, symbol, title, ...) -> Figure` | Factor time-series line chart |
| `plot_factor_cross_section` | `(fdf, dt, topn, ...) -> Figure` | Cross-section bar chart (top N by abs value) |
| `plot_heatmap` | `(fdf, symbols, ...) -> Figure` | Factor heatmap (time x symbol) |
| `save_fig` | `(fig, path) -> Path` | Write image via kaleido |
| `plot_kline_with_factor` | `(kline_df, factor_df, symbol, ...) -> Figure` | Combined K-line + factor subplot |

**Issues Found:** None -- clean, well-documented.

---

#### `alpha101_factory/viz/factor_summary.py` (209 lines)

**Purpose:** High-level factor visualization orchestrator. Generates timeseries, cross-section, and heatmap images.

**Classes/Functions:**

| Name | Type | Description |
|------|------|-------------|
| `FactorVisualArtifacts` | dataclass | Stores factor name, symbol, date, and output paths/figures |
| `_coerce_datetime` | function | Ensures datetime column is properly typed |
| `_select_symbol` | function | Chooses symbol for timeseries (preferred or first available) |
| `_select_heatmap_symbols` | function | Selects symbols for heatmap by coverage |
| `_load_factor_frame` | function | Reads factor parquet file |
| `generate_factor_visuals` | function | Single factor visualization |
| `generate_all_factor_visuals` | function | Batch visualization for multiple factors |

**Issues Found:**
- [LOW] `_load_factor_frame` checks `if df is None` but `read_parquet` never returns None -- unreachable but harmless

---

### 2.6 Utilities

#### `alpha101_factory/utils/ops.py` (276 lines)

**Purpose:** Financial quantitative computation utilities with optional bottleneck/numba acceleration.

**Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `rolling_sum` | `(s: Series, n: int) -> Series` | Bottleneck-accelerated rolling sum |
| `rolling_min` | `(s: Series, n: int) -> Series` | Bottleneck-accelerated rolling min |
| `rolling_max` | `(s: Series, n: int) -> Series` | Bottleneck-accelerated rolling max |
| `rolling_std` | `(s: Series, n: int) -> Series` | Bottleneck-accelerated rolling std (ddof=0) |
| `rolling_cov` | `(s1, s2, n) -> Series` | Rolling covariance (pandas) |
| `rolling_corr` | `(s1, s2, n) -> Series` | Rolling correlation (pandas) |
| `ts_rank` | `(s: Series, n: int) -> Series` | Numba-accelerated time-series percentile rank |
| `decay_linear` | `(s: Series, n: int) -> Series` | Numba-accelerated linear decay weighted average |
| `delay` | `(s: Series, n: int = 1) -> Series` | Lag by n periods |
| `delta` | `(s: Series, n: int = 1) -> Series` | Difference: current - n periods ago |
| `returns` | `(close: Series) -> Series` | Percentage change |
| `vwap_from_amount` | `(close, high, low, volume, amount) -> Series` | VWAP = amount / volume |
| `adv` | `(volume: Series, n: int) -> Series` | Average daily volume |
| `cs_rank` | `(s: Series) -> Series` | Cross-sectional percentile rank (groupby level 0) |
| `cs_zscore` | `(s: Series) -> Series` | Cross-sectional z-score |
| `by_symbol` | `(df, col, func, *args) -> Series` | Groupby symbol helper |

**Issues Found:**
- [CRITICAL] Missing `argmin` function -- referenced by Alpha098 but not implemented
- [LOW] `cs_rank` assumes MultiIndex level 0 -- some callers in alphas_basic.py pass plain Series

---

#### `alpha101_factory/utils/io.py` (53 lines)

**Purpose:** Safe parquet read/write with error handling and auto-directory creation.

**Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `read_parquet` | `(path: Path) -> DataFrame` | Safe read, returns empty DataFrame on error |
| `write_parquet` | `(df: DataFrame, path: Path) -> None` | Safe write, auto-creates parent dirs |

**Issues Found:** None -- clean, minimal, robust.

---

#### `alpha101_factory/utils/log.py` (62 lines)

**Purpose:** Loguru logging initialization.

**Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `setup_logger` | `() -> logger` | Configures console + file logging |

**Issues Found:**
- [LOW] Console handler `lambda msg: print(msg, end="")` double-prints and breaks loguru colorization
- [LOW] Duplicate `from pathlib import Path` import (lines 3 and 19)

---

### 2.7 CLI

#### `alpha101_factory/cli.py` (144 lines)

**Purpose:** Top-level CLI with argparse subcommands.

**Subcommands:**

| Command | Handler | Description |
|---------|---------|-------------|
| `fetch` | `cmd_fetch` | Fetch spot + batch klines |
| `fetch-one` | `cmd_fetch_one` | Load/fetch single stock with `--stock`, `--start`, `--end`, `--adjust` |
| `tmp` | `cmd_tmp` | Build tmp features (optional `--stock`) |
| `factor` | `cmd_factor` | Compute factors with `--factors`, `--all`, `--stock` |
| `check` | `cmd_check` | Check kline integrity |
| `visualize` | `cmd_visualize` | Generate factor visualizations with `--all`, `--factors`, `--prefix`, etc. |

**Issues Found:** None -- clean implementation.

---

### 2.8 Config

#### `alpha101_factory/config.py` (48 lines)

**Purpose:** Central configuration via environment variables. Creates all directories at import time.

**Settings:**

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_ROOT` | `./data` | Root data directory |
| `ADJUST` | `qfq` | Forward-adjusted prices |
| `START_DATE` | `20200101` | Start date for data fetching |
| `END_DATE` | `20250917` | End date for data fetching |
| `MAX_WORKERS` | `1` | Concurrent workers (unused) |
| `REQUEST_PAUSE` | `0.6` | Rate limiting pause (seconds) |
| `LIMIT_STOCKS` | `0` | Debug limit (0 = all) |

**Directory Structure Created:**
- `data/spot`, `data/klines_daily`, `data/tmp_features`, `data/factors`, `data/logs`
- `data/images/klines`, `data/images/backtest`
- `data/images/factors/{timeseries,cross_section,heatmap}`

**Issues Found:** None.

---

### 2.9 Tests

#### `tests/test_loader_paths.py` (18 lines)

2 tests for `_resolve_kline_path`:
- `test_resolve_kline_path_full_range` -- verifies path with date range and exchange prefix stripping
- `test_resolve_kline_path_without_range` -- verifies fallback path without dates

#### `tests/test_factor_visuals.py` (58 lines)

2 tests for visualization:
- `test_generate_factor_visuals_returns_figures` -- verifies figure generation with synthetic data
- `test_generate_all_factor_visuals_reads_parquet` -- verifies batch visualization from parquet (writes/reads temp file)

---

## 3. Bug & Issue Registry

### Critical (Crashes / Incorrect Results)

| # | Location | Description | Status |
|---|----------|-------------|--------|
| 1 | `alphas_basic.py` | `_g(df, None, ...)` pattern captures outer `df` in lambdas. Functionally works due to pandas index alignment but is a code smell. 27 occurrences across factors. | NOTED (design pattern, not changed) |
| 2 | `compute_factor.py:43,127` | `_load_join(symbols=None)` reads full parquet filename stems (e.g. `600000_20200101_20250917_qfq`) as symbol codes, causing kline file lookup failures. | **FIXED** — uses `p.stem.split("_")[0]` |
| 3 | `utils/ops.py` | Missing `argmin` function referenced by Alpha098. Silently falls back to `a*0`. | **FIXED** — added `argmin()` function |

### Medium (Logic / Data Issues)

| # | Location | Description | Status |
|---|----------|-------------|--------|
| 4 | `alphas_basic.py` Alpha031 | `requires = ["close","volume"]` but compute uses `df["low"]` | **FIXED** — added "low" to requires |
| 5 | `alphas_basic.py` Alpha099 | `requires = ["high","low","volume"]` but compute uses `df["close"]` | **FIXED** — added "close" to requires |
| 6 | `alphas_basic.py` Alpha095 | Computes unused variable `b` | **FIXED** — `b` now used in final comparison |
| 7 | `alphas_basic.py` Alpha064/086 | `if False else` dead code branches | **FIXED** — removed dead code; Alpha064 requires updated |
| 8 | `loader.py` | `_fetch_kline_fallback` double-strips digits and double-logs | **FIXED** — removed redundant strip/log |
| 9 | `baostock_api.py` | `bs_code` doesn't handle 9xx codes (Shanghai B-shares) | **FIXED** — added "9" prefix handling |

### Low (Code Quality)

| # | Location | Description | Status |
|---|----------|-------------|--------|
| 10 | `metrics.py` | Uses `print()` instead of `logger` | **FIXED** — replaced with loguru logger |
| 11 | `log.py` | Console handler `lambda msg: print(msg, end="")` double-prints | **FIXED** — uses `sys.stderr` sink |
| 12 | `log.py` | Duplicate `from pathlib import Path` import | **FIXED** — removed duplicate |
| 13 | Multiple files | `sys.path.append` at module level -- fragile | NOTED (systemic, not changed) |
| 14 | `alphas_basic.py` | Inconsistent error handling (first half has try/except, second half doesn't) | NOTED (style, not changed) |

---

## 4. Dependency Analysis

```
requirements.txt (UPDATED):
  akshare>=1.13         -- A-share data API (primary)
  baostock              -- fallback data source
  pandas>=2.0           -- data manipulation core
  numpy>=1.24           -- numerical computation
  pyarrow>=14.0         -- parquet read/write engine
  tqdm>=4.66            -- progress bars
  loguru>=0.7           -- structured logging
  plotly>=5.24          -- interactive visualization
  bottleneck>=1.3       -- rolling window acceleration
  numba>=0.59           -- JIT compilation for ts_rank, decay_linear
  kaleido>=0.2.1        -- plotly static image export (PNG)
```

**Resolved:**
- Removed `scipy` (never imported anywhere)
- Removed `fastparquet` (redundant with pyarrow)

**Remaining:**
- `pytest` not in requirements.txt (needed for testing)
- No pinned version for `baostock`

---

## 5. Test Coverage Gap Analysis

**Current:** 2 test files, 4 tests, ~76 lines

| Area | Lines of Code | Tests | Coverage |
|------|--------------|-------|----------|
| Factor computation (alphas_basic.py) | 2,100 | 0 | None |
| Data normalization (loader.normalize_k) | ~40 | 0 | None |
| Backtest metrics (IC/RankIC, quantile) | 220 | 0 | None |
| Pipeline orchestration (compute_factor) | 149 | 0 | None |
| Tmp feature building | 150 | 0 | None |
| Utils/ops functions | 276 | 0 | None |
| CLI commands | 144 | 0 | None |
| Baostock API mapping | 118 | 0 | None |
| Path resolution (loader) | 20 | 2 | Covered |
| Visualization (factor_summary) | 209 | 2 | Partial |

**Critical gaps:** Factor computation and backtest metrics have zero test coverage despite being the core value of the project.
