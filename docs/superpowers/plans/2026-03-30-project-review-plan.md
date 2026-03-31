# alpha101_factory Comprehensive Review & Reconstruction Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Perform a complete code review, build a full project analysis markdown, run all tests, fix bugs, and revalidate.

**Architecture:** 4 sequential phases — Phase 1 builds `PROJECT_ANALYSIS.md` with file-by-file reconstruction organized by category. Phase 2 runs all tests and workflows. Phase 3 fixes all bugs found. Phase 4 reruns tests and updates the analysis.

**Tech Stack:** Python 3.10+, pandas, numpy, plotly, numba, bottleneck, baostock, akshare, pytest

---

## Phase 1: Build PROJECT_ANALYSIS.md

### Task 1: Create PROJECT_ANALYSIS.md with Overview and Data Layer

**Files:**
- Create: `PROJECT_ANALYSIS.md`

- [ ] **Step 1: Write the project overview section**

Write `PROJECT_ANALYSIS.md` with these sections:

```markdown
# alpha101_factory — Project Analysis

## 1. Project Overview

**Purpose:** Pluggable Alpha101 factor factory for A-share (Chinese stock market) daily kline data. End-to-end pipeline: data fetching → feature caching → factor computation → backtesting → visualization.

**Architecture:**
```
CLI (cli.py)
  ├── fetch / fetch-one → data/loader.py (AkShare→Baostock fallback)
  ├── tmp              → factors/tmp_features.py (returns, vwap, advN)
  ├── factor           → pipeline/compute_factor.py → factors/registry.py → alphas_*.py
  ├── check            → pipeline/check_data.py
  └── visualize        → viz/factor_summary.py → viz/plots.py

Backtest: backtest/run_bt.py → backtest/metrics.py
Config:   config.py (env vars, paths)
Utils:    utils/ops.py (bottleneck/numba), utils/io.py, utils/log.py
```

**Data Flow:**
1. `fetch` → AkShare/Baostock → `data/klines_daily/{sym}_{start}_{end}_{adj}.parquet` + PNG
2. `tmp` → klines → returns/vwap/advN → `data/tmp_features/{sym}_{start}_{end}_{adj}.parquet`
3. `factor` → merge(klines, tmp) → Factor.compute() → `data/factors/{AlphaName}.parquet`
4. `backtest` → factors + prices → IC/RankIC + quantile portfolios → CSV + PNG
5. `visualize` → factors → timeseries/cross-section/heatmap PNGs

**Stats:** ~4,700 lines of Python across 20 modules, 2 test files, 55+ Alpha factors implemented.
```

- [ ] **Step 2: Write the Data Layer section**

Append the Data Layer analysis covering 3 files:

**`alpha101_factory/data/loader.py` (320 lines)**
- Purpose: A-share kline data fetching, normalization, caching, integrity checks
- Key functions:
  - `_resolve_kline_path(symbol, start_date, end_date, adjust) -> Path` — generates parquet path, strips exchange prefix
  - `normalize_k(df) -> DataFrame` — renames Chinese column names, converts types
  - `_fetch_kline_ak(symbol, start_date, end_date, adjust) -> DataFrame` — AkShare API call
  - `_fetch_kline_fallback(symbol, start_date, end_date, adjust) -> DataFrame` — AkShare with Baostock fallback
  - `_save_kline_png(sym, df, start_date, end_date, adjust) -> Path` — K-line chart via plotly+kaleido
  - `fetch_spot(save=True) -> DataFrame` — gets/caches A-share real-time snapshot
  - `fetch_klines_from_spot(spot) -> int` — batch download klines for universe
  - `check_klines_integrity() -> DataFrame` — validates local parquet files
  - `load_or_fetch_symbol(symbol, start_date, end_date, adjust, save_image) -> DataFrame` — load local or download single stock
- Imports: pandas, akshare, tqdm, loguru, config, utils.io, viz.plots, data.baostock_api
- Potential issues:
  - **BUG:** `_fetch_kline_fallback` calls `_fetch_kline_ak` which ALSO strips digits and logs — double-logging and double-stripping
  - `fetch_spot` caches aggressively — no staleness check, always returns cached if file exists
  - `sys.path.append` at module level is fragile

**`alpha101_factory/data/baostock_api.py` (118 lines)**
- Purpose: Baostock K-line data fetching as fallback
- Key functions:
  - `bs_code(symbol) -> str` — maps 6-digit code to `sh.XXXXXX` or `sz.XXXXXX`
  - `_map_adjustflag(adjust) -> str` — maps qfq/hfq to Baostock flags
  - `fetch_kline_bs(symbol, start_date, end_date, period, adjust) -> DataFrame` — full Baostock workflow
- Potential issues:
  - `bs_code` only handles `6xx` as Shanghai — misses `9xx` (BSE/北交所) stocks which are also Shanghai
  - Login/logout in every call — no connection pooling

**`alpha101_factory/data/universe.py` (54 lines)**
- Purpose: Load stock universe from cached spot parquet
- Key functions:
  - `load_universe(limit=0) -> Series` — reads a_spot.parquet, returns 6-digit codes
- Potential issues: None significant, clean implementation

- [ ] **Step 3: Commit Phase 1a**

```bash
git add PROJECT_ANALYSIS.md
git commit -m "docs: add PROJECT_ANALYSIS.md Phase 1a — overview and data layer"
```

### Task 2: Add Factor System to PROJECT_ANALYSIS.md

**Files:**
- Modify: `PROJECT_ANALYSIS.md`

- [ ] **Step 1: Write the Factor System section**

Append analysis of 5 files:

**`alpha101_factory/factors/base.py` (26 lines)**
- Purpose: Abstract base class for all factors
- Key classes:
  - `Factor(ABC)` — `name: str`, `requires: List[str]`, abstract `compute(df) -> Series`, static `as_cs_series(df, values) -> Series`
- Clean, minimal design

**`alpha101_factory/factors/registry.py` (121 lines)**
- Purpose: Factor registration and auto-discovery
- Key functions:
  - `register(cls) -> cls` — decorator, stores in `_REGISTRY` dict
  - `_ensure_loaded()` — auto-discovers `alphas_*.py` modules via `pkgutil.iter_modules`
  - `get_factor(name) -> Type[Factor]` — lookup by name
  - `list_factors() -> list[str]` — sorted registered names
- Clean design with proper error handling

**`alpha101_factory/factors/__init__.py` (12 lines)**
- Purpose: Triggers `@register` decorators by importing factor modules
- Explicitly imports `alphas_basic` and optionally `alphas_more`

**`alpha101_factory/factors/alphas_basic.py` (2100 lines)**
- Purpose: 55 Alpha factor implementations (Alpha001–Alpha101)
- Key helpers:
  - `_cs_rank(df, s) -> Series` — cross-sectional rank with MultiIndex
  - `_g(df, col, fn, *args) -> Series` — groupby symbol + apply
- Implemented factors: Alpha001, 003, 004, 005, 006, 009, 010, 011, 012, 013, 014, 016, 018, 019, 020, 021, 022, 023, 024, 025, 026, 030, 031, 032, 033, 034, 035, 036, 037, 038, 039, 040, 041, 042, 043, 044, 045, 046, 047, 049, 050, 051, 052, 053, 054, 055, 060, 061, 064, 065, 071, 083, 084, 085, 086, 094, 095, 096, 098, 099, 101
- Potential issues:
  - **BUG:** `_g(df, None, lambda *_: ...)` pattern used in Alpha022, 026, 035, 045, 050, etc. — passes the whole DataFrame but the lambda ignores the group context, meaning the function operates on the full DataFrame rather than per-symbol. This is INCORRECT for rolling operations that should be per-symbol.
  - **BUG:** Alpha030 calls `_cs_rank(df, ...)` which returns a MultiIndex Series, then tries to multiply with per-symbol grouped Series — index mismatch potential
  - **BUG:** Alpha031 `requires` lists `["close","volume"]` but `compute` references `df["low"]` — missing from requires
  - **BUG:** Alpha064 has dead code: `if False else` branch
  - **BUG:** Alpha086 has dead code: `.groupby(level=0, group_keys=False) if False else`
  - **BUG:** Alpha095 computes `b` (ts_rank) but never uses it
  - **BUG:** Alpha098 uses `ops.argmin` which doesn't exist in `utils/ops.py` — falls back to `a*0` silently
  - **BUG:** Alpha099 `requires` lists `["high","low","volume"]` but `compute` references `df["close"]` — missing from requires
  - Many factors wrap everything in try/except RuntimeError — masks bugs during development
  - Later factors (Alpha036 onwards) lack try/except and docstrings — inconsistent style

**`alpha101_factory/factors/alphas_more.py` (9 lines)**
- Purpose: Placeholder for additional factors
- Currently empty (just imports)

**`alpha101_factory/factors/tmp_features.py` (150 lines)**
- Purpose: Build intermediate features (returns, vwap, advN) per symbol
- Key functions:
  - `build_tmp_for_symbol(sym) -> bool` — single stock feature builder
  - `build_tmp_all(symbols) -> int` — batch builder with tqdm
  - `load_panel(symbols) -> DataFrame` — merge tmp files into long table
- Clean implementation, proper error handling

- [ ] **Step 2: Commit Phase 1b**

```bash
git add PROJECT_ANALYSIS.md
git commit -m "docs: add factor system analysis to PROJECT_ANALYSIS.md"
```

### Task 3: Add Pipeline, Backtest, Viz, Utils, CLI, Config, Tests sections

**Files:**
- Modify: `PROJECT_ANALYSIS.md`

- [ ] **Step 1: Write Pipeline section**

**`alpha101_factory/pipeline/compute_factor.py` (149 lines)**
- Purpose: Load kline+tmp data, compute factors, save results
- Key functions:
  - `_load_join(symbols) -> DataFrame` — merges kline and tmp data via outer join
  - `compute_and_save(factor_name, symbols=None) -> None` — orchestrates factor computation
  - `main()` — batch computation of all registered factors
- Potential issues:
  - `_load_join` with `symbols=None` reads tmp directory stems as symbol names — these include `{sym}_{start}_{end}_{adj}` which is the FULL filename stem, not the symbol code. This will cause mismatches when trying to find kline files.

**`alpha101_factory/pipeline/check_data.py` (59 lines)**
- Purpose: Kline integrity checking script
- Key functions: `main()` — runs check and saves CSV report
- Clean wrapper around `loader.check_klines_integrity()`

**`alpha101_factory/pipeline/build_tmp.py` (54 lines)**
- Purpose: Tmp feature building script
- Key functions: `main()` — loads universe and builds all tmp files
- Clean wrapper

- [ ] **Step 2: Write Backtest section**

**`alpha101_factory/backtest/metrics.py` (220 lines)**
- Purpose: IC/RankIC computation and quantile portfolio construction
- Key functions:
  - `make_forward_return(price_df, horizon=1) -> Series` — forward returns with MultiIndex
  - `_t_stat(x) -> float` — t-statistic
  - `_pearson(g) / _spearman(g) -> float` — cross-sectional IC/RankIC
  - `ic_rankic(factor_df, price_df, horizon=1) -> Dict` — full IC analysis
  - `quantile_portfolios(factor_df, price_df, horizon=1, q=5) -> Dict` — quantile portfolio returns
- Potential issues:
  - Uses `print()` instead of `logger` for error messages — inconsistent with rest of codebase
  - `_pearson` and `_spearman` check `g["symbol"].nunique()` but `g` is a DataFrame with `value` and `fwd_ret` columns — `symbol` column may not exist after groupby in some paths

**`alpha101_factory/backtest/run_bt.py` (139 lines)**
- Purpose: CLI for factor backtesting
- Key functions:
  - `_load_prices(symbols) -> DataFrame` — loads kline close prices
  - `main()` — argparse CLI with IC/RankIC + quantile portfolio analysis
- Clean implementation with good error handling

- [ ] **Step 3: Write Visualization section**

**`alpha101_factory/viz/plots.py` (276 lines)**
- Purpose: Plotly visualization functions
- Key functions:
  - `_ensure_datetime_series(s) -> Series` — handles ns/ms/s timestamp heuristics
  - `_datetime_array_for_plot(s) -> ndarray` — converts to Python datetime for Plotly
  - `plot_kline(df, title, ...) -> Figure` — candlestick chart
  - `plot_factor_timeseries(fdf, symbol, title, ...) -> Figure` — factor line chart
  - `plot_factor_cross_section(fdf, dt, topn, ...) -> Figure` — cross-section bar chart
  - `plot_heatmap(fdf, symbols, ...) -> Figure` — factor heatmap
  - `save_fig(fig, path) -> Path` — write image via kaleido
  - `plot_kline_with_factor(kline_df, factor_df, symbol, ...) -> Figure` — combined K-line + factor subplot
- Clean, well-documented implementation

**`alpha101_factory/viz/factor_summary.py` (209 lines)**
- Purpose: High-level factor visualization orchestrator
- Key classes/functions:
  - `FactorVisualArtifacts` — dataclass for visualization outputs
  - `generate_factor_visuals(factor_name, ...) -> FactorVisualArtifacts` — single factor
  - `generate_all_factor_visuals(...) -> Mapping` — batch visualization
- Potential issues:
  - `_load_factor_frame` returns `pd.DataFrame` but also checks `if df is None` — `read_parquet` never returns None (returns empty DataFrame), so this check is unreachable but harmless

- [ ] **Step 4: Write Utils, CLI, Config, Tests sections**

**`alpha101_factory/utils/ops.py` (276 lines)**
- Purpose: Financial quantitative computation utilities with optional acceleration
- Key functions: `rolling_sum/min/max/std/cov/corr`, `ts_rank`, `decay_linear`, `delay`, `delta`, `returns`, `vwap_from_amount`, `adv`, `cs_rank`, `cs_zscore`, `by_symbol`
- Uses bottleneck for rolling ops, numba for ts_rank and decay_linear
- Potential issues:
  - `cs_rank` assumes input has a MultiIndex with level 0 as datetime — but some callers in `alphas_basic.py` pass plain Series without proper index, causing groupby to fail silently
  - Missing `argmin` function — referenced by Alpha098 but not implemented

**`alpha101_factory/utils/io.py` (53 lines)**
- Purpose: Safe parquet read/write
- Clean, minimal, robust

**`alpha101_factory/utils/log.py` (62 lines)**
- Purpose: Loguru initialization
- Potential issues:
  - Console handler uses `lambda msg: print(msg, end="")` — double-prints since loguru already formats messages; also breaks loguru's colorization
  - Duplicate `from pathlib import Path` import

**`alpha101_factory/cli.py` (144 lines)**
- Purpose: Top-level CLI with argparse subcommands
- Subcommands: `fetch`, `fetch-one`, `tmp`, `factor`, `check`, `visualize`
- Clean implementation

**`alpha101_factory/config.py` (48 lines)**
- Purpose: Central configuration via environment variables
- Defines all directory paths, creates them at import time
- Clean, no issues

**`alpha101_factory/__init__.py` (3 lines)**
- Purpose: Package marker
- Only exports `config`

**Tests (2 files, 76 lines total)**

`tests/test_loader_paths.py` (18 lines):
  - `test_resolve_kline_path_full_range` — verifies path with date range
  - `test_resolve_kline_path_without_range` — verifies path without dates

`tests/test_factor_visuals.py` (58 lines):
  - `test_generate_factor_visuals_returns_figures` — verifies figure generation with sample data
  - `test_generate_all_factor_visuals_reads_parquet` — verifies batch visualization from parquet

- [ ] **Step 5: Write Bug Registry and Coverage Analysis sections**

## Bug & Issue Registry

### Critical (Crashes / Incorrect Results)
1. **`alphas_basic.py` — `_g(df, None, ...)` pattern ignores per-symbol grouping**: Operations like `ops.delta(corr, 5)` in Alpha022 operate on the full DataFrame instead of per-symbol, producing incorrect results when multiple symbols are present.
2. **`compute_factor.py` — `_load_join(symbols=None)` uses full filename stems as symbols**: When no symbols are specified, it reads `{sym}_{start}_{end}_{adj}` as the symbol, which won't match kline file paths.
3. **`ops.py` — Missing `argmin` function**: Alpha098 references `ops.argmin` which doesn't exist, silently falling back to zeros.

### Medium (Logic Issues)
4. **`alphas_basic.py` — Alpha031 missing `"low"` in requires**: Won't fail but could cause confusion if requires is used for validation.
5. **`alphas_basic.py` — Alpha099 missing `"close"` in requires**: Same issue.
6. **`alphas_basic.py` — Alpha095 computes unused variable `b`**: Dead computation.
7. **`alphas_basic.py` — Alpha064/086 contain `if False else` dead code**: Always takes one branch.
8. **`loader.py` — Double symbol stripping and double logging**: `_fetch_kline_fallback` strips digits then calls `_fetch_kline_ak` which strips again and logs again.
9. **`baostock_api.py` — `bs_code` doesn't handle 8xx/4xx/9xx codes**: BSE (北交所) stocks use codes starting with 8/4/9 but are not Shanghai.

### Low (Code Quality)
10. **`metrics.py` — Uses `print()` instead of `logger`**: Inconsistent with rest of codebase.
11. **`log.py` — Console handler double-prints**: `lambda msg: print(msg, end="")` duplicates output.
12. **`log.py` — Duplicate import**: `from pathlib import Path` imported twice.
13. **Multiple files — `sys.path.append` at module level**: Fragile, should use proper package installation.
14. **`alphas_basic.py` — Inconsistent error handling**: First half has try/except+docstrings, second half has neither or only partial.

## Test Coverage Gap Analysis

**Current coverage:** 2 test files, 4 test functions
- `test_loader_paths.py`: Path resolution only (2 tests)
- `test_factor_visuals.py`: Visualization with mock data (2 tests)

**Not covered:**
- Factor computation (alphas_basic.py — 2100 lines, 0 tests)
- Data normalization (normalize_k)
- Backtest metrics (IC/RankIC, quantile portfolios)
- Pipeline orchestration (compute_factor.py)
- Tmp feature building
- Utils/ops functions
- CLI commands
- Baostock API mapping

## Dependency Analysis

```
requirements.txt:
  akshare>=1.13        — A-share data API
  baostock             — fallback data source
  pandas>=2.0          — data manipulation
  numpy>=1.24          — numerical computation
  pyarrow>=14.0        — parquet read/write
  fastparquet>=2024.2.0 — alternative parquet engine (unused? both pyarrow and fastparquet listed)
  tqdm>=4.66           — progress bars
  loguru>=0.7          — logging
  plotly>=5.24         — visualization
  bottleneck>=1.3      — rolling window acceleration
  numba>=0.59          — JIT compilation
  kaleido>=0.2.1       — plotly static image export
  scipy                — listed but NOT imported anywhere in the codebase
```

Issues:
- **scipy** is listed but never imported — unnecessary dependency
- Both **pyarrow** and **fastparquet** are listed but only pyarrow is needed for `pd.read_parquet`
- No **pytest** in requirements.txt (needed for testing)
- No pinned versions for baostock or scipy

- [ ] **Step 6: Commit Phase 1c**

```bash
git add PROJECT_ANALYSIS.md
git commit -m "docs: complete PROJECT_ANALYSIS.md — pipeline, backtest, viz, utils, bug registry"
```

- [ ] **Step 7: User review gate — present PROJECT_ANALYSIS.md for review**

Pause and ask: "Phase 1 complete. PROJECT_ANALYSIS.md is ready for review. Please check it and let me know when to proceed to Phase 2 (test execution)."

---

## Phase 2: Run Tests & Workflows

### Task 4: Run existing tests

**Files:**
- None modified

- [ ] **Step 1: Run pytest**

```bash
cd /Users/xd/Desktop/codes/mygithubs/alpha101_factory
python -m pytest tests/ -v
```

Expected: 4 tests. Document pass/fail status.

- [ ] **Step 2: Run import validation**

```bash
python -c "import alpha101_factory; from alpha101_factory import config, cli; from alpha101_factory.factors import registry; print('All imports OK'); print('Registered factors:', registry.list_factors())"
```

- [ ] **Step 3: Run CLI smoke test (check command — no network)**

```bash
python -m alpha101_factory.cli check
```

Expected: Reports on local kline files.

- [ ] **Step 4: Document all results**

Create a test results summary in the conversation. Include:
- pytest output (pass/fail per test)
- Import validation results
- CLI smoke test output
- Any errors or warnings

- [ ] **Step 5: User review gate**

Pause: "Phase 2 complete. Here are the test results. Ready to proceed to Phase 3 (bug fixes)?"

---

## Phase 3: Fix Bugs

### Task 5: Fix critical bugs

**Files:**
- Modify: `alpha101_factory/pipeline/compute_factor.py`
- Modify: `alpha101_factory/utils/ops.py`

- [ ] **Step 1: Fix `_load_join` symbol parsing when symbols=None**

In `compute_factor.py:43`, the current code reads full parquet stems (e.g. `600000_20200101_20250917_qfq`) as symbols. Fix to extract just the symbol code:

```python
# Before:
symbols = sorted({p.stem for p in (PARQ_DIR_TMP).glob("*.parquet")})

# After:
symbols = sorted({p.stem.split("_")[0] for p in (PARQ_DIR_TMP).glob("*.parquet")})
```

Apply the same fix in `main()` at line 127.

Run: `python -m pytest tests/ -v`
Expected: All existing tests still pass.

- [ ] **Step 2: Add `argmin` function to ops.py**

Add after the `decay_linear` function:

```python
def argmin(s: pd.Series, n: int) -> pd.Series:
    """计算滚动窗口内最小值位置（从窗口末尾倒数）。"""
    return s.rolling(n, min_periods=n).apply(lambda x: np.argmin(x), raw=True)
```

Run: `python -m pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 3: Commit critical fixes**

```bash
git add alpha101_factory/pipeline/compute_factor.py alpha101_factory/utils/ops.py
git commit -m "fix: correct symbol parsing in _load_join and add missing argmin to ops"
```

### Task 6: Fix medium bugs

**Files:**
- Modify: `alpha101_factory/factors/alphas_basic.py`
- Modify: `alpha101_factory/data/loader.py`
- Modify: `alpha101_factory/data/baostock_api.py`

- [ ] **Step 1: Fix `requires` lists for Alpha031 and Alpha099**

In `alphas_basic.py`:

Alpha031 (line ~1673): Change `requires = ["close","volume"]` to `requires = ["close","volume","low"]`

Alpha099 (line ~2083): Change `requires = ["high","low","volume"]` to `requires = ["high","low","volume","close"]`

- [ ] **Step 2: Remove dead code in Alpha064, Alpha086, Alpha095**

Alpha064 (line ~1794): Remove `if False else` dead branch
Alpha086 (line ~1955): Remove `.groupby(...) if False else` dead branch
Alpha095 (line ~2013): Remove unused variable `b`

- [ ] **Step 3: Fix double-stripping in loader.py**

In `_fetch_kline_fallback` (line 132), remove the `symbol = ''.join(filter(str.isdigit, symbol))` line — `_fetch_kline_ak` already does this.

- [ ] **Step 4: Improve bs_code for BSE stocks**

In `baostock_api.py`, update `bs_code`:

```python
def bs_code(symbol: str) -> str:
    s = str(symbol).zfill(6)
    if s.startswith(("6", "9")):
        return f"sh.{s}"
    return f"sz.{s}"
```

Note: 8xx/4xx codes for BSE (北交所) are handled by `sz.` prefix in Baostock as they trade on NEEQ/BSE which maps to Shenzhen in Baostock's system. However 9xx Shanghai B-shares should be `sh.`.

- [ ] **Step 5: Commit medium fixes**

```bash
git add alpha101_factory/factors/alphas_basic.py alpha101_factory/data/loader.py alpha101_factory/data/baostock_api.py
git commit -m "fix: correct requires lists, remove dead code, fix double-stripping and bs_code"
```

### Task 7: Fix low-priority issues

**Files:**
- Modify: `alpha101_factory/backtest/metrics.py`
- Modify: `alpha101_factory/utils/log.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Replace print() with logger in metrics.py**

Replace all `print(f"[...]` calls with `logger.error(...)` or `logger.warning(...)`. Add `from loguru import logger` import at top.

- [ ] **Step 2: Fix log.py console handler and duplicate import**

Remove duplicate `from pathlib import Path` import. Replace the console handler:

```python
# Before:
logger.add(lambda msg: print(msg, end=""))

# After:
logger.add(sys.stderr, level="DEBUG")
```

- [ ] **Step 3: Clean up requirements.txt**

Remove `scipy` (unused). Add `pytest>=7.0` to dev dependencies or note it.

- [ ] **Step 4: Commit low-priority fixes**

```bash
git add alpha101_factory/backtest/metrics.py alpha101_factory/utils/log.py requirements.txt
git commit -m "fix: use logger in metrics, fix log handler, clean requirements"
```

---

## Phase 4: Rerun & Validate

### Task 8: Rerun all tests and validate

**Files:**
- Modify: `PROJECT_ANALYSIS.md`

- [ ] **Step 1: Rerun pytest**

```bash
python -m pytest tests/ -v
```

Expected: All 4 tests pass.

- [ ] **Step 2: Rerun import validation**

```bash
python -c "import alpha101_factory; from alpha101_factory.factors import registry; print('Factors:', registry.list_factors())"
```

Expected: All 55+ factors listed, no import errors.

- [ ] **Step 3: Rerun CLI smoke test**

```bash
python -m alpha101_factory.cli check
```

Expected: No errors.

- [ ] **Step 4: Verify fixed bugs**

Run targeted checks:
```bash
python -c "from alpha101_factory.utils.ops import argmin; print('argmin exists:', argmin)"
python -c "from alpha101_factory.pipeline.compute_factor import _load_join; print('_load_join importable')"
```

- [ ] **Step 5: Update PROJECT_ANALYSIS.md with final state**

Update the Bug Registry section to mark all fixed issues. Add a "Validation Results" section documenting all test passes.

- [ ] **Step 6: Final commit**

```bash
git add PROJECT_ANALYSIS.md
git commit -m "docs: update PROJECT_ANALYSIS.md with validation results and fixed bug status"
```
