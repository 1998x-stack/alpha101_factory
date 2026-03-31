# Design Spec: alpha101_factory Comprehensive Review & Reconstruction

**Date:** 2026-03-30
**Status:** Approved

## Goal

Perform a comprehensive code review, build a full project analysis markdown, run all tests and workflows, fix bugs, and revalidate — in 4 sequential phases with review gates.

## Project Context

- **Project:** alpha101_factory — pluggable Alpha factor factory for A-share daily data
- **Size:** ~4,700 lines of Python across 20 modules + 2 test files
- **Stack:** Python 3.10+, pandas, numpy, plotly, numba, bottleneck, baostock, akshare
- **Data:** ~150+ parquet files of daily klines in `data/klines_daily/`

## Phases

### Phase 1: PROJECT_ANALYSIS.md (Full Reconstruction)

Build a comprehensive analysis document organized by category:

1. **Project Overview** — purpose, architecture, data flow diagram
2. **Category-by-Category File Reconstruction:**
   - Data Layer (`data/loader.py`, `baostock_api.py`, `universe.py`)
   - Factor System (`factors/base.py`, `registry.py`, `alphas_basic.py`, `alphas_more.py`, `tmp_features.py`)
   - Pipeline (`pipeline/compute_factor.py`, `check_data.py`, `build_tmp.py`)
   - Backtest (`backtest/metrics.py`, `run_bt.py`)
   - Visualization (`viz/plots.py`, `factor_summary.py`)
   - Utilities (`utils/ops.py`, `io.py`, `log.py`)
   - CLI (`cli.py`)
   - Config (`config.py`)
   - Tests (`tests/`)
3. **Bug & Issue Registry** — categorized by severity (crash/incorrect/quality)
4. **Dependency Analysis** — requirements.txt review
5. **Test Coverage Gap Analysis**

Each file entry: purpose, all functions/classes with signatures, imports, data flow, potential issues.

**Review gate:** User reviews PROJECT_ANALYSIS.md before Phase 2.

### Phase 2: Run Tests & Workflows

- `pytest -v` on existing tests
- Import validation for all modules
- CLI smoke tests using local data (no network)
- Document all results

**Review gate:** Share results before Phase 3.

### Phase 3: Fix Bugs

Priority order:
1. Crashes / import errors
2. Incorrect computation results
3. Logic bugs / edge cases
4. Code quality issues affecting correctness

Each fix is a distinct logical change.

### Phase 4: Rerun & Validate

- Rerun all tests
- Confirm all fixes resolved
- Update PROJECT_ANALYSIS.md with final state

## Output Artifacts

- `PROJECT_ANALYSIS.md` — comprehensive project analysis
- `docs/superpowers/specs/2026-03-30-project-review-design.md` — this spec
- Bug fixes committed to codebase
- Test results documented

## Success Criteria

- Every Python file analyzed with function-level detail
- All existing tests pass
- All runnable workflows execute without errors
- All identified bugs fixed and validated
- PROJECT_ANALYSIS.md is complete and accurate
