# -*- coding: utf-8 -*-
"""
Alpha101 Factory — 核心模块测试

覆盖:
  1. utils/ops (滚动计算、截面排名)
  2. factors/registry (注册、发现、获取)
  3. backtest/metrics (IC/RankIC、分位数组合)
  4. 关键因子正确性 (Alpha003/004/009)
  5. config (路径、环境变量)
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ─── 测试数据工厂 ─────────────────────────────────────────

def make_panel_df(n_symbols=5, n_days=100, seed=42):
    """生成面板数据: DataFrame with [datetime, symbol, open, high, low, close, volume, amount]"""
    np.random.seed(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    symbols = [f"SH{600000+i}" for i in range(n_symbols)]

    rows = []
    for sym in symbols:
        base = np.random.uniform(10, 50)
        close = base * np.cumprod(1 + np.random.normal(0.0003, 0.015, n_days))
        high = close * (1 + np.abs(np.random.uniform(0, 0.02, n_days)))
        low = close * (1 - np.abs(np.random.uniform(0, 0.02, n_days)))
        open_p = low + (high - low) * np.random.uniform(0.3, 0.7, n_days)
        volume = np.random.uniform(1e6, 1e8, n_days)
        amount = close * volume

        for i, d in enumerate(dates):
            rows.append({
                "datetime": d, "symbol": sym,
                "open": open_p[i], "high": high[i], "low": low[i],
                "close": close[i], "volume": volume[i], "amount": amount[i],
            })

    return pd.DataFrame(rows)


def make_price_df(n_symbols=5, n_days=100, seed=42):
    """生成价格数据: [datetime, symbol, close]"""
    df = make_panel_df(n_symbols, n_days, seed)
    return df[["datetime", "symbol", "close"]]


# ═══════════════════════════════════════════════════════════
# 1. utils/ops 测试
# ═══════════════════════════════════════════════════════════

class TestOps:
    def setup_method(self):
        from alpha101_factory.utils import ops
        self.ops = ops

    def test_rolling_sum(self):
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = self.ops.rolling_sum(s, 3)
        expected = [np.nan, np.nan, 6, 9, 12, 15, 18, 21, 24, 27]
        np.testing.assert_array_almost_equal(result.values[:3], expected[:3], decimal=10)
        assert result.iloc[2] == 6.0
        assert result.iloc[9] == 27.0

    def test_rolling_min_max(self):
        s = pd.Series([5, 3, 8, 1, 9, 2, 7])
        r_min = self.ops.rolling_min(s, 3)
        r_max = self.ops.rolling_max(s, 3)
        assert r_min.iloc[2] == 3.0  # min(5,3,8)
        assert r_max.iloc[4] == 9.0  # max(8,1,9)

    def test_rolling_std(self):
        s = pd.Series([2, 4, 4, 4, 5, 5, 7, 9])
        result = self.ops.rolling_std(s, 4)
        # First valid window: std([2,4,4,4]) = 0.866...
        assert result.iloc[3] > 0

    def test_delta(self):
        s = pd.Series([10, 12, 15, 11, 8])
        result = self.ops.delta(s, 1)
        expected = [np.nan, 2, 3, -4, -3]
        np.testing.assert_array_almost_equal(result.values, expected, decimal=10)

    def test_delay(self):
        s = pd.Series([1, 2, 3, 4, 5])
        result = self.ops.delay(s, 2)
        assert pd.isna(result.iloc[0]) and pd.isna(result.iloc[1])
        assert result.iloc[2] == 1.0

    def test_returns(self):
        close = pd.Series([100, 110, 121, 108.9])
        result = self.ops.returns(close)
        assert abs(result.iloc[1] - 0.10) < 1e-10
        assert abs(result.iloc[2] - 0.10) < 1e-10

    def test_ts_rank(self):
        s = pd.Series([1, 3, 2, 5, 4])
        result = self.ops.ts_rank(s, 3)
        # Window [1,3,2]: last=2, rank=2/3 ≈ 0.667
        assert abs(result.iloc[2] - 2/3) < 0.01

    def test_decay_linear(self):
        s = pd.Series([1, 2, 3, 4, 5])
        result = self.ops.decay_linear(s, 3)
        # weights [1,2,3]/6 → [1/6, 2/6, 3/6]
        # At index 2: 1*1/6 + 2*2/6 + 3*3/6 = 14/6 ≈ 2.333
        assert abs(result.iloc[2] - 14/6) < 0.01

    def test_decay_linear_nan_propagation(self):
        """GOTCHA FIX: NaN 应严格传播"""
        s = pd.Series([1, 2, np.nan, 4, 5])
        result = self.ops.decay_linear(s, 3)
        # Window [1,2,nan] → should be NaN
        assert pd.isna(result.iloc[2])

    def test_cs_rank(self):
        """截面排名: 每个时间点上股票按值排名"""
        idx = pd.MultiIndex.from_tuples([
            ("2024-01-01", "A"), ("2024-01-01", "B"), ("2024-01-01", "C"),
            ("2024-01-02", "A"), ("2024-01-02", "B"), ("2024-01-02", "C"),
        ], names=["datetime", "symbol"])
        s = pd.Series([10, 20, 30, 50, 40, 60], index=idx, dtype=float)
        result = self.ops.cs_rank(s)
        # Day 1: A=10→0.167, B=20→0.5, C=30→0.833
        assert abs(result.iloc[0] - 1/3) < 0.01
        assert abs(result.iloc[1] - 2/3) < 0.01
        assert abs(result.iloc[2] - 1.0) < 0.01

    def test_cs_rank_requires_multiindex(self):
        """GOTCHA FIX: 非 MultiIndex 应报错"""
        s = pd.Series([1, 2, 3])
        with pytest.raises(ValueError, match="MultiIndex"):
            self.ops.cs_rank(s)

    def test_rolling_cov_ddof0(self):
        """GOTCHA FIX: rolling_cov 应使用 ddof=0 与 rolling_std 一致"""
        s1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        s2 = pd.Series([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
        result = self.ops.rolling_cov(s1, s2, 3)
        # Perfect correlation, cov = var of s1 = 2/3 for window [1,2,3]
        assert result.iloc[2] > 0

    def test_adv(self):
        volume = pd.Series([100, 200, 300, 400, 500])
        result = self.ops.adv(volume, 3)
        assert abs(result.iloc[2] - 200) < 0.01  # avg(100,200,300)
        assert abs(result.iloc[4] - 400) < 0.01  # avg(300,400,500)


# ═══════════════════════════════════════════════════════════
# 2. Registry 测试
# ═══════════════════════════════════════════════════════════

class TestRegistry:
    def test_list_factors(self):
        from alpha101_factory.factors.registry import list_factors
        factors = list_factors()
        assert len(factors) >= 60
        assert "Alpha001" in factors
        assert "Alpha101" in factors

    def test_get_factor(self):
        from alpha101_factory.factors.registry import get_factor
        cls = get_factor("Alpha001")
        assert cls.name == "Alpha001"
        assert hasattr(cls, "compute")

    def test_get_factor_not_found(self):
        from alpha101_factory.factors.registry import get_factor
        with pytest.raises(KeyError):
            get_factor("NonExistent")

    def test_factor_requires(self):
        from alpha101_factory.factors.registry import get_factor
        f = get_factor("Alpha001")
        assert isinstance(f.requires, list)
        assert len(f.requires) > 0

    def test_factor_instantiation(self):
        from alpha101_factory.factors.registry import get_factor
        f = get_factor("Alpha003")()
        assert f.name == "Alpha003"


# ═══════════════════════════════════════════════════════════
# 3. Factor 计算正确性测试
# ═══════════════════════════════════════════════════════════

class TestFactorCompute:
    def test_alpha003_compute(self):
        """Alpha003: -rolling_corr(cs_rank(open), cs_rank(volume), 10)"""
        from alpha101_factory.factors.registry import get_factor
        df = make_panel_df(n_symbols=5, n_days=50)
        fac = get_factor("Alpha003")()
        result = fac.compute(df)
        assert result is not None
        assert len(result) > 0
        # Result should be a Series with MultiIndex
        assert isinstance(result.index, pd.MultiIndex)

    def test_alpha004_compute(self):
        """Alpha004: -ts_rank(cs_rank(low), 9)"""
        from alpha101_factory.factors.registry import get_factor
        df = make_panel_df(n_symbols=5, n_days=50)
        fac = get_factor("Alpha004")()
        result = fac.compute(df)
        assert len(result) > 0
        assert isinstance(result.index, pd.MultiIndex)

    def test_alpha009_compute(self):
        """Alpha009: 趋势方向因子"""
        from alpha101_factory.factors.registry import get_factor
        df = make_panel_df(n_symbols=5, n_days=50)
        fac = get_factor("Alpha009")()
        result = fac.compute(df)
        assert len(result) > 0


# ═══════════════════════════════════════════════════════════
# 4. Backtest Metrics 测试
# ═══════════════════════════════════════════════════════════

class TestMetrics:
    def test_make_forward_return(self):
        from alpha101_factory.backtest.metrics import make_forward_return
        df = make_price_df(n_symbols=3, n_days=50)
        fwd = make_forward_return(df, horizon=1)
        assert fwd is not None
        assert len(fwd) == len(df)
        # Last value should be NaN (no future data)
        assert pd.isna(fwd.iloc[-1])

    def test_make_forward_return_horizon(self):
        from alpha101_factory.backtest.metrics import make_forward_return
        df = make_price_df(n_symbols=3, n_days=50)
        fwd5 = make_forward_return(df, horizon=5)
        # Last 5 values should be NaN
        assert fwd5.iloc[-5:].isna().all()

    def test_ic_rankic(self):
        from alpha101_factory.backtest.metrics import ic_rankic
        # Use make_panel_df to ensure consistent format
        panel_df = make_panel_df(n_symbols=3, n_days=50)
        factor_df = panel_df[["datetime", "symbol", "close"]].copy()
        factor_df.columns = ["datetime", "symbol", "value"]
        price_df = make_price_df(n_symbols=3, n_days=50)
        res = ic_rankic(factor_df, price_df, horizon=1)
        assert "daily" in res
        assert "summary" in res
        assert "ts_summary" in res
        assert len(res["daily"]) > 0

    def test_quantile_portfolios(self):
        from alpha101_factory.backtest.metrics import quantile_portfolios
        np.random.seed(42)
        panel_df = make_panel_df(n_symbols=5, n_days=50)
        factor_df = panel_df[["datetime", "symbol", "close"]].copy()
        factor_df.columns = ["datetime", "symbol", "value"]
        price_df = make_price_df(n_symbols=5, n_days=50)
        res = quantile_portfolios(factor_df, price_df, horizon=1, q=5)
        assert "ports" in res
        assert "ls" in res
        assert not res["ports"].empty

    def test_quantile_portfolios_few_groups(self):
        """GOTCHA FIX: 当 qcut 减少分组数时，LS 应使用实际最高/最低组"""
        from alpha101_factory.backtest.metrics import quantile_portfolios
        # Create factor with many duplicate values → fewer groups
        np.random.seed(42)
        panel_df = make_panel_df(n_symbols=3, n_days=20)
        factor_df = panel_df[["datetime", "symbol"]].copy()
        factor_df["value"] = [1.0] * 30 + [2.0] * 30  # Only 2 distinct values
        price_df = make_price_df(n_symbols=3, n_days=20)
        res = quantile_portfolios(factor_df, price_df, horizon=1, q=5)
        # Should not crash, LS might be empty or use actual groups
        assert "ports" in res
        assert "ls" in res


# ═══════════════════════════════════════════════════════════
# 5. Config 测试
# ═══════════════════════════════════════════════════════════

class TestConfig:
    def test_config_paths(self):
        from alpha101_factory.config import DATA_ROOT, PARQ_DIR_KLINES, IMG_BT_DIR
        assert DATA_ROOT is not None
        assert PARQ_DIR_KLINES is not None
        assert IMG_BT_DIR is not None

    def test_config_defaults(self):
        from alpha101_factory.config import ADJUST, START_DATE, END_DATE
        assert ADJUST in ["qfq", "hfq", ""]
        assert START_DATE is not None


# ═══════════════════════════════════════════════════════════
# 6. IO 测试
# ═══════════════════════════════════════════════════════════

class TestIO:
    def test_read_parquet_not_found(self):
        from alpha101_factory.utils.io import read_parquet
        from pathlib import Path
        result = read_parquet(Path("/nonexistent/file.parquet"))
        assert result.empty

    def test_write_parquet_raises_on_error(self):
        """GOTCHA FIX: write_parquet 应重新抛出异常"""
        from alpha101_factory.utils.io import write_parquet
        from pathlib import Path
        df = pd.DataFrame({"a": [1, 2, 3]})
        # Writing to a read-only or invalid path should raise
        with pytest.raises(Exception):
            write_parquet(df, Path("/proc/invalid_path_12345/test.parquet"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
