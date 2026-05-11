# -*- coding: utf-8 -*-
"""Alpha101因子库 - 完整版.

所有 101 个 Alpha 因子实现，基于 WorldQuant Alpha101 论文.
统一使用正确的截面排名和多索引操作.
"""

import numpy as np
import pandas as pd
from alpha101_factory.factors.base import Factor
from alpha101_factory.factors.registry import register
from alpha101_factory.utils import ops


# ═══════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════

def _g(df, col, fn, *args):
    """按股票分组应用函数, 返回与 df 对齐的 MultiIndex Series."""
    m = df.set_index(["datetime", "symbol"])
    # 使用 transform 保留原始索引, 然后通过 values 对齐到 MultiIndex
    result = df.groupby("symbol")[col].transform(lambda s: fn(s, *args))
    return pd.Series(result.values, index=m.index)


def _cs(s: pd.Series) -> pd.Series:
    """截面排名: 在每个时间点上对所有股票排名 (MultiIndex: datetime, symbol)."""
    if not isinstance(s.index, pd.MultiIndex):
        raise ValueError(f"_cs 需要 MultiIndex(datetime, symbol)")
    # level=0 是 datetime → 截面排名
    return s.groupby(level=0).rank(pct=True)


def _mi(df):
    """快速设置 MultiIndex."""
    return df.set_index(["datetime", "symbol"])


def _ts_rank_mi(s: pd.Series, n: int) -> pd.Series:
    """时间序列排名: 每只股票独立做 ts_rank (MultiIndex: datetime, symbol)."""
    # level=1 是 symbol → 按股票分组做时间序列排名
    return s.groupby(level=1).transform(lambda x: ops.ts_rank(x, n))




# ═══════════════════════════════════════════════════════════
# Alpha001~Alpha101
# ═══════════════════════════════════════════════════════════

@register
class Alpha001(Factor):
    """(-1 * ts_rank(rank(log(volume)), 5)) * rank(((close - open) / open))"""
    name = "Alpha001"
    requires = ["returns", "close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        val = -ops.ts_rank(_cs(np.log(m["volume"])), 5) * _cs((m["close"] - m["open"]) / m["open"])
        return Factor.as_cs_series(df, val)


@register
class Alpha002(Factor):
    """(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))"""
    name = "Alpha002"
    requires = ["close", "open", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        rank_vol = _cs(ops.delta(np.log(m["volume"]), 2))
        rank_ret = _cs((m["close"] - m["open"]) / m["open"])
        return Factor.as_cs_series(df, -ops.rolling_corr(rank_vol, rank_ret, 6))


@register
class Alpha003(Factor):
    """(-1 * rolling_corr(rank(open), rank(volume), 10))"""
    name = "Alpha003"
    requires = ["open", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        return Factor.as_cs_series(df, -ops.rolling_corr(_cs(m["open"]), _cs(m["volume"]), 10))


@register
class Alpha004(Factor):
    """(-1 * ts_rank(cs_rank(low), 9))"""
    name = "Alpha004"
    requires = ["low"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        return Factor.as_cs_series(df, -ops.ts_rank(_cs(m["low"]), 9))


@register
class Alpha005(Factor):
    """(-1 * rank(open - (sum(vwap, 10) / 10))) * (-1 * rank(abs(close - vwap)))"""
    name = "Alpha005"
    requires = ["open", "vwap", "close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        vwap_avg = _g(df, "vwap", lambda s: ops.rolling_sum(s, 10) / 10)
        rank1 = _cs(-(m["open"] - vwap_avg))
        rank2 = _cs(-np.abs(m["close"] - m["vwap"]))
        return Factor.as_cs_series(df, rank1 * rank2)


@register
class Alpha006(Factor):
    """(-1 * rolling_corr(open, volume, 10))"""
    name = "Alpha006"
    requires = ["open", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        return Factor.as_cs_series(df, -ops.rolling_corr(m["open"], m["volume"], 10))


@register
class Alpha007(Factor):
    """(-1 * rank(abs(delta(close, 3)) * (volume / adv20)))"""
    name = "Alpha007"
    requires = ["close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv20 = _g(df, "volume", lambda s: ops.adv(s, 20))
        val = -_cs(np.abs(ops.delta(m["close"], 3)) * (m["volume"] / adv20))
        return Factor.as_cs_series(df, val)


@register
class Alpha008(Factor):
    """-rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10)))"""
    name = "Alpha008"
    requires = ["open", "close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        returns = m["close"].pct_change()
        sum_open = ops.rolling_sum(m["open"], 5)
        sum_ret = ops.rolling_sum(returns, 5)
        prod = sum_open * sum_ret
        return Factor.as_cs_series(df, -_cs(prod - ops.delay(prod, 10)))


@register
class Alpha009(Factor):
    """((0 < ts_min(delta(close, 1), 5)) * (-1)) + ((0 < ts_max(delta(close, 1), 5)) * 1)"""
    name = "Alpha009"
    requires = ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        delta = ops.delta(m["close"], 1)
        ts_min = delta.groupby(level=1).transform(lambda x: ops.rolling_min(x, 5))
        ts_max = delta.groupby(level=1).transform(lambda x: ops.rolling_max(x, 5))
        val = ((0 < ts_min).astype(float) * (-1)) + ((0 < ts_max).astype(float) * 1)
        return Factor.as_cs_series(df, val)


@register
class Alpha010(Factor):
    """rank(((0 < ts_min(delta(close, 1), 4))) * (-1)) + rank(((0 < ts_max(delta(close, 1), 4))) * 1)"""
    name = "Alpha010"
    requires = ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        delta = ops.delta(m["close"], 1)
        ts_min = delta.groupby(level=1).transform(lambda x: ops.rolling_min(x, 4))
        ts_max = delta.groupby(level=1).transform(lambda x: ops.rolling_max(x, 4))
        val1 = _cs((0 < ts_min).astype(float) * (-1))
        val2 = _cs((0 < ts_max).astype(float) * 1)
        return Factor.as_cs_series(df, val1 + val2)


@register
class Alpha011(Factor):
    """((rank(ts_rank((vwap - close), 3)) + (-1 * rank(ts_rank((close - vwap), 3)))) * rank(volume / adv20))"""
    name = "Alpha011"
    requires = ["vwap", "close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv20 = _g(df, "volume", lambda s: ops.adv(s, 20))
        ts1 = ops.ts_rank(m["vwap"] - m["close"], 3)
        ts2 = ops.ts_rank(m["close"] - m["vwap"], 3)
        rank1 = _cs(ts1 + (-ts2))
        rank2 = _cs(m["volume"] / adv20)
        return Factor.as_cs_series(df, rank1 * rank2)


@register
class Alpha012(Factor):
    """(-1 * sign((volume * delta(close, 1))))"""
    name = "Alpha012"
    requires = ["close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        return Factor.as_cs_series(df, -np.sign(m["volume"] * ops.delta(m["close"], 1)))


@register
class Alpha013(Factor):
    """(-1 * rank(covariance(rank(close), rank(volume), 5)))"""
    name = "Alpha013"
    requires = ["close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        return Factor.as_cs_series(df, -_cs(ops.rolling_cov(_cs(m["close"]), _cs(m["volume"]), 5)))


@register
class Alpha014(Factor):
    """rank(((open - delay(close, 1)) * (correlation(open, volume, 20))))"""
    name = "Alpha014"
    requires = ["open", "volume", "returns"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        delay_close = ops.delay(m["close"], 1)
        corr = ops.rolling_corr(m["open"], m["volume"], 20)
        return Factor.as_cs_series(df, _cs((m["open"] - delay_close) * corr))


@register
class Alpha015(Factor):
    """(-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))"""
    name = "Alpha015"
    requires = ["high", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        corr = ops.rolling_corr(_cs(m["high"]), _cs(m["volume"]), 3)
        return Factor.as_cs_series(df, -ops.rolling_sum(_cs(corr), 3))


@register
class Alpha016(Factor):
    """(-1 * rank(covariance(rank(high), rank(volume), 5)))"""
    name = "Alpha016"
    requires = ["high", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        return Factor.as_cs_series(df, -_cs(ops.rolling_cov(_cs(m["high"]), _cs(m["volume"]), 5)))


@register
class Alpha017(Factor):
    """(-1 * rank(rank(close) * rank(volume) * rank(((adv20 / low) * rank(((high - low) / (high + low)))))))"""
    name = "Alpha017"
    requires = ["high", "low", "close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv20 = _g(df, "volume", lambda s: ops.adv(s, 20))
        adv_low = adv20 / m["low"]
        hl_ratio = (m["high"] - m["low"]) / (m["high"] + m["low"])
        return Factor.as_cs_series(df, -_cs(_cs(m["close"]) * _cs(m["volume"]) * _cs(adv_low * _cs(hl_ratio))))


@register
class Alpha018(Factor):
    """(-1 * rank((std(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10)))"""
    name = "Alpha018"
    requires = ["close", "open"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        std_val = ops.rolling_std(np.abs(m["close"] - m["open"]), 5)
        diff = m["close"] - m["open"]
        corr = ops.rolling_corr(m["close"], m["open"], 10)
        return Factor.as_cs_series(df, -_cs(std_val + diff + corr))


@register
class Alpha019(Factor):
    """((-1 * sign((close - delay(close, 7)) * correlation(close, delay(close, 7), 250))))"""
    name = "Alpha019"
    requires = ["close", "returns"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        delay7 = ops.delay(m["close"], 7)
        corr = ops.rolling_corr(m["close"], delay7, 250)
        return Factor.as_cs_series(df, -np.sign((m["close"] - delay7) * corr))


@register
class Alpha020(Factor):
    """((-1 * rank((open - delay(high, 1)) * correlation(open, volume, 10))))"""
    name = "Alpha020"
    requires = ["open", "high", "low", "close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        delay_high = ops.delay(m["high"], 1)
        corr = ops.rolling_corr(m["open"], m["volume"], 10)
        return Factor.as_cs_series(df, -_cs((m["open"] - delay_high) * corr))


@register
class Alpha021(Factor):
    """((((-1 * mean(returns, 20)) - open) * close) * volume)"""
    name = "Alpha021"
    requires = ["close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        returns = m["close"].pct_change()
        mean_ret = ops.rolling_sum(returns, 20) / 20
        val = ((-mean_ret - m["open"]) * m["close"]) * m["volume"]
        return Factor.as_cs_series(df, val)


@register
class Alpha022(Factor):
    """(-1 * (delta(correlation(high, volume, 5), 5) * rank(std(close, 20))))"""
    name = "Alpha022"
    requires = ["high", "volume", "close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        corr = ops.rolling_corr(m["high"], m["volume"], 5)
        delta_corr = ops.delta(corr, 5)
        std_close = ops.rolling_std(m["close"], 20)
        return Factor.as_cs_series(df, -(delta_corr * _cs(std_close)))


@register
class Alpha023(Factor):
    """(-1 * rank((sum(high, 20) / 20) * high))"""
    name = "Alpha023"
    requires = ["high", "close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        sum_high = ops.rolling_sum(m["high"], 20) / 20
        return Factor.as_cs_series(df, -_cs(sum_high * m["high"]))


@register
class Alpha024(Factor):
    """(-1 * rank(delta(sum(close, 5), 5)))"""
    name = "Alpha024"
    requires = ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        sum_close = ops.rolling_sum(m["close"], 5)
        return Factor.as_cs_series(df, -_cs(ops.delta(sum_close, 5)))


@register
class Alpha025(Factor):
    """(-1 * rank((((close - open) / delay(close, 7)) * correlation(close, volume, 250)) * rank(returns)))"""
    name = "Alpha025"
    requires = ["returns", "vwap", "high", "close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        delay7 = ops.delay(m["close"], 7)
        corr = ops.rolling_corr(m["close"], m["volume"], 250)
        ret = m["close"].pct_change()
        return Factor.as_cs_series(df, -_cs(((m["close"] - m["open"]) / delay7) * corr * _cs(ret)))


@register
class Alpha026(Factor):
    """(-1 * ts_rank(volume, 5))"""
    name = "Alpha026"
    requires = ["volume", "high"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        return Factor.as_cs_series(df, -ops.ts_rank(m["volume"], 5))


@register
class Alpha027(Factor):
    """((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 6) / 6))) * (-1))"""
    name = "Alpha027"
    requires = ["volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        corr = ops.rolling_corr(_cs(m["volume"]), _cs(m["vwap"]), 6)
        val = (0.5 < _cs(ops.rolling_sum(corr, 6) / 6)).astype(float) * (-1)
        return Factor.as_cs_series(df, val)


@register
class Alpha028(Factor):
    """scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))"""
    name = "Alpha028"
    requires = ["high", "low", "close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv20 = _g(df, "volume", lambda s: ops.adv(s, 20))
        corr = ops.rolling_corr(adv20, m["low"], 5)
        hl_avg = (m["high"] + m["low"]) / 2
        val = corr + hl_avg - m["close"]
        g = val.groupby(level=0)
        return Factor.as_cs_series(df, (val - g.transform("mean")) / g.transform(lambda x: np.sum(np.abs(x))).replace(0, np.nan))


@register
class Alpha029(Factor):
    """(min(product(rank(rank(scale(log(sum(rank(rank(-1 * rank(delta((close - 1), 5))))), 2))))), 5) + ts_rank(delay(1, 1), 5))"""
    name = "Alpha029"
    requires = ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        delta_close = ops.delta(m["close"] - 1, 5)
        rank1 = _cs(-delta_close)
        rank2 = _cs(rank1)
        g = rank2.groupby(level=0)
        scale_val = (rank2 - g.transform("mean")) / g.transform(lambda x: np.sum(np.abs(x))).replace(0, np.nan)
        log_sum = ops.rolling_sum(np.log(scale_val.clip(lower=1e-10)), 2)
        rank3 = _cs(log_sum)
        rank4 = _cs(rank3)
        min_val = ops.rolling_min(rank4, 5)
        delay_val = ops.delay(pd.Series(1.0, index=m.index), 1)
        ts_rank_val = _ts_rank_mi(delay_val, 5)
        return Factor.as_cs_series(df, min_val + ts_rank_val)


@register
class Alpha030(Factor):
    """(-1 * (rank(1 - (rank(returns) * rank(volume))) * rank(returns)))"""
    name = "Alpha030"
    requires = ["close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        ret = m["close"].pct_change()
        rank_ret = _cs(ret)
        rank_vol = _cs(m["volume"])
        return Factor.as_cs_series(df, -(1 - (rank_ret * rank_vol)) * rank_ret)


@register
class Alpha031(Factor):
    """(-1 * rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10))))"""
    name = "Alpha031"
    requires = ["close", "volume", "low"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        delta = ops.delta(m["close"], 10)
        decay = ops.decay_linear(_cs(_cs(-delta)), 10)
        return Factor.as_cs_series(df, -_cs(_cs(_cs(decay))))


@register
class Alpha032(Factor):
    """(-1 * sum(rank(correlation(rank(volume), rank(vwap), 5)), 5))"""
    name = "Alpha032"
    requires = ["close", "vwap"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        corr = ops.rolling_corr(_cs(m["volume"]), _cs(m["vwap"]), 5)
        return Factor.as_cs_series(df, -ops.rolling_sum(_cs(corr), 5))


@register
class Alpha033(Factor):
    """(((-1 * ((min(low, 5) - delay(min(low, 5), 5)) / min(low, 5))) * sum(volume, 10)) / sum(volume, 5))"""
    name = "Alpha033"
    requires = ["open", "close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        min5 = ops.rolling_min(m["low"], 5)
        delay_min5 = ops.delay(min5, 5)
        sum_vol10 = ops.rolling_sum(m["volume"], 10)
        sum_vol5 = ops.rolling_sum(m["volume"], 5)
        return Factor.as_cs_series(df, ((-((min5 - delay_min5) / min5)) * sum_vol10) / sum_vol5)


@register
class Alpha034(Factor):
    """((((rank(returns) * rank(volume)) / rank(close - open)) * rank(returns))"""
    name = "Alpha034"
    requires = ["returns", "close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        ret = m["close"].pct_change()
        rank_ret = _cs(ret)
        rank_vol = _cs(m["volume"])
        rank_diff = _cs(m["close"] - m["open"])
        return Factor.as_cs_series(df, ((rank_ret * rank_vol) / rank_diff) * rank_ret)


@register
class Alpha035(Factor):
    """Alpha035"""
    name = "Alpha035"
    requires = ["volume", "close", "high", "low", "returns"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        vol_ratio = _cs(m["volume"] / ops.rolling_sum(m["volume"], 15))
        hl_ratio = _cs((m["high"] - m["low"]) / (m["high"] + m["low"]))
        ret = _cs(m["close"].pct_change())
        return Factor.as_cs_series(df, vol_ratio * hl_ratio * ret)


@register
class Alpha036(Factor):
    """((-1 * rank(ts_rank(correlation(((high * 0.5) + (low * 0.5)), adv20, 10), 15))))"""
    name = "Alpha036"
    requires = ["close", "open", "volume", "vwap", "returns"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv20 = _g(df, "volume", lambda s: ops.adv(s, 20))
        hl_avg = (m["high"] * 0.5) + (m["low"] * 0.5)
        corr = ops.rolling_corr(hl_avg, adv20, 10)
        ts = ops.ts_rank(corr, 15)
        return Factor.as_cs_series(df, -_cs(ts))


@register
class Alpha037(Factor):
    """(((-1 * rank(ts_rank((delay(close, 1) / close), 10))) * rank(correlation(open, volume, 10))))"""
    name = "Alpha037"
    requires = ["open", "close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        delay_close = ops.delay(m["close"], 1)
        ts = ops.ts_rank(delay_close / m["close"], 10)
        corr = ops.rolling_corr(m["open"], m["volume"], 10)
        return Factor.as_cs_series(df, -_cs(ts) * _cs(corr))


@register
class Alpha038(Factor):
    """((-1 * rank(ts_rank(close, 10))) * rank((close / open)))"""
    name = "Alpha038"
    requires = ["close", "open"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        ts = ops.ts_rank(m["close"], 10)
        return Factor.as_cs_series(df, -_cs(ts) * _cs(m["close"] / m["open"]))


@register
class Alpha039(Factor):
    """((-1 * rank(((decay_linear(delta(close, 7), 8) / decay_linear(correlation(((close * 0.5) + (vwap * 0.5)), adv20, 9), 10))))"""
    name = "Alpha039"
    requires = ["close", "volume", "returns"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv20 = _g(df, "volume", lambda s: ops.adv(s, 20))
        delta7 = ops.delta(m["close"], 7)
        decay1 = ops.decay_linear(delta7, 8)
        cv_avg = (m["close"] * 0.5) + (m["vwap"] * 0.5)
        corr = ops.rolling_corr(cv_avg, adv20, 9)
        decay2 = ops.decay_linear(corr, 10)
        return Factor.as_cs_series(df, -_cs(decay1 / decay2))


@register
class Alpha040(Factor):
    """((-1 * rank(std(high, 10))) * correlation(high, volume, 10))"""
    name = "Alpha040"
    requires = ["high", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        std_high = ops.rolling_std(m["high"], 10)
        corr = ops.rolling_corr(m["high"], m["volume"], 10)
        return Factor.as_cs_series(df, -_cs(std_high) * corr)


@register
class Alpha041(Factor):
    """(((high * low)^0.5) - vwap)"""
    name = "Alpha041"
    requires = ["high", "low", "vwap"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        return Factor.as_cs_series(df, np.sqrt(m["high"] * m["low"]) - m["vwap"])


@register
class Alpha042(Factor):
    """(-1 * rank(std(high, 10) * correlation(high, volume, 10)))"""
    name = "Alpha042"
    requires = ["vwap", "close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        std_high = ops.rolling_std(m["high"], 10)
        corr = ops.rolling_corr(m["high"], m["volume"], 10)
        return Factor.as_cs_series(df, -_cs(std_high * corr))


@register
class Alpha043(Factor):
    """(ts_rank(volume / adv20, 20) * ts_rank((-1 * delta(close, 7)), 8))"""
    name = "Alpha043"
    requires = ["volume", "close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv20 = _g(df, "volume", lambda s: ops.adv(s, 20))
        ts1 = _ts_rank_mi(m["volume"] / adv20, 20)
        ts2 = ops.ts_rank(-ops.delta(m["close"], 7), 8)
        return Factor.as_cs_series(df, ts1 * ts2)


@register
class Alpha044(Factor):
    """((-1 * rank(ts_rank(correlation(high, rank(volume), 5), 5))) * rank(returns))"""
    name = "Alpha044"
    requires = ["high", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        rank_vol = _cs(m["volume"])
        corr = ops.rolling_corr(m["high"], rank_vol, 5)
        ts = ops.ts_rank(corr, 5)
        ret = _cs(m["close"].pct_change())
        return Factor.as_cs_series(df, -ts * ret)


@register
class Alpha045(Factor):
    """((-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2))))"""
    name = "Alpha045"
    requires = ["close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        delay5 = ops.delay(m["close"], 5)
        sum_delay = ops.rolling_sum(delay5, 20) / 20
        corr = ops.rolling_corr(m["close"], m["volume"], 2)
        return Factor.as_cs_series(df, -_cs(sum_delay) * corr)


@register
class Alpha046(Factor):
    """(-1 * rank(delta((((close * 0.375) + (open * 0.625)), 1))))"""
    name = "Alpha046"
    requires = ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        mix = (m["close"] * 0.375) + (m["open"] * 0.625)
        return Factor.as_cs_series(df, -_cs(ops.delta(mix, 1)))


@register
class Alpha047(Factor):
    """((rank(1 / close) * volume) / adv20)"""
    name = "Alpha047"
    requires = ["close", "high", "vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv20 = _g(df, "volume", lambda s: ops.adv(s, 20))
        return Factor.as_cs_series(df, (_cs(1 / m["close"]) * m["volume"]) / adv20)


@register
class Alpha048(Factor):
    """(correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close"""
    name = "Alpha048"
    requires = ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        delta_close = ops.delta(m["close"], 1)
        delta_delay = ops.delta(ops.delay(m["close"], 1), 1)
        corr = ops.rolling_corr(delta_close, delta_delay, 250)
        return Factor.as_cs_series(df, (corr * delta_close) / m["close"])


@register
class Alpha049(Factor):
    """(-1 * rank(((sum(high, 15) - sum(low, 15)) / 15)))"""
    name = "Alpha049"
    requires = ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        sum_high = ops.rolling_sum(m["high"], 15)
        sum_low = ops.rolling_sum(m["low"], 15)
        return Factor.as_cs_series(df, -_cs((sum_high - sum_low) / 15))


@register
class Alpha050(Factor):
    """(-1 * ts_rank(rank(correlation(rank(volume), rank(vwap), 5)), 5))"""
    name = "Alpha050"
    requires = ["volume", "vwap"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        rank_vol = _cs(m["volume"])
        rank_vwap = _cs(m["vwap"])
        corr = ops.rolling_corr(rank_vol, rank_vwap, 5)
        return Factor.as_cs_series(df, -ops.ts_rank(_cs(corr), 5))


@register
class Alpha051(Factor):
    """(-1 * rank(((sum(high, 20) - sum(low, 20)) / 20)))"""
    name = "Alpha051"
    requires = ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        sum_high = ops.rolling_sum(m["high"], 20)
        sum_low = ops.rolling_sum(m["low"], 20)
        return Factor.as_cs_series(df, -_cs((sum_high - sum_low) / 20))


@register
class Alpha052(Factor):
    """((-1 * delta((((close - low) - (high - close)) / (close - low)), 9)))"""
    name = "Alpha052"
    requires = ["low", "returns", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        ratio = ((m["close"] - m["low"]) - (m["high"] - m["close"])) / (m["close"] - m["low"])
        return Factor.as_cs_series(df, -_cs(ops.delta(ratio, 9)))


@register
class Alpha053(Factor):
    """(-1 * delta((((close - low) - (high - close)) / (high - low)), 9))"""
    name = "Alpha053"
    requires = ["close", "low", "high"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        ratio = ((m["close"] - m["low"]) - (m["high"] - m["close"])) / (m["high"] - m["low"])
        return Factor.as_cs_series(df, -ops.delta(ratio, 9))


@register
class Alpha054(Factor):
    """((-1 * ((low - close) * (open ** 5))) / ((low - high) * (close ** 5)))"""
    name = "Alpha054"
    requires = ["low", "close", "open", "high"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        numerator = (m["low"] - m["close"]) * (m["open"] ** 5)
        denominator = (m["low"] - m["high"]) * (m["close"] ** 5)
        return Factor.as_cs_series(df, -(numerator / denominator))


@register
class Alpha055(Factor):
    """(-1 * correlation(rank(((close - min(low, 12)) / (max(high, 12) - min(low, 12)))), rank(volume), 6))"""
    name = "Alpha055"
    requires = ["close", "high", "low", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        min12 = ops.rolling_min(m["low"], 12)
        max12 = ops.rolling_max(m["high"], 12)
        ratio = (m["close"] - min12) / (max12 - min12)
        return Factor.as_cs_series(df, -ops.rolling_corr(_cs(ratio), _cs(m["volume"]), 6))


@register
class Alpha056(Factor):
    """(0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))"""
    name = "Alpha056"
    requires = ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        returns = m["close"].pct_change()
        sum_ret_10 = ops.rolling_sum(returns, 10)
        sum_ret_2 = ops.rolling_sum(returns, 2)
        sum_sum_ret = ops.rolling_sum(sum_ret_2, 3)
        return Factor.as_cs_series(df, -_cs(sum_ret_10 / sum_sum_ret) * _cs(returns))


@register
class Alpha057(Factor):
    """(0 - (1 * ((close - vwap) / decay_linear(rank(ts_rank(close, 30)), 2))))"""
    name = "Alpha057"
    requires = ["close", "vwap"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        ts_rank_val = _ts_rank_mi(m["close"], 30)
        decay = ops.decay_linear(_cs(ts_rank_val), 2)
        return Factor.as_cs_series(df, -(m["close"] - m["vwap"]) / decay)


@register
class Alpha058(Factor):
    """(-1 * Ts_Rank(decay_linear(correlation(vwap, volume, 4), 8), 6))"""
    name = "Alpha058"
    requires = ["close", "vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        corr = ops.rolling_corr(m["vwap"], m["volume"], 4)
        decay = ops.decay_linear(corr, 8)
        ts = _ts_rank_mi(decay, 6)
        return Factor.as_cs_series(df, -ts)


@register
class Alpha059(Factor):
    """(-1 * Ts_Rank(decay_linear(correlation(vwap, volume, 4), 16), 8))"""
    name = "Alpha059"
    requires = ["close", "vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        corr = ops.rolling_corr(m["vwap"], m["volume"], 4)
        decay = ops.decay_linear(corr, 16)
        ts = _ts_rank_mi(decay, 8)
        return Factor.as_cs_series(df, -ts)


@register
class Alpha060(Factor):
    """(-1 * correlation(rank(((close - min(low, 12)) / (max(high, 12) - min(low, 12)))), rank(volume), 6))"""
    name = "Alpha060"
    requires = ["high", "low", "close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        min12 = ops.rolling_min(m["low"], 12)
        max12 = ops.rolling_max(m["high"], 12)
        ratio = (m["close"] - min12) / (max12 - min12)
        return Factor.as_cs_series(df, -ops.rolling_corr(_cs(ratio), _cs(m["volume"]), 6))


@register
class Alpha061(Factor):
    """(rank((vwap - min(vwap, 16))) < rank(correlation(vwap, adv180, 18)))"""
    name = "Alpha061"
    requires = ["vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv180 = _g(df, "volume", lambda s: ops.adv(s, 180))
        min16 = ops.rolling_min(m["vwap"], 16)
        rank1 = _cs(m["vwap"] - min16)
        corr = ops.rolling_corr(m["vwap"], adv180, 18)
        rank2 = _cs(corr)
        return Factor.as_cs_series(df, (rank1 < rank2).astype(float))


@register
class Alpha062(Factor):
    """(-1 * correlation(vwap, rank(adv5), 20))"""
    name = "Alpha062"
    requires = ["vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv5 = _g(df, "volume", lambda s: ops.adv(s, 5))
        return Factor.as_cs_series(df, -ops.rolling_corr(m["vwap"], _cs(adv5), 20))


@register
class Alpha063(Factor):
    """((-1 * correlation(rank(((rank(close) / rank(volume)))), rank(((vwap / 2.51051) + ((vwap * (1 - 0.728317))))), 10))"""
    name = "Alpha063"
    requires = ["close", "volume", "vwap"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        rank1 = _cs(_cs(m["close"]) / _cs(m["volume"]))
        vwap_adj = (m["vwap"] / 2.51051) + (m["vwap"] * (1 - 0.728317))
        rank2 = _cs(vwap_adj)
        return Factor.as_cs_series(df, -ops.rolling_corr(rank1, rank2, 10))


@register
class Alpha064(Factor):
    """((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12), sum(adv120, 12), 16)) < rank(correlation(rank(vwap), rank(volume), 4))) * (-1))"""
    name = "Alpha064"
    requires = ["open", "low", "high", "vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv120 = _g(df, "volume", lambda s: ops.adv(s, 120))
        ol_mix = (m["open"] * 0.178404) + (m["low"] * (1 - 0.178404))
        sum_ol = ops.rolling_sum(ol_mix, 12)
        sum_adv = ops.rolling_sum(adv120, 12)
        corr1 = ops.rolling_corr(sum_ol, sum_adv, 16)
        rank1 = _cs(corr1)
        rank_vwap = _cs(m["vwap"])
        rank_vol = _cs(m["volume"])
        corr2 = ops.rolling_corr(rank_vwap, rank_vol, 4)
        rank2 = _cs(corr2)
        return Factor.as_cs_series(df, (rank1 < rank2).astype(float) * (-1))


@register
class Alpha065(Factor):
    """(rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60, 60), 9)) < rank(((open - ts_min(open, 13)) / sum(adv60, 60))))"""
    name = "Alpha065"
    requires = ["open", "vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv60 = _g(df, "volume", lambda s: ops.adv(s, 60))
        ov_mix = (m["open"] * 0.00817205) + (m["vwap"] * (1 - 0.00817205))
        sum_adv = ops.rolling_sum(adv60, 60)
        corr = ops.rolling_corr(ov_mix, sum_adv, 9)
        rank1 = _cs(corr)
        min13 = ops.rolling_min(m["open"], 13)
        rank2 = _cs((m["open"] - min13) / sum_adv)
        return Factor.as_cs_series(df, (rank1 < rank2).astype(float))


@register
class Alpha066(Factor):
    """((close - delay(close, 6)) / delay(close, 6)) * volume"""
    name = "Alpha066"
    requires = ["close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        delay6 = ops.delay(m["close"], 6)
        return Factor.as_cs_series(df, ((m["close"] - delay6) / delay6) * m["volume"])


@register
class Alpha067(Factor):
    """(-1 * correlation(delta(close, 1), delay(close, 1), 250) * (close / delay(close, 1)))"""
    name = "Alpha067"
    requires = ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        delta_close = ops.delta(m["close"], 1)
        delay_close = ops.delay(m["close"], 1)
        corr = ops.rolling_corr(delta_close, delay_close, 250)
        return Factor.as_cs_series(df, -corr * (m["close"] / delay_close))


@register
class Alpha068(Factor):
    """Ts_Rank(correlation(rank(high), rank(adv15), 9), 14)"""
    name = "Alpha068"
    requires = ["high", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv15 = _g(df, "volume", lambda s: ops.adv(s, 15))
        corr = ops.rolling_corr(_cs(m["high"]), _cs(adv15), 9)
        ts = _ts_rank_mi(corr, 14)
        return Factor.as_cs_series(df, ts)


@register
class Alpha069(Factor):
    """((rank(delta(close, 1) / delay(close, 1)) * rank(volume / adv20)) * rank(high - low))"""
    name = "Alpha069"
    requires = ["high", "low", "close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv20 = _g(df, "volume", lambda s: ops.adv(s, 20))
        ret = ops.delta(m["close"], 1) / ops.delay(m["close"], 1)
        return Factor.as_cs_series(df, _cs(ret) * _cs(m["volume"] / adv20) * _cs(m["high"] - m["low"]))


@register
class Alpha070(Factor):
    """(-1 * rank(delta(((close - low) / (high - low)), 1)))"""
    name = "Alpha070"
    requires = ["high", "low", "close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        hl_range = m["high"] - m["low"]
        ratio = (m["close"] - m["low"]) / hl_range
        return Factor.as_cs_series(df, -_cs(ops.delta(ratio, 1)))


@register
class Alpha071(Factor):
    """max(rank(decay_linear(delta(vwap, 5), 17)), Ts_Rank(delta(((close * 0.496803) + (vwap * 0.503197)), 2), 5)) * (-1)"""
    name = "Alpha071"
    requires = ["close", "low", "open", "vwap"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        delta_vwap = ops.delta(m["vwap"], 5)
        decay = ops.decay_linear(delta_vwap, 17)
        rank1 = _cs(decay)
        cv_mix = (m["close"] * 0.496803) + (m["vwap"] * 0.503197)
        delta_cv = ops.delta(cv_mix, 2)
        ts = _ts_rank_mi(delta_cv, 5)
        return Factor.as_cs_series(df, pd.Series(np.maximum(rank1.values, ts.values), index=m.index) * (-1))


@register
class Alpha072(Factor):
    """(rank(decay_linear(correlation(((high + low) / 2), adv40, 9), 14)) - rank(decay_linear(correlation(rank(vwap), rank(volume), 4), 12)))"""
    name = "Alpha072"
    requires = ["high", "low", "vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv40 = _g(df, "volume", lambda s: ops.adv(s, 40))
        hl_avg = (m["high"] + m["low"]) / 2
        corr1 = ops.rolling_corr(hl_avg, adv40, 9)
        decay1 = ops.decay_linear(corr1, 14)
        rank1 = _cs(decay1)
        rank_vwap = _cs(m["vwap"])
        rank_vol = _cs(m["volume"])
        corr2 = ops.rolling_corr(rank_vwap, rank_vol, 4)
        decay2 = ops.decay_linear(corr2, 12)
        rank2 = _cs(decay2)
        return Factor.as_cs_series(df, rank1 - rank2)


@register
class Alpha073(Factor):
    """(max(rank(decay_linear(delta(vwap, 5), 17)), Ts_Rank(delta(((close * 0.496803) + (vwap * 0.503197)), 2), 5)) * (-1))"""
    name = "Alpha073"
    requires = ["close", "vwap"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        delta_vwap = ops.delta(m["vwap"], 5)
        decay = ops.decay_linear(delta_vwap, 17)
        rank1 = _cs(decay)
        cv_mix = (m["close"] * 0.496803) + (m["vwap"] * 0.503197)
        delta_cv = ops.delta(cv_mix, 2)
        ts = _ts_rank_mi(delta_cv, 5)
        return Factor.as_cs_series(df, pd.Series(np.maximum(rank1.values, ts.values), index=m.index) * (-1))


@register
class Alpha074(Factor):
    """((rank(correlation(close, sum(adv30, 37), 15)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11))) * (-1))"""
    name = "Alpha074"
    requires = ["high", "close", "vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv30 = _g(df, "volume", lambda s: ops.adv(s, 30))
        sum_adv = ops.rolling_sum(adv30, 37)
        corr1 = ops.rolling_corr(m["close"], sum_adv, 15)
        rank1 = _cs(corr1)
        hv_mix = (m["high"] * 0.0261661) + (m["vwap"] * (1 - 0.0261661))
        rank_hv = _cs(hv_mix)
        rank_vol = _cs(m["volume"])
        corr2 = ops.rolling_corr(rank_hv, rank_vol, 11)
        rank2 = _cs(corr2)
        return Factor.as_cs_series(df, (rank1 < rank2).astype(float) * (-1))


@register
class Alpha075(Factor):
    """(correlation(volume, vwap, 4) * (-1))"""
    name = "Alpha075"
    requires = ["volume", "vwap"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        return Factor.as_cs_series(df, ops.rolling_corr(m["volume"], m["vwap"], 4) * (-1))


@register
class Alpha076(Factor):
    """(max(rank(decay_linear(delta(vwap, 1), 12)), Ts_Rank(decay_linear(((close * 0.383476) + (vwap * 0.616524)), 19), 8)) * (-1))"""
    name = "Alpha076"
    requires = ["close", "vwap"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        delta_vwap = ops.delta(m["vwap"], 1)
        decay1 = ops.decay_linear(delta_vwap, 12)
        rank1 = _cs(decay1)
        cv_mix = (m["close"] * 0.383476) + (m["vwap"] * 0.616524)
        decay2 = ops.decay_linear(cv_mix, 19)
        ts = _ts_rank_mi(decay2, 8)
        return Factor.as_cs_series(df, pd.Series(np.maximum(rank1.values, ts.values), index=m.index) * (-1))


@register
class Alpha077(Factor):
    """(min(rank(decay_linear(((high + low) / 2) + ((high - low) / 2), 20)), Ts_Rank(decay_linear(correlation(((high + low) / 2), adv40, 3), 6), 4)) * (-1))"""
    name = "Alpha077"
    requires = ["high", "low", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv40 = _g(df, "volume", lambda s: ops.adv(s, 40))
        hl_avg = (m["high"] + m["low"]) / 2
        hl_diff = (m["high"] - m["low"]) / 2
        hl_combined = hl_avg + hl_diff
        decay1 = ops.decay_linear(hl_combined, 20)
        rank1 = _cs(decay1)
        corr = ops.rolling_corr(hl_avg, adv40, 3)
        decay2 = ops.decay_linear(corr, 6)
        ts = _ts_rank_mi(decay2, 4)
        return Factor.as_cs_series(df, pd.Series(np.minimum(rank1.values, ts.values), index=m.index) * (-1))


@register
class Alpha078(Factor):
    """(rank(correlation(sum(((low * 0.352233) + (vwap * 0.647767)), 20), sum(adv40, 20), 7)) * rank(rank(volume / adv20)))"""
    name = "Alpha078"
    requires = ["low", "vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv20 = _g(df, "volume", lambda s: ops.adv(s, 20))
        adv40 = _g(df, "volume", lambda s: ops.adv(s, 40))
        lv_mix = (m["low"] * 0.352233) + (m["vwap"] * 0.647767)
        sum_lv = ops.rolling_sum(lv_mix, 20)
        sum_adv = ops.rolling_sum(adv40, 20)
        corr = ops.rolling_corr(sum_lv, sum_adv, 7)
        rank1 = _cs(corr)
        rank_vol = _cs(_cs(m["volume"] / adv20))
        return Factor.as_cs_series(df, rank1 * rank_vol)


@register
class Alpha079(Factor):
    """(rank(delta(((close * 0.607189) + (open * 0.392811)), 1)) < rank(correlation(rank(vwap), rank(adv150), 10)))"""
    name = "Alpha079"
    requires = ["close", "open", "vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv150 = _g(df, "volume", lambda s: ops.adv(s, 150))
        co_mix = (m["close"] * 0.607189) + (m["open"] * 0.392811)
        delta_co = ops.delta(co_mix, 1)
        rank1 = _cs(delta_co)
        rank_vwap = _cs(m["vwap"])
        rank_adv = _cs(adv150)
        corr = ops.rolling_corr(rank_vwap, rank_adv, 10)
        rank2 = _cs(corr)
        return Factor.as_cs_series(df, (rank1 < rank2).astype(float))


@register
class Alpha080(Factor):
    """((rank(Sign(delta(((open * 0.868128) + (high * 0.131872)), 4))) * rank(correlation(((high * 0.51827) + (low * 0.48173)), sum(adv30, 30), 14)))"""
    name = "Alpha080"
    requires = ["high", "low", "open", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv30 = _g(df, "volume", lambda s: ops.adv(s, 30))
        oh_mix = (m["open"] * 0.868128) + (m["high"] * 0.131872)
        delta_oh = ops.delta(oh_mix, 4)
        rank1 = _cs(np.sign(delta_oh))
        hl_mix = (m["high"] * 0.51827) + (m["low"] * 0.48173)
        sum_adv = ops.rolling_sum(adv30, 30)
        corr = ops.rolling_corr(hl_mix, sum_adv, 14)
        rank2 = _cs(corr)
        return Factor.as_cs_series(df, rank1 * rank2)


@register
class Alpha081(Factor):
    """((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 50), 8))^4)), 15))) < rank(correlation(rank(vwap), rank(volume), 15))) * (-1))"""
    name = "Alpha081"
    requires = ["vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv10 = _g(df, "volume", lambda s: ops.adv(s, 10))
        sum_adv = ops.rolling_sum(adv10, 50)
        corr = ops.rolling_corr(m["vwap"], sum_adv, 8)
        rank_corr = _cs(corr)
        prod = rank_corr ** 4
        rank_prod = _cs(prod)
        log_prod = np.log(rank_prod)
        rank1 = _cs(ops.rolling_sum(log_prod, 15))
        rank_vwap = _cs(m["vwap"])
        rank_vol = _cs(m["volume"])
        corr2 = ops.rolling_corr(rank_vwap, rank_vol, 15)
        rank2 = _cs(corr2)
        return Factor.as_cs_series(df, (rank1 < rank2).astype(float) * (-1))


@register
class Alpha082(Factor):
    """(min(rank(decay_linear(delta(open, 1), 15)), Ts_Rank(decay_linear(correlation(volume, low, 6), 17), 7)) * (-1))"""
    name = "Alpha082"
    requires = ["open", "low", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        delta_open = ops.delta(m["open"], 1)
        decay1 = ops.decay_linear(delta_open, 15)
        rank1 = _cs(decay1)
        corr = ops.rolling_corr(m["volume"], m["low"], 6)
        decay2 = ops.decay_linear(corr, 17)
        ts = decay2.groupby(level=1).apply(lambda x: ops.ts_rank(x, 7))
        return Factor.as_cs_series(df, pd.Series(np.minimum(rank1.values, ts.values), index=m.index) * (-1))


@register
class Alpha083(Factor):
    """((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high - low) / (sum(close, 5) / 5)) * (correlation(volume, ((high - low) / (sum(close, 5) / 5)), 5))))"""
    name = "Alpha083"
    requires = ["high", "low", "close", "vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        sum_close5 = ops.rolling_sum(m["close"], 5) / 5
        hl_ratio = (m["high"] - m["low"]) / sum_close5
        delay_hl = ops.delay(hl_ratio, 2)
        rank1 = _cs(delay_hl)
        rank2 = _cs(_cs(m["volume"]))
        corr = ops.rolling_corr(m["volume"], hl_ratio, 5)
        return Factor.as_cs_series(df, (rank1 * rank2) / (hl_ratio * corr))


@register
class Alpha084(Factor):
    """(-1 * power((vwap - ts_min(vwap, 14)), 3))"""
    name = "Alpha084"
    requires = ["vwap", "close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        ts_min = ops.rolling_min(m["vwap"], 14)
        return Factor.as_cs_series(df, -(m["vwap"] - ts_min) ** 3)


@register
class Alpha085(Factor):
    """(rank((volume / adv20)) * rank((high - low) / close))"""
    name = "Alpha085"
    requires = ["high", "close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv20 = _g(df, "volume", lambda s: ops.adv(s, 20))
        rank_vol = _cs(m["volume"] / adv20)
        rank_hl = _cs((m["high"] - m["low"]) / m["close"])
        return Factor.as_cs_series(df, rank_vol * rank_hl)


@register
class Alpha086(Factor):
    """((delay((correlation(close, volume, 10)), 5) * rank(((sum(close, 20) / 20) * volume))) * rank(volume / adv60))"""
    name = "Alpha086"
    requires = ["close", "open", "vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv60 = _g(df, "volume", lambda s: ops.adv(s, 60))
        corr = ops.rolling_corr(m["close"], m["volume"], 10)
        delay_corr = ops.delay(corr, 5)
        sum_close = ops.rolling_sum(m["close"], 20) / 20
        rank1 = _cs(delay_corr * (sum_close * m["volume"]))
        rank2 = _cs(m["volume"] / adv60)
        return Factor.as_cs_series(df, rank1 * rank2)


@register
class Alpha087(Factor):
    """(rank(decay_linear(correlation(((high * 0.876703) + (close * 0.123297)), adv30, 10), 13)) - rank(decay_linear(correlation(rank(vwap), rank(volume), 4), 12)))"""
    name = "Alpha087"
    requires = ["high", "close", "vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv30 = _g(df, "volume", lambda s: ops.adv(s, 30))
        hc_mix = (m["high"] * 0.876703) + (m["close"] * 0.123297)
        corr1 = ops.rolling_corr(hc_mix, adv30, 10)
        decay1 = ops.decay_linear(corr1, 13)
        rank1 = _cs(decay1)
        rank_vwap = _cs(m["vwap"])
        rank_vol = _cs(m["volume"])
        corr2 = ops.rolling_corr(rank_vwap, rank_vol, 4)
        decay2 = ops.decay_linear(corr2, 12)
        rank2 = _cs(decay2)
        return Factor.as_cs_series(df, rank1 - rank2)


@register
class Alpha088(Factor):
    """((Ts_Rank(decay_linear(correlation(((close * 0.496803) + (vwap * 0.503197)), adv20, 5), 7), 5) < rank(decay_linear(correlation(rank(vwap), rank(volume), 4), 12))) * (-1))"""
    name = "Alpha088"
    requires = ["close", "vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv20 = _g(df, "volume", lambda s: ops.adv(s, 20))
        cv_mix = (m["close"] * 0.496803) + (m["vwap"] * 0.503197)
        corr = ops.rolling_corr(cv_mix, adv20, 5)
        decay = ops.decay_linear(corr, 7)
        ts = _ts_rank_mi(decay, 5)
        rank_vwap = _cs(m["vwap"])
        rank_vol = _cs(m["volume"])
        corr2 = ops.rolling_corr(rank_vwap, rank_vol, 4)
        decay2 = ops.decay_linear(corr2, 12)
        rank_decay = _cs(decay2)
        return Factor.as_cs_series(df, pd.Series((ts.values < rank_decay.values).astype(float), index=m.index) * (-1))


@register
class Alpha089(Factor):
    """(Ts_Rank(decay_linear(correlation(low, adv10, 6), 9), 4) - Ts_Rank(decay_linear(Ts_Rank(correlation(vwap, adv20, 5), 18), 16), 9))"""
    name = "Alpha089"
    requires = ["low", "vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv10 = _g(df, "volume", lambda s: ops.adv(s, 10))
        adv20 = _g(df, "volume", lambda s: ops.adv(s, 20))
        corr1 = ops.rolling_corr(m["low"], adv10, 6)
        decay1 = ops.decay_linear(corr1, 9)
        ts1 = _ts_rank_mi(decay1, 4)
        corr2 = ops.rolling_corr(m["vwap"], adv20, 5)
        ts2_inner = _ts_rank_mi(corr2, 18)
        decay2 = ops.decay_linear(ts2_inner, 16)
        ts2 = _ts_rank_mi(decay2, 9)
        return Factor.as_cs_series(df, pd.Series(ts1.values - ts2.values, index=m.index))


@register
class Alpha090(Factor):
    """((rank((close - min(close, 5))) ^ rank(decay_linear((vwap * rank(((high * 0.51827) + (low * 0.48173)))), 14))) * (-1))"""
    name = "Alpha090"
    requires = ["high", "low", "close", "vwap"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        min_close = ops.rolling_min(m["close"], 5)
        rank1 = _cs(m["close"] - min_close)
        hl_mix = (m["high"] * 0.51827) + (m["low"] * 0.48173)
        rank_hl = _cs(hl_mix)
        prod = m["vwap"] * rank_hl
        decay = ops.decay_linear(prod, 14)
        rank2 = _cs(decay)
        return Factor.as_cs_series(df, pd.Series(-(rank1.values ** rank2.values), index=m.index))


@register
class Alpha091(Factor):
    """((rank((close - min(close, 5))) * rank(decay_linear(vwap, 14))) * (-1))"""
    name = "Alpha091"
    requires = ["close", "vwap"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        min_close = ops.rolling_min(m["close"], 5)
        rank1 = _cs(m["close"] - min_close)
        decay = ops.decay_linear(m["vwap"], 14)
        rank2 = _cs(decay)
        return Factor.as_cs_series(df, pd.Series(-(rank1.values * rank2.values), index=m.index))


@register
class Alpha092(Factor):
    """((Ts_Rank(decay_linear(correlation(((high * 0.51827) + (low * 0.48173)), adv30, 5), 7), 5) < rank(decay_linear(correlation(rank(vwap), rank(volume), 4), 12))) * (-1))"""
    name = "Alpha092"
    requires = ["high", "low", "vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv30 = _g(df, "volume", lambda s: ops.adv(s, 30))
        hl_mix = (m["high"] * 0.51827) + (m["low"] * 0.48173)
        corr = ops.rolling_corr(hl_mix, adv30, 5)
        decay = ops.decay_linear(corr, 7)
        ts = _ts_rank_mi(decay, 5)
        rank_vwap = _cs(m["vwap"])
        rank_vol = _cs(m["volume"])
        corr2 = ops.rolling_corr(rank_vwap, rank_vol, 4)
        decay2 = ops.decay_linear(corr2, 12)
        rank_decay = _cs(decay2)
        return Factor.as_cs_series(df, pd.Series((ts.values < rank_decay.values).astype(float), index=m.index) * (-1))


@register
class Alpha093(Factor):
    """(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 4), 12), 5) - Ts_Rank(decay_linear(correlation(((high * 0.51827) + (low * 0.48173)), adv30, 5), 7), 5))"""
    name = "Alpha093"
    requires = ["high", "low", "vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv30 = _g(df, "volume", lambda s: ops.adv(s, 30))
        rank_vwap = _cs(m["vwap"])
        rank_vol = _cs(m["volume"])
        corr1 = ops.rolling_corr(rank_vwap, rank_vol, 4)
        decay1 = ops.decay_linear(corr1, 12)
        ts1 = _ts_rank_mi(decay1, 5)
        hl_mix = (m["high"] * 0.51827) + (m["low"] * 0.48173)
        corr2 = ops.rolling_corr(hl_mix, adv30, 5)
        decay2 = ops.decay_linear(corr2, 7)
        ts2 = decay2.groupby(level=1).apply(lambda x: ops.ts_rank(x, 5))
        return Factor.as_cs_series(df, pd.Series(ts1.values - ts2.values, index=m.index))


@register
class Alpha094(Factor):
    """((rank(((vwap - min(vwap, 11))) / (ts_rank(correlation(vwap, adv150, 6), 15)))) * rank(volume / adv20))"""
    name = "Alpha094"
    requires = ["vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv20 = _g(df, "volume", lambda s: ops.adv(s, 20))
        adv150 = _g(df, "volume", lambda s: ops.adv(s, 150))
        min11 = ops.rolling_min(m["vwap"], 11)
        corr = ops.rolling_corr(m["vwap"], adv150, 6)
        ts = corr.groupby(level=1).transform(lambda x: ops.ts_rank(x, 15))
        rank1 = _cs((m["vwap"] - min11) / ts)
        rank2 = _cs(m["volume"] / adv20)
        return Factor.as_cs_series(df, rank1 * rank2)


@register
class Alpha095(Factor):
    """(rank((open - ts_min(open, 12))) < rank(correlation(((high + low) / 2), sum(adv40, 40), 10)))"""
    name = "Alpha095"
    requires = ["open", "high", "low", "close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv40 = _g(df, "volume", lambda s: ops.adv(s, 40))
        min12 = ops.rolling_min(m["open"], 12)
        rank1 = _cs(m["open"] - min12)
        hl_avg = (m["high"] + m["low"]) / 2
        sum_adv = ops.rolling_sum(adv40, 40)
        corr = ops.rolling_corr(hl_avg, sum_adv, 10)
        rank2 = _cs(corr)
        return Factor.as_cs_series(df, (rank1 < rank2).astype(float))


@register
class Alpha096(Factor):
    """(-1 * max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 4), 4), 8), Ts_Rank(decay_linear(Ts_Rank(correlation(rank(close), rank(adv20), 5), 7), 17), 9)))"""
    name = "Alpha096"
    requires = ["vwap", "volume", "close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv20 = _g(df, "volume", lambda s: ops.adv(s, 20))
        rank_vwap = _cs(m["vwap"])
        rank_vol = _cs(m["volume"])
        corr1 = ops.rolling_corr(rank_vwap, rank_vol, 4)
        decay1 = ops.decay_linear(corr1, 4)
        ts1 = _ts_rank_mi(decay1, 8)
        rank_close = _cs(m["close"])
        rank_adv = _cs(adv20)
        corr2 = ops.rolling_corr(rank_close, rank_adv, 5)
        ts2_inner = _ts_rank_mi(corr2, 7)
        decay2 = ops.decay_linear(ts2_inner, 17)
        ts2 = _ts_rank_mi(decay2, 9)
        return Factor.as_cs_series(df, pd.Series(-np.maximum(ts1.values, ts2.values), index=m.index))


@register
class Alpha097(Factor):
    """((rank(decay_linear(delta(((low * 0.721002) + (vwap * 0.278998)), 3), 7)) < Ts_Rank(decay_linear(Ts_Rank(correlation(close, adv20, 5), 11), 20), 8)) * (-1))"""
    name = "Alpha097"
    requires = ["low", "vwap", "close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv20 = _g(df, "volume", lambda s: ops.adv(s, 20))
        lv_mix = (m["low"] * 0.721002) + (m["vwap"] * 0.278998)
        delta_lv = ops.delta(lv_mix, 3)
        decay1 = ops.decay_linear(delta_lv, 7)
        rank1 = _cs(decay1)
        corr = ops.rolling_corr(m["close"], adv20, 5)
        ts_inner = corr.groupby(level=1).apply(lambda x: ops.ts_rank(x, 11))
        decay2 = ops.decay_linear(ts_inner, 20)
        ts = _ts_rank_mi(decay2, 8)
        return Factor.as_cs_series(df, pd.Series((rank1.values < ts.values).astype(float), index=m.index) * (-1))


@register
class Alpha098(Factor):
    """(rank(decay_linear(correlation(vwap, sum(adv5, 26), 5), 8)) - rank(decay_linear(Ts_Rank(correlation(rank(close), rank(volume), 4), 16), 4)))"""
    name = "Alpha098"
    requires = ["vwap", "volume", "open"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv5 = _g(df, "volume", lambda s: ops.adv(s, 5))
        sum_adv = ops.rolling_sum(adv5, 26)
        corr1 = ops.rolling_corr(m["vwap"], sum_adv, 5)
        decay1 = ops.decay_linear(corr1, 8)
        rank1 = _cs(decay1)
        rank_close = _cs(m["close"])
        rank_vol = _cs(m["volume"])
        corr2 = ops.rolling_corr(rank_close, rank_vol, 4)
        ts = corr2.groupby(level=1).transform(lambda x: ops.ts_rank(x, 16))
        decay2 = ops.decay_linear(ts, 4)
        rank2 = _cs(decay2)
        return Factor.as_cs_series(df, rank1 - rank2)


@register
class Alpha099(Factor):
    """(-1 * rank(correlation(((high + low) / 2), sum(adv60, 40), 9)))"""
    name = "Alpha099"
    requires = ["high", "low", "volume", "close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv60 = _g(df, "volume", lambda s: ops.adv(s, 60))
        hl_avg = (m["high"] + m["low"]) / 2
        sum_adv = ops.rolling_sum(adv60, 40)
        corr = ops.rolling_corr(hl_avg, sum_adv, 9)
        return Factor.as_cs_series(df, -_cs(corr))


@register
class Alpha100(Factor):
    """(0 - (1 * ((1.5 * scale(vwap)) * scale(correlation(((high * 0.51827) + (low * 0.48173)), adv30, 15)))))"""
    name = "Alpha100"
    requires = ["high", "low", "vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        adv30 = _g(df, "volume", lambda s: ops.adv(s, 30))
        g1 = m["vwap"].groupby(level=0)
        scale1 = (m["vwap"] - g1.transform("mean")) / g1.transform(lambda x: np.sum(np.abs(x))).replace(0, np.nan)
        hl_mix = (m["high"] * 0.51827) + (m["low"] * 0.48173)
        corr = ops.rolling_corr(hl_mix, adv30, 15)
        g2 = corr.groupby(level=0)
        scale2 = (corr - g2.transform("mean")) / g2.transform(lambda x: np.sum(np.abs(x))).replace(0, np.nan)
        return Factor.as_cs_series(df, pd.Series(-(1.5 * scale1.values * scale2.values), index=m.index))


@register
class Alpha101(Factor):
    """((close - open) / ((high - low) + 0.001))"""
    name = "Alpha101"
    requires = ["open", "high", "low", "close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = _mi(df)
        return Factor.as_cs_series(df, (m["close"] - m["open"]) / ((m["high"] - m["low"]) + 0.001))
