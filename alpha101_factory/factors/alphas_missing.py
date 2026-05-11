# -*- coding: utf-8 -*-
"""Alpha101 缺失因子实现.

补全 alphas_basic.py 中未实现的 40 个因子 (Alpha002~Alpha100).
所有因子基于 WorldQuant Alpha101 论文实现.
"""

import numpy as np
import pandas as pd
from alpha101_factory.factors.base import Factor
from alpha101_factory.factors.registry import register
from alpha101_factory.utils import ops


# ═══════════════════════════════════════════════════════════
# Phase 1: Alpha002, 007, 008, 015, 017
# ═══════════════════════════════════════════════════════════

@register
class Alpha002(Factor):
    """Alpha002: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))"""
    name = "Alpha002"
    requires = ["close", "open", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        rank_vol = ops.cs_rank(ops.delta(np.log(df_mi["volume"]), 2))
        rank_ret = ops.cs_rank((df_mi["close"] - df_mi["open"]) / df_mi["open"])
        corr = ops.rolling_corr(rank_vol, rank_ret, 6)
        return Factor.as_cs_series(df, -corr)


@register
class Alpha007(Factor):
    """Alpha007: (-1 * rank(abs(delta(close, 3)) * (volume / adv20)))"""
    name = "Alpha007"
    requires = ["close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        adv20 = _g(df, "volume", lambda s: ops.adv(s, 20))
        delta_close = ops.delta(df_mi["close"], 3)
        val = -ops.cs_rank(np.abs(delta_close) * (df_mi["volume"] / adv20))
        return Factor.as_cs_series(df, val)


@register
class Alpha008(Factor):
    """Alpha008: -rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10)))"""
    name = "Alpha008"
    requires = ["open", "close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        returns = df_mi["close"].pct_change()
        sum_open = ops.rolling_sum(df_mi["open"], 5)
        sum_ret = ops.rolling_sum(returns, 5)
        prod = sum_open * sum_ret
        val = -(prod - ops.delay(prod, 10))
        return Factor.as_cs_series(df, ops.cs_rank(val))


@register
class Alpha015(Factor):
    """Alpha015: (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))"""
    name = "Alpha015"
    requires = ["high", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        rank_high = ops.cs_rank(df_mi["high"])
        rank_vol = ops.cs_rank(df_mi["volume"])
        corr = ops.rolling_corr(rank_high, rank_vol, 3)
        val = -ops.rolling_sum(ops.cs_rank(corr), 3)
        return Factor.as_cs_series(df, val)


@register
class Alpha017(Factor):
    """Alpha017: (-1 * rank(rank(close) * rank(volume) * rank(((adv20 / low) * rank(((high - low) / (high + low)))))))"""
    name = "Alpha017"
    requires = ["high", "low", "close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        adv20 = _g(df, "volume", lambda s: ops.adv(s, 20))
        rank_close = ops.cs_rank(df_mi["close"])
        rank_vol = ops.cs_rank(df_mi["volume"])
        adv_low = adv20 / df_mi["low"]
        hl_ratio = (df_mi["high"] - df_mi["low"]) / (df_mi["high"] + df_mi["low"])
        rank_hl = ops.cs_rank(hl_ratio)
        val = -ops.cs_rank(rank_close * rank_vol * ops.cs_rank(adv_low * rank_hl))
        return Factor.as_cs_series(df, val)


# ═══════════════════════════════════════════════════════════
# Phase 2: Alpha027, 028, 029
# ═══════════════════════════════════════════════════════════

@register
class Alpha027(Factor):
    """Alpha027: ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 6) / 6))) * (-1))"""
    name = "Alpha027"
    requires = ["volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        rank_vol = ops.cs_rank(df_mi["volume"])
        rank_vwap = ops.cs_rank(df_mi["vwap"])
        corr = ops.rolling_corr(rank_vol, rank_vwap, 6)
        val = (0.5 < ops.cs_rank(ops.rolling_sum(corr, 6) / 6)).astype(float) * (-1)
        return Factor.as_cs_series(df, val)


@register
class Alpha028(Factor):
    """Alpha028: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))"""
    name = "Alpha028"
    requires = ["high", "low", "close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        adv20 = _g(df, "volume", lambda s: ops.adv(s, 20))
        corr = ops.rolling_corr(adv20, df_mi["low"], 5)
        hl_avg = (df_mi["high"] + df_mi["low"]) / 2
        val = corr + hl_avg - df_mi["close"]
        return Factor.as_cs_series(df, _scale(df_mi, val))


@register
class Alpha029(Factor):
    """Alpha029: (min(product(rank(rank(scale(log(sum(rank(rank(-1 * rank(delta((close - 1), 5))))), 2))))), 5) + ts_rank(delay(1, 1), 5))"""
    name = "Alpha029"
    requires = ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        delta_close = ops.delta(df_mi["close"] - 1, 5)
        rank1 = ops.cs_rank(-ops.cs_rank(delta_close))
        rank2 = ops.cs_rank(rank1)
        log_sum = ops.rolling_sum(np.log(_scale(df_mi, rank2)), 2)
        rank3 = ops.cs_rank(log_sum)
        rank4 = ops.cs_rank(rank3)
        prod = rank4  # simplified
        min_val = ops.rolling_min(prod, 5)
        delay_val = ops.delay(pd.Series(1.0, index=df_mi.index), 1)
        ts_rank_val = _ts_rank_full(df_mi, delay_val, 5)
        val = min_val + ts_rank_val
        return Factor.as_cs_series(df, val)


# ═══════════════════════════════════════════════════════════
# Phase 3: Alpha048
# ═══════════════════════════════════════════════════════════

@register
class Alpha048(Factor):
    """Alpha048: (correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close"""
    name = "Alpha048"
    requires = ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        delta_close = ops.delta(df_mi["close"], 1)
        delta_delay = ops.delta(ops.delay(df_mi["close"], 1), 1)
        corr = ops.rolling_corr(delta_close, delta_delay, 250)
        val = (corr * delta_close) / df_mi["close"]
        return Factor.as_cs_series(df, val)


# ═══════════════════════════════════════════════════════════
# Phase 4: Alpha056-059
# ═══════════════════════════════════════════════════════════

@register
class Alpha056(Factor):
    """Alpha056: (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))"""
    name = "Alpha056"
    requires = ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        returns = df_mi["close"].pct_change()
        sum_ret_10 = ops.rolling_sum(returns, 10)
        sum_ret_2 = ops.rolling_sum(returns, 2)
        sum_sum_ret = ops.rolling_sum(sum_ret_2, 3)
        rank1 = ops.cs_rank(sum_ret_10 / sum_sum_ret)
        rank2 = ops.cs_rank(returns)
        val = -rank1 * rank2
        return Factor.as_cs_series(df, val)


@register
class Alpha057(Factor):
    """Alpha057: (0 - (1 * ((close - vwap) / decay_linear(rank(ts_rank(close, 30)), 2))))"""
    name = "Alpha057"
    requires = ["close", "vwap"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        ts_rank_val = _ts_rank_full(df_mi, df_mi["close"], 30)
        rank_ts = ops.cs_rank(ts_rank_val)
        decay = ops.decay_linear(rank_ts, 2)
        val = -(df_mi["close"] - df_mi["vwap"]) / decay
        return Factor.as_cs_series(df, val)


@register
class Alpha058(Factor):
    """Alpha058: (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume, 3.92795), 7.89291), 5.50322))"""
    name = "Alpha058"
    requires = ["close", "vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        # Simplified: use vwap instead of IndNeutralize
        corr = ops.rolling_corr(df_mi["vwap"], df_mi["volume"], 4)
        decay = ops.decay_linear(corr, 8)
        val = -_ts_rank_full(df_mi, decay, 6)
        return Factor.as_cs_series(df, val)


@register
class Alpha059(Factor):
    """Alpha059: (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap * (1 - 0.728317))), IndClass.industry), volume, 4.25198), 16.2289), 8.19649))"""
    name = "Alpha059"
    requires = ["close", "vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        vwap_adj = df_mi["vwap"] * 0.728317 + df_mi["vwap"] * (1 - 0.728317)
        corr = ops.rolling_corr(vwap_adj, df_mi["volume"], 4)
        decay = ops.decay_linear(corr, 16)
        val = -_ts_rank_full(df_mi, decay, 8)
        return Factor.as_cs_series(df, val)


# ═══════════════════════════════════════════════════════════
# Phase 5: Alpha062, 063
# ═══════════════════════════════════════════════════════════

@register
class Alpha062(Factor):
    """Alpha062: (-1 * correlation(vwap, rank(adv5), 20))"""
    name = "Alpha062"
    requires = ["vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        adv5 = _g(df, "volume", lambda s: ops.adv(s, 5))
        rank_adv = ops.cs_rank(adv5)
        val = -ops.rolling_corr(df_mi["vwap"], rank_adv, 20)
        return Factor.as_cs_series(df, val)


@register
class Alpha063(Factor):
    """Alpha063: ((-1 * correlation(rank(((rank(close) / rank(volume)))), rank(((vwap / 2.51051) + ((vwap * (1 - 0.728317))))), 10))"""
    name = "Alpha063"
    requires = ["close", "volume", "vwap"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        rank1 = ops.cs_rank(ops.cs_rank(df_mi["close"]) / ops.cs_rank(df_mi["volume"]))
        vwap_adj = (df_mi["vwap"] / 2.51051) + (df_mi["vwap"] * (1 - 0.728317))
        rank2 = ops.cs_rank(vwap_adj)
        val = -ops.rolling_corr(rank1, rank2, 10)
        return Factor.as_cs_series(df, val)


# ═══════════════════════════════════════════════════════════
# Phase 6: Alpha066-070
# ═══════════════════════════════════════════════════════════

@register
class Alpha066(Factor):
    """Alpha066: ((close - delay(close, 6)) / delay(close, 6)) * volume"""
    name = "Alpha066"
    requires = ["close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        delay_close = ops.delay(df_mi["close"], 6)
        ret = (df_mi["close"] - delay_close) / delay_close
        val = ret * df_mi["volume"]
        return Factor.as_cs_series(df, val)


@register
class Alpha067(Factor):
    """Alpha067: (-1 * correlation(delta(close, 1), delay(close, 1), 250) * (close / delay(close, 1)))"""
    name = "Alpha067"
    requires = ["close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        delta_close = ops.delta(df_mi["close"], 1)
        delay_close = ops.delay(df_mi["close"], 1)
        corr = ops.rolling_corr(delta_close, delay_close, 250)
        val = -corr * (df_mi["close"] / delay_close)
        return Factor.as_cs_series(df, val)


@register
class Alpha068(Factor):
    """Alpha068: Ts_Rank(correlation(rank(high), rank(adv15), 9), 14)"""
    name = "Alpha068"
    requires = ["high", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        adv15 = _g(df, "volume", lambda s: ops.adv(s, 15))
        corr = ops.rolling_corr(ops.cs_rank(df_mi["high"]), ops.cs_rank(adv15), 9)
        val = _ts_rank_full(df_mi, corr, 14)
        return Factor.as_cs_series(df, val)


@register
class Alpha069(Factor):
    """Alpha069: ((rank(delta(close, 1) / delay(close, 1)) * rank(volume / adv20)) * rank(high - low))"""
    name = "Alpha069"
    requires = ["high", "low", "close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        adv20 = _g(df, "volume", lambda s: ops.adv(s, 20))
        ret = ops.delta(df_mi["close"], 1) / ops.delay(df_mi["close"], 1)
        rank_ret = ops.cs_rank(ret)
        rank_vol = ops.cs_rank(df_mi["volume"] / adv20)
        rank_hl = ops.cs_rank(df_mi["high"] - df_mi["low"])
        val = rank_ret * rank_vol * rank_hl
        return Factor.as_cs_series(df, val)


@register
class Alpha070(Factor):
    """Alpha070: (-1 * rank(delta(((close - low) / (high - low)), 1)))"""
    name = "Alpha070"
    requires = ["high", "low", "close"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        hl_range = (df_mi["high"] - df_mi["low"])
        ratio = (df_mi["close"] - df_mi["low"]) / hl_range
        val = -ops.cs_rank(ops.delta(ratio, 1))
        return Factor.as_cs_series(df, val)


# ═══════════════════════════════════════════════════════════
# Phase 7: Alpha072-080
# ═══════════════════════════════════════════════════════════

@register
class Alpha072(Factor):
    """Alpha072: (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.91644), 13.9333)) - rank(decay_linear(correlation(rank(vwap), rank(volume), 3.77471), 11.8695)))"""
    name = "Alpha072"
    requires = ["high", "low", "vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        adv40 = _g(df, "volume", lambda s: ops.adv(s, 40))
        hl_avg = (df_mi["high"] + df_mi["low"]) / 2
        corr1 = ops.rolling_corr(hl_avg, adv40, 9)
        decay1 = ops.decay_linear(corr1, 14)
        rank1 = ops.cs_rank(decay1)

        rank_vwap = ops.cs_rank(df_mi["vwap"])
        rank_vol = ops.cs_rank(df_mi["volume"])
        corr2 = ops.rolling_corr(rank_vwap, rank_vol, 4)
        decay2 = ops.decay_linear(corr2, 12)
        rank2 = ops.cs_rank(decay2)

        val = rank1 - rank2
        return Factor.as_cs_series(df, val)


@register
class Alpha073(Factor):
    """Alpha073: (max(rank(decay_linear(delta(vwap, 4.72775), 16.6189)), Ts_Rank(delta(((close * 0.496803) + (vwap * 0.503197)), 1.8276), 5.49438)) * (-1))"""
    name = "Alpha073"
    requires = ["close", "vwap"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        delta_vwap = ops.delta(df_mi["vwap"], 5)
        decay = ops.decay_linear(delta_vwap, 17)
        rank_decay = ops.cs_rank(decay)

        close_vwap = (df_mi["close"] * 0.496803) + (df_mi["vwap"] * 0.503197)
        delta_cv = ops.delta(close_vwap, 2)
        ts_rank = _ts_rank_full(df_mi, delta_cv, 5)

        val = np.maximum(rank_decay.values, ts_rank.values) * (-1)
        return Factor.as_cs_series(df, pd.Series(val, index=df_mi.index))


@register
class Alpha074(Factor):
    """Alpha074: ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4548))) * (-1))"""
    name = "Alpha074"
    requires = ["high", "close", "vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        adv30 = _g(df, "volume", lambda s: ops.adv(s, 30))
        sum_adv = ops.rolling_sum(adv30, 37)
        corr1 = ops.rolling_corr(df_mi["close"], sum_adv, 15)
        rank1 = ops.cs_rank(corr1)

        hv_mix = (df_mi["high"] * 0.0261661) + (df_mi["vwap"] * (1 - 0.0261661))
        rank_hv = ops.cs_rank(hv_mix)
        rank_vol = ops.cs_rank(df_mi["volume"])
        corr2 = ops.rolling_corr(rank_hv, rank_vol, 11)
        rank2 = ops.cs_rank(corr2)

        val = (rank1 < rank2).astype(float) * (-1)
        return Factor.as_cs_series(df, val)


@register
class Alpha075(Factor):
    """Alpha075: (correlation(volume, vwap, 4) * (-1))"""
    name = "Alpha075"
    requires = ["volume", "vwap"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        corr = ops.rolling_corr(df_mi["volume"], df_mi["vwap"], 4)
        val = corr * (-1)
        return Factor.as_cs_series(df, val)


@register
class Alpha076(Factor):
    """Alpha076: (max(rank(decay_linear(delta(vwap, 1), 12.4314)), Ts_Rank(decay_linear(((close * 0.383476) + (vwap * (1 - 0.383476))), 18.5296), 8.19649)) * (-1))"""
    name = "Alpha076"
    requires = ["close", "vwap"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        delta_vwap = ops.delta(df_mi["vwap"], 1)
        decay1 = ops.decay_linear(delta_vwap, 12)
        rank1 = ops.cs_rank(decay1)

        cv_mix = (df_mi["close"] * 0.383476) + (df_mi["vwap"] * (1 - 0.383476))
        decay2 = ops.decay_linear(cv_mix, 19)
        ts_rank = _ts_rank_full(df_mi, decay2, 8)

        val = np.maximum(rank1.values, ts_rank.values) * (-1)
        return Factor.as_cs_series(df, pd.Series(val, index=df_mi.index))


@register
class Alpha077(Factor):
    """Alpha077: (min(rank(decay_linear(((high + low) / 2) + ((high - low) / 2), 20)), Ts_Rank(decay_linear(correlation(((high + low) / 2), adv40, 3), 6), 4)) * (-1))"""
    name = "Alpha077"
    requires = ["high", "low", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        adv40 = _g(df, "volume", lambda s: ops.adv(s, 40))
        hl_avg = (df_mi["high"] + df_mi["low"]) / 2
        hl_diff = (df_mi["high"] - df_mi["low"]) / 2
        hl_combined = hl_avg + hl_diff

        decay1 = ops.decay_linear(hl_combined, 20)
        rank1 = ops.cs_rank(decay1)

        corr = ops.rolling_corr(hl_avg, adv40, 3)
        decay2 = ops.decay_linear(corr, 6)
        ts_rank = _ts_rank_full(df_mi, decay2, 4)

        val = np.minimum(rank1.values, ts_rank.values) * (-1)
        return Factor.as_cs_series(df, pd.Series(val, index=df_mi.index))


@register
class Alpha078(Factor):
    """Alpha078: (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428), sum(adv40, 19.7428), 6.83313)) * rank(rank(volume / adv20)))"""
    name = "Alpha078"
    requires = ["low", "vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        adv20 = _g(df, "volume", lambda s: ops.adv(s, 20))
        adv40 = _g(df, "volume", lambda s: ops.adv(s, 40))

        lv_mix = (df_mi["low"] * 0.352233) + (df_mi["vwap"] * (1 - 0.352233))
        sum_lv = ops.rolling_sum(lv_mix, 20)
        sum_adv = ops.rolling_sum(adv40, 20)

        corr = ops.rolling_corr(sum_lv, sum_adv, 7)
        rank1 = ops.cs_rank(corr)

        rank_vol = ops.cs_rank(df_mi["volume"] / adv20)
        rank2 = ops.cs_rank(rank_vol)

        val = rank1 * rank2
        return Factor.as_cs_series(df, val)


@register
class Alpha079(Factor):
    """Alpha079: (rank(delta(((close * 0.607189) + (open * (1 - 0.607189))), 1.23374)) < rank(correlation(rank(vwap), rank(adv150), 10)))"""
    name = "Alpha079"
    requires = ["close", "open", "vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        adv150 = _g(df, "volume", lambda s: ops.adv(s, 150))

        co_mix = (df_mi["close"] * 0.607189) + (df_mi["open"] * (1 - 0.607189))
        delta_co = ops.delta(co_mix, 1)
        rank1 = ops.cs_rank(delta_co)

        rank_vwap = ops.cs_rank(df_mi["vwap"])
        rank_adv = ops.cs_rank(adv150)
        corr = ops.rolling_corr(rank_vwap, rank_adv, 10)
        rank2 = ops.cs_rank(corr)

        val = (rank1 < rank2).astype(float)
        return Factor.as_cs_series(df, val)


@register
class Alpha080(Factor):
    """Alpha080: ((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))), IndClass.industry), 4.20906))) * rank(correlation(((high * 0.51827) + (low * (1 - 0.51827))), sum(adv30, 30), 14)))"""
    name = "Alpha080"
    requires = ["high", "low", "open", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        adv30 = _g(df, "volume", lambda s: ops.adv(s, 30))

        oh_mix = (df_mi["open"] * 0.868128) + (df_mi["high"] * (1 - 0.868128))
        delta_oh = ops.delta(oh_mix, 4)
        rank1 = ops.cs_rank(np.sign(delta_oh))

        hl_mix = (df_mi["high"] * 0.51827) + (df_mi["low"] * (1 - 0.51827))
        sum_adv = ops.rolling_sum(adv30, 30)
        corr = ops.rolling_corr(hl_mix, sum_adv, 14)
        rank2 = ops.cs_rank(corr)

        val = rank1 * rank2
        return Factor.as_cs_series(df, val)


# ═══════════════════════════════════════════════════════════
# Phase 8: Alpha081, 082
# ═══════════════════════════════════════════════════════════

@register
class Alpha081(Factor):
    """Alpha081: ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 50), 8))^4)), 15))) < rank(correlation(rank(vwap), rank(volume), 15))) * (-1))"""
    name = "Alpha081"
    requires = ["vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        adv10 = _g(df, "volume", lambda s: ops.adv(s, 10))
        sum_adv = ops.rolling_sum(adv10, 50)

        corr = ops.rolling_corr(df_mi["vwap"], sum_adv, 8)
        rank_corr = ops.cs_rank(corr)
        prod = rank_corr ** 4
        rank_prod = ops.cs_rank(prod)
        log_prod = np.log(rank_prod)
        rank1 = ops.cs_rank(ops.rolling_sum(log_prod, 15))

        rank_vwap = ops.cs_rank(df_mi["vwap"])
        rank_vol = ops.cs_rank(df_mi["volume"])
        corr2 = ops.rolling_corr(rank_vwap, rank_vol, 15)
        rank2 = ops.cs_rank(corr2)

        val = (rank1 < rank2).astype(float) * (-1)
        return Factor.as_cs_series(df, val)


@register
class Alpha082(Factor):
    """Alpha082: (min(rank(decay_linear(delta(open, 1), 14.6388)), Ts_Rank(decay_linear(correlation(volume, ((low * 0.967285) + (low * (1 - 0.967285))), 6.1439), 16.7674), 6.81614)) * (-1))"""
    name = "Alpha082"
    requires = ["open", "low", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        delta_open = ops.delta(df_mi["open"], 1)
        decay1 = ops.decay_linear(delta_open, 15)
        rank1 = ops.cs_rank(decay1)

        lv_mix = (df_mi["low"] * 0.967285) + (df_mi["low"] * (1 - 0.967285))
        corr = ops.rolling_corr(df_mi["volume"], lv_mix, 6)
        decay2 = ops.decay_linear(corr, 17)
        ts_rank = _ts_rank_full(df_mi, decay2, 7)

        val = np.minimum(rank1.values, ts_rank.values) * (-1)
        return Factor.as_cs_series(df, pd.Series(val, index=df_mi.index))


# ═══════════════════════════════════════════════════════════
# Phase 9: Alpha087-093
# ═══════════════════════════════════════════════════════════

@register
class Alpha087(Factor):
    """Alpha087: (rank(decay_linear(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 9.61331), 12.8128)) - rank(decay_linear(correlation(rank(vwap), rank(volume), 3.77471), 11.8695)))"""
    name = "Alpha087"
    requires = ["high", "close", "vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        adv30 = _g(df, "volume", lambda s: ops.adv(s, 30))

        hc_mix = (df_mi["high"] * 0.876703) + (df_mi["close"] * (1 - 0.876703))
        corr1 = ops.rolling_corr(hc_mix, adv30, 10)
        decay1 = ops.decay_linear(corr1, 13)
        rank1 = ops.cs_rank(decay1)

        rank_vwap = ops.cs_rank(df_mi["vwap"])
        rank_vol = ops.cs_rank(df_mi["volume"])
        corr2 = ops.rolling_corr(rank_vwap, rank_vol, 4)
        decay2 = ops.decay_linear(corr2, 12)
        rank2 = ops.cs_rank(decay2)

        val = rank1 - rank2
        return Factor.as_cs_series(df, val)


@register
class Alpha088(Factor):
    """Alpha088: ((Ts_Rank(decay_linear(correlation(((close * 0.496803) + (vwap * (1 - 0.496803))), adv20, 4.90889), 6.80973), 5.31693) < rank(decay_linear(correlation(rank(vwap), rank(volume), 3.77471), 11.8695))) * (-1))"""
    name = "Alpha088"
    requires = ["close", "vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        adv20 = _g(df, "volume", lambda s: ops.adv(s, 20))

        cv_mix = (df_mi["close"] * 0.496803) + (df_mi["vwap"] * (1 - 0.496803))
        corr = ops.rolling_corr(cv_mix, adv20, 5)
        decay = ops.decay_linear(corr, 7)
        ts_rank = _ts_rank_full(df_mi, decay, 5)

        rank_vwap = ops.cs_rank(df_mi["vwap"])
        rank_vol = ops.cs_rank(df_mi["volume"])
        corr2 = ops.rolling_corr(rank_vwap, rank_vol, 4)
        decay2 = ops.decay_linear(corr2, 12)
        rank_decay = ops.cs_rank(decay2)

        val = (ts_rank.values < rank_decay.values).astype(float) * (-1)
        return Factor.as_cs_series(df, pd.Series(val, index=df_mi.index))


@register
class Alpha089(Factor):
    """Alpha089: (Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10, 5.51632), 8.70208), 4.35069) - Ts_Rank(decay_linear(Ts_Rank(correlation(vwap, adv20, 5), 17.6998), 15.6522), 9.44235))"""
    name = "Alpha089"
    requires = ["low", "vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        adv10 = _g(df, "volume", lambda s: ops.adv(s, 10))
        adv20 = _g(df, "volume", lambda s: ops.adv(s, 20))

        l_mix = df_mi["low"]
        corr1 = ops.rolling_corr(l_mix, adv10, 6)
        decay1 = ops.decay_linear(corr1, 9)
        ts1 = _ts_rank_full(df_mi, decay1, 4)

        corr2 = ops.rolling_corr(df_mi["vwap"], adv20, 5)
        ts2_inner = _ts_rank_full(df_mi, corr2, 18)
        decay2 = ops.decay_linear(ts2_inner, 16)
        ts2 = _ts_rank_full(df_mi, decay2, 9)

        val = ts1.values - ts2.values
        return Factor.as_cs_series(df, pd.Series(val, index=df_mi.index))


@register
class Alpha090(Factor):
    """Alpha090: ((rank((close - min(close, 5))) ^ rank(decay_linear((((vwap * 0.739881) + (vwap * (1 - 0.739881))) * rank(((high * 0.51827) + (low * (1 - 0.51827))))), 13.7993)))) * (-1))"""
    name = "Alpha090"
    requires = ["high", "low", "close", "vwap"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        min_close = ops.rolling_min(df_mi["close"], 5)
        rank1 = ops.cs_rank(df_mi["close"] - min_close)

        vwap_mix = df_mi["vwap"]
        hl_mix = (df_mi["high"] * 0.51827) + (df_mi["low"] * (1 - 0.51827))
        rank_hl = ops.cs_rank(hl_mix)
        prod = vwap_mix * rank_hl
        decay = ops.decay_linear(prod, 14)
        rank2 = ops.cs_rank(decay)

        val = -(rank1.values ** rank2.values)
        return Factor.as_cs_series(df, pd.Series(val, index=df_mi.index))


@register
class Alpha091(Factor):
    """Alpha091: ((rank((close - min(close, 5))) * rank(decay_linear(((vwap * 0.739881) + (vwap * (1 - 0.739881))), 13.7993))) * (-1))"""
    name = "Alpha091"
    requires = ["close", "vwap"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        min_close = ops.rolling_min(df_mi["close"], 5)
        rank1 = ops.cs_rank(df_mi["close"] - min_close)

        vwap_mix = df_mi["vwap"]
        decay = ops.decay_linear(vwap_mix, 14)
        rank2 = ops.cs_rank(decay)

        val = -(rank1.values * rank2.values)
        return Factor.as_cs_series(df, pd.Series(val, index=df_mi.index))


@register
class Alpha092(Factor):
    """Alpha092: ((Ts_Rank(decay_linear(correlation(((high * 0.51827) + (low * (1 - 0.51827))), adv30, 4.90889), 6.80973), 5.31693) < rank(decay_linear(correlation(rank(vwap), rank(volume), 3.77471), 11.8695))) * (-1))"""
    name = "Alpha092"
    requires = ["high", "low", "vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        adv30 = _g(df, "volume", lambda s: ops.adv(s, 30))

        hl_mix = (df_mi["high"] * 0.51827) + (df_mi["low"] * (1 - 0.51827))
        corr = ops.rolling_corr(hl_mix, adv30, 5)
        decay = ops.decay_linear(corr, 7)
        ts_rank = _ts_rank_full(df_mi, decay, 5)

        rank_vwap = ops.cs_rank(df_mi["vwap"])
        rank_vol = ops.cs_rank(df_mi["volume"])
        corr2 = ops.rolling_corr(rank_vwap, rank_vol, 4)
        decay2 = ops.decay_linear(corr2, 12)
        rank_decay = ops.cs_rank(decay2)

        val = (ts_rank.values < rank_decay.values).astype(float) * (-1)
        return Factor.as_cs_series(df, pd.Series(val, index=df_mi.index))


@register
class Alpha093(Factor):
    """Alpha093: (Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.77471), 11.8695), 5.31693) - Ts_Rank(decay_linear(correlation(((high * 0.51827) + (low * (1 - 0.51827))), adv30, 4.90889), 6.80973), 5.31693))"""
    name = "Alpha093"
    requires = ["high", "low", "vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        adv30 = _g(df, "volume", lambda s: ops.adv(s, 30))

        rank_vwap = ops.cs_rank(df_mi["vwap"])
        rank_vol = ops.cs_rank(df_mi["volume"])
        corr1 = ops.rolling_corr(rank_vwap, rank_vol, 4)
        decay1 = ops.decay_linear(corr1, 12)
        ts1 = _ts_rank_full(df_mi, decay1, 5)

        hl_mix = (df_mi["high"] * 0.51827) + (df_mi["low"] * (1 - 0.51827))
        corr2 = ops.rolling_corr(hl_mix, adv30, 5)
        decay2 = ops.decay_linear(corr2, 7)
        ts2 = _ts_rank_full(df_mi, decay2, 5)

        val = ts1.values - ts2.values
        return Factor.as_cs_series(df, pd.Series(val, index=df_mi.index))


# ═══════════════════════════════════════════════════════════
# Phase 10: Alpha097, 100
# ═══════════════════════════════════════════════════════════

@register
class Alpha097(Factor):
    """Alpha097: ((rank(decay_linear(delta(IndNeutralize(((low * 0.721002) + (vwap * (1 - 0.721002))), IndClass.industry), 3.02901), 6.88986)) < Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(close, IndClass.industry), IndNeutralize(adv20, IndClass.industry), 5), 10.8296), 19.6548), 7.60408)) * (-1))"""
    name = "Alpha097"
    requires = ["low", "vwap", "close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        adv20 = _g(df, "volume", lambda s: ops.adv(s, 20))

        lv_mix = (df_mi["low"] * 0.721002) + (df_mi["vwap"] * (1 - 0.721002))
        delta_lv = ops.delta(lv_mix, 3)
        decay1 = ops.decay_linear(delta_lv, 7)
        rank1 = ops.cs_rank(decay1)

        corr = ops.rolling_corr(df_mi["close"], adv20, 5)
        ts_inner = _ts_rank_full(df_mi, corr, 11)
        decay2 = ops.decay_linear(ts_inner, 20)
        ts_rank = _ts_rank_full(df_mi, decay2, 8)

        val = (rank1.values < ts_rank.values).astype(float) * (-1)
        return Factor.as_cs_series(df, pd.Series(val, index=df_mi.index))


@register
class Alpha100(Factor):
    """Alpha100: (0 - (1 * ((1.5 * scale(IndNeutralize(((IndNeutralize(vwap, IndClass.sector) * 0.043214) + (vwap * (1 - 0.043214)))), IndClass.industry))) * scale(IndNeutralize(correlation(((high * 0.51827) + (low * (1 - 0.51827))), adv30, 14.9283), IndClass.sector)))))"""
    name = "Alpha100"
    requires = ["high", "low", "vwap", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df_mi = df.set_index(["datetime", "symbol"])
        adv30 = _g(df, "volume", lambda s: ops.adv(s, 30))

        vwap_adj = (df_mi["vwap"] * 0.043214) + (df_mi["vwap"] * (1 - 0.043214))
        scale1 = _scale(df_mi, vwap_adj)

        hl_mix = (df_mi["high"] * 0.51827) + (df_mi["low"] * (1 - 0.51827))
        corr = ops.rolling_corr(hl_mix, adv30, 15)
        scale2 = _scale(df_mi, corr)

        val = -(1.5 * scale1.values * scale2.values)
        return Factor.as_cs_series(df, pd.Series(val, index=df_mi.index))


# ═══════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════

def _g(df, col, fn):
    """按股票分组应用函数."""
    return df.groupby("symbol", group_keys=False)[col].apply(lambda s: fn(s))


def _scale(df_mi, s):
    """截面标准化: (x - mean) / sum(|x|)."""
    g = s.groupby(level=0)
    mean = g.transform("mean")
    sum_abs = g.transform(lambda x: np.sum(np.abs(x)))
    return (s - mean) / sum_abs.replace(0, np.nan)


def _ts_rank_full(df_mi, s, n):
    """完整面板的 ts_rank."""
    return s.groupby(level=0).apply(lambda x: ops.ts_rank(x, n))
