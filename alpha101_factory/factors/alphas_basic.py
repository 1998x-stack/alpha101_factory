# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from .base import Factor
from .registry import register
from ..utils import ops

# ---- 工具：截面 rank ----
def _cs_rank(df: pd.DataFrame, s: pd.Series) -> pd.Series:
    idx = pd.MultiIndex.from_frame(df[["datetime","symbol"]], names=["datetime","symbol"])
    s = pd.Series(s.values, index=idx)
    return s.groupby(level=0).rank(pct=True)

# 便捷：取列并按 symbol 应用 rolling 函数
def _g(df, col, fn, *a):
    return df.groupby("symbol", group_keys=False)[col].apply(lambda x: fn(x, *a))

# ===== Alpha 实现（示例若干）=====

@register
class Alpha001(Factor):
    name = "Alpha001"
    requires = ["returns","close"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        x = df["returns"].copy()
        part = np.where(x < 0, _g(df, "returns", ops.rolling_std, 20), df["close"])
        val = _g(df.assign(part=part), "part", lambda s: ops.ts_rank( (s**2), 5))
        out = _cs_rank(df, val) - 0.5
        return Factor.as_cs_series(df, out)

@register
class Alpha003(Factor):
    name = "Alpha003"
    requires = ["open","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        val = - _g(df, "open", lambda s: ops.rolling_corr(ops.cs_rank(s), ops.cs_rank(df.loc[s.index, "volume"]), 10))
        return Factor.as_cs_series(df, val)

@register
class Alpha004(Factor):
    name = "Alpha004"
    requires = ["low"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        val = - _g(df, "low", lambda s: ops.ts_rank(ops.cs_rank(s), 9))
        return Factor.as_cs_series(df, val)

@register
class Alpha005(Factor):
    name = "Alpha005"
    requires = ["open","vwap","close"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        a = df.groupby("symbol")["open"].apply(lambda s: ops.rolling_sum(s, 10)/10)
        b = ops.cs_rank(df["open"] - a.values)
        c = - np.abs(ops.cs_rank(df["close"] - df["vwap"]))
        val = b * c
        return Factor.as_cs_series(df, val)

@register
class Alpha006(Factor):
    name = "Alpha006"
    requires = ["open","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        val = - _g(df, "open", lambda s: ops.rolling_corr(s, df.loc[s.index,"volume"], 10))
        return Factor.as_cs_series(df, val)

@register
class Alpha009(Factor):
    name = "Alpha009"
    requires = ["close"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        d1 = _g(df, "close", ops.delta, 1)
        cond1 = _g(df, "close", lambda s: ops.rolling_min(ops.delta(s,1), 5)) > 0
        cond2 = _g(df, "close", lambda s: ops.rolling_max(ops.delta(s,1), 5)) < 0
        val = np.where(cond1, d1, np.where(cond2, d1, -d1))
        return Factor.as_cs_series(df, pd.Series(val))

@register
class Alpha010(Factor):
    name = "Alpha010"
    requires = ["close"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        d1 = _g(df, "close", ops.delta, 1)
        cond1 = _g(df, "close", lambda s: ops.rolling_min(ops.delta(s,1), 4)) > 0
        cond2 = _g(df, "close", lambda s: ops.rolling_max(ops.delta(s,1), 4)) < 0
        val = ops.cs_rank(np.where(cond1, d1, np.where(cond2, d1, -d1)))
        return Factor.as_cs_series(df, val)

@register
class Alpha011(Factor):
    name = "Alpha011"
    requires = ["vwap","close","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        a = _g(df, "vwap", lambda s: ops.ts_rank(s - df.loc[s.index,"close"], 3))
        b = _g(df, "vwap", lambda s: ops.ts_rank(df.loc[s.index,"close"] - s, 3))
        c = _g(df, "volume", lambda s: ops.ts_rank(ops.delta(s,3), 3))
        val = (ops.cs_rank(a) + ops.cs_rank(b)) * ops.cs_rank(c)
        return Factor.as_cs_series(df, val)

@register
class Alpha012(Factor):
    name = "Alpha012"
    requires = ["close","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        val = np.sign(_g(df, "volume", ops.delta, 1)) * (- _g(df, "close", ops.delta, 1))
        return Factor.as_cs_series(df, pd.Series(val))

@register
class Alpha013(Factor):
    name = "Alpha013"
    requires = ["close","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        val = - _g(df, "close", lambda s: ops.rolling_cov(ops.cs_rank(s), ops.cs_rank(df.loc[s.index,"volume"]), 5))
        return Factor.as_cs_series(df, val)

@register
class Alpha014(Factor):
    name = "Alpha014"
    requires = ["open","volume","returns"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        val = (- ops.cs_rank(_g(df,"returns", ops.delta, 3))) * _g(df,"open", lambda s: ops.rolling_corr(s, df.loc[s.index,"volume"], 10))
        return Factor.as_cs_series(df, val)

@register
class Alpha016(Factor):
    name = "Alpha016"
    requires = ["high","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        val = - _g(df,"high", lambda s: ops.rolling_cov(ops.cs_rank(s), ops.cs_rank(df.loc[s.index,"volume"]), 5))
        return Factor.as_cs_series(df, val)

@register
class Alpha018(Factor):
    name = "Alpha018"
    requires = ["close","open"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        a = _g(df, "close", lambda s: ops.rolling_std(np.abs(s - df.loc[s.index,"open"]), 5))
        b = df["close"] - df["open"]
        c = _g(df, "close", lambda s: ops.rolling_corr(s, df.loc[s.index,"open"], 10))
        val = - ops.cs_rank(a + b + c)
        return Factor.as_cs_series(df, val)

@register
class Alpha019(Factor):
    name = "Alpha019"
    requires = ["close","returns"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        val = - np.sign(df["close"] - _g(df,"close", ops.delay, 7) + _g(df,"close", ops.delta, 7)) * (1 + ops.cs_rank(1 + _g(df,"returns", ops.rolling_sum, 250)))
        return Factor.as_cs_series(df, pd.Series(val))

@register
class Alpha020(Factor):
    name = "Alpha020"
    requires = ["open","high","low","close"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        val = (- ops.cs_rank(df["open"] - _g(df,"high", ops.delay, 1))) * ops.cs_rank(df["open"] - _g(df,"close", ops.delay, 1)) * ops.cs_rank(df["open"] - _g(df,"low", ops.delay, 1))
        return Factor.as_cs_series(df, val)

@register
class Alpha021(Factor):
    name = "Alpha021"
    requires = ["close","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        s8 = _g(df,"close", lambda s: ops.rolling_sum(s, 8) / 8)
        sd8 = _g(df,"close", lambda s: ops.rolling_std(s, 8))
        s2 = _g(df,"close", lambda s: ops.rolling_sum(s, 2) / 2)
        adv20 = _g(df,"volume", lambda s: ops.adv(s, 20))
        cond = ((s8 + sd8) < s2) * (-1) + ((s2 < (s8 - sd8)) * 1)
        cond2 = (((df["volume"]/adv20) >= 1) * 1) + (((df["volume"]/adv20) < 1) * (-1))
        val = np.where(cond != 0, cond, cond2)
        return Factor.as_cs_series(df, pd.Series(val))

@register
class Alpha022(Factor):
    name = "Alpha022"
    requires = ["high","volume","close"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        corr = _g(df,"high", lambda s: ops.rolling_corr(s, df.loc[s.index,"volume"], 5))
        val = - _g(df, None, lambda *_: ops.delta(corr, 5)) * ops.cs_rank(_g(df,"close", ops.rolling_std, 20))
        return Factor.as_cs_series(df, val)

@register
class Alpha023(Factor):
    name = "Alpha023"
    requires = ["high","close"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        cond = (_g(df,"close", ops.rolling_sum, 20)/20 < df["high"])
        val = np.where(cond, - _g(df,"high", ops.delta, 2), 0)
        return Factor.as_cs_series(df, pd.Series(val))

@register
class Alpha025(Factor):
    name = "Alpha025"
    requires = ["returns","vwap","high","close","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        adv20 = _g(df,"volume", lambda s: ops.adv(s, 20))
        val = ops.cs_rank((-df["returns"]) * adv20 * df["vwap"] * (df["high"] - df["close"]))
        return Factor.as_cs_series(df, val)

@register
class Alpha026(Factor):
    name = "Alpha026"
    requires = ["volume","high"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        a = _g(df,"volume", lambda s: ops.ts_rank(s, 5))
        b = _g(df,"high", lambda s: ops.ts_rank(s, 5))
        val = - _g(df, None, lambda *_: ops.rolling_max(ops.rolling_corr(a, b, 5), 3))
        return Factor.as_cs_series(df, val)

@register
class Alpha033(Factor):
    name = "Alpha033"
    requires = ["open","close"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        val = ops.cs_rank(- (1 - (df["open"]/df["close"])))
        return Factor.as_cs_series(df, val)

@register
class Alpha034(Factor):
    name = "Alpha034"
    requires = ["returns","close"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        a = 1 - ops.cs_rank(_g(df,"returns", lambda s: ops.rolling_std(s, 2) / ops.rolling_std(s,5)))
        b = 1 - ops.cs_rank(_g(df,"close", ops.delta, 1))
        val = a + b
        return Factor.as_cs_series(df, val)

@register
class Alpha035(Factor):
    name = "Alpha035"
    requires = ["volume","close","high","low","returns"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        a = _g(df,"volume", lambda s: ops.ts_rank(s,32))
        b = 1 - _g(df, None, lambda *_: ops.ts_rank(((df["close"]+df["high"])-df["low"]), 16))
        c = 1 - _g(df,"returns", lambda s: ops.ts_rank(s,32))
        val = a * b * c
        return Factor.as_cs_series(df, val)

@register
class Alpha038(Factor):
    name = "Alpha038"
    requires = ["close","open"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        val = - _g(df,"close", lambda s: ops.ts_rank(s, 10)) * ops.cs_rank(df["close"]/df["open"])
        return Factor.as_cs_series(df, val)

@register
class Alpha040(Factor):
    name = "Alpha040"
    requires = ["high","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        val = (- ops.cs_rank(_g(df,"high", ops.rolling_std, 10))) * _g(df,"high", lambda s: ops.rolling_corr(s, df.loc[s.index,"volume"], 10))
        return Factor.as_cs_series(df, val)

@register
class Alpha041(Factor):
    name = "Alpha041"
    requires = ["high","low","vwap"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        val = (df["high"]*df["low"])**0.5 - df["vwap"]
        return Factor.as_cs_series(df, val)

@register
class Alpha042(Factor):
    name = "Alpha042"
    requires = ["vwap","close"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        val = ops.cs_rank(df["vwap"] - df["close"]) / ops.cs_rank(df["vwap"] + df["close"])
        return Factor.as_cs_series(df, val)

@register
class Alpha043(Factor):
    name = "Alpha043"
    requires = ["volume","close","returns","vwap"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        adv20 = _g(df,"volume", lambda s: ops.adv(s, 20))
        val = _g(df, None, lambda *_: ops.ts_rank(df["volume"]/adv20, 20)) * _g(df,"close", lambda s: ops.ts_rank(- ops.delta(s,7), 8))
        return Factor.as_cs_series(df, val)

@register
class Alpha045(Factor):
    name = "Alpha045"
    requires = ["close","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        a = ops.cs_rank(_g(df,"close", lambda s: ops.rolling_sum(ops.delay(s,5), 20)/20))
        b = _g(df,"close", lambda s: ops.rolling_corr(s, df.loc[s.index,"volume"], 2))
        c = _g(df,"close", lambda s: ops.rolling_corr(_g(df,"close", lambda s2: ops.rolling_sum(s2,5)), _g(df,"close", lambda s2: ops.rolling_sum(s2,20)), 2))
        val = - (a * b * ops.cs_rank(c))
        return Factor.as_cs_series(df, val)

@register
class Alpha046(Factor):
    name = "Alpha046"
    requires = ["close"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        a = (_g(df,"close", ops.delay,20) - _g(df,"close", ops.delay,10))/10 - (_g(df,"close", ops.delay,10) - df["close"])/10
        val = np.where(a > 0.25, -1, np.where(a < 0, 1, -1*(df["close"] - _g(df,"close", ops.delay,1))))
        return Factor.as_cs_series(df, pd.Series(val))

@register
class Alpha049(Factor):
    name = "Alpha049"
    requires = ["close"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        a = (_g(df,"close", ops.delay,20) - _g(df,"close", ops.delay,10))/10 - (_g(df,"close", ops.delay,10) - df["close"])/10
        val = np.where(a < -0.1, 1, -1*(df["close"] - _g(df,"close", ops.delay,1)))
        return Factor.as_cs_series(df, pd.Series(val))

@register
class Alpha050(Factor):
    name = "Alpha050"
    requires = ["volume","vwap"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        val = - _g(df, None, lambda *_: ops.rolling_max(
            ops.cs_rank(_g(df,"volume", lambda s: ops.rolling_corr(ops.cs_rank(s), ops.cs_rank(df.loc[s.index,"vwap"]), 5))), 5))
        return Factor.as_cs_series(df, val)

@register
class Alpha051(Factor):
    name = "Alpha051"
    requires = ["close"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        a = (_g(df,"close", ops.delay,20) - _g(df,"close", ops.delay,10))/10 - (_g(df,"close", ops.delay,10) - df["close"])/10
        val = np.where(a < -0.05, 1, -1*(df["close"] - _g(df,"close", ops.delay,1)))
        return Factor.as_cs_series(df, pd.Series(val))

@register
class Alpha052(Factor):
    name = "Alpha052"
    requires = ["low","returns","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        part = (- _g(df,"low", lambda s: ops.rolling_min(s,5)) + _g(df,"low", lambda s: ops.delay(ops.rolling_min(s,5),5))) * ops.cs_rank((_g(df,"returns", ops.rolling_sum,240) - _g(df,"returns", ops.rolling_sum,20))/220) * _g(df,"volume", lambda s: ops.ts_rank(s,5))
        return Factor.as_cs_series(df, part)

@register
class Alpha053(Factor):
    name = "Alpha053"
    requires = ["close","low","high"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        x = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["close"] - df["low"]).replace(0,np.nan)
        val = - _g(df, None, lambda *_: ops.delta(x, 9))
        return Factor.as_cs_series(df, val)

@register
class Alpha055(Factor):
    name = "Alpha055"
    requires = ["close","high","low","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        num = (df["close"] - _g(df,"low", lambda s: ops.rolling_min(s,12))) / (_g(df,"high", lambda s: ops.rolling_max(s,12)) - _g(df,"low", lambda s: ops.rolling_min(s,12))).replace(0,np.nan)
        val = - _g(df, None, lambda *_: ops.rolling_corr(ops.cs_rank(num), ops.cs_rank(df["volume"]), 6))
        return Factor.as_cs_series(df, val)

@register
class Alpha060(Factor):
    name = "Alpha060"
    requires = ["high","low","close","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        x = (((df["close"]-df["low"]) - (df["high"]-df["close"])) / (df["high"]-df["low"]).replace(0,np.nan)) * df["volume"]
        val = - ( 2*ops.cs_rank(ops.decay_linear(x, 10)) - ops.cs_rank(_g(df,"close", lambda s: ops.ts_rank(s,10))) )
        return Factor.as_cs_series(df, val)

@register
class Alpha101(Factor):
    name = "Alpha101"
    requires = ["open","high","low","close"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        val = (df["close"] - df["open"]) / ((df["high"] - df["low"]) + 0.001)
        return Factor.as_cs_series(df, val)
