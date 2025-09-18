# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from .base import Factor
from .registry import register
from ..utils import ops

def _cs_rank(df: pd.DataFrame, s: pd.Series) -> pd.Series:
    idx = pd.MultiIndex.from_frame(df[["datetime","symbol"]], names=["datetime","symbol"])
    s = pd.Series(s.values, index=idx)
    return s.groupby(level=0).rank(pct=True)

def _g(df, col, fn, *a):
    return df.groupby("symbol", group_keys=False)[col].apply(lambda x: fn(x, *a))

@register
class Alpha024(Factor):
    name = "Alpha024"
    requires = ["close"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        s100 = _g(df,"close", lambda s: ops.rolling_sum(s,100) / 100)
        d = _g(df,"close", lambda s: ops.delta(ops.rolling_sum(s,100)/100, 100)) / _g(df,"close", lambda s: ops.delay(s,100))
        cond = (d <= 0.05)
        val = np.where(cond, - (df["close"] - _g(df,"close", ops.rolling_min, 100)),
                              - _g(df,"close", ops.delta, 3))
        return Factor.as_cs_series(df, pd.Series(val))

@register
class Alpha030(Factor):
    name = "Alpha030"
    requires = ["close","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        s5 = _g(df,"volume", lambda s: ops.rolling_sum(s,5))
        s20 = _g(df,"volume", lambda s: ops.rolling_sum(s,20))
        sig = (1.0 - _cs_rank(df, (np.sign(ops.delta(_g(df,"close", ops.delay,1),1)) +
                                   np.sign(ops.delta(_g(df,"close", ops.delay,2),1)) +
                                   np.sign(ops.delta(_g(df,"close", ops.delay,3),1)))))
        val = (sig * s5) / s20.replace(0,np.nan)
        return Factor.as_cs_series(df, val)

@register
class Alpha031(Factor):
    name = "Alpha031"
    requires = ["close","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        a = ops.cs_rank(ops.decay_linear(- _cs_rank(_g(df,"close", lambda s: ops.delta(s,10))), 10))
        b = ops.cs_rank(- _g(df,"close", ops.delta, 3))
        adv20 = _g(df,"volume", lambda s: ops.adv(s,20))
        c = np.sign(ops.cs_rank(_g(df,"volume", lambda s: ops.rolling_corr(adv20, df.loc[s.index,"low"] if "low" in df.columns else s, 12))))
        val = a + b + c
        return Factor.as_cs_series(df, val)

@register
class Alpha032(Factor):
    name = "Alpha032"
    requires = ["close","vwap"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        part1 = ops.cs_rank((_g(df,"close", lambda s: ops.rolling_sum(s,7)/7) - df["close"]))
        part2 = 20 * ops.cs_rank(_g(df,"close", lambda s: ops.rolling_corr(df.loc[s.index,"vwap"], _g(df,"close", ops.delay,5), 230)))
        return Factor.as_cs_series(df, part1 + part2)

@register
class Alpha036(Factor):
    name = "Alpha036"
    requires = ["close","open","volume","vwap","returns"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        a = 2.21 * ops.cs_rank(_g(df, None, lambda *_: ops.rolling_corr(df["close"]-df["open"], _g(df,"volume", ops.delay,1), 15)))
        b = 0.7 * ops.cs_rank(df["open"] - df["close"])
        c = 0.73 * ops.cs_rank(_g(df, None, lambda *_: ops.ts_rank(ops.delay(-df["returns"],6), 5)))
        d = ops.cs_rank(np.abs(_g(df, None, lambda *_: ops.rolling_corr(df["vwap"], _g(df,"volume", lambda s: ops.adv(s,20)), 6))))
        e = 0.6 * ops.cs_rank((_g(df,"close", lambda s: ops.rolling_sum(s,200)/200) - df["open"]) * (df["close"] - df["open"]))
        val = a + b + c + d + e
        return Factor.as_cs_series(df, val)

@register
class Alpha037(Factor):
    name = "Alpha037"
    requires = ["open","close"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        a = ops.cs_rank(_g(df, None, lambda *_: ops.rolling_corr(ops.delay(df["open"]-df["close"],1), df["close"], 200)))
        b = ops.cs_rank(df["open"] - df["close"])
        return Factor.as_cs_series(df, a + b)

@register
class Alpha039(Factor):
    name = "Alpha039"
    requires = ["close","volume","returns"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        adv20 = _g(df,"volume", lambda s: ops.adv(s,20))
        part = - ops.cs_rank(_g(df,"close", lambda s: ops.delta(s,7)) * (1 - ops.cs_rank(ops.decay_linear(df["volume"]/adv20, 9))))
        val = part * (1 + ops.cs_rank(_g(df,"returns", lambda s: ops.rolling_sum(s,250))))
        return Factor.as_cs_series(df, val)

@register
class Alpha044(Factor):
    name = "Alpha044"
    requires = ["high","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        val = - _g(df,"high", lambda s: ops.rolling_corr(s, ops.cs_rank(df.loc[s.index,"volume"]), 5))
        return Factor.as_cs_series(df, val)

@register
class Alpha047(Factor):
    name = "Alpha047"
    requires = ["close","high","vwap","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        adv20 = _g(df,"volume", lambda s: ops.adv(s,20))
        part1 = (ops.cs_rank(1/df["close"]) * df["volume"]) / adv20
        part2 = (df["high"] * ops.cs_rank(df["high"]-df["close"])) / (_g(df,"high", lambda s: ops.rolling_sum(s,5))/5)
        val = part1 * part2 - ops.cs_rank(df["vwap"] - _g(df,"vwap", ops.delay,5))
        return Factor.as_cs_series(df, val)

@register
class Alpha054(Factor):
    name = "Alpha054"
    requires = ["low","close","open","high"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        val = - ((df["low"] - df["close"]) * (df["open"]**5)) / ((df["low"] - df["high"]) * (df["close"]**5))
        return Factor.as_cs_series(df, val)

@register
class Alpha061(Factor):
    name = "Alpha061"
    requires = ["vwap","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        adv180 = _g(df,"volume", lambda s: ops.adv(s,180))
        a = ops.cs_rank(df["vwap"] - _g(df,"vwap", lambda s: ops.rolling_min(s, int(16.1219))))
        b = ops.cs_rank(_g(df, None, lambda *_: ops.rolling_corr(df["vwap"], adv180, int(17.9282))))
        val = (a < b).astype(float)
        return Factor.as_cs_series(df, val)

@register
class Alpha064(Factor):
    name = "Alpha064"
    requires = ["open","low","vwap","close"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        a = ops.cs_rank(_g(df, None, lambda *_: ops.rolling_corr((_g(df,"open", lambda s: 0.178404*s) + (df["low"]*(1-0.178404))), _g(df,"volume", lambda s: ops.adv(s,120)), int(16.6208))))
        b = ops.cs_rank(_g(df, None, lambda *_: ops.delta((((df["high"]+df["low"])/2)*0.178404 + df["vwap"]*(1-0.178404)), int(3.69741))))
        val = (a < b).astype(float) * -1
        return Factor.as_cs_series(df, val)

@register
class Alpha065(Factor):
    name = "Alpha065"
    requires = ["open","vwap","low"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        a = ops.cs_rank(_g(df, None, lambda *_: ops.rolling_corr(0.00817205*df["open"] + (1-0.00817205)*df["vwap"], _g(df,"volume", lambda s: ops.adv(s,60)), int(6.40374))))
        b = ops.cs_rank(df["open"] - _g(df,"open", lambda s: ops.rolling_min(s, int(13.635))))
        val = (a < b).astype(float) * -1
        return Factor.as_cs_series(df, val)

@register
class Alpha071(Factor):
    name = "Alpha071"
    requires = ["close","volume","low","open","vwap"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        a = _g(df, None, lambda *_: ops.ts_rank(ops.decay_linear(_g(df,"close", lambda s: ops.ts_rank(s, int(3.43976))), int(4.20501)), int(15.6948)))
        b = _g(df, None, lambda *_: ops.ts_rank(ops.decay_linear(ops.cs_rank(((df["low"]+df["open"])-(df["vwap"]+df["vwap"]))**2), int(16.4662)), int(4.4388)))
        val = np.maximum(a, b)
        return Factor.as_cs_series(df, val)

@register
class Alpha083(Factor):
    name = "Alpha083"
    requires = ["high","low","close","vwap","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        num = ops.cs_rank(ops.delay(((df["high"]-df["low"]) / (_g(df,"close", lambda s: ops.rolling_sum(s,5))/5)), 2)) * ops.cs_rank(ops.cs_rank(df["volume"]))
        den = ((df["high"]-df["low"]) / (_g(df,"close", lambda s: ops.rolling_sum(s,5))/5)) / (df["vwap"] - df["close"]).replace(0,np.nan)
        val = num / den.replace(0,np.nan)
        return Factor.as_cs_series(df, val)

@register
class Alpha084(Factor):
    name = "Alpha084"
    requires = ["vwap","close"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        val = np.sign(_g(df,"close", ops.delta, int(4.96796))) * _g(df,"vwap", lambda s: ops.ts_rank(s - _g(df,"vwap", ops.rolling_max, int(15.3217)), int(20.7127)))
        return Factor.as_cs_series(df, val)

@register
class Alpha085(Factor):
    name = "Alpha085"
    requires = ["high","close","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        a = _g(df, None, lambda *_: ops.rolling_corr(0.876703*df["high"] + (1-0.876703)*df["close"], _g(df,"volume", lambda s: ops.adv(s,30)), int(9.61331)))
        b = _g(df, None, lambda *_: ops.rolling_corr(_g(df,"close", lambda s: ops.ts_rank((df["high"]+df["low"])/2, int(3.70596))), _g(df,"volume", lambda s: ops.ts_rank(df["volume"], int(10.1595))), int(7.11408)))
        val = ops.cs_rank(a) ** ops.cs_rank(b)
        return Factor.as_cs_series(df, val)

@register
class Alpha086(Factor):
    name = "Alpha086"
    requires = ["close","open","vwap","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        a = _g(df,"close", lambda s: ops.ts_rank(ops.rolling_corr(s, _g(df,"volume", lambda s2: ops.adv(s2,20)) .groupby(level=0, group_keys=False) if False else _g(df,"volume", lambda s2: ops.adv(s2,20)), int(6.00049)), int(20.4195)))
        b = ops.cs_rank((df["open"] + df["close"]) - (df["vwap"] + df["open"]))
        val = (a < b).astype(float) * -1
        return Factor.as_cs_series(df, val)

@register
class Alpha094(Factor):
    name = "Alpha094"
    requires = ["vwap","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        adv60 = _g(df,"volume", lambda s: ops.adv(s,60))
        a = ops.cs_rank(df["vwap"] - _g(df,"vwap", lambda s: ops.rolling_min(s, int(11.5783))))
        b = _g(df, None, lambda *_: ops.ts_rank(ops.rolling_corr(_g(df,"vwap", lambda s: ops.ts_rank(s, int(19.6462))),
                                                                 _g(df,"volume", lambda s: ops.ts_rank(adv60, int(4.02992))), int(18.0926)), int(2.70756)))
        val = (a ** b) * -1
        return Factor.as_cs_series(df, val)

@register
class Alpha095(Factor):
    name = "Alpha095"
    requires = ["open","high","low","close","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        a = ops.cs_rank(_g(df, None, lambda *_: ops.rolling_corr(_g(df,"close", lambda s: ops.rolling_sum((df["high"]+df["low"])/2, int(19.1351))),
                                                                 _g(df,"volume", lambda s: ops.adv(s,40)), int(12.8742))) ** 5)
        b = _g(df,"open", lambda s: ops.ts_rank(s - _g(df,"open", lambda s2: ops.rolling_min(s2, int(12.4105))), 1))
        val = (ops.cs_rank(df["open"] - _g(df,"open", lambda s: ops.rolling_min(s, int(12.4105)))) < a).astype(float)
        return Factor.as_cs_series(df, val)

@register
class Alpha096(Factor):
    name = "Alpha096"
    requires = ["vwap","volume","close"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        a = _g(df, None, lambda *_: ops.ts_rank(ops.decay_linear(ops.rolling_corr(ops.cs_rank(df["vwap"]), ops.cs_rank(df["volume"]), int(3.83878)), int(4.16783)), int(8.38151)))
        b = _g(df, None, lambda *_: ops.ts_rank(ops.decay_linear(ops.ts_rank(_g(df,"close", lambda s: ops.rolling_corr(ops.cs_rank(s), _g(df,"volume", lambda s2: ops.adv(s2,60)), int(4.13242))), int(7.45404)), int(14.0365)), int(13.4143)))
        val = - np.maximum(a, b)
        return Factor.as_cs_series(df, val)

@register
class Alpha098(Factor):
    name = "Alpha098"
    requires = ["vwap","volume","open"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        adv5 = _g(df,"volume", lambda s: ops.adv(s,5))
        a = ops.cs_rank(_g(df, None, lambda *_: ops.rolling_corr(df["vwap"], _g(df,"volume", lambda s: ops.rolling_sum(adv5, int(26.4719))), int(4.58418))))
        b = _g(df, None, lambda *_: ops.ts_rank(ops.ts_rank(ops.argmin(ops.rolling_corr(ops.cs_rank(df["open"]), _g(df,"volume", lambda s: ops.adv(s,15)), int(20.8187))), int(6.95668)), int(8.07206))) if hasattr(np, "argmin") else a*0
        val = a - b
        return Factor.as_cs_series(df, val)

@register
class Alpha099(Factor):
    name = "Alpha099"
    requires = ["high","low","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        a = _g(df, None, lambda *_: ops.rolling_corr(_g(df,"close", lambda s: ops.rolling_sum((df["high"]+df["low"])/2, int(19.8975))), _g(df,"volume", lambda s: ops.adv(s,60)), int(8.8136)))
        b = _g(df, None, lambda *_: ops.rolling_corr(df["low"], df["volume"], int(6.28259)))
        val = (ops.cs_rank(a) < ops.cs_rank(b)).astype(float) * -1
        return Factor.as_cs_series(df, val)
