# Alpha101 Factory — 101 Alpha 因子核心公式

基于 WorldQuant [Alpha101 论文](https://arxiv.org/abs/1601.00991) 实现的 61/101 个量化因子。

---

## 因子系统

### 因子基类

```python
class Factor(ABC):
    name: str           # 因子名称
    requires: List[str]  # 所需列 (close, volume, vwap, etc.)

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """输入面板数据 (datetime × symbol)，返回 MultiIndex Series"""
```

### 注册机制

```python
from alpha101_factory.factors.registry import register

@register
class AlphaXXX(Factor):
    name = "AlphaXXX"
    requires = ["close", "volume"]

    def compute(self, df):
        ...
        return Factor.as_cs_series(df, values)
```

使用 `@register` 装饰器后，因子自动被 `pkgutil` 发现，无需额外配置。

### 算子体系 (utils/ops.py)

| 算子 | 签名 | 说明 | 加速 |
|------|------|------|------|
| `rolling_sum(s, n)` | → Series | 滚动求和 | bottleneck |
| `rolling_min(s, n)` | → Series | 滚动最小值 | bottleneck |
| `rolling_max(s, n)` | → Series | 滚动最大值 | bottleneck |
| `rolling_std(s, n)` | → Series | 滚动标准差 (ddof=0) | bottleneck |
| `rolling_cov(s1, s2, n)` | → Series | 滚动协方差 (ddof=0) | pandas |
| `rolling_corr(s1, s2, n)` | → Series | 滚动相关系数 | pandas |
| `ts_rank(s, n)` | → Series | 时间序列百分位排名 | numba |
| `decay_linear(s, n)` | → Series | 线性衰减加权平均 | numba |
| `delay(s, n=1)` | → Series | 滞后 n 期 | pandas |
| `delta(s, n=1)` | → Series | 差分: s[t] - s[t-n] | pandas |
| `cs_rank(s)` | → Series | 截面百分位排名 (level=0) | pandas |
| `cs_zscore(s)` | → Series | 截面 z-score | pandas |
| `adv(volume, n)` | → Series | 平均日成交量 | pandas |

### 辅助函数 (alphas_complete.py)

| 函数 | 说明 |
|------|------|
| `_g(df, col, fn, *args)` | 按股票分组应用函数，返回与 df 对齐的 MultiIndex |
| `_cs(s)` | 截面排名：`s.groupby(level=0).rank(pct=True)` |
| `_mi(df)` | 快速设置 MultiIndex：`df.set_index(["datetime", "symbol"])` |
| `_ts_rank_mi(s, n)` | 时间序列排名：`s.groupby(level=1).transform(ts_rank)` |

---

## 因子分类目录

### 量价相关 (Price-Volume Correlation)
滚动相关系数类因子，捕捉价格与成交量的联动关系。

| 因子 | 核心公式 | requires |
|------|----------|----------|
| **Alpha003** | `-corr(rank(open), rank(volume), 10)` | open, volume |
| **Alpha006** | `-corr(open, volume, 10)` | open, volume |
| **Alpha013** | `-rank(cov(rank(close), rank(volume), 5))` | close, volume |
| **Alpha016** | `-rank(cov(rank(high), rank(volume), 5))` | high, volume |
| **Alpha040** | `-rank(std(high, 10)) × corr(high, volume, 10)` | high, volume |
| **Alpha042** | `-rank(std(high, 10) × corr(high, volume, 10))` | vwap, close |
| **Alpha055** | `-corr(rank((close - min(low, 12)) / (max(high, 12) - min(low, 12))), rank(volume), 6)` | close, high, low, volume |
| **Alpha060** | 同 Alpha055（去 rank 版本） | high, low, close, volume |
| **Alpha075** | `-corr(volume, vwap, 4)` | volume, vwap |

### 动量/反转 (Momentum/Reversal)
捕捉价格趋势与反转信号。

| 因子 | 核心公式 | requires |
|------|----------|----------|
| **Alpha002** | `-corr(rank(Δlog(volume), 2), rank((close-open)/open), 6)` | close, open, volume |
| **Alpha009** | `(0 < ts_min(Δclose, 5))×(-1) + (0 < ts_max(Δclose, 5))×1` | close |
| **Alpha010** | `rank((0 < ts_min(Δclose, 4))×(-1)) + rank((0 < ts_max(Δclose, 4))×1)` | close |
| **Alpha012** | `-sign(volume × Δclose)` | close, volume |
| **Alpha024** | `-rank(Δ(sum(close, 5), 5))` | close |
| **Alpha030** | `-((1 - rank(ret)×rank(vol)) × rank(ret))` | close, volume |
| **Alpha046** | `-rank(Δ((close×0.375 + open×0.625), 1))` | close |
| **Alpha048** | `corr(Δclose, Δdelay(close, 1), 250) × Δclose / close` | close |
| **Alpha052** | `-rank(Δ(((close-low) - (high-close)) / (close-low), 9))` | low, returns, volume |
| **Alpha053** | `-Δ(((close-low) - (high-close)) / (high-low), 9)` | close, low, high |
| **Alpha066** | `((close - delay(close, 6)) / delay(close, 6)) × volume` | close, volume |
| **Alpha067** | `-corr(Δclose, delay(close, 1), 250) × (close / delay(close, 1))` | close |
| **Alpha069** | `rank(Δclose/delay(close,1)) × rank(volume/adv20) × rank(high-low)` | high, low, close, volume |
| **Alpha070** | `-rank(Δ((close-low)/(high-low), 1))` | high, low, close |

### 波动率 (Volatility)
捕捉价格波动幅度的变化。

| 因子 | 核心公式 | requires |
|------|----------|----------|
| **Alpha001** | `-ts_rank(cs_rank(log(volume)), 5) × cs_rank((close-open)/open)` | returns, close |
| **Alpha004** | `-ts_rank(cs_rank(low), 9)` | low |
| **Alpha018** | `-rank(std(abs(close-open), 5) + (close-open) + corr(close, open, 10))` | close, open |
| **Alpha023** | `-rank((sum(high, 20)/20) × high)` | high, close |
| **Alpha031** | 多层层叠 rank + decay_linear | close, volume, low |
| **Alpha035** | `cs_rank(vol/sum(vol,15)) × cs_rank((high-low)/(high+low)) × cs_rank(ret)` | volume, close, high, low |
| **Alpha049** | `-rank((sum(high,15) - sum(low,15)) / 15)` | close |
| **Alpha051** | `-rank((sum(high,20) - sum(low,20)) / 20)` | close |
| **Alpha084** | `-(vwap - ts_min(vwap, 14))³` | vwap, close |

### 量价比 (Volume-Price Valuation)
成交量与价格的比率关系。

| 因子 | 核心公式 | requires |
|------|----------|----------|
| **Alpha007** | `-rank(|Δclose| × (volume / adv20))` | close, volume |
| **Alpha017** | 多层 rank 乘积 | high, low, close, volume |
| **Alpha033** | `((-((min(low,5) - delay(min(low,5),5)) / min(low,5))) × sum(vol,10)) / sum(vol,5)` | open, close |
| **Alpha034** | `(rank(ret) × rank(vol) / rank(close-open)) × rank(ret)` | returns, close |
| **Alpha041** | `√(high×low) - vwap` | high, low, vwap |
| **Alpha043** | `ts_rank(volume/adv20, 20) × ts_rank(-Δclose, 8)` | volume, close |
| **Alpha047** | `(rank(1/close) × volume) / adv20` | close, high, vwap, volume |
| **Alpha054** | `-((low-close) × open⁵) / ((low-high) × close⁵)` | low, close, open, high |
| **Alpha061** | `rank(vwap - min(vwap,16)) < rank(corr(vwap, adv180, 18))` | vwap, volume |
| **Alpha085** | `rank(volume/adv20) × rank((high-low)/close)` | high, close, volume |
| **Alpha101** | `(close - open) / ((high - low) + 0.001)` | open, high, low, close |

### 截面排名 (Cross-Sectional Rank)
排名比较类因子。

| 因子 | 核心公式 | requires |
|------|----------|----------|
| **Alpha008** | `-rank(sum(open,5)×sum(ret,5) - delay(..., 10))` | open, close, volume |
| **Alpha011** | `(rank(ts_rank(vwap-close, 3)) - rank(ts_rank(close-vwap, 3))) × rank(vol/adv20)` | vwap, close, volume |
| **Alpha015** | `-sum(rank(corr(rank(high), rank(volume), 3)), 3)` | high, volume |
| **Alpha022** | `-(Δ(corr(high, volume, 5), 5) × rank(std(close, 20)))` | high, volume, close |
| **Alpha025** | `-rank(((close-open)/delay(close,7) × corr(close,vol,250) × rank(returns)))` | returns, vwap, high, close, volume |
| **Alpha026** | `-ts_rank(volume, 5)` | volume, high |
| **Alpha027** | `(0.5 < rank(mean(corr(rank(vol), rank(vwap), 6), 6))) × (-1)` | volume |
| **Alpha028** | `scale(corr(adv20, low, 5) + (high+low)/2 - close)` | high, low, close, volume |
| **Alpha032** | `-sum(rank(corr(rank(vol), rank(vwap), 5)), 5)` | close, vwap |
| **Alpha036** | `-rank(ts_rank(corr((high+low)/2, adv20, 10), 15))` | close, open, volume, vwap |
| **Alpha037** | `-rank(ts_rank(delay(close,1)/close, 10)) × rank(corr(open, vol, 10))` | open, close |
| **Alpha038** | `-rank(ts_rank(close, 10)) × rank(close/open)` | close, open |
| **Alpha044** | `-rank(ts_rank(corr(high, rank(vol), 5), 5)) × rank(returns)` | high, volume |
| **Alpha045** | `-rank(mean(delay(close,5), 20)) × corr(close, volume, 2)` | close, volume |
| **Alpha050** | `-ts_rank(rank(corr(rank(vol), rank(vwap), 5)), 5)` | volume, vwap |
| **Alpha056** | `-rank(sum(ret,10) / sum(sum(ret,2), 3)) × rank(returns)` | close |
| **Alpha057** | `-(close - vwap) / decay_linear(rank(ts_rank(close, 30)), 2)` | close, vwap |
| **Alpha068** | `ts_rank(corr(rank(high), rank(adv15), 9), 14)` | high, volume |
| **Alpha083** | 复杂 HL 比率交叉项 | high, low, close, vwap, volume |
| **Alpha094** | `rank((vwap-min(vwap,11)) / ts_rank(corr(vwap,adv150,6),15)) × rank(vol/adv20)` | vwap, volume |
| **Alpha095** | `rank(open - ts_min(open, 12)) < rank(corr((high+low)/2, sum(adv40,40), 10))` | open, high, low, close, volume |
| **Alpha099** | `-rank(corr((high+low)/2, sum(adv60, 40), 9))` | high, low, volume, close |

### 趋势/衰减 (Trend/Decay)
使用 decay_linear 加权捕捉趋势。

| 因子 | 核心公式 | requires |
|------|----------|----------|
| **Alpha039** | `-rank(decay_linear(Δclose,8) / decay_linear(corr((close+vwap)/2, adv20, 9), 10))` | close, volume, returns |
| **Alpha058** | `-ts_rank(decay_linear(corr(vwap, vol, 4), 8), 6)` | close, vwap, volume |
| **Alpha059** | `-ts_rank(decay_linear(corr(vwap, vol, 4), 16), 8)` | close, vwap, volume |
| **Alpha087** | `rank(decay_linear(corr(high×0.877+close×0.123, adv30, 10), 13)) - rank(decay_linear(corr(rank(vwap), rank(vol), 4), 12))` | high, close, vwap, volume |
| **Alpha088** | `(ts_rank(decay_linear(corr(close×0.497+vwap×0.503, adv20, 5), 7), 5) < rank(decay_linear(...))) × (-1)` | close, vwap, volume |
| **Alpha092** | 同 Alpha088 结构（high×0.518+low×0.482） | high, low, vwap, volume |
| **Alpha093** | `ts_rank(decay_linear(corr(rank(vwap), rank(vol), 4), 12), 5) - ts_rank(decay_linear(corr(high×0.518+low×0.482, adv30, 5), 7), 5)` | high, low, vwap, volume |
| **Alpha096** | `-max(ts_rank(decay_linear(..), 8), ts_rank(decay_linear(ts_rank(..), 17), 9))` | vwap, volume, close |
| **Alpha098** | `rank(decay_linear(corr(vwap, sum(adv5,26), 5), 8)) - rank(decay_linear(ts_rank(corr(rank(close), rank(vol), 4), 16), 4))` | vwap, volume, open |

### 比较/条件 (Comparison/Conditional)
布尔比较类因子。

| 因子 | 核心公式 | requires |
|------|----------|----------|
| **Alpha019** | `-sign((close - delay(close,7)) × corr(close, delay(close,7), 250))` | close, returns |
| **Alpha062** | `-corr(vwap, rank(adv5), 20)` | vwap, volume |
| **Alpha063** | `-corr(rank(rank(close)/rank(vol)), rank(vwap/2.51 + vwap×0.272), 10)` | close, volume, vwap |
| **Alpha064** | `(rank(corr(sum(open×0.178+low×0.822,12), sum(adv120,12), 16)) < rank(corr(rank(vwap), rank(vol), 4))) × (-1)` | open, low, high, vwap, volume |
| **Alpha065** | `rank(corr(open×0.008+vwap×0.992, sum(adv60,60), 9)) < rank((open-ts_min(open,13)) / sum(adv60,60))` | open, vwap, volume |
| **Alpha071** | `max(rank(decay_linear(Δvwap, 17)), ts_rank(Δ(close×0.497+vwap×0.503, 2), 5)) × (-1)` | close, low, open, vwap |
| **Alpha072** | `rank(decay_linear(corr((high+low)/2, adv40, 9), 14)) - rank(decay_linear(corr(rank(vwap), rank(vol), 4), 12))` | high, low, vwap, volume |
| **Alpha073** | 同 Alpha071 | close, vwap |
| **Alpha074** | `(rank(corr(close, sum(adv30,37), 15)) < rank(corr(rank(high×0.026+vwap×0.974), rank(vol), 11))) × (-1)` | high, close, vwap, volume |
| **Alpha076** | `max(rank(decay_linear(Δvwap, 12)), ts_rank(decay_linear(close×0.383+vwap×0.617, 19), 8)) × (-1)` | close, vwap |
| **Alpha077** | `min(rank(decay_linear((high+low)/2+(high-low)/2, 20)), ts_rank(decay_linear(corr((high+low)/2, adv40, 3), 6), 4)) × (-1)` | high, low, volume |
| **Alpha078** | `rank(corr(sum(low×0.352+vwap×0.648, 20), sum(adv40,20), 7)) × rank(rank(vol/adv20))` | low, vwap, volume |
| **Alpha079** | `rank(Δ(close×0.607+open×0.393, 1)) < rank(corr(rank(vwap), rank(adv150), 10))` | close, open, vwap, volume |
| **Alpha080** | `rank(sign(Δ(open×0.868+high×0.132, 4))) × rank(corr(high×0.518+low×0.482, sum(adv30,30), 14))` | high, low, open, volume |
| **Alpha081** | `(rank(log(product(rank(corr(vwap,sum(adv10,50),8))⁴, 15))) < rank(corr(rank(vwap),rank(vol),15))) × (-1)` | vwap, volume |
| **Alpha082** | `min(rank(decay_linear(Δopen, 15)), ts_rank(decay_linear(corr(vol, low, 6), 17), 7)) × (-1)` | open, low, volume |
| **Alpha089** | `ts_rank(decay_linear(corr(low, adv10, 6), 9), 4) - ts_rank(decay_linear(ts_rank(corr(vwap, adv20, 5), 18), 16), 9)` | low, vwap, volume |
| **Alpha090** | `-(rank(close - min(close, 5))^rank(decay_linear(vwap × rank(high×0.518+low×0.482), 14)))` | high, low, close, vwap |
| **Alpha091** | `-(rank(close - min(close, 5)) × rank(decay_linear(vwap, 14)))` | close, vwap |
| **Alpha097** | `(rank(decay_linear(Δ(low×0.721+vwap×0.279, 3), 7)) < ts_rank(decay_linear(ts_rank(corr(close, adv20, 5), 11), 20), 8)) × (-1)` | low, vwap, close, volume |
| **Alpha100** | `-1.5 × scale(vwap) × scale(corr(high×0.518+low×0.482, adv30, 15))` | high, low, vwap, volume |

### 简单形态 (Simple Patterns)

| 因子 | 核心公式 | requires |
|------|----------|----------|
| **Alpha005** | `rank(-(open - mean(vwap, 10))) × rank(-abs(close - vwap))` | open, vwap, close |
| **Alpha014** | `rank((open - delay(close,1)) × corr(open, volume, 20))` | open, volume, returns |
| **Alpha020** | `-rank((open - delay(high,1)) × corr(open, volume, 10))` | open, high, low, close |
| **Alpha021** | `((-mean(ret,20) - open) × close) × volume` | close, volume |
| **Alpha029** | 多层嵌套 rank+scale+log 复合 | close |
| **Alpha086** | `delay(corr(close, volume, 10), 5) × rank(mean(close,20) × volume) × rank(vol/adv60)` | close, open, vwap, volume |

---

## 已实现因子清单 (61/101)

| 状态 | 因子编号 |
|------|----------|
| ✅ 已实现 | 001, 002, 003, 004, 005, 006, 007, 008, 009, 010, 011, 012, 013, 014, 015, 016, 017, 018, 019, 020, 021, 022, 023, 024, 025, 026, 027, 028, 029, 030, 031, 032, 033, 034, 035, 036, 037, 038, 039, 040, 041, 042, 043, 044, 045, 046, 047, 048, 049, 050, 051, 052, 053, 054, 055, 056, 057, 058, 059, 060, 061, 062, 063, 064, 065, 066, 067, 068, 069, 070, 071, 072, 073, 074, 075, 076, 077, 078, 079, 080, 081, 082, 083, 084, 085, 086, 087, 088, 089, 090, 091, 092, 093, 094, 095, 096, 097, 098, 099, 100, 101 |
| ❌ 待实现 | — |

---

## 关键 Gotchas（公式实现陷阱）

1. **截面 vs 时间序列：** `groupby(level=0)` = 按日期（截面），`groupby(level=1)` = 按股票（时间序列）。截面操作（cs_rank, cs_zscore）必须使用 level=0。

2. **MultiIndex 要求：** `cs_rank()` 要求输入 Series 具有 `(datetime, symbol)` 的 MultiIndex。传入普通 Series 会抛出 ValueError。

3. **索引保持：** `pd.Series(val)` 会丢失索引 → 始终使用 `pd.Series(val, index=df.index)`。

4. **ddof 一致性：** `rolling_cov` 使用 `ddof=0`（与 `rolling_std(ddof=0)` 一致），而非 pandas 默认的 `ddof=1`。

5. **requires 完整性：** 因子 `requires` 必须列出 `compute()` 中使用的**每一个**列。缺失列会导致运行时错误（如 Alpha031 缺失 "low"，Alpha099 缺失 "close"）。

6. **_g 函数：** 使用 `groupby.transform` 保留原始索引，再通过 `.values` 对齐到 MultiIndex。直接 concat 会导致 RangeIndex 无法匹配。

7. **Alpha053 分母：** 公式为 `(high - low)`，而非早期实现中的 `(close - low)`。

8. **Alpha029 复杂嵌套：** 多层层叠 rank+scale+log 操作，需注意 scale 的分母为 `sum(abs(x))` 而非 `std(x)`。

---

## 添加新因子

```python
from alpha101_factory.factors.base import Factor
from alpha101_factory.factors.registry import register

@register
class AlphaXXX(Factor):
    """公式描述"""
    name = "AlphaXXX"
    requires = ["close", "volume"]  # 列出 ALL 使用的列

    def compute(self, df: pd.DataFrame) -> pd.Series:
        m = df.set_index(["datetime", "symbol"])
        val = ...  # 你的计算逻辑
        return Factor.as_cs_series(df, val)
```

注册后自动可用：
```bash
python -m alpha101_factory.cli factor --factors AlphaXXX
```
