# Alpha 因子完整文档

Alpha101 因子工厂的权威参考指南，涵盖全部 60 个 Alpha 因子的公式、依赖字段、金融直觉与实现细节。

## 概述

本项目的 Alpha 因子体系源自 Kakushadzi 2016 年的经典论文《101 Formulaic Alphas》，并结合 A 股市场特性进行了适配。所有因子均通过统一的计算框架实现，支持横截面 (Cross-Sectional) 和时间序列 (Time-Series) 两种视角的分析。

### 核心概念

- **requires**: 因子计算所需的原始或中间数据字段
- **ts_rank**: 时间序列排名，衡量当前值在过去 N 期的相对位置
- **cs_rank**: 截面排名，衡量当天所有股票中的相对排序
- **rolling_corr/cov**: 滚动窗口相关系数/协方差
- **delta/delay**: 差分与滞后变换

### 因子分类

| 类别 | 数量 | 描述 |
|------|------|------|
| 量价相关 | 30 | 基于价格、成交量及其相互关系的因子 |
| 价格动量 | 18 | 捕捉价格趋势与动量效应的因子 |
| 波动率 | 10 | 度量价格波动与风险特征的因子 |
| 成交量异常 | 4 | 识别成交量异常波动的因子 |

---

## 因子列表总览

| 编号 | 名称 | 依赖字段 | 简要描述 |
|------|------|----------|----------|
| Alpha001 | 波动率排名 | returns, close | 负收益时的波动率排名与截面标准化 |
| Alpha003 | 量价相关性 | open, volume | 开盘价与成交量的滚动相关性的反向排名 |
| Alpha004 | 最低价排名 | low | 最低价的跨期截面排名倒数 |
| Alpha005 | 开盘偏离度 | open, vwap, close | 开盘价偏离与收盘价-VWAP 偏离的乘积 |
| Alpha006 | 量价滚动相关 | open, volume | 开盘价与成交量的滚动相关性的反向值 |
| Alpha009 | 价格变化符号 | close | 基于价格变化的条件符号变换 |
| Alpha010 | 价格变化截面排名 | close | Alpha009 的截面排名版本 |
| Alpha011 | VWAP-价格偏离 | vwap, close, volume | VWAP 与收盘价偏离及成交量排名的组合 |
| Alpha012 | 量价方向 | close, volume | 成交量变化符号与价格变化的反向乘积 |
| Alpha013 | 价量协方差 | close, volume | 收盘价与成交量的滚动协方差的反向排名 |
| Alpha014 | 收益率动量 | open, volume, returns | 收益率变化与量价相关的组合 |
| Alpha016 | 高价量协方差 | high, volume | 最高价与成交量的滚动协方差的反向排名 |
| Alpha018 | 振幅波动 | close, open | 实体幅度、波动率与相关性的组合 |
| Alpha019 | 长期价格动量 | close, returns | 7 日价格变化与年化收益率排名的组合 |
| Alpha020 | 高低开关系 | open, high, low, close | 开盘价与前一日高低收的关系乘积 |
| Alpha021 | 均值回归信号 | close, volume | 短期与中期均值的偏离及成交量信号 |
| Alpha022 | 量价相关变化 | high, volume, close | 高量相关的变化与波动率排名的乘积 |
| Alpha023 | 高位动量 | high, close | 20 日均线上方的最高价变化 |
| Alpha024 | 长期趋势强度 | close | 100 日移动平均的变化率与价格位置 |
| Alpha025 | 反转量价 | returns, vwap, high, close, volume | 负收益率与量价特征的截面排名 |
| Alpha026 | 量价排名相关 | volume, high | 成交量与最高价排名的滚动相关最大值 |
| Alpha030 | 成交量信号 | close, volume | 价格方向信号与成交量比例的乘积 |
| Alpha031 | 多源信号组合 | close, volume | 衰减线性、价格变化与量价相关的组合 |
| Alpha032 | VWAP 偏离 | close, vwap | 短期均值偏离与 VWAP-延迟收盘价相关 |
| Alpha033 | 开盘收盘比 | open, close | 开盘价与收盘价比率的截面排名 |
| Alpha034 | 波动率比率 | returns, close | 短期与长期波动率比率及价格变化 |
| Alpha035 | 多维排名乘积 | volume, close, high, low, returns | 成交量、价差区间与收益率的排名乘积 |
| Alpha036 | 五因子组合 | close, open, volume, vwap, returns | 五个加权截面排名因子的加权和 |
| Alpha037 | 开盘差相关 | open, close | 延迟开盘差与收盘价的长期相关 |
| Alpha038 | 价格位置比率 | close, open | 收盘价排名与开收盘比的乘积 |
| Alpha039 | 量价动量 | close, volume, returns | 价格变化、成交量衰减与收益率排名 |
| Alpha040 | 高量相关 | high, volume | 最高价波动率排名与量价相关的乘积 |
| Alpha041 | VWAP 偏离 | high, low, vwap | 高低几何平均与 VWAP 的差值 |
| Alpha042 | VWAP-收盘价 | vwap, close | VWAP 与收盘价差的排名比率 |
| Alpha043 | 量价排名 | volume, close, returns, vwap | 成交量比例与价格变化的排名乘积 |
| Alpha044 | 量价相关 | high, volume | 最高价与成交量排名的滚动相关 |
| Alpha045 | 延迟量价 | close, volume | 延迟收盘价、量价相关的复杂组合 |
| Alpha046 | 加速度阈值 | close | 价格加速度超过阈值的分段函数 |
| Alpha047 | 多空量价 | close, high, vwap, volume | 多重量价关系的组合 |
| Alpha049 | 加速度阈值 | close | 简化版加速度阈值因子 |
| Alpha050 | VWAP 量相关 | volume, vwap | 成交量与 VWAP 相关的最大排名 |
| Alpha051 | 加速度阈值 | close | 另一版本加速度阈值因子 |
| Alpha052 | 低波动量价 | low, returns, volume | 最低价变化、收益率差与成交量排名 |
| Alpha053 | 价格位置 | close, low, high | 收盘价在高低区间的位置变化 |
| Alpha054 | OHLC 比率 | low, close, open, high | 基于 OHLC 四价的复杂比率 |
| Alpha055 | 量价协方差 | close, high, low, volume | KDJ 指标与成交量的滚动相关 |
| Alpha060 | 量价加权 | high, low, close, volume | 价格位置与成交量的加权组合 |
| Alpha056 | VWAP 最小值 | vwap, volume | VWAP 相对最小值与长期 ADV 的相关 |
| Alpha064 | 加权相关 | open, low, vwap, close | 加权价格与 ADV 的相关比较 |
| Alpha065 | 开盘 VWAP 相关 | open, vwap, low | 开盘价与 VWAP 的加权相关比较 |
| Alpha071 | 双重排名 | close, volume, low, open, vwap | 价格排名与波动率的复合排名 |
| Alpha083 | 波动率比率 | high, low, close, vwap, volume | 价格区间比率与 VWAP 偏离的组合 |
| Alpha084 | VWAP 趋势 | vwap, close | 价格变化符号与 VWAP 排名的乘积 |
| Alpha085 | 双相关指数 | high, close, volume | 两个滚动相关系数的截面排名幂次 |
| Alpha086 | VWAP 相关排名 | close, open, vwap, volume | 量价相关排名与 VWAP 偏离的比较 |
| Alpha094 | VWAP 幂次 | vwap, volume | VWAP 偏离与量价相关的幂次组合 |
| Alpha095 | 多因子阈值 | open, high, low, close, volume | 均价相关与开盘排名的阈值比较 |
| Alpha096 | 双重衰减 | vwap, volume, close | 两个衰减线性相关的最大值反向 |
| Alpha098 | VWAP 开放相关 | vwap, volume, open | VWAP 与成交量相关减去开放排名 |
| Alpha099 | 双相关比较 | high, low, volume | 均价相关与低量相关的比较 |
| Alpha101 | 开盘收盘比率 | open, high, low, close | 收盘价相对于价格区间的偏移 |

---

## 详细因子文档

### 量价相关因子 (Price-Volume)

#### Alpha003 - 量价相关性

**依赖字段**: `open`, `volume`

**计算公式**:
$$\text{Alpha003} = -\text{RollingCorr}(\text{CSRank}(\text{open}), \text{CSRank}(\text{volume}), 10)$$

**金融直觉**: 该因子衡量开盘价与成交量之间的负相关关系。当开盘价上涨伴随成交量萎缩时，可能预示上涨动能不足。反向处理使得负相关产生正信号。

**代码示例**:
```python
@register
class Alpha003(Factor):
    name = "Alpha003"
    requires = ["open", "volume"]
    def compute(self, df):
        val = -self._g(df, "open", lambda s: ops.rolling_corr(ops.cs_rank(s), ops.cs_rank(df.loc[s.index, "volume"]), 10))
        return self.as_cs_series(df, val)
```

---

#### Alpha006 - 量价滚动相关

**依赖字段**: `open`, `volume`

**计算公式**:
$$\text{Alpha006} = -\text{RollingCorr}(\text{open}, \text{volume}, 10)$$

**金融直觉**: 直接衡量开盘价与成交量的滚动相关性，不做截面排名处理。捕捉量价配合的基本模式。

**代码示例**:
```python
@register
class Alpha006(Factor):
    name = "Alpha006"
    requires = ["open", "volume"]
    def compute(self, df):
        val = -self._g(df, "open", lambda s: ops.rolling_corr(s, df.loc[s.index, "volume"], 10))
        return self.as_cs_series(df, val)
```

---

#### Alpha013 - 价量协方差

**依赖字段**: `close`, `volume`

**计算公式**:
$$\text{Alpha013} = -\text{RollingCov}(\text{CSRank}(\text{close}), \text{CSRank}(\text{volume}), 5)$$

**金融直觉**: 收盘价与成交量的协方差反映两者联动性。负协方差表示价格上涨时成交量下降，可能预示趋势衰竭。

**代码示例**:
```python
@register
class Alpha013(Factor):
    name = "Alpha013"
    requires = ["close", "volume"]
    def compute(self, df):
        val = -self._g(df, "close", lambda s: ops.rolling_cov(ops.cs_rank(s), ops.cs_rank(df.loc[s.index, "volume"]), 5))
        return self.as_cs_series(df, val)
```

---

#### Alpha014 - 收益率动量

**依赖字段**: `open`, `volume`, `returns`

**计算公式**:
$$\text{Alpha014} = -\text{CSRank}(\Delta(\text{returns}, 3)) \times \text{RollingCorr}(\text{open}, \text{volume}, 10)$$

**金融直觉**: 结合收益率的动量变化与量价相关性。收益率加速变化与量价背离时产生信号。

**代码示例**:
```python
@register
class Alpha014(Factor):
    name = "Alpha014"
    requires = ["open", "volume", "returns"]
    def compute(self, df):
        returns_rank = -ops.cs_rank(self._g(df, "returns", ops.delta, 3))
        open_volume_corr = self._g(df, "open", lambda s: ops.rolling_corr(s, df.loc[s.index, "volume"], 10))
        val = returns_rank * open_volume_corr
        return self.as_cs_series(df, val)
```

---

#### Alpha016 - 高价量协方差

**依赖字段**: `high`, `volume`

**计算公式**:
$$\text{Alpha016} = -\text{RollingCov}(\text{CSRank}(\text{high}), \text{CSRank}(\text{volume}), 5)$$

**金融直觉**: 类似 Alpha013 但使用最高价。最高价突破时的成交量配合程度是重要的趋势确认信号。

**代码示例**:
```python
@register
class Alpha016(Factor):
    name = "Alpha016"
    requires = ["high", "volume"]
    def compute(self, df):
        val = -self._g(df, "high", lambda s: ops.rolling_cov(ops.cs_rank(s), ops.cs_rank(df.loc[s.index, "volume"]), 5))
        return self.as_cs_series(df, val)
```

---

#### Alpha022 - 量价相关变化

**依赖字段**: `high`, `volume`, `close`

**计算公式**:
$$\text{Alpha022} = -\Delta(\text{RollingCorr}(\text{high}, \text{volume}, 5), 5) \times \text{CSRank}(\text{RollingStd}(\text{close}, 20))$$

**金融直觉**: 量价相关性的变化率乘以价格波动率排名。相关性突然改变且波动率高时，往往预示趋势转折。

**代码示例**:
```python
@register
class Alpha022(Factor):
    name = "Alpha022"
    requires = ["high", "volume", "close"]
    def compute(self, df):
        corr = self._g(df, "high", lambda s: ops.rolling_corr(s, df.loc[s.index, "volume"], 5))
        corr_delta = self._g(df, None, lambda *_: ops.delta(corr, 5))
        close_std_rank = ops.cs_rank(self._g(df, "close", ops.rolling_std, 20))
        val = -corr_delta * close_std_rank
        return self.as_cs_series(df, val)
```

---

#### Alpha026 - 量价排名相关

**依赖字段**: `volume`, `high`

**计算公式**:
$$\text{Alpha026} = -\text{RollingMax}(\text{RollingCorr}(\text{TSRank}(\text{volume}, 5), \text{TSRank}(\text{high}, 5), 5), 3)$$

**金融直觉**: 成交量与最高价的排名相关性。高排名同时出现可能预示突破，反向处理捕捉假突破。

**代码示例**:
```python
@register
class Alpha026(Factor):
    name = "Alpha026"
    requires = ["volume", "high"]
    def compute(self, df):
        a = self._g(df, "volume", lambda s: ops.ts_rank(s, 5))
        b = self._g(df, "high", lambda s: ops.ts_rank(s, 5))
        val = -self._g(df, None, lambda *_: ops.rolling_max(ops.rolling_corr(a, b, 5), 3))
        return self.as_cs_series(df, val)
```

---

#### Alpha030 - 成交量信号

**依赖字段**: `close`, `volume`

**计算公式**:
$$\text{Alpha030} = \frac{(1 - \text{CSRank}(\sum \text{Sign}(\Delta^k(\text{delay}(\text{close}, 1), 1)))) \times \text{RollingSum}(\text{volume}, 5)}{\text{RollingSum}(\text{volume}, 20)}$$

其中 $k=1,2,3$。

**金融直觉**: 价格方向信号的复杂度与近期成交量占 20 日均量的比例。复杂的 price action 配合放量可能预示重要转折。

**代码示例**:
```python
@register
class Alpha030(Factor):
    name = "Alpha030"
    requires = ["close", "volume"]
    def compute(self, df):
        s5 = self._g(df, "volume", lambda s: ops.rolling_sum(s, 5))
        s20 = self._g(df, "volume", lambda s: ops.rolling_sum(s, 20))
        sig = (1.0 - self._cs_rank(df, (np.sign(ops.delta(self._g(df, "close", ops.delay, 1), 1)) + np.sign(ops.delta(self._g(df, "close", ops.delay, 2), 1)) + np.sign(ops.delta(self._g(df, "close", ops.delay, 3), 1)))))
        val = (sig * s5) / s20.replace(0, np.nan)
        return self.as_cs_series(df, val)
```

---

#### Alpha031 - 多源信号组合

**依赖字段**: `close`, `volume`

**计算公式**:
$$\text{Alpha031} = \text{CSRank}(\text{DecayLinear}(-\text{CSRank}(\Delta(\text{close}, 10)), 10)) + \text{CSRank}(-\Delta(\text{close}, 3)) + \text{Sign}(\text{CSRank}(\text{RollingCorr}(\text{ADV}_{20}, \text{low}, 12)))$$

**金融直觉**: 三个独立信号的叠加：10 日价格变化的衰减排名、3 日价格变化、以及成交量与低价的相关性符号。

**代码示例**:
```python
@register
class Alpha031(Factor):
    name = "Alpha031"
    requires = ["close", "volume"]
    def compute(self, df):
        a = ops.cs_rank(ops.decay_linear(-self._cs_rank(self._g(df, "close", lambda s: ops.delta(s, 10))), 10))
        b = ops.cs_rank(-self._g(df, "close", ops.delta, 3))
        adv20 = self._g(df, "volume", lambda s: ops.adv(s, 20))
        c = np.sign(ops.cs_rank(self._g(df, "volume", lambda s: ops.rolling_corr(adv20, df.loc[s.index, "low"] if "low" in df.columns else s, 12))))
        val = a + b + c
        return self.as_cs_series(df, val)
```

---

#### Alpha035 - 多维排名乘积

**依赖字段**: `volume`, `close`, `high`, `low`, `returns`

**计算公式**:
$$\text{Alpha035} = \text{TSRank}(\text{volume}, 32) \times (1 - \text{TSRank}((\text{close} + \text{high}) - \text{low}, 16)) \times (1 - \text{TSRank}(\text{returns}, 32))$$

**金融直觉**: 成交量排名、价格区间位置和收益率排名的乘积。三者同时处于极端位置时产生强信号。

**代码示例**:
```python
@register
class Alpha035(Factor):
    name = "Alpha035"
    requires = ["volume", "close", "high", "low", "returns"]
    def compute(self, df):
        a = self._g(df, "volume", lambda s: ops.ts_rank(s, 32))
        b = 1 - self._g(df, None, lambda *_: ops.ts_rank(((df["close"] + df["high"]) - df["low"]), 16))
        c = 1 - self._g(df, "returns", lambda s: ops.ts_rank(s, 32))
        val = a * b * c
        return self.as_cs_series(df, val)
```

---

#### Alpha036 - 五因子组合

**依赖字段**: `close`, `open`, `volume`, `vwap`, `returns`

**计算公式**:
$$\begin{aligned}
\text{Alpha036} &= 2.21 \times \text{CSRank}(\text{RollingCorr}(\text{close}-\text{open}, \text{delay}(\text{volume}, 1), 15)) \\
&+ 0.7 \times \text{CSRank}(\text{open}-\text{close}) \\
&+ 0.73 \times \text{CSRank}(\text{TSRank}(\text{delay}(-\text{returns}, 6), 5)) \\
&+ \text{CSRank}(|\text{RollingCorr}(\text{vwap}, \text{ADV}_{20}, 6)|) \\
&+ 0.6 \times \text{CSRank}((\frac{\text{RollingSum}(\text{close}, 200)}{200} - \text{open}) \times (\text{close}-\text{open}))
\end{aligned}$$

**金融直觉**: 五个加权因子的线性组合，涵盖量价相关、开盘收盘差、收益率滞后、VWAP 相关和长期均线偏离。

**代码示例**:
```python
@register
class Alpha036(Factor):
    name = "Alpha036"
    requires = ["close", "open", "volume", "vwap", "returns"]
    def compute(self, df):
        a = 2.21 * ops.cs_rank(self._g(df, None, lambda *_: ops.rolling_corr(df["close"] - df["open"], self._g(df, "volume", ops.delay, 1), 15)))
        b = 0.7 * ops.cs_rank(df["open"] - df["close"])
        c = 0.73 * ops.cs_rank(self._g(df, None, lambda *_: ops.ts_rank(ops.delay(-df["returns"], 6), 5)))
        d = ops.cs_rank(np.abs(self._g(df, None, lambda *_: ops.rolling_corr(df["vwap"], self._g(df, "volume", lambda s: ops.adv(s, 20)), 6))))
        e = 0.6 * ops.cs_rank((self._g(df, "close", lambda s: ops.rolling_sum(s, 200) / 200) - df["open"]) * (df["close"] - df["open"]))
        val = a + b + c + d + e
        return self.as_cs_series(df, val)
```

---

#### Alpha038 - 价格位置比率

**依赖字段**: `close`, `open`

**计算公式**:
$$\text{Alpha038} = -\text{TSRank}(\text{close}, 10) \times \text{CSRank}\left(\frac{\text{close}}{\text{open}}\right)$$

**金融直觉**: 近期价格位置与当日开收盘比的乘积。高位股票若高开低走（比率小）产生负面信号。

**代码示例**:
```python
@register
class Alpha038(Factor):
    name = "Alpha038"
    requires = ["close", "open"]
    def compute(self, df):
        close_ts_rank = self._g(df, "close", lambda s: ops.ts_rank(s, 10))
        close_open_ratio_rank = ops.cs_rank(df["close"] / df["open"])
        val = -close_ts_rank * close_open_ratio_rank
        return self.as_cs_series(df, val)
```

---

#### Alpha039 - 量价动量

**依赖字段**: `close`, `volume`, `returns`

**计算公式**:
$$\text{Alpha039} = -\text{CSRank}(\Delta(\text{close}, 7) \times (1 - \text{CSRank}(\text{DecayLinear}(\frac{\text{volume}}{\text{ADV}_{20}}, 9)))) \times (1 + \text{CSRank}(\text{RollingSum}(\text{returns}, 250)))$$

**金融直觉**: 7 日价格变化、成交量相对水平的衰减排名、以及年化收益率排名的组合。捕捉量价动量的多维度特征。

**代码示例**:
```python
@register
class Alpha039(Factor):
    name = "Alpha039"
    requires = ["close", "volume", "returns"]
    def compute(self, df):
        adv20 = self._g(df, "volume", lambda s: ops.adv(s, 20))
        part = -ops.cs_rank(self._g(df, "close", lambda s: ops.delta(s, 7)) * (1 - ops.cs_rank(ops.decay_linear(df["volume"] / adv20, 9))))
        val = part * (1 + ops.cs_rank(self._g(df, "returns", lambda s: ops.rolling_sum(s, 250))))
        return self.as_cs_series(df, val)
```

---

#### Alpha040 - 高量相关

**依赖字段**: `high`, `volume`

**计算公式**:
$$\text{Alpha040} = -\text{CSRank}(\text{RollingStd}(\text{high}, 10)) \times \text{RollingCorr}(\text{high}, \text{volume}, 10)$$

**金融直觉**: 最高价波动率排名与量价相关性的乘积。高波动且量价配合良好时产生信号。

**代码示例**:
```python
@register
class Alpha040(Factor):
    name = "Alpha040"
    requires = ["high", "volume"]
    def compute(self, df):
        val = (-ops.cs_rank(self._g(df, "high", ops.rolling_std, 10))) * self._g(df, "high", lambda s: ops.rolling_corr(s, df.loc[s.index, "volume"], 10))
        return self.as_cs_series(df, val)
```

---

#### Alpha043 - 量价排名

**依赖字段**: `volume`, `close`, `returns`, `vwap`

**计算公式**:
$$\text{Alpha043} = \text{TSRank}\left(\frac{\text{volume}}{\text{ADV}_{20}}, 20\right) \times \text{TSRank}(-\Delta(\text{close}, 7), 8)$$

**金融直觉**: 成交量相对水平的排名与价格变化排名的乘积。放量且价格下跌（负 delta）时产生信号。

**代码示例**:
```python
@register
class Alpha043(Factor):
    name = "Alpha043"
    requires = ["volume", "close", "returns", "vwap"]
    def compute(self, df):
        adv20 = self._g(df, "volume", lambda s: ops.adv(s, 20))
        val = self._g(df, None, lambda *_: ops.ts_rank(df["volume"] / adv20, 20)) * self._g(df, "close", lambda s: ops.ts_rank(-ops.delta(s, 7), 8))
        return self.as_cs_series(df, val)
```

---

#### Alpha044 - 量价相关

**依赖字段**: `high`, `volume`

**计算公式**:
$$\text{Alpha044} = -\text{RollingCorr}(\text{high}, \text{CSRank}(\text{volume}), 5)$$

**金融直觉**: 最高价与成交量排名的滚动相关。成交量排名领先于价格变动时产生信号。

**代码示例**:
```python
@register
class Alpha044(Factor):
    name = "Alpha044"
    requires = ["high", "volume"]
    def compute(self, df):
        val = -self._g(df, "high", lambda s: ops.rolling_corr(s, ops.cs_rank(df.loc[s.index, "volume"]), 5))
        return self.as_cs_series(df, val)
```

---

#### Alpha045 - 延迟量价

**依赖字段**: `close`, `volume`

**计算公式**:
$$\text{Alpha045} = -\text{CSRank}(\text{RollingSum}(\text{delay}(\text{close}, 5), 20)/20) \times \text{RollingCorr}(\text{close}, \text{volume}, 2) \times \text{CSRank}(\text{RollingCorr}(\text{RollingSum}(\text{close}, 5), \text{RollingSum}(\text{close}, 20), 2))$$

**金融直觉**: 延迟收盘价的 20 日均值、短期量价相关、以及不同周期均值的相互相关的复杂组合。

**代码示例**:
```python
@register
class Alpha045(Factor):
    name = "Alpha045"
    requires = ["close", "volume"]
    def compute(self, df):
        a = ops.cs_rank(self._g(df, "close", lambda s: ops.rolling_sum(ops.delay(s, 5), 20) / 20))
        b = self._g(df, "close", lambda s: ops.rolling_corr(s, df.loc[s.index, "volume"], 2))
        c = self._g(df, "close", lambda s: ops.rolling_corr(self._g(df, "close", lambda s2: ops.rolling_sum(s2, 5)), self._g(df, "close", lambda s2: ops.rolling_sum(s2, 20)), 2))
        val = -(a * b * ops.cs_rank(c))
        return self.as_cs_series(df, val)
```

---

#### Alpha055 - KDJ 量价协方差

**依赖字段**: `close`, `high`, `low`, `volume`

**计算公式**:
$$\text{Alpha055} = -\text{RollingCorr}(\text{CSRank}(\text{RSV}), \text{CSRank}(\text{volume}), 6)$$

其中 $\text{RSV} = \frac{\text{close} - \text{RollingMin}(\text{low}, 12)}{\text{RollingMax}(\text{high}, 12) - \text{RollingMin}(\text{low}, 12)}$

**金融直觉**: KDJ 指标的 RSV 分量与成交量的相关关系。RSV 高位放量可能预示回调。

**代码示例**:
```python
@register
class Alpha055(Factor):
    name = "Alpha055"
    requires = ["close", "high", "low", "volume"]
    def compute(self, df):
        num = (df["close"] - self._g(df, "low", lambda s: ops.rolling_min(s, 12))) / (self._g(df, "high", lambda s: ops.rolling_max(s, 12)) - self._g(df, "low", lambda s: ops.rolling_min(s, 12))).replace(0, np.nan)
        val = -self._g(df, None, lambda *_: ops.rolling_corr(ops.cs_rank(num), ops.cs_rank(df["volume"]), 6))
        return self.as_cs_series(df, val)
```

---

#### Alpha060 - 量价加权

**依赖字段**: `high`, `low`, `close`, `volume`

**计算公式**:
$$\text{Alpha060} = -\left(2 \times \text{CSRank}(\text{DecayLinear}(x, 10)) - \text{CSRank}(\text{TSRank}(\text{close}, 10))\right)$$

其中 $x = \frac{(\text{close} - \text{low}) - (\text{high} - \text{close})}{\text{high} - \text{low}} \times \text{volume}$

**金融直觉**: 价格位置加权成交量与价格排名的组合。价格靠近高点且放量时产生信号。

**代码示例**:
```python
@register
class Alpha060(Factor):
    name = "Alpha060"
    requires = ["high", "low", "close", "volume"]
    def compute(self, df):
        x = (((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"]).replace(0, np.nan)) * df["volume"]
        val = -(2 * ops.cs_rank(ops.decay_linear(x, 10)) - ops.cs_rank(self._g(df, "close", lambda s: ops.ts_rank(s, 10))))
        return self.as_cs_series(df, val)
```

---

#### Alpha061 - VWAP 最小值

**依赖字段**: `vwap`, `volume`

**计算公式**:
$$\text{Alpha061} = \mathbb{I}(\text{CSRank}(\text{vwap} - \text{RollingMin}(\text{vwap}, 16.12)) < \text{CSRank}(\text{RollingCorr}(\text{vwap}, \text{ADV}_{180}, 17.93)))$$

**金融直觉**: VWAP 相对近期最小值的地位与 VWAP 和长期成交量的相关性的比较。VWAP 低位且与量能正相关时产生买入信号。

**代码示例**:
```python
@register
class Alpha061(Factor):
    name = "Alpha061"
    requires = ["vwap", "volume"]
    def compute(self, df):
        adv180 = self._g(df, "volume", lambda s: ops.adv(s, 180))
        a = ops.cs_rank(df["vwap"] - self._g(df, "vwap", lambda s: ops.rolling_min(s, int(16.1219))))
        b = ops.cs_rank(self._g(df, None, lambda *_: ops.rolling_corr(df["vwap"], adv180, int(17.9282))))
        val = (a < b).astype(float)
        return self.as_cs_series(df, val)
```

---

#### Alpha064 - 加权相关

**依赖字段**: `open`, `low`, `vwap`, `close`

**计算公式**:
$$\text{Alpha064} = -\mathbb{I}(\text{CSRank}(\text{RollingCorr}(0.178 \times \text{open} + 0.822 \times \text{low}, \text{ADV}_{120}, 16.62)) < \text{CSRank}(\Delta(\frac{\text{high}+\text{low}}{2} \times 0.178 + \text{vwap} \times 0.822, 3.70)))$$

**金融直觉**: 加权价格（偏向低价）与长期成交量的相关性，与加权高低 VWAP 的变化率的比较。

**代码示例**:
```python
@register
class Alpha064(Factor):
    name = "Alpha064"
    requires = ["open", "low", "vwap", "close"]
    def compute(self, df):
        a = ops.cs_rank(self._g(df, None, lambda *_: ops.rolling_corr((self._g(df, "open", lambda s: 0.178404 * s) + (df["low"] * (1 - 0.178404))), self._g(df, "volume", lambda s: ops.adv(s, 120)), int(16.6208))))
        b = ops.cs_rank(self._g(df, None, lambda *_: ops.delta((((df["high"] + df["low"]) / 2) * 0.178404 + df["vwap"] * (1 - 0.178404)), int(3.69741))))
        val = (a < b).astype(float) * -1
        return self.as_cs_series(df, val)
```

---

#### Alpha065 - 开盘 VWAP 相关

**依赖字段**: `open`, `vwap`, `low`

**计算公式**:
$$\text{Alpha065} = -\mathbb{I}(\text{CSRank}(\text{RollingCorr}(0.008 \times \text{open} + 0.992 \times \text{vwap}, \text{ADV}_{60}, 6.40)) < \text{CSRank}(\text{open} - \text{RollingMin}(\text{open}, 13.64)))$$

**金融直觉**: 开盘价与 VWAP 的高度加权相关（偏向 VWAP）与开盘价相对最小值的比较。

**代码示例**:
```python
@register
class Alpha065(Factor):
    name = "Alpha065"
    requires = ["open", "vwap", "low"]
    def compute(self, df):
        a = ops.cs_rank(self._g(df, None, lambda *_: ops.rolling_corr(0.00817205 * df["open"] + (1 - 0.00817205) * df["vwap"], self._g(df, "volume", lambda s: ops.adv(s, 60)), int(6.40374))))
        b = ops.cs_rank(df["open"] - self._g(df, "open", lambda s: ops.rolling_min(s, int(13.635))))
        val = (a < b).astype(float) * -1
        return self.as_cs_series(df, val)
```

---

#### Alpha071 - 双重排名

**依赖字段**: `close`, `volume`, `low`, `open`, `vwap`

**计算公式**:
$$\text{Alpha071} = \max(a, b)$$

其中
$$a = \text{TSRank}(\text{DecayLinear}(\text{TSRank}(\text{close}, 3.44), 4.21), 15.69)$$
$$b = \text{TSRank}(\text{DecayLinear}(\text{CSRank}((\text{low} + \text{open} - 2 \times \text{vwap})^2), 16.47), 4.44)$$

**金融直觉**: 价格排名的衰减加权与波动率指标（偏离 VWAP 的平方）的排名的最大值。捕捉价格趋势与波动率的综合信号。

**代码示例**:
```python
@register
class Alpha071(Factor):
    name = "Alpha071"
    requires = ["close", "volume", "low", "open", "vwap"]
    def compute(self, df):
        a = self._g(df, None, lambda *_: ops.ts_rank(ops.decay_linear(self._g(df, "close", lambda s: ops.ts_rank(s, int(3.43976))), int(4.20501)), int(15.6948)))
        b = self._g(df, None, lambda *_: ops.ts_rank(ops.decay_linear(ops.cs_rank(((df["low"] + df["open"]) - (df["vwap"] + df["vwap"]))**2), int(16.4662)), int(4.4388)))
        val = np.maximum(a, b)
        return self.as_cs_series(df, val)
```

---

#### Alpha083 - 波动率比率

**依赖字段**: `high`, `low`, `close`, `vwap`, `volume`

**计算公式**:
$$\text{Alpha083} = \frac{\text{CSRank}(\text{delay}(\frac{\text{high}-\text{low}}{\text{MA}_5(\text{close})}, 2)) \times \text{CSRank}(\text{CSRank}(\text{volume}))}{\frac{\text{high}-\text{low}}{\text{MA}_5(\text{close})} / (\text{vwap} - \text{close})}$$

**金融直觉**: 价格区间与均值的比率滞后排名的乘积，除以区间与 VWAP 偏离的比率。捕捉波动率与 VWAP 偏离的综合效应。

**代码示例**:
```python
@register
class Alpha083(Factor):
    name = "Alpha083"
    requires = ["high", "low", "close", "vwap", "volume"]
    def compute(self, df):
        num = ops.cs_rank(ops.delay(((df["high"] - df["low"]) / (self._g(df, "close", lambda s: ops.rolling_sum(s, 5)) / 5)), 2)) * ops.cs_rank(ops.cs_rank(df["volume"]))
        den = ((df["high"] - df["low"]) / (self._g(df, "close", lambda s: ops.rolling_sum(s, 5)) / 5)) / (df["vwap"] - df["close"]).replace(0, np.nan)
        val = num / den.replace(0, np.nan)
        return self.as_cs_series(df, val)
```

---

#### Alpha084 - VWAP 趋势

**依赖字段**: `vwap`, `close`

**计算公式**:
$$\text{Alpha084} = \text{Sign}(\Delta(\text{close}, 4.97)) \times \text{TSRank}(\text{vwap} - \text{RollingMax}(\text{vwap}, 15.32), 20.71)$$

**金融直觉**: 价格变化的符号与 VWAP 相对其最大值的排名的乘积。价格上涨且 VWAP 接近高位时产生信号。

**代码示例**:
```python
@register
class Alpha084(Factor):
    name = "Alpha084"
    requires = ["vwap", "close"]
    def compute(self, df):
        val = np.sign(self._g(df, "close", ops.delta, int(4.96796))) * self._g(df, "vwap", lambda s: ops.ts_rank(s - self._g(df, "vwap", ops.rolling_max, int(15.3217)), int(20.7127)))
        return self.as_cs_series(df, val)
```

---

#### Alpha085 - 双相关指数

**依赖字段**: `high`, `close`, `volume`

**计算公式**:
$$\text{Alpha085} = (\text{CSRank}(a))^{\text{CSRank}(b)}$$

其中
$$a = \text{RollingCorr}(0.877 \times \text{high} + 0.123 \times \text{close}, \text{ADV}_{30}, 9.61)$$
$$b = \text{RollingCorr}(\text{TSRank}(\frac{\text{high}+\text{low}}{2}, 3.71), \text{TSRank}(\text{volume}, 10.16), 7.11)$$

**金融直觉**: 两个滚动相关系数的截面排名的幂次关系。第一个衡量高价与成交量的相关，第二个衡量价格排名与成交量排名的相关。

**代码示例**:
```python
@register
class Alpha085(Factor):
    name = "Alpha085"
    requires = ["high", "close", "volume"]
    def compute(self, df):
        a = self._g(df, None, lambda *_: ops.rolling_corr(0.876703 * df["high"] + (1 - 0.876703) * df["close"], self._g(df, "volume", lambda s: ops.adv(s, 30)), int(9.61331)))
        b = self._g(df, None, lambda *_: ops.rolling_corr(self._g(df, "close", lambda s: ops.ts_rank((df["high"] + df["low"]) / 2, int(3.70596))), self._g(df, "volume", lambda s: ops.ts_rank(df["volume"], int(10.1595))), int(7.11408)))
        val = ops.cs_rank(a) ** ops.cs_rank(b)
        return self.as_cs_series(df, val)
```

---

#### Alpha086 - VWAP 相关排名

**依赖字段**: `close`, `open`, `vwap`, `volume`

**计算公式**:
$$\text{Alpha086} = -\mathbb{I}(\text{TSRank}(\text{RollingCorr}(\text{close}, \text{ADV}_{20}, 6.00), 20.42) < \text{CSRank}((\text{open} + \text{close}) - (\text{vwap} + \text{open})))$$

**金融直觉**: 收盘价与成交量的相关排名与 VWAP 偏离的比较。相关性弱且 VWAP 偏离大时产生信号。

**代码示例**:
```python
@register
class Alpha086(Factor):
    name = "Alpha086"
    requires = ["close", "open", "vwap", "volume"]
    def compute(self, df):
        a = self._g(df, "close", lambda s: ops.ts_rank(ops.rolling_corr(s, self._g(df, "volume", lambda s2: ops.adv(s2, 20)), int(6.00049)), int(20.4195)))
        b = ops.cs_rank((df["open"] + df["close"]) - (df["vwap"] + df["open"]))
        val = (a < b).astype(float) * -1
        return self.as_cs_series(df, val)
```

---

#### Alpha094 - VWAP 幂次

**依赖字段**: `vwap`, `volume`

**计算公式**:
$$\text{Alpha094} = -(\text{CSRank}(\text{vwap} - \text{RollingMin}(\text{vwap}, 11.58)))^{\text{TSRank}(\text{RollingCorr}(\text{TSRank}(\text{vwap}, 19.65), \text{TSRank}(\text{ADV}_{60}, 4.03), 18.09), 2.71)}$$

**金融直觉**: VWAP 相对最小值的排名作为底数，量价相关的排名作为指数的幂次组合。VWAP 低位且与量能正相关时产生强信号。

**代码示例**:
```python
@register
class Alpha094(Factor):
    name = "Alpha094"
    requires = ["vwap", "volume"]
    def compute(self, df):
        adv60 = self._g(df, "volume", lambda s: ops.adv(s, 60))
        a = ops.cs_rank(df["vwap"] - self._g(df, "vwap", lambda s: ops.rolling_min(s, int(11.5783))))
        b = self._g(df, None, lambda *_: ops.ts_rank(ops.rolling_corr(self._g(df, "vwap", lambda s: ops.ts_rank(s, int(19.6462))), self._g(df, "volume", lambda s: ops.ts_rank(adv60, int(4.02992))), int(18.0926)), int(2.70756)))
        val = (a ** b) * -1
        return self.as_cs_series(df, val)
```

---

#### Alpha095 - 多因子阈值

**依赖字段**: `open`, `high`, `low`, `close`, `volume`

**计算公式**:
$$\text{Alpha095} = \mathbb{I}(\text{CSRank}(\text{open} - \text{RollingMin}(\text{open}, 12.41)) < (\text{CSRank}(\text{RollingCorr}(\text{RollingSum}(\frac{\text{high}+\text{low}}{2}, 19.14), \text{ADV}_{40}, 12.87)))^5)$$

**金融直觉**: 开盘价相对最小值的排名与均价和成交量相关的五次幂的比较。强势开盘且量价配合良好时产生信号。

**代码示例**:
```python
@register
class Alpha095(Factor):
    name = "Alpha095"
    requires = ["open", "high", "low", "close", "volume"]
    def compute(self, df):
        a = ops.cs_rank(self._g(df, None, lambda *_: ops.rolling_corr(self._g(df, "close", lambda s: ops.rolling_sum((df["high"] + df["low"]) / 2, int(19.1351))), self._g(df, "volume", lambda s: ops.adv(s, 40)), int(12.8742))) ** 5)
        b = self._g(df, "open", lambda s: ops.ts_rank(s - self._g(df, "open", lambda s2: ops.rolling_min(s2, int(12.4105))), 1))
        val = (ops.cs_rank(df["open"] - self._g(df, "open", lambda s: ops.rolling_min(s, int(12.4105)))) < a).astype(float)
        return self.as_cs_series(df, val)
```

---

#### Alpha096 - 双重衰减

**依赖字段**: `vwap`, `volume`, `close`

**计算公式**:
$$\text{Alpha096} = -\max(a, b)$$

其中
$$a = \text{TSRank}(\text{DecayLinear}(\text{RollingCorr}(\text{CSRank}(\text{vwap}), \text{CSRank}(\text{volume}), 3.84), 4.17), 8.38)$$
$$b = \text{TSRank}(\text{DecayLinear}(\text{TSRank}(\text{RollingCorr}(\text{CSRank}(\text{close}), \text{ADV}_{60}, 4.13), 7.45), 14.04), 13.41)$$

**金融直觉**: 两个不同周期的衰减线性相关的最大值的反向。捕捉 VWAP-成交量和收盘价 - 成交量相关性的多重时间尺度信号。

**代码示例**:
```python
@register
class Alpha096(Factor):
    name = "Alpha096"
    requires = ["vwap", "volume", "close"]
    def compute(self, df):
        a = self._g(df, None, lambda *_: ops.ts_rank(ops.decay_linear(ops.rolling_corr(ops.cs_rank(df["vwap"]), ops.cs_rank(df["volume"]), int(3.83878)), int(4.16783)), int(8.38151)))
        b = self._g(df, None, lambda *_: ops.ts_rank(ops.decay_linear(ops.ts_rank(self._g(df, "close", lambda s: ops.rolling_corr(ops.cs_rank(s), self._g(df, "volume", lambda s2: ops.adv(s2, 60)), int(4.13242))), int(7.45404)), int(14.0365)), int(13.4143)))
        val = -np.maximum(a, b)
        return self.as_cs_series(df, val)
```

---

#### Alpha098 - VWAP 开放相关

**依赖字段**: `vwap`, `volume`, `open`

**计算公式**:
$$\text{Alpha098} = \text{CSRank}(\text{RollingCorr}(\text{vwap}, \text{RollingSum}(\text{ADV}_5, 26.47), 4.58)) - \text{TSRank}(\text{TSRank}(\text{ArgMin}(\text{RollingCorr}(\text{CSRank}(\text{open}), \text{ADV}_{15}, 20.82)), 6.96), 8.07)$$

**金融直觉**: VWAP 与短期成交量的相关排名减去开盘价与成交量相关的极值点排名。捕捉 VWAP 与量能的关系及开盘价的领先滞后特征。

**代码示例**:
```python
@register
class Alpha098(Factor):
    name = "Alpha098"
    requires = ["vwap", "volume", "open"]
    def compute(self, df):
        adv5 = self._g(df, "volume", lambda s: ops.adv(s, 5))
        a = ops.cs_rank(self._g(df, None, lambda *_: ops.rolling_corr(df["vwap"], self._g(df, "volume", lambda s: ops.rolling_sum(adv5, int(26.4719))), int(4.58418))))
        b = self._g(df, None, lambda *_: ops.ts_rank(ops.ts_rank(ops.argmin(ops.rolling_corr(ops.cs_rank(df["open"]), self._g(df, "volume", lambda s: ops.adv(s, 15)), int(20.8187))), int(6.95668)), int(8.07206))) if hasattr(np, "argmin") else a * 0
        val = a - b
        return self.as_cs_series(df, val)
```

---

#### Alpha099 - 双相关比较

**依赖字段**: `high`, `low`, `volume`

**计算公式**:
$$\text{Alpha099} = -\mathbb{I}(\text{CSRank}(\text{RollingCorr}(\text{RollingSum}(\frac{\text{high}+\text{low}}{2}, 19.90), \text{ADV}_{60}, 8.81)) < \text{CSRank}(\text{RollingCorr}(\text{low}, \text{volume}, 6.28)))$$

**金融直觉**: 均价与长期成交量的相关和低量相关的截面排名比较。低价与成交量更相关时产生信号。

**代码示例**:
```python
@register
class Alpha099(Factor):
    name = "Alpha099"
    requires = ["high", "low", "volume"]
    def compute(self, df):
        a = self._g(df, None, lambda *_: ops.rolling_corr(self._g(df, "close", lambda s: ops.rolling_sum((df["high"] + df["low"]) / 2, int(19.8975))), self._g(df, "volume", lambda s: ops.adv(s, 60)), int(8.8136)))
        b = self._g(df, None, lambda *_: ops.rolling_corr(df["low"], df["volume"], int(6.28259)))
        val = (ops.cs_rank(a) < ops.cs_rank(b)).astype(float) * -1
        return self.as_cs_series(df, val)
```

---

#### Alpha101 - 开盘收盘比率

**依赖字段**: `open`, `high`, `low`, `close`

**计算公式**:
$$\text{Alpha101} = \frac{\text{close} - \text{open}}{(\text{high} - \text{low}) + 0.001}$$

**金融直觉**: 最简单的因子之一，衡量收盘价相对于当日价格区间的偏移。值接近 1 表示收于高点，接近 -1 表示收于低点。

**代码示例**:
```python
@register
class Alpha101(Factor):
    name = "Alpha101"
    requires = ["open", "high", "low", "close"]
    def compute(self, df):
        val = (df["close"] - df["open"]) / ((df["high"] - df["low"]) + 0.001)
        return self.as_cs_series(df, val)
```

---

### 价格动量因子 (Price Momentum)

#### Alpha004 - 最低价排名

**依赖字段**: `low`

**计算公式**:
$$\text{Alpha004} = -\text{TSRank}(\text{CSRank}(\text{low}), 9)$$

**金融直觉**: 最低价截面排名的时间序列排名取反。低价股若持续创新低（TSRank 低）则产生正面信号。

**代码示例**:
```python
@register
class Alpha004(Factor):
    name = "Alpha004"
    requires = ["low"]
    def compute(self, df):
        val = -self._g(df, "low", lambda s: ops.ts_rank(ops.cs_rank(s), 9))
        return self.as_cs_series(df, val)
```

---

#### Alpha009 - 价格变化符号

**依赖字段**: `close`

**计算公式**:
$$\text{Alpha009} = \begin{cases} \Delta(\text{close}, 1) & \text{if } \text{RollingMin}(\Delta(\text{close}, 1), 5) > 0 \\ \Delta(\text{close}, 1) & \text{if } \text{RollingMax}(\Delta(\text{close}, 1), 5) < 0 \\ -\Delta(\text{close}, 1) & \text{otherwise} \end{cases}$$

**金融直觉**: 根据过去 5 日价格变化的极值判断当前变化的持续性。若持续上涨或下跌则保持原符号，否则反转。

**代码示例**:
```python
@register
class Alpha009(Factor):
    name = "Alpha009"
    requires = ["close"]
    def compute(self, df):
        d1 = self._g(df, "close", ops.delta, 1)
        cond1 = self._g(df, "close", lambda s: ops.rolling_min(ops.delta(s, 1), 5)) > 0
        cond2 = self._g(df, "close", lambda s: ops.rolling_max(ops.delta(s, 1), 5)) < 0
        val = np.where(cond1, d1, np.where(cond2, d1, -d1))
        return self.as_cs_series(df, pd.Series(val))
```

---

#### Alpha010 - 价格变化截面排名

**依赖字段**: `close`

**计算公式**:
$$\text{Alpha010} = \text{CSRank}(\text{Alpha009})$$

**金融直觉**: Alpha009 的截面排名版本，将绝对值转换为相对排序，便于横截面比较。

**代码示例**:
```python
@register
class Alpha010(Factor):
    name = "Alpha010"
    requires = ["close"]
    def compute(self, df):
        d1 = self._g(df, "close", ops.delta, 1)
        cond1 = self._g(df, "close", lambda s: ops.rolling_min(ops.delta(s, 1), 4)) > 0
        cond2 = self._g(df, "close", lambda s: ops.rolling_max(ops.delta(s, 1), 4)) < 0
        val = ops.cs_rank(np.where(cond1, d1, np.where(cond2, d1, -d1)))
        return self.as_cs_series(df, val)
```

---

#### Alpha019 - 长期价格动量

**依赖字段**: `close`, `returns`

**计算公式**:
$$\text{Alpha019} = -\text{Sign}(\text{close} - \text{delay}(\text{close}, 7) + \Delta(\text{close}, 7)) \times (1 + \text{CSRank}(1 + \text{RollingSum}(\text{returns}, 250)))$$

**金融直觉**: 7 日价格变化加上最近 7 日变化，乘以年化收益率排名的反向。捕捉中长期动量效应。

**代码示例**:
```python
@register
class Alpha019(Factor):
    name = "Alpha019"
    requires = ["close", "returns"]
    def compute(self, df):
        price_change = df["close"] - self._g(df, "close", ops.delay, 7) + self._g(df, "close", ops.delta, 7)
        returns_sum_rank = ops.cs_rank(1 + self._g(df, "returns", ops.rolling_sum, 250))
        val = -np.sign(price_change) * (1 + returns_sum_rank)
        return self.as_cs_series(df, pd.Series(val))
```

---

#### Alpha023 - 高位动量

**依赖字段**: `high`, `close`

**计算公式**:
$$\text{Alpha023} = \begin{cases} -\Delta(\text{high}, 2) & \text{if } \text{MA}_{20}(\text{close}) < \text{high} \\ 0 & \text{otherwise} \end{cases}$$

**金融直觉**: 当最高价高于 20 日均价时，捕捉最高价的向下变化。高位回落可能预示调整。

**代码示例**:
```python
@register
class Alpha023(Factor):
    name = "Alpha023"
    requires = ["high", "close"]
    def compute(self, df):
        close_ma20 = self._g(df, "close", ops.rolling_sum, 20) / 20
        cond = close_ma20 < df["high"]
        val = np.where(cond, -self._g(df, "high", ops.delta, 2), 0)
        return self.as_cs_series(df, pd.Series(val))
```

---

#### Alpha024 - 长期趋势强度

**依赖字段**: `close`

**计算公式**:
$$d = \frac{\Delta(\text{MA}_{100}(\text{close}), 100)}{\text{delay}(\text{close}, 100)}$$

$$\text{Alpha024} = \begin{cases} -(\text{close} - \text{RollingMin}(\text{close}, 100)) & \text{if } d \leq 0.05 \\ -\Delta(\text{close}, 3) & \text{otherwise} \end{cases}$$

**金融直觉**: 100 日均线斜率较缓时关注价格相对最低位的距离，斜率陡峭时关注短期价格变化。适应不同趋势强度的市场环境。

**代码示例**:
```python
@register
class Alpha024(Factor):
    name = "Alpha024"
    requires = ["close"]
    def compute(self, df):
        s100 = self._g(df, "close", lambda s: ops.rolling_sum(s, 100) / 100)
        d = self._g(df, "close", lambda s: ops.delta(ops.rolling_sum(s, 100) / 100, 100)) / self._g(df, "close", lambda s: ops.delay(s, 100))
        cond = (d <= 0.05)
        val = np.where(cond, -(df["close"] - self._g(df, "close", ops.rolling_min, 100)), -self._g(df, "close", ops.delta, 3))
        return self.as_cs_series(df, pd.Series(val))
```

---

#### Alpha033 - 开盘收盘比

**依赖字段**: `open`, `close`

**计算公式**:
$$\text{Alpha033} = \text{CSRank}\left(-\left(1 - \frac{\text{open}}{\text{close}}\right)\right)$$

**金融直觉**: 开盘收盘比率的截面排名。高开低走（open/close 大）时值为负，反之为正。

**代码示例**:
```python
@register
class Alpha033(Factor):
    name = "Alpha033"
    requires = ["open", "close"]
    def compute(self, df):
        val = ops.cs_rank(-(1 - (df["open"] / df["close"])))
        return self.as_cs_series(df, val)
```

---

#### Alpha034 - 波动率比率

**依赖字段**: `returns`, `close`

**计算公式**:
$$\text{Alpha034} = 1 - \text{CSRank}\left(\frac{\text{RollingStd}(\text{returns}, 2)}{\text{RollingStd}(\text{returns}, 5)}\right) + 1 - \text{CSRank}(\Delta(\text{close}, 1))$$

**金融直觉**: 短期与长期波动率比率及价格变化的截面排名。波动率收缩后扩张通常伴随趋势启动。

**代码示例**:
```python
@register
class Alpha034(Factor):
    name = "Alpha034"
    requires = ["returns", "close"]
    def compute(self, df):
        a = 1 - ops.cs_rank(self._g(df, "returns", lambda s: ops.rolling_std(s, 2) / ops.rolling_std(s, 5)))
        b = 1 - ops.cs_rank(self._g(df, "close", ops.delta, 1))
        val = a + b
        return self.as_cs_series(df, val)
```

---

#### Alpha037 - 开盘差相关

**依赖字段**: `open`, `close`

**计算公式**:
$$\text{Alpha037} = \text{CSRank}(\text{RollingCorr}(\text{delay}(\text{open}-\text{close}, 1), \text{close}, 200)) + \text{CSRank}(\text{open}-\text{close})$$

**金融直觉**: 开盘收盘差与收盘价的长期相关，加上当期开盘收盘差的排名。捕捉开盘信息的预测能力。

**代码示例**:
```python
@register
class Alpha037(Factor):
    name = "Alpha037"
    requires = ["open", "close"]
    def compute(self, df):
        a = ops.cs_rank(self._g(df, None, lambda *_: ops.rolling_corr(ops.delay(df["open"] - df["close"], 1), df["close"], 200)))
        b = ops.cs_rank(df["open"] - df["close"])
        return self.as_cs_series(df, a + b)
```

---

#### Alpha041 - VWAP 偏离

**依赖字段**: `high`, `low`, `vwap`

**计算公式**:
$$\text{Alpha041} = \sqrt{\text{high} \times \text{low}} - \text{vwap}$$

**金融直觉**: 高低价的几何平均与 VWAP 的差值。几何平均对极端值不敏感，反映 VWAP 相对于价格区间的偏离。

**代码示例**:
```python
@register
class Alpha041(Factor):
    name = "Alpha041"
    requires = ["high", "low", "vwap"]
    def compute(self, df):
        val = (df["high"] * df["low"])**0.5 - df["vwap"]
        return self.as_cs_series(df, val)
```

---

#### Alpha042 - VWAP-收盘价

**依赖字段**: `vwap`, `close`

**计算公式**:
$$\text{Alpha042} = \frac{\text{CSRank}(\text{vwap} - \text{close})}{\text{CSRank}(\text{vwap} + \text{close})}$$

**金融直觉**: VWAP 与收盘价差的排名除以两者和的排名。标准化 VWAP 偏离，消除量纲影响。

**代码示例**:
```python
@register
class Alpha042(Factor):
    name = "Alpha042"
    requires = ["vwap", "close"]
    def compute(self, df):
        val = ops.cs_rank(df["vwap"] - df["close"]) / ops.cs_rank(df["vwap"] + df["close"])
        return self.as_cs_series(df, val)
```

---

#### Alpha046 - 加速度阈值

**依赖字段**: `close`

**计算公式**:
$$a = \frac{\text{delay}(\text{close}, 20) - \text{delay}(\text{close}, 10)}{10} - \frac{\text{delay}(\text{close}, 10) - \text{close}}{10}$$

$$\text{Alpha046} = \begin{cases} -1 & \text{if } a > 0.25 \\ 1 & \text{if } a < 0 \\ -(\text{close} - \text{delay}(\text{close}, 1)) & \text{otherwise} \end{cases}$$

**金融直觉**: 价格加速度（二阶差分）的三个区域：加速上涨、加速下跌、平稳。不同区域采用不同的信号生成逻辑。

**代码示例**:
```python
@register
class Alpha046(Factor):
    name = "Alpha046"
    requires = ["close"]
    def compute(self, df):
        a = (self._g(df, "close", ops.delay, 20) - self._g(df, "close", ops.delay, 10)) / 10 - (self._g(df, "close", ops.delay, 10) - df["close"]) / 10
        val = np.where(a > 0.25, -1, np.where(a < 0, 1, -1 * (df["close"] - self._g(df, "close", ops.delay, 1))))
        return self.as_cs_series(df, pd.Series(val))
```

---

#### Alpha047 - 多空量价

**依赖字段**: `close`, `high`, `vwap`, `volume`

**计算公式**:
$$\text{Alpha047} = \frac{\text{CSRank}(1/\text{close}) \times \text{volume}}{\text{ADV}_{20}} \times \frac{\text{high} \times \text{CSRank}(\text{high}-\text{close})}{\text{MA}_5(\text{high})/5} - \text{CSRank}(\text{vwap} - \text{delay}(\text{vwap}, 5))$$

**金融直觉**: 多重量价关系的组合：低价股的成交量权重、高价相对收盘价的排名、VWAP 的变化。捕捉不同市值股票的量价特征。

**代码示例**:
```python
@register
class Alpha047(Factor):
    name = "Alpha047"
    requires = ["close", "high", "vwap", "volume"]
    def compute(self, df):
        adv20 = self._g(df, "volume", lambda s: ops.adv(s, 20))
        part1 = (ops.cs_rank(1 / df["close"]) * df["volume"]) / adv20
        part2 = (df["high"] * ops.cs_rank(df["high"] - df["close"])) / (self._g(df, "high", lambda s: ops.rolling_sum(s, 5)) / 5)
        val = part1 * part2 - ops.cs_rank(df["vwap"] - self._g(df, "vwap", ops.delay, 5))
        return self.as_cs_series(df, val)
```

---

#### Alpha049 - 加速度阈值

**依赖字段**: `close`

**计算公式**:
$$a = \frac{\text{delay}(\text{close}, 20) - \text{delay}(\text{close}, 10)}{10} - \frac{\text{delay}(\text{close}, 10) - \text{close}}{10}$$

$$\text{Alpha049} = \begin{cases} 1 & \text{if } a < -0.1 \\ -(\text{close} - \text{delay}(\text{close}, 1)) & \text{otherwise} \end{cases}$$

**金融直觉**: Alpha046 的简化版，只关注加速下跌的情况。下跌加速时给出正面信号（预期反弹）。

**代码示例**:
```python
@register
class Alpha049(Factor):
    name = "Alpha049"
    requires = ["close"]
    def compute(self, df):
        a = (self._g(df, "close", ops.delay, 20) - self._g(df, "close", ops.delay, 10)) / 10 - (self._g(df, "close", ops.delay, 10) - df["close"]) / 10
        val = np.where(a < -0.1, 1, -1 * (df["close"] - self._g(df, "close", ops.delay, 1)))
        return self.as_cs_series(df, pd.Series(val))
```

---

#### Alpha051 - 加速度阈值

**依赖字段**: `close`

**计算公式**:
$$a = \frac{\text{delay}(\text{close}, 20) - \text{delay}(\text{close}, 10)}{10} - \frac{\text{delay}(\text{close}, 10) - \text{close}}{10}$$

$$\text{Alpha051} = \begin{cases} 1 & \text{if } a < -0.05 \\ -(\text{close} - \text{delay}(\text{close}, 1)) & \text{otherwise} \end{cases}$$

**金融直觉**: 另一版本的加速度阈值，阈值设为 -0.05，比 Alpha049 更敏感。

**代码示例**:
```python
@register
class Alpha051(Factor):
    name = "Alpha051"
    requires = ["close"]
    def compute(self, df):
        a = (self._g(df, "close", ops.delay, 20) - self._g(df, "close", ops.delay, 10)) / 10 - (self._g(df, "close", ops.delay, 10) - df["close"]) / 10
        val = np.where(a < -0.05, 1, -1 * (df["close"] - self._g(df, "close", ops.delay, 1)))
        return self.as_cs_series(df, pd.Series(val))
```

---

#### Alpha053 - 价格位置

**依赖字段**: `close`, `low`, `high`

**计算公式**:
$$x = \frac{(\text{close} - \text{low}) - (\text{high} - \text{close})}{\text{close} - \text{low}}$$

$$\text{Alpha053} = -\Delta(x, 9)$$

**金融直觉**: 收盘价在高低区间的位置变化。位置从偏低转向偏高时产生信号。

**代码示例**:
```python
@register
class Alpha053(Factor):
    name = "Alpha053"
    requires = ["close", "low", "high"]
    def compute(self, df):
        x = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["close"] - df["low"]).replace(0, np.nan)
        val = -self._g(df, None, lambda *_: ops.delta(x, 9))
        return self.as_cs_series(df, val)
```

---

#### Alpha054 - OHLC 比率

**依赖字段**: `low`, `close`, `open`, `high`

**计算公式**:
$$\text{Alpha054} = -\frac{(\text{low} - \text{close}) \times \text{open}^5}{(\text{low} - \text{high}) \times \text{close}^5}$$

**金融直觉**: 基于 OHLC 四价的复杂比率，五次幂放大差异。捕捉开盘、收盘相对于高低点的相对位置。

**代码示例**:
```python
@register
class Alpha054(Factor):
    name = "Alpha054"
    requires = ["low", "close", "open", "high"]
    def compute(self, df):
        val = -((df["low"] - df["close"]) * (df["open"]**5)) / ((df["low"] - df["high"]) * (df["close"]**5))
        return self.as_cs_series(df, val)
```

---

### 波动率因子 (Volatility)

#### Alpha001 - 波动率排名

**依赖字段**: `returns`, `close`

**计算公式**:
$$\text{part} = \begin{cases} \text{RollingStd}(\text{returns}, 20) & \text{if } \text{returns} < 0 \\ \text{close} & \text{otherwise} \end{cases}$$

$$\text{Alpha001} = \text{CSRank}(\text{TSRank}(\text{part}^2, 5)) - 0.5$$

**金融直觉**: 负收益时使用波动率，否则使用价格。波动率平方的时间排名经截面标准化。捕捉波动率聚集效应。

**代码示例**:
```python
@register
class Alpha001(Factor):
    name = "Alpha001"
    requires = ["returns", "close"]
    def compute(self, df):
        x = df["returns"].copy()
        part = np.where(x < 0, self._g(df, "returns", ops.rolling_std, 20), df["close"])
        val = self._g(df.assign(part=part), "part", lambda s: ops.ts_rank(s**2, 5))
        out = self._cs_rank(df, val) - 0.5
        return self.as_cs_series(df, out)
```

---

#### Alpha005 - 开盘偏离度

**依赖字段**: `open`, `vwap`, `close`

**计算公式**:
$$a = \text{CSRank}\left(\frac{\text{RollingSum}(\text{open}, 10)}{10} - \text{open}\right)$$

$$c = -|\text{CSRank}(\text{close} - \text{vwap})|$$

$$\text{Alpha005} = a \times c$$

**金融直觉**: 开盘价偏离 10 日均值的排名与收盘价-VWAP 偏离的乘积。开盘偏离与盘中偏离方向相反时产生信号。

**代码示例**:
```python
@register
class Alpha005(Factor):
    name = "Alpha005"
    requires = ["open", "vwap", "close"]
    def compute(self, df):
        a = df.groupby("symbol")["open"].apply(lambda s: ops.rolling_sum(s, 10) / 10)
        b = ops.cs_rank(df["open"] - a.values)
        c = -np.abs(ops.cs_rank(df["close"] - df["vwap"]))
        val = b * c
        return self.as_cs_series(df, val)
```

---

#### Alpha018 - 振幅波动

**依赖字段**: `close`, `open`

**计算公式**:
$$\text{Alpha018} = -\text{CSRank}(\text{RollingStd}(|\text{close}-\text{open}|, 5) + (\text{close}-\text{open}) + \text{RollingCorr}(\text{close}, \text{open}, 10))$$

**金融直觉**: 实体幅度的波动率、实体幅度本身、以及开收盘的相关性的组合。振幅大且不稳定时产生信号。

**代码示例**:
```python
@register
class Alpha018(Factor):
    name = "Alpha018"
    requires = ["close", "open"]
    def compute(self, df):
        a = self._g(df, "close", lambda s: ops.rolling_std(np.abs(s - df.loc[s.index, "open"]), 5))
        b = df["close"] - df["open"]
        c = self._g(df, "close", lambda s: ops.rolling_corr(s, df.loc[s.index, "open"], 10))
        val = -ops.cs_rank(a + b + c)
        return self.as_cs_series(df, val)
```

---

#### Alpha021 - 均值回归信号

**依赖字段**: `close`, `volume`

**计算公式**:
$$s_8 = \frac{\text{RollingSum}(\text{close}, 8)}{8}, \quad sd_8 = \text{RollingStd}(\text{close}, 8)$$

$$s_2 = \frac{\text{RollingSum}(\text{close}, 2)}{2}, \quad \text{ADV}_{20} = \text{Adv}(\text{volume}, 20)$$

$$\text{cond} = \mathbb{I}(s_8 + sd_8 < s_2) \times (-1) + \mathbb{I}(s_2 < s_8 - sd_8) \times 1$$

$$\text{cond2} = \mathbb{I}(\frac{\text{volume}}{\text{ADV}_{20}} \geq 1) \times 1 + \mathbb{I}(\frac{\text{volume}}{\text{ADV}_{20}} < 1) \times (-1)$$

$$\text{Alpha021} = \begin{cases} \text{cond} & \text{if } \text{cond} \neq 0 \\ \text{cond2} & \text{otherwise} \end{cases}$$

**金融直觉**: 短期与中期均值的偏离判断均值回归信号，若不明显则使用成交量相对水平。捕捉均值回归与成交量突破。

**代码示例**:
```python
@register
class Alpha021(Factor):
    name = "Alpha021"
    requires = ["close", "volume"]
    def compute(self, df):
        s8 = self._g(df, "close", lambda s: ops.rolling_sum(s, 8) / 8)
        sd8 = self._g(df, "close", lambda s: ops.rolling_std(s, 8))
        s2 = self._g(df, "close", lambda s: ops.rolling_sum(s, 2) / 2)
        adv20 = self._g(df, "volume", lambda s: ops.adv(s, 20))
        cond = ((s8 + sd8) < s2) * (-1) + ((s2 < (s8 - sd8)) * 1)
        cond2 = (((df["volume"] / adv20) >= 1) * 1) + (((df["volume"] / adv20) < 1) * (-1))
        val = np.where(cond != 0, cond, cond2)
        return self.as_cs_series(df, pd.Series(val))
```

---

#### Alpha025 - 反转量价

**依赖字段**: `returns`, `vwap`, `high`, `close`, `volume`

**计算公式**:
$$\text{Alpha025} = \text{CSRank}(-\text{returns} \times \text{ADV}_{20} \times \text{vwap} \times (\text{high} - \text{close}))$$

**金融直觉**: 负收益率、高成交量、高 VWAP、收盘价低于高价的组合。反转策略：坏消息配合放量可能预示底部。

**代码示例**:
```python
@register
class Alpha025(Factor):
    name = "Alpha025"
    requires = ["returns", "vwap", "high", "close", "volume"]
    def compute(self, df):
        adv20 = self._g(df, "volume", lambda s: ops.adv(s, 20))
        val = ops.cs_rank((-df["returns"]) * adv20 * df["vwap"] * (df["high"] - df["close"]))
        return self.as_cs_series(df, val)
```

---

#### Alpha032 - VWAP 偏离

**依赖字段**: `close`, `vwap`

**计算公式**:
$$\text{Alpha032} = \text{CSRank}(\text{MA}_7(\text{close}) - \text{close}) + 20 \times \text{CSRank}(\text{RollingCorr}(\text{vwap}, \text{delay}(\text{close}, 5), 230))$$

**金融直觉**: 短期均值偏离与 VWAP 和延迟收盘价长期相关的组合。VWAP 领先价格变动时产生信号。

**代码示例**:
```python
@register
class Alpha032(Factor):
    name = "Alpha032"
    requires = ["close", "vwap"]
    def compute(self, df):
        part1 = ops.cs_rank((self._g(df, "close", lambda s: ops.rolling_sum(s, 7) / 7) - df["close"]))
        part2 = 20 * ops.cs_rank(self._g(df, "close", lambda s: ops.rolling_corr(df.loc[s.index, "vwap"], self._g(df, "close", ops.delay, 5), 230)))
        return self.as_cs_series(df, part1 + part2)
```

#### Alpha050 - VWAP 量相关

**依赖字段**: `volume`, `vwap`

**计算公式**:
$$\text{Alpha050} = -\text{RollingMax}(\text{CSRank}(\text{RollingCorr}(\text{CSRank}(\text{volume}), \text{CSRank}(\text{vwap}), 5)), 5)$$

**金融直觉**: 成交量排名与 VWAP 排名的滚动相关性的最大值的反向。量价同步达到高位后可能回调。

**代码示例**:
```python
@register
class Alpha050(Factor):
    name = "Alpha050"
    requires = ["volume", "vwap"]
    def compute(self, df):
        val = -self._g(df, None, lambda *_: ops.rolling_max(ops.cs_rank(self._g(df, "volume", lambda s: ops.rolling_corr(ops.cs_rank(s), ops.cs_rank(df.loc[s.index, "vwap"]), 5))), 5))
        return self.as_cs_series(df, val)
```

---

#### Alpha052 - 低波动量价

**依赖字段**: `low`, `returns`, `volume`

**计算公式**:
$$\text{Alpha052} = (-\text{RollingMin}(\text{low}, 5) + \text{delay}(\text{RollingMin}(\text{low}, 5), 5)) \times \text{CSRank}\left(\frac{\text{RollingSum}(\text{returns}, 240) - \text{RollingSum}(\text{returns}, 20)}{220}\right) \times \text{TSRank}(\text{volume}, 5)$$

**金融直觉**: 最低价的变化、长期与短期收益率差的排名、以及成交量排名的乘积。捕捉低位反弹的量价配合。

**代码示例**:
```python
@register
class Alpha052(Factor):
    name = "Alpha052"
    requires = ["low", "returns", "volume"]
    def compute(self, df):
        part = (-self._g(df, "low", lambda s: ops.rolling_min(s, 5)) + self._g(df, "low", lambda s: ops.delay(ops.rolling_min(s, 5), 5))) * ops.cs_rank((self._g(df, "returns", ops.rolling_sum, 240) - self._g(df, "returns", ops.rolling_sum, 20)) / 220) * self._g(df, "volume", lambda s: ops.ts_rank(s, 5))
        return self.as_cs_series(df, part)
```

---

### 成交量异常因子 (Volume Anomaly)

#### Alpha011 - VWAP-价格偏离

**依赖字段**: `vwap`, `close`, `volume`

**计算公式**:
$$a = \text{TSRank}(\text{vwap} - \text{close}, 3)$$

$$b = \text{TSRank}(\text{close} - \text{vwap}, 3)$$

$$c = \text{TSRank}(\Delta(\text{volume}, 3), 3)$$

$$\text{Alpha011} = (\text{CSRank}(a) + \text{CSRank}(b)) \times \text{CSRank}(c)$$

**金融直觉**: VWAP 与收盘价的偏离方向及成交量变化的排名。VWAP 偏离配合成交量异常变化时产生信号。

**代码示例**:
```python
@register
class Alpha011(Factor):
    name = "Alpha011"
    requires = ["vwap", "close", "volume"]
    def compute(self, df):
        a = self._g(df, "vwap", lambda s: ops.ts_rank(s - df.loc[s.index, "close"], 3))
        b = self._g(df, "vwap", lambda s: ops.ts_rank(df.loc[s.index, "close"] - s, 3))
        c = self._g(df, "volume", lambda s: ops.ts_rank(ops.delta(s, 3), 3))
        val = (ops.cs_rank(a) + ops.cs_rank(b)) * ops.cs_rank(c)
        return self.as_cs_series(df, val)
```

---

#### Alpha012 - 量价方向

**依赖字段**: `close`, `volume`

**计算公式**:
$$\text{Alpha012} = \text{Sign}(\Delta(\text{volume}, 1)) \times (-\Delta(\text{close}, 1))$$

**金融直觉**: 成交量变化的符号与价格变化的反向乘积。放量下跌或缩量上涨时产生信号。

**代码示例**:
```python
@register
class Alpha012(Factor):
    name = "Alpha012"
    requires = ["close", "volume"]
    def compute(self, df):
        val = np.sign(self._g(df, "volume", ops.delta, 1)) * (-self._g(df, "close", ops.delta, 1))
        return self.as_cs_series(df, pd.Series(val))
```

---

#### Alpha020 - 高低开关系

**依赖字段**: `open`, `high`, `low`, `close`

**计算公式**:
$$\text{rank1} = -\text{CSRank}(\text{open} - \text{delay}(\text{high}, 1))$$

$$\text{rank2} = \text{CSRank}(\text{open} - \text{delay}(\text{close}, 1))$$

$$\text{rank3} = \text{CSRank}(\text{open} - \text{delay}(\text{low}, 1))$$

$$\text{Alpha020} = \text{rank1} \times \text{rank2} \times \text{rank3}$$

**金融直觉**: 开盘价与前一日高低收盘价的相对位置的排名乘积。开盘位置相对于前一日区间的意义。

**代码示例**:
```python
@register
class Alpha020(Factor):
    name = "Alpha020"
    requires = ["open", "high", "low", "close"]
    def compute(self, df):
        rank1 = -ops.cs_rank(df["open"] - self._g(df, "high", ops.delay, 1))
        rank2 = ops.cs_rank(df["open"] - self._g(df, "close", ops.delay, 1))
        rank3 = ops.cs_rank(df["open"] - self._g(df, "low", ops.delay, 1))
        val = rank1 * rank2 * rank3
        return self.as_cs_series(df, val)
```

## 算子参考

本章节介绍 `utils/ops.py` 中提供的所有基础算子，这些是构建 Alpha 因子的基本构件。

### 时间序列滚动窗口函数

#### rolling_sum

计算滚动窗口的和。

```python
def rolling_sum(s: pd.Series, n: int) -> pd.Series
```

- **参数**: 
  - `s`: 输入序列
  - `n`: 窗口大小
- **返回**: 滚动求和结果
- **说明**: 优先使用 bottleneck.move_sum 加速

#### rolling_min

计算滚动窗口的最小值。

```python
def rolling_min(s: pd.Series, n: int) -> pd.Series
```

#### rolling_max

计算滚动窗口的最大值。

```python
def rolling_max(s: pd.Series, n: int) -> pd.Series
```

#### rolling_std

计算滚动窗口的标准差（无偏差调整，ddof=0）。

```python
def rolling_std(s: pd.Series, n: int) -> pd.Series
```

#### rolling_cov

计算滚动窗口的协方差。

```python
def rolling_cov(s1: pd.Series, s2: pd.Series, n: int) -> pd.Series
```

#### rolling_corr

计算滚动窗口的相关系数。

```python
def rolling_corr(s1: pd.Series, s2: pd.Series, n: int) -> pd.Series
```

### 时间序列排名与加权

#### ts_rank

计算时间序列排名（窗口最后值的百分位排名）。

```python
def ts_rank(s: pd.Series, n: int) -> pd.Series
```

- **说明**: 对于每个位置 i，计算 s[i] 在过去 n 期中的百分位排名

#### decay_linear

计算线性衰减加权平均，越新的值权重越大。

```python
def decay_linear(s: pd.Series, n: int) -> pd.Series
```

- **权重**: $w_i = \frac{i}{\sum_{j=1}^n j}$，其中 $i=1,2,\dots,n$

### 基础变换函数

#### delay

计算滞后 n 期。

```python
def delay(s: pd.Series, n: int = 1) -> pd.Series
```

等价于 `s.shift(n)`

#### delta

计算差分：当前值减去 n 期前的值。

```python
def delta(s: pd.Series, n: int = 1) -> pd.Series
```

等价于 `s - s.shift(n)`

#### returns

计算收益率：相邻价格的百分比变化。

```python
def returns(close: pd.Series) -> pd.Series
```

等价于 `close.pct_change()`

#### vwap_from_amount

计算成交均价 (VWAP)。

```python
def vwap_from_amount(close, high, low, volume, amount) -> pd.Series
```

使用成交额 / 成交量计算。

#### adv

计算平均成交量（n 日均量）。

```python
def adv(volume: pd.Series, n: int) -> pd.Series
```

### 截面计算工具

#### cs_rank

截面分位数排名：对每个时间点上的股票进行排序。

```python
def cs_rank(s: pd.Series) -> pd.Series
```

- **实现**: 按日期分组后调用 pandas.rank(pct=True)

#### cs_zscore

截面标准化（Z-score）。

```python
def cs_zscore(s: pd.Series) -> pd.Series
```

- **公式**: $z = \frac{x - \mu}{\sigma}$，其中 $\mu, \sigma$ 为当天的截面均值和标准差

### 其他工具

#### by_symbol

对 DataFrame 按 symbol 分组后在指定列上应用函数。

```python
def by_symbol(df: pd.DataFrame, col: str, func, *args, **kwargs) -> pd.Series
```

---

## 添加新因子指南

本章节介绍如何在 alpha101_factory 中添加自定义 Alpha 因子。

### 步骤一：创建因子文件

在 `alpha101_factory/factors/` 目录下创建新文件，命名为 `alphas_custom.py`（或其他以 `alphas_` 开头的文件名）。

### 步骤二：继承 Factor 基类

创建新的因子类，继承自 `Factor`，并设置必要的属性：

```python
from alpha101_factory.factors.base import Factor
from alpha101_factory.factors.registry import register
from alpha101_factory.utils import ops
import numpy as np
import pandas as pd

@register
class MyCustomAlpha(Factor):
    name = "MyCustomAlpha"
    requires = ["close", "volume"]  # 依赖的字段
    
    def compute(self, df):
        """
        计算因子值
        
        Args:
            df: 长表格式的 DataFrame，MultiIndex 包含 (datetime, symbol)
        
        Returns:
            pd.Series: 因子值，MultiIndex 格式
        """
        # 你的计算逻辑
        close = df["close"]
        volume = df["volume"]
        
        # 示例：简单的量价因子
        ret = ops.returns(close)
        adv = ops.adv(volume, 20)
        
        # 计算因子值
        val = ret * (volume / adv)
        
        # 返回截面排名格式
        return self.as_cs_series(df, val)
```

### 步骤三：使用算子构建因子

利用 `utils/ops.py` 提供的算子构建复杂的因子逻辑：

```python
@register
class AdvancedAlpha(Factor):
    name = "AdvancedAlpha"
    requires = ["open", "high", "low", "close", "volume"]
    
    def compute(self, df):
        # 多个算子的组合使用
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        
        # 1. 价格位置
        price_pos = (close - ops.rolling_min(low, 20)) / (ops.rolling_max(high, 20) - ops.rolling_min(low, 20))
        
        # 2. 成交量排名
        vol_rank = ops.ts_rank(volume, 20)
        
        # 3. 波动率
        volatility = ops.rolling_std(ops.returns(close), 20)
        
        # 4. 组合
        factor_value = price_pos * vol_rank - volatility
        
        return self.as_cs_series(df, factor_value)
```

### 步骤四：自动注册

使用 `@register` 装饰器后，因子会自动被注册系统发现，无需手动修改任何配置文件。

### 步骤五：测试因子

运行因子计算命令：

```bash
# 计算单个因子
python -m alpha101_factory.cli factor --factors MyCustomAlpha

# 计算所有因子（包括新添加的）
python -m alpha101_factory.cli factor --all
```

### 完整示例

以下是一个完整的自定义因子示例，包含错误处理和文档：

```python
# alpha101_factory/factors/alphas_custom.py
# -*- coding: utf-8 -*-
"""
自定义 Alpha 因子示例
"""
import numpy as np
import pandas as pd
from alpha101_factory.factors.base import Factor
from alpha101_factory.factors.registry import register
from alpha101_factory.utils import ops


@register
class VolumeBreakout(Factor):
    """
    成交量突破因子
    
    当成交量显著高于平均水平且价格朝同一方向运动时产生信号。
    
    依赖字段:
        - close: 收盘价
        - volume: 成交量
        - returns: 收益率（可选，如未提供会自动计算）
    """
    name = "VolumeBreakout"
    requires = ["close", "volume"]
    
    def compute(self, df):
        """
        计算成交量突破因子值
        
        公式:
            factor = sign(ret) * max(0, volume_ratio - 1) * ts_rank(vol_ratio, 20)
        
        其中:
            - ret: 当日收益率
            - volume_ratio: 成交量 / 20 日均量
            - ts_rank: 时间序列排名
        """
        close = df["close"]
        volume = df["volume"]
        
        # 确保有收益率数据
        if "returns" not in df.columns:
            ret = ops.returns(close)
        else:
            ret = df["returns"]
        
        # 计算成交量比率
        adv = ops.adv(volume, 20)
        volume_ratio = volume / adv.replace(0, np.nan)
        
        # 计算因子值
        sign_ret = np.sign(ret)
        excess_vol = np.maximum(0, volume_ratio - 1)
        vol_rank = ops.ts_rank(volume_ratio, 20)
        
        factor_value = sign_ret * excess_vol * vol_rank
        
        # 处理 NaN
        factor_value = factor_value.fillna(0)
        
        return self.as_cs_series(df, factor_value)


@register
class MeanReversionAlpha(Factor):
    """
    均值回归因子
    
    当价格偏离均值过远时产生反转信号。
    """
    name = "MeanReversionAlpha"
    requires = ["close"]
    
    def compute(self, df):
        close = df["close"]
        
        # 计算多周期移动平均
        ma5 = ops.rolling_sum(close, 5) / 5
        ma20 = ops.rolling_sum(close, 20) / 20
        ma60 = ops.rolling_sum(close, 60) / 60
        
        # 计算偏离度
        dev5 = (close - ma5) / ma5
        dev20 = (close - ma20) / ma20
        dev60 = (close - ma60) / ma60
        
        # 组合：短期偏离权重更高
        factor_value = -2 * dev5 - dev20 - 0.5 * dev60
        
        # 截面标准化
        return self.as_cs_series(df, factor_value)
```

### 调试技巧

1. **查看中间结果**: 在 compute 方法中添加打印语句查看各步骤的值

```python
def compute(self, df):
    close = df["close"]
    print(f"Close shape: {close.shape}")
    print(f"Close sample:\n{close.head()}")
    
    ret = ops.returns(close)
    print(f"Returns NaN count: {ret.isna().sum()}")
    
    # ... rest of computation
```

2. **单股票测试**: 使用 `--stock` 参数限制计算范围

```bash
python -m alpha101_factory.cli factor --factors MyCustomAlpha --stock 600000
```

3. **检查输出**: 查看生成的 `factors/{factor_name}.jsonl` 文件

```bash
head data/factors/MyCustomAlpha.jsonl
```

### 最佳实践

1. **命名规范**: 因子名使用 PascalCase，如 `VolumeBreakout`、`MeanReversionAlpha`
2. **文档字符串**: 为每个因子添加清晰的 docstring，说明逻辑和公式
3. **依赖声明**: 准确列出 requires 字段，避免遗漏
4. **NaN 处理**: 确保输出不包含过多 NaN，必要时进行填充或掩码
5. **性能考虑**: 尽量使用向量化操作，避免 Python 循环

---

## 附录

### 常见错误及解决方案

**Q: 因子计算结果为全 NaN？**  
A: 检查依赖字段是否正确提供，确认数据完整性。

**Q: 运行时提示字段不存在？**  
A: 检查 `requires` 列表是否包含所有使用的字段。

**Q: 内存不足？**  
A: 尝试减少并行 worker 数 (`ALPHA101_MAX_WORKERS`) 或分批计算。

### 参考资料

- Kakushadzi, Z. (2016). "101 Formulaic Alphas". arXiv preprint arXiv:1601.06434.
- Alpha101 Factory README.md
- utils/ops.py 源码注释
