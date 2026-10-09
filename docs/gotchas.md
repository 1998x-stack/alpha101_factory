# Gotchas — Lessons Learned

> 沉淀从项目审查、修复和重构中发现的坑与经验教训
> 创建: 2026-05-11 | 项目: alpha101_factory + investment_portfolio

---

## 🔴 Critical

### G001: `write_parquet` 静默吞异常 — 数据丢失不知情
**项目**: alpha101_factory | **文件**: `utils/io.py`

```python
# ❌ 错误写法
def write_parquet(df, path):
    try:
        df.to_parquet(path)
    except Exception as e:
        logger.error(f"写入失败: {e}")  # 只记录日志，调用方不知情！
```

**根因**: `except` 块只记录日志不重新抛出，导致数据写入失败但调用方仍认为成功。

**修复**:
```python
# ✅ 正确写法
def write_parquet(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)  # 失败会自然抛出异常
```

**教训**: I/O 操作的异常必须让调用方知情，除非有明确的降级策略。

---

### G002: `_cs_rank` 截面排名结果恒为 1.0
**项目**: alpha101_factory | **文件**: `factors/alphas_basic.py`

```python
# ❌ 错误写法
idx = pd.MultiIndex.from_frame(df[["datetime", "symbol"]], names=["datetime", "symbol"])
s_indexed = pd.Series(s.values, index=idx)
return s_indexed.groupby(level=0).rank(pct=True)
# → level=0 是 (datetime, symbol) 元组，每组只有一行，rank 永远是 1.0
```

**根因**: `MultiIndex.from_frame` 创建的新索引中，`level=0` 是完整的 `(datetime, symbol)` 元组组合，而非仅 `datetime`。导致 `groupby(level=0)` 每个组只有一个元素，`rank(pct=True)` 永远返回 1.0。

**修复**: 使用已有的 `ops.cs_rank(s)`，它假设输入 Series 的 index 已经是 `(datetime, symbol)` MultiIndex，直接 `groupby(level=0).rank(pct=True)`。

**教训**: 
- 创建 MultiIndex 时要理解 level 的语义
- `groupby(level=N)` 分组的是该 level 的唯一值
- 用 `pd.MultiIndex.from_arrays` 比 `from_frame` 更直观

---

### G003: `_g(df, None, lambda *_: ...)` 反模式 — N 倍重复计算
**项目**: alpha101_factory | **文件**: `factors/alphas_basic.py` | **影响**: ~25 个因子

```python
# ❌ 错误写法
val = _g(df, None, lambda *_: ops.rolling_corr(df["close"], df["volume"], 10))
# → 对每个 symbol 执行完全相同的操作，返回 N 个相同的 Series
# → groupby.apply 将它们堆叠，产生 N 倍数据
```

**根因**: 当 `col=None` 时，`_g` 对每个 symbol 分组执行 lambda，但 lambda 完全忽略分组参数 `s`，直接操作全局 `df`。结果每个分组返回完全相同的全局 Series，`groupby.apply` 将它们堆叠 N 次。

**修复**:
```python
# ✅ 正确写法 — 直接调用算子
val = ops.rolling_corr(df["close"], df["volume"], 10)

# ✅ 正确写法 — 如果需要截面排名
rank_close = ops.cs_rank(df.set_index(["datetime","symbol"])["close"])
df = df.copy()
df["_rank_close"] = rank_close.values
val = _g(df, "_rank_close", lambda s: ops.rolling_corr(s, ...))
```

**教训**:
- `_g` 只用于**需要按股票分组计算**的场景（如滚动窗口）
- 截面操作（cs_rank）应该在分组前完成
- lambda 如果忽略 `s` 参数，说明不需要 `_g`

---

### G004: Alpha053 分母公式错误
**项目**: alpha101_factory | **文件**: `factors/alphas_basic.py`

```python
# ❌ 错误: 分母用 (close - low)
x = ((close - low) - (high - close)) / (close - low)

# ✅ 正确: 分母应为 (high - low) — 全天价格区间
x = ((close - low) - (high - close)) / (high - low)
```

**根因**: 直接翻译代码时未理解公式的物理含义。`(high - low)` 是价格振幅，`(close - low)` 只是收盘相对低点的偏移。

**教训**: 实现论文公式时，必须理解每个符号的物理/金融含义，不能盲目照搬。

---

### G005: `pd.Series(val)` 丢失索引 — 因子值可能错位
**项目**: alpha101_factory | **文件**: `factors/alphas_basic.py` | **影响**: 9 个因子

```python
# ❌ 错误写法
val = np.where(cond1, d1, np.where(cond2, d1, -d1))
return Factor.as_cs_series(df, pd.Series(val))
# → pd.Series(val) 创建 RangeIndex，而非原有的 MultiIndex
# → as_cs_series 通过位置对齐重建索引，极其脆弱
```

**修复**:
```python
# ✅ 正确写法
return Factor.as_cs_series(df, pd.Series(val, index=df.index))
```

**教训**: 从 numpy array 创建 Series 时，**永远显式传递 index**。

---

### G006: `cs_rank` 在 `_g` 内部调用时丢失 MultiIndex
**项目**: alpha101_factory | **文件**: `factors/alphas_basic.py` | **影响**: 6 个因子

```python
# ❌ 错误写法
_g(df, "open", lambda s: ops.rolling_corr(ops.cs_rank(s), ...))
# → s 是 per-symbol Series（RangeIndex），cs_rank 需要 MultiIndex
```

**根因**: `ops.cs_rank(s)` 要求 `s` 的 index 是 `(datetime, symbol)` MultiIndex，但 `_g` 传入的 `s` 是单只股票的数据（RangeIndex）。

**修复**:
```python
# ✅ 正确写法: 预计算 cs_rank 并添加为 df 列
df = df.copy()
df["_rank_open"] = ops.cs_rank(df.set_index(["datetime","symbol"])["open"]).values
val = _g(df, "_rank_open", lambda s: ops.rolling_corr(s, ...))
```

**教训**: 
- 截面操作和时序操作不能混在同一层
- 先做截面操作（全面板），再做时序操作（per-symbol）

---

## 🟡 Warning

### G007: `sys.path.append` 污染 14 个文件
**项目**: alpha101_factory

几乎每个文件开头都有：
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
```

**根因**: 项目有正确的包结构（`alpha101_factory/__init__.py`），但从未安装为包，导致各文件自己添加路径。

**修复**: 
1. 创建 `pyproject.toml`
2. `pip install -e .`
3. 删除所有 `sys.path.append`

**教训**: 有包结构就**应该安装包**，不要手动操作 `sys.path`。

---

### G008: `config.py` 导入时创建目录（副作用）
**项目**: alpha101_factory

```python
# ❌ 在模块顶层执行
for p in [data_dir, img_dir, ...]:
    p.mkdir(parents=True, exist_ok=True)
```

**根因**: 导入配置时即创建目录。如果 `DATA_ROOT` 指向无权限路径，`import config` 会抛出 `OSError`，整个包不可用。

**教训**: 模块导入应无副作用。目录创建应延迟到首次实际需要时。

---

### G009: `rolling_cov` 与 `rolling_std` 的 ddof 不一致
**项目**: alpha101_factory | **文件**: `utils/ops.py`

```python
# rolling_std 使用 ddof=0
def rolling_std(s, n):
    return s.rolling(n).std(ddof=0)  # 总体标准差

# rolling_cov 使用默认 ddof=1
def rolling_cov(s1, s2, n):
    return s1.rolling(n).cov(s2)  # 样本协方差 ← 不一致！
```

**修复**: `rolling_cov` 也加 `ddof=0`。

**教训**: 统计函数族应保持一致的 ddof 设置，文档应明确说明。

---

### G010: `decay_linear` pandas 回退与 numba 版本 NaN 处理不同
**项目**: alpha101_factory | **文件**: `utils/ops.py`

```python
# numba 版本: 窗口内任何 NaN → 返回 NaN
for j in range(i-n+1, i+1):
    if np.isnan(arr[j]): nanhit = True; break
out[i] = np.nan if nanhit else acc

# pandas 回退: min_periods=n 只要求 n 个非 NaN 就计算
return s.rolling(n, min_periods=n).apply(lambda x: np.dot(x, w), raw=True)
# → 如果 n 个值中有 NaN，np.dot 会传播 NaN，但 min_periods 的语义不同
```

**修复**: pandas 回退也严格检查窗口内无 NaN：
```python
return s.rolling(n, min_periods=n).apply(
    lambda x: np.nan if np.any(np.isnan(x)) else np.dot(x, w), raw=True)
```

**教训**: 有回退实现时，**必须保证行为一致**，尤其 NaN 处理。

---

### G011: `quantile_portfolios` LS 组合硬编码 Q{q}
**项目**: alpha101_factory | **文件**: `backtest/metrics.py`

```python
# ❌ 硬编码
if "Q1" in port_pivot.columns and f"Q{q}" in port_pivot.columns:
    ls = port_pivot[f"Q{q}"] - port_pivot["Q1"]
```

**根因**: `pd.qcut(duplicates="drop")` 可能因重复值减少实际分组数。当 q=5 但实际只有 3 组时，Q5 不存在，LS 被静默跳过。

**修复**:
```python
# ✅ 动态使用实际最高/最低组
if len(port_pivot.columns) >= 2:
    ls = port_pivot.iloc[:, -1] - port_pivot.iloc[:, 0]
```

---

### G012: Python 3.10+ 语法在 3.8 上报错
**项目**: alpha101_factory

```python
# ❌ Python 3.10+ 语法
def func(x: str | None) -> list[str]: ...

# ✅ Python 3.8 兼容写法
from __future__ import annotations
def func(x: str | None) -> "list[str]": ...
```

**教训**: 使用新语法时加 `from __future__ import annotations`，或在 `pyproject.toml` 中声明 `requires-python >= 3.10`。

---

### G013: Baostock 每次 login/logout
**项目**: alpha101_factory | **文件**: `data/baostock_api.py`

```python
# ❌ 每次调用都建立新连接
def fetch_kline_bs(...):
    lg = bs.login()
    ...
    finally:
        bs.logout()
```

**修复**: 全局连接状态管理：
```python
_bs_connected = False

def _ensure_connected():
    global _bs_connected
    if not _bs_connected:
        bs.login()
        _bs_connected = True
    return _bs_connected
```

**教训**: 外部 API 连接应复用，尤其批量调用时。

---

## 💡 Suggestion

### G014: `requires` 声明不准确
**项目**: alpha101_factory

多个因子的 `requires` 列与实际使用不一致：
- Alpha043: requires `["volume","close","returns","vwap"]`，实际只用 `volume, close`
- Alpha065: requires `["open","vwap","low"]`，实际用 `open, vwap, volume`

**教训**: `requires` 应精确声明，避免因子计算时加载不必要的数据。

---

### G015: `_load_join` outer merge 冗余
**项目**: alpha101_factory | **文件**: `pipeline/compute_factor.py`

```python
# ❌ 用 8 列做 outer merge
m = pd.merge(k, t, on=["symbol", "datetime", "open", "high", "low", "close", "volume", "amount"], how="outer")
```

**修复**: `inner merge on ["symbol", "datetime"]`。

---

### G016: `fillna(0)` 累计收益失真
**项目**: alpha101_factory | **文件**: `backtest/run_bt.py`

```python
# ❌ NaN 表示缺失，不是零收益
cum = (1 + port_df.fillna(0)).cumprod()
```

**修复**: 不 fillna，或明确标记缺失。

---

## 🔧 系统级 Gotchas

### G017: DNS 解析失败但网络正常
**系统**: Ubuntu 20.04 | **症状**: `ssh github.com` 失败，`curl` 直连 IP 成功

**根因**: `systemd-resolved` 的上游 DNS 只配了路由器 `192.168.2.1`，但路由器 DNS 不响应。`/etc/resolv.conf` 指向 `127.0.0.53`（stub resolver），无法解析。

**修复**:
```bash
# 方案1: SSH config 用 IP 替代
cat >> ~/.ssh/config << EOF
Host github.com
    HostName 20.205.243.166
EOF

# 方案2: 修改 systemd-resolved 配置
sudo vim /etc/systemd/resolved.conf
# 添加: DNS=8.8.8.8 114.114.114.114
sudo systemctl restart systemd-resolved
```

**教训**: 网络通 ≠ DNS 通。DNS 问题优先排查 `systemd-resolved` 和 `/etc/resolv.conf`。

---

### G018: akshare 在 Python 3.8 上依赖冲突
**系统**: Python 3.8.10 | **症状**: `pip install akshare` 报错 `ResolutionImpossible`

**根因**: akshare 新版本依赖的包需要 Python 3.9+，与 3.8 不兼容。

**修复**: 
- 方案1: 升级 Python 到 3.10+
- 方案2: 锁定 akshare 旧版本 `pip install akshare==1.13.0`

**教训**: Python 3.8 已进入 end-of-life，新项目应使用 3.10+。

---

## 📊 统计

| 严重级别 | 数量 |
|----------|------|
| 🔴 Critical | 6 |
| 🟡 Warning | 7 |
| 💡 Suggestion | 3 |
| 🔧 系统级 | 2 |
| **总计** | **18** |

---

*最后更新: 2026-05-11*
