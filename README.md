# Alpha101 Factory

> 可插拔 Alpha101 因子工厂 — A 股日线量化研究工具

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 简介

基于 WorldQuant Alpha101 论文实现的可插拔因子工厂，提供从**数据获取 → 特征缓存 → 因子计算 → 回测评估 → 可视化**的完整量化研究流水线。

核心目标：让新人"拿来即用"，也能"随插随扩"。

---

## 特性

- **61 个 Alpha 因子**：覆盖 Alpha001~Alpha101 中已实现的因子
- **可插拔架构**：通过 `@register` 装饰器 + 自动发现机制扩展因子
- **双源数据**：AkShare（主）+ Baostock（备），自动容错
- **二段式计算**：中间特征缓存（tmp）避免重复计算
- **加速算子**：bottleneck / numba 可选加速，pandas 回退
- **完整回测**：IC/RankIC、分位数组合、多空组合
- **可视化**：K线图、因子时序、截面分布、热力图（Plotly）

---

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/1998x-stack/alpha101_factory.git
cd alpha101_factory

# 安装为开发包（推荐）
pip install -e .

# 或安装依赖
pip install pandas numpy akshare baostock pyarrow loguru tqdm plotly kaleido

# 可选：性能加速
pip install bottleneck numba
```

### 使用流程

```bash
# 1. 获取数据（全量或单只）
python -m alpha101_factory.cli fetch
python -m alpha101_factory.cli fetch-one --stock 600000 --start 20200101 --end 20240101 --adjust qfq

# 2. 生成中间特征（可缓存复用）
python -m alpha101_factory.cli tmp

# 3. 计算因子（单个或多个，或 --all）
python -m alpha101_factory.cli factor --factors Alpha001
python -m alpha101_factory.cli factor --all

# 4. 回测评估
python -m alpha101_factory.backtest.run_bt --alpha Alpha001 --horizon 1 --quantiles 5

# 5. 可视化
python -m alpha101_factory.cli visualize --all --prefix Alpha
```

### Python API

```python
from alpha101_factory.factors.registry import list_factors, get_factor
from alpha101_factory.data.loader import load_or_fetch_symbol
from alpha101_factory.backtest.metrics import ic_rankic, quantile_portfolios

# 列出所有可用因子
print(list_factors())  # ['Alpha001', 'Alpha003', ..., 'Alpha101']

# 获取并计算因子
fac = get_factor("Alpha001")()
result = fac.compute(df)  # df: 面板数据 DataFrame

# 回测评估
res = ic_rankic(factor_df, price_df, horizon=1)
print(res["summary"])
```

---

## 项目结构

```
alpha101_factory/
├── cli.py                   # CLI 入口 (fetch/tmp/factor/check/visualize)
├── config.py                # 配置 (路径/环境变量/日期范围)
│
├── data/                    # 数据层
│   ├── loader.py            # AkShare→Baostock 双源获取 + 增量更新
│   ├── baostock_api.py      # Baostock 备用接口 (连接复用)
│   └── universe.py          # 股票池定义
│
├── factors/                 # 因子层 (可插拔)
│   ├── base.py              # Factor ABC (name, requires, compute)
│   ├── registry.py          # @register + 自动发现 + get_factor()
│   ├── alphas_basic.py      # 61 个 Alpha101 因子实现
│   ├── alphas_more.py       # 扩展因子 (预留)
│   └── tmp_features.py      # 中间特征缓存 (returns/vwap/adv)
│
├── pipeline/                # 计算管线
│   ├── build_tmp.py         # 构建中间特征
│   ├── compute_factor.py    # 因子计算 → Parquet 输出
│   └── check_data.py        # 数据完整性检查
│
├── backtest/                # 回测评估
│   ├── metrics.py           # IC/RankIC + 分位数组合
│   └── run_bt.py            # 回测 CLI + 图表输出
│
├── utils/                   # 工具
│   ├── ops.py               # 加速算子 (bottleneck/numba)
│   ├── io.py                # Parquet 读写 (异常重抛)
│   └── log.py               # loguru 配置
│
└── viz/                     # 可视化
    ├── plots.py             # K线/因子时序/热力图 (Plotly)
    └── factor_summary.py    # 批量因子可视化

data/                        # 数据存储
├── klines_daily/            # ~496 只股票日线 Parquet
├── tmp_features/            # 中间特征缓存
├── factors/                 # 因子计算结果
├── spot/                    # A 股实时快照
└── index.json               # 数据索引
```

---

## 数据流

```
┌─────────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐    ┌─────────┐
│ AkShare │───▶│ normalize│───▶│ Parquet │───▶│ tmp     │───▶│ Factor  │
│ Baostock│    │   _k()   │    │  Cache  │    │ Features │    │ Compute │
└─────────┘    └──────────┘    └─────────┘    └─────────┘    └─────────┘
                                                                    │
                                                                    ▼
┌─────────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐    ┌─────────┐
│  PNG    │◀───│ Plotly   │◀───│  CSV   │◀───│ Backtest │◀───│ Parquet │
│  Images │    │  Viz     │    │ Results│    │ Metrics  │    │ Factors │
└─────────┘    └──────────┘    └─────────┘    └──────────┘    └─────────┘
```

---

## 扩展指南

### 新增因子

1. 在 `factors/` 目录新建 `alphas_*.py`
2. 继承 `Factor`，实现 `compute()` 方法
3. 用 `@register` 装饰器注册
4. 使用 `utils/ops` 的滚动/截面工具

```python
from alpha101_factory.factors.base import Factor
from alpha101_factory.factors.registry import register
from alpha101_factory.utils import ops

@register
class AlphaXXX(Factor):
    name = "AlphaXXX"
    requires = ["close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # 实现因子计算逻辑
        val = ...
        return Factor.as_cs_series(df, val)
```

5. 运行计算：
```bash
python -m alpha101_factory.cli factor --factors AlphaXXX
python -m alpha101_factory.backtest.run_bt --alpha AlphaXXX
```

### 更换数据源

1. 在 `data/` 新增 `XXX_api.py`，仿照 `baostock_api.py` 返回标准化 DataFrame
2. 在 `loader.py` 的 fallback 链中插入新数据源

### 自定义股票池

修改 `data/universe.py`，支持从指数成分、CSV、白名单等加载。

---

## 配置

通过环境变量覆盖默认配置：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ALPHA101_DATA_ROOT` | 数据根目录 | `./data` |
| `ALPHA101_ADJUST` | 复权方式 | `qfq` |
| `ALPHA101_START` | 起始日期 | `20200101` |
| `ALPHA101_END` | 结束日期 | `20250917` |
| `ALPHA101_LIMIT` | 限制股票数 (调试) | `0` (全量) |
| `ALPHA101_PAUSE` | 请求间隔 (秒) | `0.6` |

---

## 数据格式

### K 线数据 (`data/klines_daily/{symbol}.parquet`)

| 列名 | 类型 | 说明 |
|------|------|------|
| symbol | str | 股票代码 |
| datetime | datetime | 交易日期 |
| open | float | 开盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| close | float | 收盘价 |
| volume | float | 成交量 |
| amount | float | 成交额 |

### 因子结果 (`data/factors/{AlphaName}.parquet`)

| 列名 | 类型 | 说明 |
|------|------|------|
| datetime | datetime | 交易日期 |
| symbol | str | 股票代码 |
| value | float | 因子值 |

---

## 已知问题与修复记录

详见 [docs/gotchas.md](docs/gotchas.md) — 18 条经验教训沉淀

### 已修复 (18/18)

- [x] `write_parquet` 静默吞异常 → 重新抛出
- [x] `_g(df, None, ...)` 反模式 → 直接调用算子
- [x] Alpha053 分母公式错误 → 修正为 (high-low)
- [x] `pd.Series(val)` 丢失索引 → 保留 index
- [x] `cs_rank` 在 `_g` 内丢失 MultiIndex → 预计算为列
- [x] `sys.path.append` 污染 → `pip install -e .`
- [x] Baostock 连接复用 → 全局连接管理
- [x] 增量更新机制 → `update_kline_incremental()`
- [x] 数据质量检查 → `check_data_quality()`
- [x] 冗余'股票代码'列 → 自动清理

---

## 依赖

### 核心依赖

- `pandas>=1.3.0`
- `numpy>=1.21.0`
- `akshare>=1.13.0` (数据获取)
- `baostock>=0.8.8` (备用数据源)
- `pyarrow>=10.0.0` (Parquet 读写)
- `loguru>=0.7.0` (日志)
- `tqdm>=4.64.0` (进度条)
- `plotly>=5.13.0` (可视化)
- `kaleido>=0.2.1` (PNG 导出)

### 可选依赖

- `bottleneck>=1.3.5` (滚动计算加速)
- `numba>=0.57.0` (循环加速)

---

## 许可证

MIT

---

## 作者

[1998x-stack](https://github.com/1998x-stack)

---

## 参考

- [Alpha101 论文](https://arxiv.org/abs/1601.00991)
- [WorldQuant Alpha101 解析](https://github.com/yli188/WorldQuant_alpha101_code)
