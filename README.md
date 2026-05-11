# Alpha101 Factory

[![Python](https://img.shields.io/badge/python-3.8+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-00A858)](LICENSE)
[![Factors](https://img.shields.io/badge/factors-61-7C3AED)]()
[![Stocks](https://img.shields.io/badge/stocks-496-F59E0B)]()
[![Tests](https://img.shields.io/badge/tests-30%20passed-10B981)]()
[![Docs](https://img.shields.io/badge/docs-online-3B82F6)](https://1998x-stack.github.io/alpha101_factory/)

> **可插拔 Alpha101 因子工厂** — 从数据获取到因子回测的完整 A 股量化研究流水线

---

## 🚀 Overview

基于 WorldQuant [Alpha101](https://arxiv.org/abs/1601.00991) 论文实现的量化因子框架。提供 **数据获取 → 特征缓存 → 因子计算 → 回测评估 → 可视化** 的一站式流水线。

| 指标 | 数值 |
|------|------|
| 已实现因子 | **61** / 101 |
| 股票覆盖 | **496** 只 A 股 |
| 数据跨度 | 2020-01-02 ~ 2025-09-17 |
| 代码行数 | **3,200+** Python |
| 测试覆盖 | **30** 个用例 |
| 文档 | 18 条 Gotchas + HTML 文档站 |

---

## ✨ Key Features

### 🧬 因子引擎
- **61 个 Alpha 因子** — 覆盖动量、反转、量价、波动率等核心模式
- **可插拔注册** — `@register` 装饰器 + `pkgutil` 自动发现，零配置扩展
- **二段式计算** — tmp 中间特征缓存 (returns / vwap / advN)，避免重复计算
- **加速算子** — bottleneck / numba 可选加速，pandas 优雅回退

### 📊 数据层
- **双源容错** — AkShare (主) + Baostock (备)，自动切换
- **增量更新** — `update_kline_incremental()` 只获取最新数据
- **数据质量** — `check_data_quality()` 自动检查缺失/间隙/异常值
- **Parquet 存储** — 496 只股票仅 27.5 MB

### 📈 回测评估
- **IC / RankIC** — 横截面 Pearson/Spearman 相关性 + t 统计量
- **分位数组合** — 5 分位累乘收益 + 多空 (Long-Short) 组合
- **时间序列** — 单只股票 TS-IC / TS-RankIC 纵向评估
- **可视化** — Plotly 交互式图表 (IC曲线、收益对比、热力图)

### 🛠️ 工程品质
- **30 个测试** — ops / registry / factor / metrics / io / config 全覆盖
- **pyproject.toml** — 标准 Python 打包，`pip install -e .` 一键安装
- **18 条 Gotchas** — 从审查中沉淀的经验教训 ([docs/gotchas.md](docs/gotchas.md))

---

## 📦 Quick Start

### Install

```bash
git clone git@github.com:1998x-stack/alpha101_factory.git
cd alpha101_factory
pip install -e .                              # 开发模式安装

# 可选：性能加速
pip install bottleneck numba
```

### 5-Minute Workflow

```bash
# ① 获取数据
python -m alpha101_factory.cli fetch

# ② 构建中间特征
python -m alpha101_factory.cli tmp

# ③ 计算因子
python -m alpha101_factory.cli factor --factors Alpha001

# ④ 回测评估
python -m alpha101_factory.backtest.run_bt --alpha Alpha001 --horizon 1

# ⑤ 可视化
python -m alpha101_factory.cli visualize --all --prefix Alpha
```

### Python API

```python
from alpha101_factory.factors.registry import list_factors, get_factor
from alpha101_factory.backtest.metrics import ic_rankic

# 所有可用因子
print(list_factors())  # ['Alpha001', 'Alpha003', ..., 'Alpha101']

# 计算因子
fac = get_factor("Alpha001")()
result = fac.compute(df)  # df: 面板数据

# 回测
res = ic_rankic(factor_df, price_df, horizon=1)
print(res["summary"])
```

---

## 🏗️ Architecture

```
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐
│ AkShare │───▶│ normalize│───▶│ Parquet  │───▶│   tmp    │───▶│ Factor  │
│ Baostock│    │   _k()   │    │  Cache   │    │ Features │    │ Compute │
└─────────┘    └──────────┘    └──────────┘    └──────────┘    └────┬────┘
                                                                    │
                   ┌─────────┐    ┌──────────┐    ┌──────────┐    ▼
                   │  PNG    │◀───│  Plotly  │◀───│   CSV    │◀───│ Backtest
                   │  Images │    │   Viz    │    │ Results  │    │ Metrics
                   └─────────┘    └──────────┘    └──────────┘    └─────────┘
```

### Project Structure

```
alpha101_factory/
├── cli.py                    # CLI 入口 (fetch / tmp / factor / visualize)
├── config.py                 # 配置中心 (路径 / 环境变量 / 日期范围)
│
├── data/                     # 📥 数据层
│   ├── loader.py             #   AkShare→Baostock 双源 + 增量更新 + 质量检查
│   ├── baostock_api.py       #   备用数据源 (连接复用)
│   └── universe.py           #   股票池管理
│
├── factors/                  # 🧬 因子层 (可插拔)
│   ├── base.py               #   Factor ABC (name / requires / compute)
│   ├── registry.py           #   @register + 自动发现 + get_factor()
│   ├── alphas_basic.py       #   61 个 Alpha101 因子实现 (2,100 行)
│   └── tmp_features.py       #   中间特征缓存 (returns / vwap / advN)
│
├── pipeline/                 # 🔄 计算管线
│   ├── compute_factor.py     #   因子计算 → Parquet 输出
│   └── check_data.py         #   数据完整性检查
│
├── backtest/                 # 📈 回测评估
│   ├── metrics.py            #   IC / RankIC / 分位数组合 / 多空
│   └── run_bt.py             #   回测 CLI + 图表输出
│
├── utils/                    # 🛠️ 工具
│   ├── ops.py                #   加速算子 (bottleneck / numba / pandas)
│   ├── io.py                 #   Parquet 读写
│   └── log.py                #   loguru 配置
│
└── viz/                      # 🎨 可视化
    ├── plots.py              #   K线图 / 因子时序 / 热力图 (Plotly)
    └── factor_summary.py     #   批量因子可视化

data/                         # 💾 数据存储 (27.5 MB)
├── klines_daily/             #   496 只股票日线
├── tmp_features/             #   51 只中间特征
├── factors/                  #   因子计算结果
├── spot/                     #   A 股实时快照 (5,428 只)
└── index.json                #   数据索引
```

---

## 🧬 Factor System

### Adding a New Factor

```python
from alpha101_factory.factors.base import Factor
from alpha101_factory.factors.registry import register
from alpha101_factory.utils import ops

@register
class AlphaXXX(Factor):
    """Factor description."""
    name = "AlphaXXX"
    requires = ["close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        val = ...  # 计算逻辑
        return Factor.as_cs_series(df, val)
```

**That's it.** The factor is automatically discovered and available via:
```bash
python -m alpha101_factory.cli factor --factors AlphaXXX
```

### Factor Categories

| Category | Examples | Pattern |
|----------|----------|---------|
| **量价相关** | Alpha003, Alpha006, Alpha040 | rolling_corr(price, volume, N) |
| **动量/反转** | Alpha009, Alpha010, Alpha030 | delta / rolling_rank |
| **波动率** | Alpha001, Alpha023, Alpha034 | rolling_std / ts_rank |
| **量价比** | Alpha005, Alpha043, Alpha054 | vwap / adv / cs_rank |
| **截面排名** | Alpha004, Alpha011, Alpha035 | cs_rank + rolling |

---

## 📊 Backtest Metrics

### Information Coefficient

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **IC** | `Corr(factor, forward_return)` | 因子预测能力 |
| **RankIC** | `Spearman(factor, forward_return)` | 单调性评估 |
| **TS-IC** | 时间序列相关 | 单只股票纵向表现 |

### Quantile Portfolios

- **5 分位累乘收益** — 按因子值分 5 组，观察 monotonicity
- **Long-Short** — Q5 - Q1 多空组合收益
- **动态分组** — 自动处理 qcut 重复值导致分组减少的情况

---

## ⚙️ Configuration

Environment variables to override defaults:

| Variable | Description | Default |
|----------|-------------|---------|
| `ALPHA101_DATA_ROOT` | 数据根目录 | `./data` |
| `ALPHA101_ADJUST` | 复权方式 | `qfq` |
| `ALPHA101_START` | 起始日期 | `20200101` |
| `ALPHA101_END` | 结束日期 | `20250917` |
| `ALPHA101_LIMIT` | 限制股票数 (调试) | `0` (全量) |
| `ALPHA101_PAUSE` | 请求间隔 (秒) | `0.6` |

---

## 📋 Dependencies

### Required
```
pandas>=1.3.0    numpy>=1.21.0     pyarrow>=10.0.0
loguru>=0.7.0    tqdm>=4.64.0      plotly>=5.13.0
kaleido>=0.2.1
```

### Data Sources
| Source | Role | Install |
|--------|------|---------|
| **AkShare** | Primary | `pip install akshare` |
| **Baostock** | Fallback | `pip install baostock` |

### Optional (Performance)
| Package | Speedup | Install |
|---------|---------|---------|
| **bottleneck** | 3-5× rolling | `pip install bottleneck` |
| **numba** | 5-10× ts_rank/decay | `pip install numba` |

---

## 📚 Documentation

| Resource | Link |
|----------|------|
| **HTML Docs** | [1998x-stack.github.io/alpha101_factory](https://1998x-stack.github.io/alpha101_factory/) |
| **Gotchas** | [docs/gotchas.md](docs/gotchas.md) — 18 条经验教训 |
| **Project Analysis** | [PROJECT_ANALYSIS.md](PROJECT_ANALYSIS.md) — 完整代码分析 |
| **Refactor Plan** | [REFACTOR_PLAN.md](REFACTOR_PLAN.md) — 架构重构计划 |

---

## 🔧 Changelog

### v0.2.0 (2026-05-11)

**Features**
- 61 Alpha101 factors implemented
- Dual data source (AkShare + Baostock) with fallback
- Incremental update mechanism
- Data quality checking
- HTML documentation site with GitHub Pages

**Bug Fixes (18 total)**
- 🔴 `_cs_rank` returning constant 1.0 → delegated to `ops.cs_rank`
- 🔴 `_g(df, None, ...)` anti-pattern in ~25 factors → direct operator calls
- 🔴 Alpha053 denominator formula → corrected to `(high - low)`
- 🔴 `pd.Series(val)` losing index → `pd.Series(val, index=df.index)`
- 🔴 `write_parquet` swallowing exceptions → re-raise
- 🟡 `rolling_cov` ddof inconsistency → unified to `ddof=0`
- 🟡 `sys.path.append` pollution → `pip install -e .`
- And 11 more fixes...

**Engineering**
- `pyproject.toml` for standard packaging
- 30 unit tests (ops / registry / factor / metrics / io / config)
- GitHub Actions workflow for docs deployment

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-factor`)
3. Add your factor with `@register` decorator
4. Write tests in `tests/`
5. Push and open a Pull Request

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [WorldQuant Alpha101 Paper](https://arxiv.org/abs/1601.00991) — The original research
- [yli188/WorldQuant_alpha101_code](https://github.com/yli188/WorldQuant_alpha101_code) — Reference implementation

---

<div align="center">

**⭐ If this project helps you, give it a star!**

[Docs](https://1998x-stack.github.io/alpha101_factory/) · [Gotchas](docs/gotchas.md) · [Issues](https://github.com/1998x-stack/alpha101_factory/issues)

</div>
