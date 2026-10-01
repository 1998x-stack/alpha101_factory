# alpha101_factory — A 股日线 Alpha 因子工厂

可插拔的 Alpha 因子计算与回测框架，支持 AkShare/BaoStock 数据源、JSONL 存储、IC/RankIC 评估与分位组合分析。

## 特性

- **多数据源**：优先 AkShare，失败自动切换 BaoStock
- **JSONL 存储**：所有数据以 JSONL 格式保存，兼容性好且易于调试
- **可插拔架构**：通过 Factory 模式实现因子、数据源、评估器、图表和 Pipeline 阶段的动态注册与扩展
- **中间特征缓存**：收益率、VWAP、ADV 等预计算避免重复开销
- **完整回测流程**：横截面 IC/RankIC、时间序列 IC、分位组合、多空收益
- **可视化输出**：Plotly 图表 + Kaleido PNG 导出
- **健壮性设计**：单股票场景自动降级为 TS-IC，少样本日自动跳过

## 架构

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐
│   fetch     │────▶│    tmp       │────▶│   factor    │────▶│  backtest   │
│ (AkShare)   │     │ (features)   │     │  (Alpha)    │     │  (IC/ports) │
└─────────────┘     └──────────────┘     └─────────────┘     └─────────────┘
      │                   │                    │                    │
      ▼                   ▼                    ▼                    ▼
quotes/daily/        features/            factors/           backtest/*/
{symbol}.jsonl       {symbol}.jsonl       {Alpha}.jsonl      daily_ic.jsonl
```

Pipeline 阶段由 `StageFactory` 管理，支持自定义阶段插入。每个阶段接收上下文字典 `ctx`，返回更新后的上下文。

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 四步工作流

```bash
# 1. 抓取全量股票 K 线数据
python -m alpha101_factory.cli fetch

# 2. 构建中间特征（可选，但推荐）
python -m alpha101_factory.cli tmp

# 3. 计算因子
python -m alpha101_factory.cli factor --factors Alpha101

# 4. 回测评估
python -m alpha101_factory.backtest.run_bt --alpha Alpha101 --horizon 1 --quantiles 5
```

单只股票测试：

```bash
python -m alpha101_factory.cli fetch-one --stock 600000 --start 20200101 --end 20240101 --adjust qfq
python -m alpha101_factory.cli factor --factors Alpha101 --stock 600000
python -m alpha101_factory.backtest.run_bt --alpha Alpha101 --horizon 1 --quantiles 5
```

## 命令行参考

| 命令 | 描述 | 关键参数 |
|------|------|----------|
| `fetch` | 抓取全量股票 K 线 | — |
| `fetch-one` | 抓取单只股票并保存 K 线 PNG | `--stock`, `--start`, `--end`, `--adjust` |
| `tmp` | 构建中间特征 | `--stock` |
| `factor` | 计算因子 | `--factors`, `--all`, `--stock` |
| `check` | 校验数据完整性 | — |
| `backtest.run_bt` | 回测评估 | `--alpha`, `--horizon`, `--quantiles` |

**fetch-one 参数说明：**
- `--stock`: 6 位股票代码，如 `600000`
- `--start`: 起始日期 `YYYYMMDD`
- `--end`: 结束日期 `YYYYMMDD`
- `--adjust`: 复权方式 `qfq` / `hfq` / `""`

**factor 参数说明：**
- `--factors`: 因子名称列表，默认 `[Alpha101]`
- `--all`: 计算所有已注册因子
- `--stock`: 限制单只股票计算

**backtest.run_bt 参数说明：**
- `--alpha`: 因子名称（对应 `factors/{alpha}.jsonl`）
- `--horizon`: 前瞻收益期，默认 `1`
- `--quantiles`: 分位数数量，默认 `5`

## 数据存储

### 目录结构

```
data/
├── universe/
│   └── stocks.jsonl              # 股票池：{"code": "600000", "name": "浦发银行"}
├── quotes/
│   ├── spot/
│   │   └── a_spot.jsonl          # 行情快照
│   └── daily/
│       ├── 600000.jsonl          # 单只股票 K 线
│       └── ...
├── features/
│   ├── 600000.jsonl              # 中间特征
│   └── ...
├── factors/
│   ├── Alpha101.jsonl            # 因子值
│   └── ...
├── backtest/
│   ├── Alpha101_h1_q5/           # 回测运行目录
│   │   ├── daily_ic.jsonl        # 每日 IC/RankIC
│   │   ├── cumrets.jsonl         # 累积收益
│   │   ├── summary.json          # 汇总统计
│   │   └── ts_summary.json       # 按股票 TS 统计
│   └── ...
├── images/
│   ├── klines/                   # K 线 PNG
│   └── backtest/                 # 回测图表 PNG
└── logs/
    └── *.log
```

### JSONL 格式示例

**K 线数据 (`quotes/daily/600000.jsonl`):**
```json
{"datetime":"2024-01-02","open":10.5,"high":10.8,"low":10.3,"close":10.6,"volume":1000000,"amount":10600000}
```

**中间特征 (`features/600000.jsonl`):**
```json
{"_meta":true,"symbol":"600000","rows":1234,"features":["returns","vwap","adv5",...]}
{"datetime":"2024-01-02","symbol":"600000","open":10.5,"high":10.8,"low":10.3,"close":10.6,"volume":1000000,"amount":10600000,"returns":0.012,"vwap":10.58,"adv5":980000}
```

**因子输出 (`factors/Alpha101.jsonl`):**
```json
{"datetime":"2024-01-02","symbol":"600000","value":0.85}
```

**回测结果 (`backtest/Alpha101_h1_q5/daily_ic.jsonl`):**
```json
{"datetime":"2024-01-02","IC":0.023,"RankIC":0.018,"N":450}
```

## 因子体系

### 添加新因子

1. 在 `alpha101_factory/factors/` 下创建或编辑 `alphas_*.py` 文件
2. 继承 `Factor` 基类，实现 `compute()` 方法：

```python
from alpha101_factory.factors.base import Factor
from alpha101_factory.factors.registry import register
from alpha101_factory.utils.ops import returns, decay_linear

@register
class MyAlpha(Factor):
    name = "MyAlpha"
    requires = ["close", "volume"]

    def compute(self, df):
        # df 为长表，索引包含 datetime 和 symbol
        close = df["close"]
        ret = returns(close)
        return self.as_cs_series(df, decay_linear(ret, 10))
```

3. 使用 `utils/ops.py` 提供的算子：
   - 滚动窗口：`rolling_sum`, `rolling_min`, `rolling_max`, `rolling_std`
   - 时间序列：`ts_rank`, `decay_linear`, `delay`, `delta`, `returns`
   - 截面计算：`cs_rank`, `cs_zscore`
   - 其他：`vwap_from_amount`, `adv`

4. 运行因子计算：
```bash
python -m alpha101_factory.cli factor --factors MyAlpha
```

### 因子注册机制

- `registry.py` 通过 `pkgutil` 自动扫描 `alphas_*.py` 模块
- 无需手动修改注册表，新增文件即被加载
- `list_factors()` 返回所有已注册因子名称

## 回测评估

### 核心指标

| 指标 | 说明 |
|------|------|
| **IC** | 横截面 Pearson 相关系数（因子值 vs 前瞻收益） |
| **RankIC** | 横截面 Spearman 相关系数 |
| **IC.t** | IC 的 t 统计量，衡量显著性 |
| **TS.IC** | 单只股票时间序列 IC |
| **Avg.N** | 日均样本股票数 |

### 单股票注意事项

- CS-IC/RankIC 需要至少两只股票同一天有数据
- 单股票场景会输出 NaN，请查看 **TS-IC/TS-RankIC**
- 分位组合在样本不足时自动跳过，不会报错

### 输出结构

每个回测运行生成独立目录 `backtest/{alpha}_h{h}_q{q}/`：

- `daily_ic.jsonl`: 每日 IC/RankIC 及样本数
- `summary.json`: 横截面 IC 汇总统计
- `ts_summary.json`: 时间序列 IC 汇总统计
- `cumrets.jsonl`: 分位组合累积收益

图表保存在 `images/backtest/`:
- `{alpha}_IC_RankIC_h{h}.png`
- `{alpha}_ports_h{h}_q{q}.png`

## 可插拔架构

本项目采用 Factory 模式实现高度可扩展的架构。所有核心组件均支持动态注册。

### FactorFactory（因子工厂）

位于 `factors/base.py` 和 `factors/registry.py`。

**添加自定义因子：**

```python
from alpha101_factory.factors.base import Factor
from alpha101_factory.factors.registry import register
from alpha101_factory.utils.ops import returns, decay_linear

@register
class MyCustomFactor(Factor):
    name = "MyCustomFactor"
    requires = ["close", "volume"]

    def compute(self, df):
        close = df["close"]
        ret = returns(close)
        vol = df["volume"]
        # 自定义逻辑：收益率加权成交量
        return self.as_cs_series(df, decay_linear(ret * vol, 5))
```

注册后自动被发现，无需修改任何配置文件。

### DataSourceFactory（数据源工厂）

位于 `data/factory.py`。

**添加自定义数据源：**

```python
from alpha101_factory.data.factory import DataSourceFactory
from alpha101_factory.data.sources import DataSource
import pandas as pd

class CustomDataSource(DataSource):
    def fetch_kline(self, symbol: str, start_date: str | None,
                    end_date: str | None, adjust: str) -> pd.DataFrame:
        # 从自定义 API 或数据库拉取数据
        # 返回规范化的 DataFrame（列：datetime, open, high, low, close, volume, amount, symbol）
        pass

    def fetch_spot(self) -> pd.DataFrame:
        # 返回全市场行情快照
        pass

# 注册数据源
DataSourceFactory.register("custom", CustomDataSource)
```

可用数据源列表：
```python
DataSourceFactory._fallback_order  # 默认 ["akshare", "baostock"]
```

### EvaluatorFactory（评估器工厂）

位于 `backtest/evaluators.py`。

**添加自定义评估器：**

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Type
from pathlib import Path
import pandas as pd
from alpha101_factory.backtest.evaluators import Evaluator

class SharpeEvaluator(Evaluator):
    name = "sharpe"

    def evaluate(self, factor_df: pd.DataFrame, price_df: pd.DataFrame,
                 horizon: int, **kwargs) -> Dict[str, pd.DataFrame]:
        # 计算夏普比率等自定义指标
        # 返回 Dict[str, pd.DataFrame]
        pass

    def save(self, results: Dict[str, pd.DataFrame], run_dir: Path) -> None:
        # 保存结果到 run_dir
        pass

# 注册评估器
from alpha101_factory.backtest.evaluators import register_evaluator
register_evaluator(SharpeEvaluator)
```

当前内置评估器：
- `ic`: IC/RankIC 评估器
- `quantile`: 分位组合评估器

### StageFactory（Pipeline 阶段工厂）

位于 `pipeline/stages.py`。

**添加自定义 Pipeline 阶段：**

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Type
from alpha101_factory.pipeline.stages import Stage, register_stage

class DataQualityStage(Stage):
    name = "quality"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        # 执行数据质量检查
        # 读取 ctx 中的上下文，写入检查结果
        ctx["quality_report"] = {...}
        return ctx

# 注册阶段
register_stage(DataQualityStage)
```

当前内置阶段：
- `fetch`: 抓取 K 线数据
- `tmp`: 构建中间特征
- `check`: 校验数据完整性
- `factor`: 计算因子
- `backtest`: 执行回测

### ChartFactory（图表工厂）

位于 `viz/charts.py`。

**添加自定义图表：**

```python
from abc import ABC, abstractmethod
from typing import Dict, Type, Optional
from pathlib import Path
import plotly.graph_objects as go
from alpha101_factory.viz.charts import Chart, register_chart
import pandas as pd

class CustomChart(Chart):
    name = "custom"

    def render(self, df: pd.DataFrame, **_) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["datetime"], y=df["value"]))
        fig.update_layout(title="Custom Chart")
        return fig

# 注册图表
register_chart(CustomChart)
```

当前内置图表：
- `kline`: K 线图
- `factor_ts`: 因子时间序列图
- `factor_cs`: 因子截面柱状图
- `factor_heatmap`: 因子热力图
- `kline_factor`: K 线 + 因子叠加图

## 配置

通过环境变量调整行为：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ALPHA101_DATA_ROOT` | `./data` | 数据根目录 |
| `ALPHA101_ADJUST` | `qfq` | 复权方式：`qfq` / `hfq` / `""` |
| `ALPHA101_START` | `20200101` | 全局抓取起始日期 |
| `ALPHA101_END` | `20250917` | 全局抓取结束日期 |
| `ALPHA101_LIMIT` | `0` | 调试时限制股票数（0=全部） |
| `ALPHA101_PAUSE` | `0.6` | 请求节流间隔（秒） |
| `ALPHA101_MAX_WORKERS` | `1` | 并行 worker 数 |

覆盖方式：
```bash
ALPHA101_START=20190101 python -m alpha101_factory.cli fetch
```

## 🧬 Factor System

### 新增数据源

1. 继承 `DataSource` 基类（位于 `data/sources.py`）
2. 实现 `fetch_kline()` 和 `fetch_spot()` 方法
3. 返回规范化的 DataFrame（列：`datetime, open, high, low, close, volume, amount, symbol`）
4. 通过 `DataSourceFactory.register()` 注册

### Factor Categories

修改 `data/universe.py`，实现从指数成分、CSV 或白名单加载股票列表。

### 添加自定义 Pipeline 阶段

1. 继承 `Stage` 基类（位于 `pipeline/stages.py`）
2. 实现 `run(ctx)` 方法，接收并返回上下文字典
3. 通过 `register_stage()` 装饰器注册

### 添加自定义评估器

1. 继承 `Evaluator` 基类（位于 `backtest/evaluators.py`）
2. 实现 `evaluate()` 和 `save()` 方法
3. 通过 `register_evaluator()` 装饰器注册

### 添加自定义图表

1. 继承 `Chart` 基类（位于 `viz/charts.py`）
2. 实现 `render()` 方法，返回 Plotly Figure
3. 通过 `register_chart()` 装饰器注册

### 性能优化

- 启用 bottleneck 加速滚动统计：`pip install bottleneck`
- 启用 numba 加速排名与加权：`pip install numba`
- 增加 `ALPHA101_MAX_WORKERS` 提升并发（注意 API 限流）

## 常见问题

**Q: IC/RankIC 全是 NaN？**  
A: 你可能只计算了单只股票。横截面 IC 需要同一天至少两只股票。请使用全量股票池或查看 TS-IC。

**Q: 分位组合为空或报错？**  
A: 样本不足的天会被自动跳过。如果每天只有 1 只股票，组合无法形成。这是正常行为。

**Q: 如何验证数据完整性？**  
A: 运行 `python -m alpha101_factory.cli check` 查看已保存 K 线的存在性与行数。

**Q: AkShare 拉取失败？**  
A: 系统会自动切换到 BaoStock。也可用 `fetch-one` 单独验证某只股票。

**Q: 如何添加新的因子？**  
A: 参考"因子体系"章节，创建 `alphas_custom.py` 并实现 `@register` 装饰的因子类。

**Q: 如何添加自定义评估器？**  
A: 参考"可插拔架构"章节，继承 `Evaluator` 并通过 `register_evaluator()` 注册。

## 依赖

- Python 3.11+
- `akshare>=1.13`
- `baostock`
- `pandas>=2.0`
- `numpy>=1.24`
- `pyarrow>=14.0`
- `fastparquet>=2024.2.0`
- `tqdm>=4.66`
- `loguru>=0.7`
- `plotly>=5.24`
- `kaleido>=0.2.1`
- `bottleneck>=1.3`（可选，加速）
- `numba>=0.59`（可选，加速）
- `scipy`

---

更多细节请参考：
- [AGENTS.md](AGENTS.md) — Agent 行为指南
- [CHANGELOG.md](CHANGELOG.md) — 版本更新记录
- [FACTORS.md](FACTORS.md) — 因子详细文档
