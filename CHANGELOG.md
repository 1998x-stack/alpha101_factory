# Changelog

所有 notable 更改都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
本项目遵循 [语义化版本](https://semver.org/spec/v2.0.0.html)。

---

## [v0.3.0] - unreleased

### 新增 (Added)

- **因子工厂模式 (FactorFactory)**：
  - `factors/base.py`：新增 `Factor` 基类，支持 `validate()`, `compute()`, `compute_batch()`, `info()` 方法
  - `factors/registry.py`：实现 `FactorFactory` 类，通过 `pkgutil` 自动扫描 `alphas_*.py` 模块
  - 无需手动修改注册表，新增因子文件即被自动加载

- **数据源工厂模式 (DataSourceFactory)**：
  - `data/loader.py`：抽象 `DataSource` 接口，实现 `AkShareSource` 和 `BaoStockSource`
  - 优先使用 AkShare，失败自动切换至 BaoStock 作为 fallback
  - 支持通过 `register()` 扩展新的数据源实现

- **评估器工厂模式 (EvaluatorFactory)**：
  - `backtest/evaluators.py`：抽象 `Evaluator` 接口
  - 实现 `ICEvaluator`（横截面 IC/RankIC）和 `QuantileEvaluator`（分位组合分析）
  - 支持通过 `register_evaluator()` 注册自定义评估器

- **管线阶段工厂模式 (StageFactory)**：
  - `pipeline/stages.py`：抽象 `Stage` 接口
  - 实现 `FetchStage`, `TmpStage`, `CheckStage`, `FactorStage`, `BacktestStage`
  - 支持通过 `register_stage()` 注册自定义管线阶段

- **图表工厂模式 (ChartFactory)**：
  - `viz/charts.py`：抽象 `Chart` 接口
  - 实现 5 种图表类型：IC 时序图、RankIC 分布图、分位组合收益图、多空收益图、累积收益图
  - 支持通过 `register_chart()` 注册自定义图表

- **回测引擎 (BacktestEngine)**：
  - `backtest/engine.py`：实现完整回测流水线 `load → evaluate → plot → save`
  - 封装单只股票或全量股票的回测逻辑，统一输出结构

- **管线引擎 (PipelineEngine)**：
  - `pipeline/engine.py`：实现阶段编排 `add() → run() → run_full()`
  - 支持顺序执行 fetch → tmp → factor → backtest 全流程

- **各模块统一导出**：
  - `factors/__init__.py`：统一导出 `Factor`, `register`, `list_factors`, `FactorFactory`
  - `data/__init__.py`：统一导出 `DataSource`, `DataSourceFactory`, `AkShareSource`, `BaoStockSource`
  - `backtest/__init__.py`：统一导出 `Evaluator`, `EvaluatorFactory`, `ICEvaluator`, `QuantileEvaluator`, `BacktestEngine`
  - `pipeline/__init__.py`：统一导出 `Stage`, `StageFactory`, `PipelineEngine`
  - `viz/__init__.py`：统一导出 `Chart`, `ChartFactory`

### 变更 (Changed)

- `factors/base.py`：
  - 新增 `validate_requires()` 实例方法，验证依赖特征是否存在
  - 新增 `_cs_rank()` 实例方法，封装截面排名计算
  - 新增 `_g()` 实例方法，简化 MultiIndex 构建

- `factors/registry.py`：
  - 新增 `FactorFactory` 类，实现因子动态加载与注册机制

- `factors/alphas_basic.py`：
  - 从 2100 行精简至 660 行
  - 移除所有因子的 `try/except` 块，依赖上层错误处理
  - 移除每个因子的列检查逻辑，改为在 `validate_requires()` 中统一处理
  - 移除模块级 `_cs_rank()` 和 `_g()` 函数，改为 `Factor` 基类的实例方法

- `data/loader.py`：
  - 从 313 行精简至 144 行
  - 使用 `DataSourceFactory` 替代硬编码的 AkShare/BaoStock 调用
  - 移除 `_fetch_kline_ak()` 和 `_fetch_kline_fallback()` 内部函数

- `data/universe.py`：
  - 修复崩溃 bug：更新为 JSONL 路径，移除对已不存在的 `PARQ_DIR_SPOT` 和 `read_parquet()` 的引用

- `backtest/run_bt.py`：
  - 从 149 行精简至 15 行
  - 改用 `BacktestEngine` 封装数据加载、计算、绘图、保存逻辑

- `pipeline/compute_factor.py`：
  - 从 106 行精简至 24 行
  - 改用 `PipelineEngine` 和 `FactorStage`

- `pipeline/build_tmp.py`：
  - 从 54 行精简至 14 行
  - 改用 `PipelineEngine` 和 `TmpStage`

- `pipeline/check_data.py`：
  - 从 59 行精简至 12 行
  - 改用 `PipelineEngine` 和 `CheckStage`

- 所有 pipeline 脚本移除 docstrings 和模块级注释，保持简洁

### 移除 (Removed)

- 所有模块中的 `sys.path.append()` 样板代码
- `factors/alphas_basic.py` 中每个因子的 `try/except` 块
- `factors/alphas_basic.py` 中每个因子的列检查逻辑
- `factors/alphas_basic.py` 中的模块级 `_cs_rank()` 和 `_g()` 函数
- `data/loader.py` 中的 `_fetch_kline_ak()` 和 `_fetch_kline_fallback()` 内部函数
- `backtest/run_bt.py` 中的内联数据加载/计算/绘图/保存逻辑
- 所有 pipeline 脚本中的 docstrings 和模块级注释

### 修复 (Fixed)

- `data/universe.py`：修复崩溃 bug，该文件仍引用已不存在的 `PARQ_DIR_SPOT` 常量和 `read_parquet()` 函数
- `viz/plots.py`：修复 `save_fig()` 无法覆盖已有文件的 bug

---

## [v0.2.0] - unreleased

### 变更 (Changed)

- 将所有数据存储从 Parquet 格式迁移至 JSONL 格式（每行一个 JSON 对象）
- 重构数据目录布局：
  - `spot/` → `universe/` + `quotes/spot/`
  - `klines_daily/` → `quotes/daily/`
  - `tmp_features/` → `features/`
- 简化文件名，移除日期和复权后缀（例如 `{sym}_{START}_{END}_{ADJ}.parquet` → `{sym}.jsonl`）
- 回测输出改为按运行目录组织：`backtest/{alpha}_h{h}_q{q}/`，包含 `daily_ic.jsonl`, `cumrets.jsonl`, `summary.json`, `ts_summary.json`
- 重命名配置变量，从 `PARQ_DIR_*` 改为 `DIR_*`（例如 `PARQ_DIR_KLINES` → `DIR_QUOTES`）
- 更新 I/O API，从 `read_parquet()`/`write_parquet()` 改为 `read_jsonl()`/`write_jsonl()`，支持可选的 `_meta` 行

### 新增 (Added)

- JSONL 元数据支持，通过可选的第一行 `{"_meta": true, ...}` 存储文件级元数据（股票代码、复权方式、日期范围、行数）
- 将股票池与 dated snapshots 分离：`universe/stocks.jsonl`（股票池）vs `quotes/spot/spot_YYYYMMDD.jsonl`（时间戳行情快照）
- 在 `data/loader.py` 中添加 `kline_path(symbol)` 辅助函数
- 实现按运行目录组织的回测结果结构，便于结果追踪

### 移除 (Removed)

- 所有 Parquet 文件 I/O（核心流程不再需要 `pyarrow`/`fastparquet`）
- 数据文件名中的日期和复权后缀（元数据改由 `_meta` 行存储）
- 回测的扁平 CSV 输出（改为按运行目录的结构化组织）

---

## [v0.1.0] - 初始发布

### 新增 (Added)

- 发布可插拔的 A 股日线 Alpha 因子工厂
- 数据抓取管线，支持 AkShare 数据源和 BaoStock 备用方案
- 基于 Parquet 的 OHLCV 行情和中间特征存储
- 因子注册系统，支持自定义 Alpha 因子注册
- IC 和 RankIC 回测功能，支持分位组合分析
- Plotly 可视化，通过 Kaleido 导出 PNG 图片
- CLI 命令行接口，支持 fetch、tmp 特征生成、因子计算和回测
- 环境变量配置，支持数据路径、日期范围和并行度设置
