# Alpha101 Factory — 测试核心标准

## 测试架构

```
tests/
├── test_core.py           # 核心测试 (25 tests) — 主测试文件
├── test_loader_paths.py   # 路径解析测试 (2 tests)
└── test_factor_visuals.py # 可视化测试 (2 tests)
```

**总计：** 30 个测试，覆盖 6 大模块。

---

## 运行测试

```bash
cd alpha101_factory

# 安装测试依赖
pip install -e ".[dev]"

# 运行全部测试
pytest tests/ -v

# 运行特定模块
pytest tests/test_core.py -v
pytest tests/test_core.py::TestOps -v          # 仅算子测试
pytest tests/test_core.py::TestRegistry -v      # 仅注册表测试
pytest tests/test_core.py::TestMetrics -v       # 仅回测测试
```

**测试数据：** 全部使用**合成数据**（`make_panel_df()` / `make_price_df()`），不依赖真实数据。

---

## 测试覆盖矩阵

### test_core.py — 25 个测试

#### 1. TestOps (11 tests) — 算子正确性

测试 `utils/ops.py` 中所有加速算子的数值正确性：

| 测试 | 验证内容 |
|------|----------|
| `test_rolling_sum` | 滚动求和正确性（`[1..10]`, window=3） |
| `test_rolling_min_max` | 滚动最小/最大值 |
| `test_rolling_std` | 滚动标准差（ddof=0） |
| `test_delta` | 一阶差分 |
| `test_delay` | 滞后算子（NaN 填充） |
| `test_returns` | 收益率计算 |
| `test_ts_rank` | 时间序列百分位排名 |
| `test_decay_linear` | 线性衰减加权平均 |
| `test_decay_linear_nan_propagation` | **Gotcha 验证：** NaN 应严格传播 |
| `test_cs_rank` | 截面排名（MultiIndex，按日期分组） |
| `test_cs_rank_requires_multiindex` | **Gotcha 验证：** 非 MultiIndex 应抛出 ValueError |
| `test_rolling_cov_ddof0` | **Gotcha 验证：** rolling_cov 使用 ddof=0 |
| `test_adv` | 平均日成交量 |

#### 2. TestRegistry (5 tests) — 因子注册系统

| 测试 | 验证内容 |
|------|----------|
| `test_list_factors` | 至少注册了 60+ 因子，包含 Alpha001 和 Alpha101 |
| `test_get_factor` | 按名称获取因子类，验证有 `compute` 方法 |
| `test_get_factor_not_found` | 不存在的因子应抛出 `KeyError` |
| `test_factor_requires` | 因子正确声明 `requires` 列表 |
| `test_factor_instantiation` | 因子可以实例化 |

#### 3. TestFactorCompute (3 tests) — 因子计算正确性

使用合成面板数据（5 只股票 × 50 天）测试代表性因子：

| 测试 | 因子 | 验证内容 |
|------|------|----------|
| `test_alpha003_compute` | Alpha003 | 输出非空，MultiIndex |
| `test_alpha004_compute` | Alpha004 | 输出非空，MultiIndex |
| `test_alpha009_compute` | Alpha009 | 输出非空 |

#### 4. TestMetrics (5 tests) — 回测指标

| 测试 | 验证内容 |
|------|----------|
| `test_make_forward_return` | 前瞻收益计算（末值为 NaN） |
| `test_make_forward_return_horizon` | 多期前瞻收益（末 N 期为 NaN） |
| `test_ic_rankic` | IC/RankIC 返回 daily/summary/ts_summary |
| `test_quantile_portfolios` | 5 分位组合 + 多空组合 |
| `test_quantile_portfolios_few_groups` | **Gotcha 验证：** qcut 分组减少时不崩溃 |

#### 5. TestConfig (2 tests) — 配置

| 测试 | 验证内容 |
|------|----------|
| `test_config_paths` | 关键路径非空 |
| `test_config_defaults` | 复权方式合法，起始日期非空 |

#### 6. TestIO (2 tests) — 文件 IO

| 测试 | 验证内容 |
|------|----------|
| `test_read_parquet_not_found` | 不存在文件返回空 DataFrame |
| `test_write_parquet_raises_on_error` | **Gotcha 验证：** write 失败应抛出异常 |

### test_loader_paths.py — 2 个测试

| 测试 | 验证内容 |
|------|----------|
| `test_resolve_kline_path_full_range` | 带日期范围的路径生成（剥离交易所前缀） |
| `test_resolve_kline_path_without_range` | 无日期范围的回退路径 |

### test_factor_visuals.py — 2 个测试

| 测试 | 验证内容 |
|------|----------|
| `test_generate_factor_visuals_returns_figures` | 单因子可视化返回 Plotly Figures |
| `test_generate_all_factor_visuals_reads_parquet` | 批量可视化从 Parquet 读取 |

---

## 合成数据工厂

测试使用两个工厂函数生成不依赖外部数据的测试数据：

### `make_panel_df(n_symbols=5, n_days=100, seed=42)`

生成面板数据（宽表格式），包含列：
```
[symbol, datetime, open, high, low, close, volume, amount]
```

- 股票代码：`SH600000` ~ `SH600004`
- 日期：最近 N 个交易日
- 价格：几何随机游走（μ=0.0003, σ=0.015）
- 成交量：均匀分布（1e6 ~ 1e8）

### `make_price_df(n_symbols=5, n_days=100, seed=42)`

生成简化价格数据，仅包含：
```
[datetime, symbol, close]
```

---

## Gotcha 验证清单

测试文件中明确标记了以下从实际 bug 中沉淀的验证点：

| Gotcha | 测试 | 验证方式 |
|--------|------|----------|
| **cs_rank 需要 MultiIndex** | `test_cs_rank_requires_multiindex` | 传入普通 Series 应抛出 `ValueError` |
| **decay_linear NaN 传播** | `test_decay_linear_nan_propagation` | 窗口含 NaN 时结果应为 NaN |
| **rolling_cov ddof=0** | `test_rolling_cov_ddof0` | 与 `rolling_std(ddof=0)` 保持一致 |
| **write_parquet 异常传播** | `test_write_parquet_raises_on_error` | 写入失败应抛出异常 |
| **qcut 分组数减少** | `test_quantile_portfolios_few_groups` | 重复值导致分组减少时不崩溃 |

---

## 添加新测试的标准

### 何时添加测试

| 场景 | 测试类型 |
|------|----------|
| 新因子 | `TestFactorCompute` 中添加至少 1 个正确性测试 |
| 新算子 | `TestOps` 中添加数值正确性 + 边界条件测试 |
| Bug 修复 | 先添加复现测试（Gotcha 验证），再修复 |
| 新功能 | 对应测试类中添加至少 1 个测试 |

### 测试命名规范

```
test_{功能描述}_{边界条件}
```

示例：`test_cs_rank_requires_multiindex` — 测试 cs_rank 的输入要求

### 测试数据原则

- **始终使用合成数据**，不依赖 `data/` 目录中的真实 Parquet 文件
- 使用固定 `seed=42` 确保可重复性
- 对于需要 MultiIndex 的测试，手动构造索引

---

## 测试覆盖率缺口

当前测试覆盖的核心模块：

| 模块 | 代码行数 | 测试数 | 状态 |
|------|---------|--------|------|
| `utils/ops.py` | 276 | 12 | ✅ 良好 |
| `factors/registry.py` | 109 | 5 | ✅ 良好 |
| `backtest/metrics.py` | 220 | 5 | ✅ 基本覆盖 |
| `factors/alphas_complete.py` | 1565 | 3 | ⚠️ 仅 3 个因子 |
| `data/loader.py` | 459 | 0 | ❌ 无覆盖 |
| `pipeline/compute_factor.py` | 149 | 0 | ❌ 无覆盖 |
| `utils/io.py` | 53 | 2 | ✅ 基本覆盖 |
| `config.py` | 48 | 2 | ✅ 基本覆盖 |
| `data/baostock_api.py` | 176 | 0 | ❌ 无覆盖 |
| `viz/` | 485 | 2 | ⚠️ 仅基本覆盖 |

**优先补充：** 因子计算（核心业务逻辑）和回测指标（核心价值输出）。
