# 项目总览：alpha101_factory

这是一个**可插拔（pluggable）Alpha 因子工厂**，围绕 A 股日线数据构建：
**数据抓取（AkShare→BaoStock 兜底） → 特征缓存（tmp features） → 因子计算（Alpha101 & 更多） → 评价回测（IC/RankIC、分位组合） → 可视化与产物落盘（Parquet/PNG/CSV）**。
核心目标：让新人“拿来即用”，也能“随插随扩”。

---

## 目录一览与职责

```
├── __init__.py
├── backtest                 # 回测与评估
│   ├── metrics.py           # IC/RankIC、前瞻收益、分位组合、健壮性处理
│   └── run_bt.py            # 回测 CLI（--alpha、--horizon、--quantiles），产出图表/CSV
├── cli.py                   # 顶层命令行：fetch / fetch-one / tmp / factor / check
├── config.py                # 目录/环境变量/采样/速率限制/图像路径
├── data                     # 数据获取与装载
│   ├── baostock_api.py      # BaoStock 封装（akshare 失败时兜底）
│   ├── loader.py            # 抓取Spot/K线、校验、读取本地或下载、保存K线图片
│   └── universe.py          # 股票池定义（可按需改造）
├── factors                  # 因子体系
│   ├── __init__.py          # 自动发现与导入因子模块（触发注册）
│   ├── alphas_basic.py      # 基础一批可实现的 Alpha *
│   ├── base.py              # Factor 抽象类与输出约定
│   ├── registry.py          # @register 装饰器 + 自动发现 + 列表/检索
│   └── tmp_features.py      # 中间变量构建：returns/vwap/adv 等并落盘
├── pipeline                 # 管线拼装
│   ├── build_tmp.py         # （可选）只做 tmp 的脚本壳
│   ├── check_data.py        # parquet 完整性检查工具
│   └── compute_factor.py    # 读取面板数据 → 计算指定因子 → 写 factors/*.parquet
├── utils                    # 通用工具
│   ├── io.py                # 读写 parquet、健壮封装
│   ├── log.py               # loguru 初始化
│   └── ops.py               # 加速算子：bottleneck/numba 的 rolling、ts_rank、decay_linear 等
└── viz
    └── plots.py             # Plotly 可视化（K线/因子时序/截面/热力）+ 保存PNG（kaleido）
```

---

## 数据流与产物

1. **fetch / fetch-one**

   * `data/loader.py`

     * 优先用 **AkShare** 获取；失败自动改用 **BaoStock**。
     * 生成：

       * `data/spot/a_spot.parquet`（行情快照）
       * `data/klines_daily/{symbol}.parquet`（日线：datetime/open/high/low/close/volume/amount…）
       * `images/klines/{symbol}_{start}_{end}_{adjust}.png`（自动保存K线）
   * `check_klines_integrity()` 校验文件是否齐全、行数、日期范围。

2. **tmp（中间变量）**

   * `factors/tmp_features.py`
   * 读取 K 线 → 按股票分组计算：`returns、vwap、advN（平均成交量）…`
   * 产出：`data/tmp_features/{symbol}.parquet`
   * 设计目的是**避免重复重算**，为所有因子复用。

3. **factor（因子计算）**

   * `pipeline/compute_factor.py` 组装面板（K线 ⨯ tmp）
   * `factors/base.py` 规定每个因子类 `compute(df)->Series[MultiIndex(datetime,symbol)]`
   * `factors/registry.py` + `factors/__init__.py`：

     * 通过 `@register` 注册因子；
     * 自动发现 `alphas_*.py`，CLI 能通过 `--factors Alpha101` 或 `--all` 调用。
   * 产出：`data/factors/{AlphaName}.parquet`（列：datetime/symbol/value）

4. **backtest（评估）**

   * `backtest/metrics.py`

     * **前瞻收益**：对 close 计算 `t→t+h` 的 forward return 并回对齐；
     * **IC / RankIC（横截面）**：每日按截面对齐 `factor vs fwd_ret`；不足 2 只股票自动跳过；
     * **TS-IC / TS-RankIC（纵向每只股票）**：当只有单一股票时也给出相关性摘要；
     * **分位组合**：每日按因子分桶（`qcut(duplicates='drop')`），少样本天自动跳过；
   * `backtest/run_bt.py`

     * CLI：`--alpha --horizon --quantiles`；
     * 生成图表：

       * `images/backtest/{alpha}_IC_RankIC_h{h}.png`
       * `images/backtest/{alpha}_ports_h{h}_q{q}.png`（累乘收益，含多空LS）
     * 生成表格：

       * `daily_ic.csv / summary.csv / ts_summary.csv / cumrets.csv`

---

## 关键设计点

### A. 配置与环境变量（`config.py`）

* `DATA_ROOT`（默认 `./data`）与各子目录：`spot / klines_daily / tmp_features / factors / images/...`
* `ADJUST`：`qfq/hfq/''`
* `START_DATE / END_DATE`：全局抓取范围（也可用 `fetch-one --start/--end` 临时覆盖）
* `LIMIT_STOCKS`：调试时限制股票数量
* `REQUEST_PAUSE`：抓取节流
* 图片目录：`images/klines`、`images/backtest`

### B. 数据获取（AkShare→BaoStock 兜底）

* `loader._fetch_kline_fallback()`：先 akshare，异常/空数据 → `baostock_api.fetch_kline_bs()`
* 读取本地 Parquet 时支持**日期裁剪**；无本地即下载。
* 每次加载或下载，都会**保存 K 线 PNG**（Plotly+Kaleido，x 轴强制 date 类型，格式 `%Y-%m-%d`，-45°）。

### C. 因子体系（可插拔）

* `factors/base.py`：

  * `Factor.name`、`Factor.requires`（依赖字段）
  * `compute(self, df) -> pd.Series(MultiIndex[datetime,symbol])`
* `factors/registry.py`：

  * `@register` 装饰器，`get_factor(name)` / `list_factors()`
  * 自动发现模块（无需手动改 registry，新增文件即可被加载）
* `utils/ops.py` 加速算子：

  * `bottleneck`：rolling min/max/mean/std
  * `numba`：`ts_rank`、`decay_linear`
  * 统一的 `rolling_corr/cov`、`delta/delay/returns/adv`、`cs_rank/cs_zscore`
* **tmp-因子二段式**：

  * 先算中间变量（可缓存），因子阶段直接组装—**节省时间**。

### D. 回测健壮性

* 单只股票或少样本场景：

  * CS-IC/RankIC 可能为 NaN，但**不会报错**；
  * 分位组合不足 2 桶则跳过；
  * 额外输出 **TS-IC/TS-RankIC** 让单标的也有可读性。
* `run_bt.py` 对分位构建**try/except**，任何异常只会跳过图表并保留可用 CSV。

---

## 典型使用流程（CLI）

```bash
# 1) 抓取数据（全量或由 ALPHA101_LIMIT 限制）
python -m alpha101_factory.cli fetch
# 或单只：读取本地或下载（并存PNG）
python -m alpha101_factory.cli fetch-one --stock 600000 --start 20200101 --end 20240101 --adjust qfq

# 2) 生成中间变量（可限定单只）
python -m alpha101_factory.cli tmp --stock 600000

# 3) 计算因子（单只或全量；或 --all）
python -m alpha101_factory.cli factor --factors Alpha101 --stock 600000

# 4) 回测评估（会自动从 factors/*.parquet + klines 读取）
python -m alpha101_factory.backtest.run_bt --alpha Alpha101 --horizon 1 --quantiles 5
```

> Colab 可用同样命令；注意 `kaleido` 负责把 Plotly figure 存为 PNG。

---

## 数据格式约定（Parquet）

* `spot/a_spot.parquet`：`[code,name,...]`
* `klines_daily/{symbol}.parquet`：
  `symbol, datetime, open, high, low, close, volume, amount, (可选 pct_change/change/amplitude/turnover)`
* `tmp_features/{symbol}.parquet`：
  必备：`symbol, datetime, returns, vwap, advN(若计算过) ...`
* `factors/{Alpha}.parquet`：
  `datetime, symbol, value`

---

## 扩展指南

### 新增因子

1. 在 `factors/` 新建或追加 `alphas_*.py`
2. 继承 `Factor`，实现 `compute()`，并用 `@register` 注册
3. 使用 `utils/ops` 的滚动/截面工具，尽量选整数窗、避免不可得数据（行业/市值）
4. 运行：

   ```bash
   python -m alpha101_factory.cli factor --factors YourAlpha
   python -m alpha101_factory.backtest.run_bt --alpha YourAlpha
   ```

### 更换/扩展数据源

* 在 `data/` 里新增 XXX_api.py，仿照 `baostock_api.py` 返回规范化 DataFrame；
* 在 `loader.py` 的 fallback 链中插入你的数据源。

### 自定义股票池

* 修改 `data/universe.py`（如从指数成分、CSV、白名单等加载）。

---

## 常见问题（FAQ）

* **为什么 IC/RankIC 全是 NaN？**
  你可能只计算了**单只**股票；CS-IC/RankIC需要**同一天至少两只**，请用更大股票池或查看 TS-IC/TS-RankIC。
* **分位组合报错或为空？**
  少样本天会被自动跳过；如果几乎每天都只有 1 只，组合会为空（不会崩）。
* **K 线横轴是 1.6e18？**
  已在 `viz/plots.py` 强制把任何输入类型转为 `datetime` 并设置 `xaxis.type='date'`，PNG 正常显示 `%Y-%m-%d` 且 -45°。
* **AkShare 拉不下来？**
  自动切 BaoStock；也可手动只跑 `fetch-one` 做单票验证。

---

## 依赖与性能

* **依赖**：`akshare、baostock、pandas、numpy、pyarrow/fastparquet、plotly+kaleido、loguru、tqdm、bottleneck、numba`
* **性能**：bottleneck 加速滚动统计，numba 加速 `ts_rank/decay_linear`；tmp 特征缓存避免重复计算。
* **可视化**：Plotly 交互、Kaleido 无头导出 PNG；所有图像统一落在 `images/`。

---