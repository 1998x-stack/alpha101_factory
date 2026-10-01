# Alpha101 Factory — 数据获取接口

## 架构概览

数据层采用**三级数据源回退架构**，确保数据获取的高可用性：

```
CLI (cli.py)
  └── fetch / fetch-one
        └── data/loader.py
              ├── AkShare (主源 — A 股官方数据)
              ├── Baostock (备源 — 免费 A 股数据)
              └── Yahoo Finance (全球备源 — 跨市场数据)
```

## 数据源

| 数据源 | 模块 | 角色 | 安装 |
|--------|------|------|------|
| **AkShare** | `data/loader.py` | 主数据源 — 最全面的 A 股数据 | `pip install akshare` |
| **Baostock** | `data/baostock_api.py` | 第一备源 — 连接复用优化 | `pip install baostock` |
| **Yahoo Finance** | `data/yfinance_api.py` | 第二备源 — 全球市场覆盖 | `pip install yfinance` |

### 回退链

```
_fetch_kline_fallback(symbol, start, end, adjust):
  1. 尝试 AkShare → 成功则返回
  2. 尝试 Baostock → 成功则返回
  3. 尝试 Yahoo Finance → 成功则返回
  4. 全部失败 → 返回空 DataFrame + 错误日志
```

---

## CLI 命令

### `fetch` — 批量下载全市场数据

```bash
python -m alpha101_factory.cli fetch
```

**流程：**
1. `fetch_spot()` — 获取 A 股实时行情快照（股票代码列表）
2. `fetch_klines_from_spot()` — 遍历快照中的每只股票，下载日线数据
3. 每只股票保存为独立 Parquet 文件：`{symbol}_{start}_{end}_{adjust}.parquet`
4. 自动绘制每只股票的 K 线图（Plotly → PNG）

**速率控制：** `REQUEST_PAUSE` (默认 0.6 秒) 防止触发 API 限流

### `fetch-one` — 单只股票下载

```bash
python -m alpha101_factory.cli fetch-one \
  --stock 600000 \
  --start 20200101 \
  --end 20240101 \
  --adjust qfq \
  --source auto
```

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `--stock` | 6 位股票代码 | `600000`, `000001` |
| `--start` | 起始日期 | `YYYYMMDD` |
| `--end` | 结束日期 | `YYYYMMDD` |
| `--adjust` | 复权方式 | `qfq`(前复权), `hfq`(后复权), `""`(不复权) |
| `--source` | 数据源 | `auto`, `akshare`, `baostock`, `yfinance` |

---

## 核心 API

### `load_or_fetch_symbol(symbol, start_date, end_date, adjust, save_image)`

加载或下载单只股票的 K 线数据。

**逻辑：**
1. 检查本地 Parquet 文件是否存在
2. 若存在 → 直接读取（可按日期范围过滤）
3. 若不存在 → 调用三级回退链获取 → 保存到 Parquet → 绘制 K 线图

```python
from alpha101_factory.data.loader import load_or_fetch_symbol, normalize_k
from alpha101_factory.config import ADJUST

# 加载单只股票
df = load_or_fetch_symbol("600519", "20230101", "20231231", ADJUST)
# df 包含列: [symbol, datetime, open, high, low, close, volume, amount]
```

### `fetch_spot(save=True)` → DataFrame

获取 A 股实时行情快照（股票代码列表）。优先读取本地缓存 `a_spot.parquet`。

### `fetch_klines_from_spot(spot)` → int

从快照批量下载所有股票的日线数据。自动跳过已下载的股票。

### `update_kline_incremental(symbol, adjust)` → DataFrame

增量更新单只股票数据。读取本地最新日期，只获取之后的新数据并追加。

### `check_klines_integrity()` → DataFrame

检查本地 K 线文件完整性，返回每只股票的存在性、行数、日期范围。

### `check_data_quality(df, symbol)` → dict

检查单只股票的 K 线数据质量：
- 缺失值统计（open/high/low/close/volume）
- 交易日间隙检查（>3 天为异常）
- 异常值检查（价格≤0、成交量为负）
- 综合质量评分（0-100）

---

## 数据格式

### 输入：原始数据 → 标准化

`normalize_k()` 函数将各数据源的不同格式统一为标准格式：

| 原始字段 | 标准化字段 | 类型 |
|----------|-----------|------|
| 日期 / date | `datetime` | `datetime64` |
| 开盘 / open | `open` | `float64` |
| 最高 / high | `high` | `float64` |
| 最低 / low | `low` | `float64` |
| 收盘 / close | `close` | `float64` |
| 成交量 / volume | `volume` | `float64` |
| 成交额 / amount | `amount` | `float64` |

### 存储格式

- **格式：** Apache Parquet（高压缩比，496 只股票仅 27.5 MB）
- **路径：** `{DATA_ROOT}/klines_daily/{symbol}_{start}_{end}_{adjust}.parquet`
- **日期范围：** 2020-01-02 ~ 2025-09-17

---

## 数据源细节

### AkShare (`data/loader.py`)

```python
def _fetch_kline_ak(symbol, start_date, end_date, adjust):
    symbol = ''.join(filter(str.isdigit, symbol))  # 去除前缀
    k = ak.stock_zh_a_hist(
        symbol=symbol,
        start_date=start_date,  # YYYYMMDD
        end_date=end_date,
        period="daily",
        adjust=adjust,          # qfq / hfq / ""
    )
    return normalize_k(k)       # 标准化列名和类型
```

**特点：**
- 最全面的 A 股数据源
- 支持前复权/后复权/不复权
- 包含涨跌幅、换手率等附加字段

### Baostock (`data/baostock_api.py`)

```python
def fetch_kline_bs(symbol, start_date, end_date, period="d", adjust="qfq"):
    # 代码映射: 6/9开头 → sh., 其他 → sz.
    code = bs_code(symbol)  # 如: "600000" → "sh.600000"
    
    # 连接复用: 全局 _bs_connected 标志
    _ensure_connected()
    
    rs = bs.query_history_k_data_plus(
        code=code,
        fields="date,open,high,low,close,volume,amount",
        start_date=start_fmt,   # YYYY-MM-DD
        end_date=end_fmt,
        frequency=period,
        adjustflag=_map_adjustflag(adjust),  # qfq→2, hfq→1, 其他→3
    )
```

**特点：**
- **连接复用：** 全局 `_bs_connected` 标志，批量下载时避免频繁 login/logout
- **上下文管理器：** `bs_connection()` 自动管理连接生命周期
- 代码映射：6/9 开头 → 上海，0/2/3/4 开头 → 深圳，8 开头 → 北交所

### Yahoo Finance (`data/yfinance_api.py`)

```python
def fetch_kline_yf(symbol, start_date, end_date, period="d", adjust="qfq"):
    ticker = to_yf_ticker(symbol)  # 如: "600519" → "600519.SS"
    
    # 代理处理: 临时移除代理环境变量（YF 直连）
    saved_env = _strip_proxy_env()
    
    # 重试机制: 最多 3 次，指数退避
    for attempt in range(1, 4):
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start, end=end, interval=interval)
        df = _normalize_df(hist, symbol, adjust)
    
    _restore_proxy_env(saved_env)
```

**特点：**
- 支持 A 股（`.SS`/`.SZ`/`.BJ`）、美股、港股等全球市场
- 自动检测并跳过代理（解决网络环境问题）
- 3 次重试 + 指数退避（应对 Rate Limit）
- 退市股票自动跳过

---

## 配置（环境变量）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ALPHA101_DATA_ROOT` | `./data` | 数据根目录 |
| `ALPHA101_ADJUST` | `qfq` | 复权方式 |
| `ALPHA101_START` | `20200101` | 数据起始日期 |
| `ALPHA101_END` | `20250917` | 数据结束日期 |
| `ALPHA101_LIMIT` | `0` | 调试用股票数量限制（0=全量） |
| `ALPHA101_PAUSE` | `0.6` | API 请求间隔（秒） |

---

## 关键 Gotchas

1. **代码前缀处理：** `_resolve_kline_path` 会自动剥离交易所前缀（`sh600000` → `600000`）。不要重复剥离。
2. **Parquet 文件名解析：** 文件名格式为 `{symbol}_{start}_{end}_{adjust}.parquet`。提取股票代码时使用 `p.stem.split("_")[0]`。
3. **Baostock 代码映射：** `bs_code()` 返回 `sh.XXXXXX` 或 `sz.XXXXXX`。6 和 9 开头 → 上海，其他 → 深圳。
4. **Yahoo Finance 代理：** `fetch_kline_yf` 会临时修改 `os.environ` 移除代理，操作完成后自动恢复。
5. **速率控制：** `fetch_klines_from_spot` 每只股票之间 `time.sleep(REQUEST_PAUSE)`，防止触发 API 限流。
6. **原子写入：** `write_parquet` 自动创建父目录，异常必须重新抛出（不吞没错误）。
