
---

## 全面修复记录 (2026-05-11)

### P0 — 核心崩溃修复

#### G019: log.py 缺少 import sys
- **文件**: `utils/log.py:30`
- **问题**: 使用 `sys.stderr` 但未导入 `sys`，导致所有日志初始化崩溃
- **修复**: 添加 `import sys`

#### G020: io.py 双重 raise 死代码
- **文件**: `utils/io.py:57-58`
- **问题**: 连续两个 `raise`，第二个是死代码
- **修复**: 删除多余的 `raise`

### P0 — 核心算法 bug

#### G021: `_cs()` 函数方向完全错误
- **文件**: `factors/alphas_complete.py:33-36`
- **问题**: `groupby(level=1)` 按**股票**分组做时间序列排名，但应该是按**日期**分组做截面排名
- **影响**: 90+ 因子值完全错误
- **修复**: `groupby(level=1)` → `groupby(level=0)`

#### G022: `_g()` 函数对齐脆弱
- **文件**: `factors/alphas_complete.py:19-27`
- **问题**: 通过 `pd.concat(results).reindex(m.index)` 对齐，但 concat 后的 RangeIndex 无法匹配 MultiIndex
- **修复**: 使用 `groupby.transform` 保留索引，再通过 `.values` 对齐

### P1 — 公式偏离修复

#### G023: Alpha009/010 完全偏离论文
- **问题**: 使用了完全不同的 `np.where` 趋势判断逻辑
- **修复**: 
  - Alpha009: `((0 < ts_min(delta(close, 1), 5)) * (-1)) + ((0 < ts_max(delta(close, 1), 5)) * 1)`
  - Alpha010: `rank(((0 < ts_min(delta(close, 1), 4))) * (-1)) + rank(((0 < ts_max(delta(close, 1), 4))) * 1)`

#### G024: scale() 截面标准化错误
- **影响因子**: Alpha028, Alpha029, Alpha100
- **问题**: `groupby(level=1)` 按股票做时间序列标准化
- **修复**: `groupby(level=0)` 按日期做截面标准化

### P2 — 工程规范修复

#### G025: __init__.py 公共 API 不完整
- **修复**: 导出 `list_factors`, `get_factor`

#### G026: CLI 引用未导入函数
- **修复**: 添加 `from alpha101_factory.data.loader import _fetch_kline_ak`

#### G027: alphas_more.py 空文件
- **修复**: 删除

#### G028: __future__ 导入位置错误
- **文件**: backtest/run_bt.py, viz/plots.py
- **修复**: 移到编码声明之后、docstring 之前

### 修复后验证

```
✅ 101/101 因子全部通过 (300天 × 5只股票)
✅ 30/30 单元测试全部通过
```

---

*最后更新: 2026-05-11 23:59*
