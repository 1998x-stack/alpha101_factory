#!/bin/bash
# =============================================================================
# setup_env.sh — 环境变量设置脚本
#
# 设置 alpha101_factory 项目的环境变量
# =============================================================================

# 设置数据根目录
export ALPHA101_DATA_ROOT="${ALPHA101_DATA_ROOT:-./data}"

# 设置复权方式 (qfq=前复权, hfq=后复权, ""=不复权)
export ALPHA101_ADJUST="${ALPHA101_ADJUST:-qfq}"

# 设置日期范围
export ALPHA101_START="${ALPHA101_START:-20200101}"
export ALPHA101_END="${ALPHA101_END:-20250917}"

# 设置调试参数
export ALPHA101_LIMIT="${ALPHA101_LIMIT:-0}"  # 0 = 全部股票，>0 = 限制数量

# 设置请求节流（秒）
export ALPHA101_PAUSE="${ALPHA101_PAUSE:-0.6}"

# 设置并行 worker 数量
export ALPHA101_MAX_WORKERS="${ALPHA101_MAX_WORKERS:-1}"

# 设置 BaoStock 用户凭证（可选，空值表示匿名访问）
export BAOSTOCK_USER_ID="${BAOSTOCK_USER_ID:-}"
export BAOSTOCK_PASSWORD="${BAOSTOCK_PASSWORD:-123456}"

# 设置代理配置（默认禁用代理）
export ALPHA101_NO_PROXY="${ALPHA101_NO_PROXY:-1}"

echo "Environment variables set:"
echo "  ALPHA101_DATA_ROOT  = $ALPHA101_DATA_ROOT"
echo "  ALPHA101_ADJUST     = $ALPHA101_ADJUST"
echo "  ALPHA101_START      = $ALPHA101_START"
echo "  ALPHA101_END        = $ALPHA101_END"
echo "  ALPHA101_LIMIT      = $ALPHA101_LIMIT"
echo "  ALPHA101_PAUSE      = $ALPHA101_PAUSE"
echo "  ALPHA101_MAX_WORKERS= $ALPHA101_MAX_WORKERS"
echo "  BAOSTOCK_USER_ID    = ${BAOSTOCK_USER_ID:+***hidden***}${BAOSTOCK_USER_ID:-(not set)}"
echo "  ALPHA101_NO_PROXY   = $ALPHA101_NO_PROXY"