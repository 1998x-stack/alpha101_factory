#!/usr/bin/env bash
# =============================================================================
# build_tmp.sh — 中间特征构建阶段
#
# 从 K 线数据计算中间特征（收益率、VWAP、ADV 等），缓存为 JSONL。
#
# 用法:
#   ./scripts/build_tmp.sh                  # 全量构建
#   ./scripts/build_tmp.sh --single 600000  # 单只股票
#
# 环境变量:
#   ALPHA101_DATA_ROOT  数据根目录 (默认 ./data)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ---------- 参数解析 ----------
SINGLE_STOCK=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --single)
            SINGLE_STOCK="$2"
            shift 2
            ;;
        -h|--help)
            head -10 "$0" | tail -8
            exit 0
            ;;
        *)
            echo "错误: 未知参数 '$1'" >&2
            exit 1
            ;;
    esac
done

# ---------- 执行 ----------
echo "=========================================="
echo "  阶段 2: 中间特征构建 (Build Tmp)"
echo "=========================================="

if [[ -n "$SINGLE_STOCK" ]]; then
    echo "  模式: 单只股票 (${SINGLE_STOCK})"
    echo "------------------------------------------"
    python -m alpha101_factory.cli tmp --stock "$SINGLE_STOCK"
else
    echo "  模式: 全量构建"
    echo "------------------------------------------"
    python -m alpha101_factory.cli tmp
fi

echo "=========================================="
echo "  特征构建阶段完成"
echo "=========================================="
