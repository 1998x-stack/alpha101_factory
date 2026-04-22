#!/usr/bin/env bash
# =============================================================================
# fetch.sh — 数据抓取阶段
#
# 从 AkShare（BaoStock 兜底）抓取 A 股日线数据，保存为 JSONL 格式。
#
# 用法:
#   ./scripts/fetch.sh                    # 全量抓取
#   ./scripts/fetch.sh --single 600000    # 单只股票
#   ./scripts/fetch.sh --single 600000 --start 20200101 --end 20240101
#
# 环境变量:
#   ALPHA101_DATA_ROOT  数据根目录 (默认 ./data)
#   ALPHA101_ADJUST     复权方式 (默认 qfq)
#   ALPHA101_START      起始日期 (默认 20200101)
#   ALPHA101_END        结束日期 (默认 20250917)
#   ALPHA101_LIMIT      限制股票数 (默认 0=全部)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ---------- 参数解析 ----------
SINGLE_STOCK=""
START_DATE="${ALPHA101_START:-20200101}"
END_DATE="${ALPHA101_END:-20250917}"
ADJUST="${ALPHA101_ADJUST:-qfq}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --single)
            SINGLE_STOCK="$2"
            shift 2
            ;;
        --start)
            START_DATE="$2"
            shift 2
            ;;
        --end)
            END_DATE="$2"
            shift 2
            ;;
        --adjust)
            ADJUST="$2"
            shift 2
            ;;
        -h|--help)
            head -12 "$0" | tail -10
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
echo "  阶段 1: 数据抓取 (Fetch)"
echo "=========================================="
echo "  数据源: AkShare → BaoStock (兜底)"
echo "  复权方式: ${ADJUST}"
echo "  日期范围: ${START_DATE} ~ ${END_DATE}"

if [[ -n "$SINGLE_STOCK" ]]; then
    echo "  模式: 单只股票 (${SINGLE_STOCK})"
    echo "------------------------------------------"
    python -m alpha101_factory.cli fetch-one \
        --stock "$SINGLE_STOCK" \
        --start "$START_DATE" \
        --end "$END_DATE" \
        --adjust "$ADJUST"
else
    LIMIT="${ALPHA101_LIMIT:-0}"
    if [[ "$LIMIT" -gt 0 ]]; then
        echo "  模式: 全量抓取 (调试模式: 限制 ${LIMIT} 只)"
    else
        echo "  模式: 全量抓取"
    fi
    echo "------------------------------------------"
    python -m alpha101_factory.cli fetch
fi

echo "=========================================="
echo "  抓取阶段完成"
echo "=========================================="
