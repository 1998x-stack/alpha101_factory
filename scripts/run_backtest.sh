#!/usr/bin/env bash
# =============================================================================
# run_backtest.sh — 回测评估阶段
#
# 对指定因子执行 IC/RankIC 分析和分位组合回测，产出图表和 JSON 结果。
#
# 用法:
#   ./scripts/run_backtest.sh --alpha Alpha101
#   ./scripts/run_backtest.sh --alpha Alpha101 --horizon 5 --quantiles 10
#
# 环境变量:
#   ALPHA101_DATA_ROOT  数据根目录 (默认 ./data)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ---------- 参数解析 ----------
ALPHA=""
HORIZON=1
QUANTILES=5

while [[ $# -gt 0 ]]; do
    case "$1" in
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        --horizon)
            HORIZON="$2"
            shift 2
            ;;
        --quantiles)
            QUANTILES="$2"
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

# ---------- 参数校验 ----------
if [[ -z "$ALPHA" ]]; then
    echo "错误: 必须指定 --alpha 参数" >&2
    echo "用法: $0 --alpha <因子名称> [--horizon N] [--quantiles N]" >&2
    exit 1
fi

# ---------- 执行 ----------
echo "=========================================="
echo "  阶段 5: 回测评估 (Backtest)"
echo "=========================================="
echo "  因子: ${ALPHA}"
echo "  前瞻期: h=${HORIZON}"
echo "  分位数: q=${QUANTILES}"
echo "------------------------------------------"
python -m alpha101_factory.backtest.run_bt \
    --alpha "$ALPHA" \
    --horizon "$HORIZON" \
    --quantiles "$QUANTILES"

echo "=========================================="
echo "  回测阶段完成"
echo "=========================================="
