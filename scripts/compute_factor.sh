#!/usr/bin/env bash
# =============================================================================
# compute_factor.sh — 因子计算阶段
#
# 从面板数据计算 Alpha 因子值，保存为 JSONL。
#
# 用法:
#   ./scripts/compute_factor.sh                    # 默认因子 (Alpha101)
#   ./scripts/compute_factor.sh --all              # 所有已注册因子
#   ./scripts/compute_factor.sh --factors Alpha001 Alpha003
#   ./scripts/compute_factor.sh --single 600000    # 单只股票
#
# 环境变量:
#   ALPHA101_DATA_ROOT  数据根目录 (默认 ./data)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ---------- 参数解析 ----------
MODE="default"       # default | all | custom
FACTORS=()
SINGLE_STOCK=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            MODE="all"
            shift
            ;;
        --factors)
            MODE="custom"
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                FACTORS+=("$1")
                shift
            done
            ;;
        --single)
            SINGLE_STOCK="$2"
            shift 2
            ;;
        -h|--help)
            head -11 "$0" | tail -9
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
echo "  阶段 3: 因子计算 (Compute Factor)"
echo "=========================================="

# 构建命令参数
CMD_ARGS=()
case "$MODE" in
    all)
        CMD_ARGS+=(--all)
        echo "  模式: 所有已注册因子"
        ;;
    custom)
        CMD_ARGS+=(--factors "${FACTORS[@]}")
        echo "  模式: 自定义因子 (${FACTORS[*]})"
        ;;
    *)
        echo "  模式: 默认因子 (Alpha101)"
        ;;
esac

if [[ -n "$SINGLE_STOCK" ]]; then
    CMD_ARGS+=(--stock "$SINGLE_STOCK")
    echo "  范围: 单只股票 (${SINGLE_STOCK})"
else
    echo "  范围: 全量股票"
fi

echo "------------------------------------------"
python -m alpha101_factory.cli factor "${CMD_ARGS[@]}"

echo "=========================================="
echo "  因子计算阶段完成"
echo "=========================================="
