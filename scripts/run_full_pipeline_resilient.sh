#!/usr/bin/env bash
# =============================================================================
# run_full_pipeline_resilient.sh — 弹性全流程执行脚本
#
# 按顺序执行: 数据抓取 → 特征构建 → 因子计算 → 数据校验 → 回测评估
# 增强错误处理和重试机制
#
# 用法:
#   ./scripts/run_full_pipeline_resilient.sh                   # 默认全流程
#   ./scripts/run_full_pipeline_resilient.sh --single 600000  # 单只股票全流程
#   ./scripts/run_full_pipeline_resilient.sh --all-factors    # 计算所有因子
#   ./scripts/run_full_pipeline_resilient.sh --skip-fetch     # 跳过抓取阶段
#   ./scripts/run_full_pipeline_resilient.sh --skip-backtest  # 跳过回测阶段
#
# 环境变量:
#   ALPHA101_DATA_ROOT  数据根目录 (默认 ./data)
#   ALPHA101_ADJUST     复权方式 (默认 qfq)
#   ALPHA101_START      起始日期 (默认 20200101)
#   ALPHA101_END        结束日期 (默认 20250917)
#   ALPHA101_LIMIT      限制股票数 (默认 0=全部)
#   MAX_RETRIES         最大重试次数 (默认 3)
#   RETRY_DELAY         重试延迟秒数 (默认 5)
# =============================================================================

set -euo pipefail

# 默认值
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_DELAY="${RETRY_DELAY:-5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ---------- 参数解析 ----------
SINGLE_STOCK=""
ALL_FACTORS=false
SKIP_FETCH=false
SKIP_TMP=false
SKIP_FACTOR=false
SKIP_CHECK=false
SKIP_BACKTEST=false
START_DATE="${ALPHA101_START:-20200101}"
END_DATE="${ALPHA101_END:-20250917}"
ADJUST="${ALPHA101_ADJUST:-qfq}"
HORIZON=1
QUANTILES=5

while [[ $# -gt 0 ]]; do
    case "$1" in
        --single)
            SINGLE_STOCK="$2"
            shift 2
            ;;
        --all-factors)
            ALL_FACTORS=true
            shift
            ;;
        --skip-fetch)
            SKIP_FETCH=true
            shift
            ;;
        --skip-tmp)
            SKIP_TMP=true
            shift
            ;;
        --skip-factor)
            SKIP_FACTOR=true
            shift
            ;;
        --skip-check)
            SKIP_CHECK=true
            shift
            ;;
        --skip-backtest)
            SKIP_BACKTEST=true
            shift
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
        --horizon)
            HORIZON="$2"
            shift 2
            ;;
        --quantiles)
            QUANTILES="$2"
            shift 2
            ;;
        -h|--help)
            head -16 "$0" | tail -14
            exit 0
            ;;
        *)
            echo "错误: 未知参数 '$1'" >&2
            exit 1
            ;;
    esac
done

# ---------- 重试函数 ----------
retry_command() {
    local cmd="$1"
    local desc="$2"
    local attempt=1
    local max_attempts=$MAX_RETRIES

    echo "  🔄 执行: $desc (最多 $max_attempts 次重试)"

    while [ $attempt -le $max_attempts ]; do
        echo "    尝试 $attempt/$max_attempts: $cmd"

        if eval "$cmd"; then
            echo "    ✓ $desc 执行成功"
            return 0
        else
            local exit_code=$?
            echo "    ⚠ $desc 第 $attempt 次尝试失败 (退出码: $exit_code)"

            if [ $attempt -lt $max_attempts ]; then
                echo "    等待 $RETRY_DELAY 秒后重试..."
                sleep $RETRY_DELAY
            fi

            ((attempt++))
        fi
    done

    echo "    ❌ $desc 执行失败，已达到最大重试次数 ($max_attempts)"
    return 1
}

# ---------- 计时 ----------
PIPELINE_START=$(date +%s)

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║       Alpha101 Factory — 弹性全流程执行             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  数据源:   AkShare → BaoStock (兜底)"
echo "  复权方式: ${ADJUST}"
echo "  日期范围: ${START_DATE} ~ ${END_DATE}"
echo "  重试次数: ${MAX_RETRIES} 次"
echo "  重试延迟: ${RETRY_DELAY} 秒"
if [[ -n "$SINGLE_STOCK" ]]; then
    echo "  目标股票: ${SINGLE_STOCK}"
else
    echo "  目标股票: 全量"
fi
echo ""

# ---------- 阶段 1: 数据抓取 ----------
if [[ "$SKIP_FETCH" == false ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ▶ 阶段 1/5: 数据抓取 (Fetch)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    FETCH_CMD=""
    if [[ -n "$SINGLE_STOCK" ]]; then
        FETCH_CMD="python -m alpha101_factory.cli fetch-one --stock \"$SINGLE_STOCK\" --start \"$START_DATE\" --end \"$END_DATE\" --adjust \"$ADJUST\""
    else
        FETCH_CMD="python -m alpha101_factory.cli fetch"
    fi

    if retry_command "$FETCH_CMD" "数据抓取"; then
        echo "  ✓ 数据抓取阶段完成"
    else
        echo "  ⚠ 数据抓取阶段失败，后续阶段可能受影响"
    fi
    echo ""
else
    echo "  ⏭ 跳过阶段 1: 数据抓取"
    echo ""
fi

# ---------- 阶段 2: 中间特征构建 ----------
if [[ "$SKIP_TMP" == false ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ▶ 阶段 2/5: 中间特征构建 (Build Tmp)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    TMP_CMD=""
    if [[ -n "$SINGLE_STOCK" ]]; then
        TMP_CMD="python -m alpha101_factory.cli tmp --stock \"$SINGLE_STOCK\""
    else
        TMP_CMD="python -m alpha101_factory.cli tmp"
    fi

    if retry_command "$TMP_CMD" "中间特征构建"; then
        echo "  ✓ 中间特征构建阶段完成"
    else
        echo "  ⚠ 中间特征构建阶段失败，后续阶段可能受影响"
    fi
    echo ""
else
    echo "  ⏭ 跳过阶段 2: 中间特征构建"
    echo ""
fi

# ---------- 阶段 3: 因子计算 ----------
if [[ "$SKIP_FACTOR" == false ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ▶ 阶段 3/5: 因子计算 (Compute Factor)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    FACTOR_ARGS=()
    if [[ "$ALL_FACTORS" == true ]]; then
        FACTOR_ARGS+=(--all)
    fi
    if [[ -n "$SINGLE_STOCK" ]]; then
        FACTOR_ARGS+=(--stock "$SINGLE_STOCK")
    fi

    FACTOR_CMD="python -m alpha101_factory.cli factor"
    if [[ "${#FACTOR_ARGS[@]}" -gt 0 ]]; then
        FACTOR_CMD="$FACTOR_CMD ${FACTOR_ARGS[*]}"
    else
        FACTOR_CMD="$FACTOR_CMD --factors Alpha101"
    fi

    if retry_command "$FACTOR_CMD" "因子计算"; then
        echo "  ✓ 因子计算阶段完成"
    else
        echo "  ⚠ 因子计算阶段失败，后续回测阶段将跳过"
        SKIP_BACKTEST=true
    fi
    echo ""
else
    echo "  ⏭ 跳过阶段 3: 因子计算"
    echo ""
fi

# ---------- 阶段 4: 数据完整性校验 ----------
if [[ "$SKIP_CHECK" == false ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ▶ 阶段 4/5: 数据完整性校验 (Check Data)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    CHECK_CMD="python -m alpha101_factory.cli check"

    if retry_command "$CHECK_CMD" "数据完整性校验"; then
        echo "  ✓ 数据完整性校验阶段完成"
    else
        echo "  ⚠ 数据完整性校验阶段失败，但这通常不影响主要功能"
    fi
    echo ""
else
    echo "  ⏭ 跳过阶段 4: 数据完整性校验"
    echo ""
fi

# ---------- 阶段 5: 回测评估 ----------
if [[ "$SKIP_BACKTEST" == false ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ▶ 阶段 5/5: 回测评估 (Backtest)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # 检查因子数据是否存在
    FACTOR_FILE="./data/factors/Alpha101.jsonl"
    if [[ ! -f "$FACTOR_FILE" ]] || [[ ! -s "$FACTOR_FILE" ]]; then
        echo "  ⚠ 因子数据文件不存在或为空: $FACTOR_FILE"
        echo "  ⏭ 跳过回测阶段"
    else
        BACKTEST_CMD="python -m alpha101_factory.backtest.run_bt --alpha Alpha101 --horizon $HORIZON --quantiles $QUANTILES"

        if retry_command "$BACKTEST_CMD" "回测评估"; then
            echo "  ✓ 回测评估阶段完成"
        else
            echo "  ⚠ 回测评估阶段失败"
        fi
    fi
    echo ""
else
    echo "  ⏭ 跳过阶段 5: 回测评估"
    echo ""
fi

# ---------- 汇总 ----------
PIPELINE_END=$(date +%s)
ELAPSED=$((PIPELINE_END - PIPELINE_START))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo "╔══════════════════════════════════════════════════════╗"
echo "║                 弹性流程执行完成                     ║"
echo "╠══════════════════════════════════════════════════════╣"
printf "║  耗时: %d 分 %d 秒                                    ║\n" "$MINUTES" "$SECONDS"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "💡 提示:"
echo "  - 如果仍有网络问题，可增加重试次数: MAX_RETRIES=5 $0"
echo "  - 可以单独运行各阶段进行调试"
echo "  - 检查 ./data/logs/ 目录中的详细日志"