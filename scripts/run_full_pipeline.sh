#!/usr/bin/env bash
# =============================================================================
# run_full_pipeline.sh — 全流程执行脚本
#
# 按顺序执行: 数据抓取 → 特征构建 → 因子计算 → 数据校验 → 回测评估
#
# 用法:
#   ./scripts/run_full_pipeline.sh                          # 默认全流程
#   ./scripts/run_full_pipeline.sh --single 600000          # 单只股票全流程
#   ./scripts/run_full_pipeline.sh --all-factors            # 计算所有因子
#   ./scripts/run_full_pipeline.sh --skip-fetch             # 跳过抓取阶段
#   ./scripts/run_full_pipeline.sh --skip-backtest          # 跳过回测阶段
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

# ---------- 计时 ----------
PIPELINE_START=$(date +%s)

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║          Alpha101 Factory — 全流程执行               ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  数据源:   AkShare → BaoStock (兜底)"
echo "  复权方式: ${ADJUST}"
echo "  日期范围: ${START_DATE} ~ ${END_DATE}"
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
    if [[ -n "$SINGLE_STOCK" ]]; then
        python -m alpha101_factory.cli fetch-one \
            --stock "$SINGLE_STOCK" \
            --start "$START_DATE" \
            --end "$END_DATE" \
            --adjust "$ADJUST"
    else
        python -m alpha101_factory.cli fetch
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
    if [[ -n "$SINGLE_STOCK" ]]; then
        python -m alpha101_factory.cli tmp --stock "$SINGLE_STOCK"
    else
        python -m alpha101_factory.cli tmp
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
    python -m alpha101_factory.cli factor "${FACTOR_ARGS[@]:---factors Alpha101}"
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
    python -m alpha101_factory.cli check
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
    # 回测默认使用 Alpha101，如果计算了所有因子则回测第一个
    BACKTEST_ALPHA="Alpha101"
    python -m alpha101_factory.backtest.run_bt \
        --alpha "$BACKTEST_ALPHA" \
        --horizon "$HORIZON" \
        --quantiles "$QUANTILES"
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
echo "║                  全流程执行完成                       ║"
echo "╠══════════════════════════════════════════════════════╣"
printf "║  耗时: %d 分 %d 秒                                       ║\n" "$MINUTES" "$SECONDS"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
