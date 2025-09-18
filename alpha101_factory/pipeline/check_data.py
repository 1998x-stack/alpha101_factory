# -*- coding: utf-8 -*-
"""
K线数据完整性检查脚本

本脚本功能：
1. 调用 `check_klines_integrity` 检查所有股票 K线数据文件的存在与行数；
2. 打印检查结果的概要统计（正常文件数量、缺失或空文件数量）；
3. 将完整的检查报告保存至日志目录 `klines_integrity.csv`。

适用于数据预处理与质量监控，确保回测因子所需的 K线数据完整可靠。
"""

import sys
from pathlib import Path
from loguru import logger

# 确保项目根目录加入 sys.path，便于模块导入
try:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
except Exception as e:
    print(f"[警告] 无法设置 sys.path: {e}")

# 项目内部依赖
from alpha101_factory.utils.log import setup_logger
from alpha101_factory.data.loader import check_klines_integrity
from alpha101_factory.config import LOG_DIR


def main() -> None:
    """执行 K线数据完整性检查，并输出与保存报告。"""
    # 初始化日志
    setup_logger()

    try:
        rpt = check_klines_integrity()
    except Exception as e:
        logger.error(f"运行 K线完整性检查失败: {e}")
        return

    if rpt.empty:
        logger.warning("检查报告为空，可能未找到任何 K线数据。")
        return

    try:
        # ===== 打印检查摘要 =====
        ok = rpt[rpt["exists"] & (rpt["rows"] > 0)]      # 正常文件
        miss = rpt[~rpt["exists"] | (rpt["rows"] <= 0)]  # 缺失或空文件
        logger.info(f"正常文件: {len(ok)} | 缺失/空文件: {len(miss)}")

        # ===== 保存完整报告 =====
        out_path = LOG_DIR / "klines_integrity.csv"
        rpt.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"完整报告已保存至: {out_path}")
    except Exception as e:
        logger.error(f"处理或保存报告失败: {e}")


if __name__ == "__main__":
    main()
