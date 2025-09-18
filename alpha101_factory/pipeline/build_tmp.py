# -*- coding: utf-8 -*-
"""
构建临时特征文件 (tmp features) 脚本

本脚本功能：
1. 加载股票池（universe），即需要处理的所有股票代码；
2. 调用 `build_tmp_all` 为每只股票构建中间特征文件；
3. 提供日志输出，记录构建进度与结果。

适用于因子计算的前置步骤，确保数据输入完整可靠。
"""

import sys
from pathlib import Path
from loguru import logger

# 确保项目根目录加入 sys.path，便于模块导入
try:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
except Exception as e:
    print(f"[警告] 无法设置 sys.path: {e}")


from alpha101_factory.utils.log import setup_logger
from alpha101_factory.data.universe import load_universe
from alpha101_factory.factors.tmp_features import build_tmp_all


def main() -> None:
    """主函数：执行股票池加载与临时特征构建流程。"""
    # 初始化日志系统
    setup_logger()

    try:
        # 加载股票池
        symbols = load_universe().tolist()
    except Exception as e:
        logger.error(f"加载股票池失败: {e}")
        return

    if not symbols:
        logger.error("股票池为空，请先运行数据获取脚本。")
        return

    try:
        # 构建所有股票的 tmp 文件
        cnt = build_tmp_all(symbols)
        logger.info(f"临时特征构建完成，共处理 {cnt} 只股票。")
    except Exception as e:
        logger.error(f"构建临时特征失败: {e}")


if __name__ == "__main__":
    main()
