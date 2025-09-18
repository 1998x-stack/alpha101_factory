# -*- coding: utf-8 -*-
"""
日志工具模块

本模块基于 loguru 封装日志初始化方法，提供以下功能：
1. 自动创建日志目录；
2. 清理默认日志配置并重新设置；
3. 同时输出日志到控制台与日志文件；
4. 文件日志支持自动分割 (rotation)，防止日志过大。

适用于量化研究与回测框架中的统一日志管理。
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))


from loguru import logger
from pathlib import Path
from alpha101_factory.config import LOG_DIR


def setup_logger() -> logger:
    """初始化并配置全局日志对象。

    Returns:
        logger: 配置完成的 loguru 日志对象。

    Notes:
        - 控制台日志：实时打印；
        - 文件日志：写入 LOG_DIR/alpha101.log，自动分割 (5 MB)；
        - 若日志目录不可用，将仅输出到控制台。
    """
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[警告] 无法创建日志目录 {LOG_DIR}，错误: {e}")
    
    # 移除默认配置，避免重复输出
    logger.remove()

    # ===== 控制台日志 =====
    try:
        logger.add(lambda msg: print(msg, end=""))  # 直接打印到标准输出
    except Exception as e:
        print(f"[警告] 控制台日志配置失败: {e}")

    # ===== 文件日志 =====
    try:
        log_file = LOG_DIR / "alpha101.log"
        logger.add(
            log_file,
            rotation="5 MB",        # 单个日志文件最大 5 MB
            encoding="utf-8",       # 统一编码
            enqueue=True,           # 多线程安全
            backtrace=True,         # 显示错误堆栈
            diagnose=True           # 调试模式下显示变量值
        )
    except Exception as e:
        print(f"[警告] 文件日志配置失败: {e}")

    return logger
