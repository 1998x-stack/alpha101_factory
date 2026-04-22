# -*- coding: utf-8 -*-
"""
日志工具模块 (Logging Utilities)

本模块基于 loguru 封装日志初始化方法，提供生产级日志配置，功能包括：

1. 自动创建日志目录，并验证目录可写性；
2. 清理 loguru 默认处理器配置，避免重复输出；
3. 同时输出日志到控制台 (stdout) 与日志文件；
4. 文件日志支持自动分割 (rotation)，防止单文件过大；
5. 完善的异常处理：磁盘满、权限不足、目录不可访问等场景均优雅降级。

适用于量化研究与回测框架中的统一日志管理。

典型用法::

    from alpha101_factory.utils.log import setup_logger

    logger = setup_logger()
    logger.info("因子计算开始")
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from alpha101_factory.config import LOG_DIR

if TYPE_CHECKING:
    from loguru import Logger


# ============================================================
# 常量定义
# ============================================================

# 单个日志文件最大体积 (5 MB)，超过后自动轮转
_MAX_LOG_FILE_SIZE = "5 MB"

# 日志文件名
_LOG_FILENAME = "alpha101.log"


def _validate_log_directory(log_dir: Path) -> bool:
    """验证日志目录是否可用（存在、可访问、可写）。

    依次检查目录是否存在、是否具有读写权限。若目录不存在则尝试创建。
    任何异常均打印警告信息并返回 False。

    Args:
        log_dir: 待验证的日志目录路径。

    Returns:
        bool: 目录可用返回 True，否则返回 False。
    """
    # 尝试创建目录（若不存在）
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"[警告] 日志目录权限不足，无法创建: {log_dir}")
        return False
    except OSError as exc:
        print(f"[警告] 无法创建日志目录 {log_dir}，系统错误: {exc}")
        return False

    # 验证目录是否可写：尝试在目录下创建临时文件
    probe_file = log_dir / ".log_write_probe"
    try:
        probe_file.touch()
        probe_file.unlink()
    except PermissionError:
        print(f"[警告] 日志目录不可写: {log_dir}")
        return False
    except OSError as exc:
        # 磁盘满等场景
        print(f"[警告] 日志目录写入失败 {log_dir}，系统错误: {exc}")
        return False

    return True


def _add_console_handler() -> None:
    """向 loguru 添加控制台输出处理器。

    将日志直接打印到标准输出 (stdout)。若添加失败，打印警告信息但不中断程序。
    """
    try:
        logger.add(
            lambda msg: print(msg, end=""),
            level="INFO",
        )
    except Exception as exc:
        print(f"[警告] 控制台日志配置失败: {exc}")


def _add_file_handler(log_dir: Path) -> bool:
    """向 loguru 添加文件输出处理器。

    配置日志文件路径、轮转策略、编码、多线程安全等参数。
    若文件写入失败（磁盘满、权限不足等），打印警告并返回 False。

    Args:
        log_dir: 日志目录路径。

    Returns:
        bool: 文件处理器添加成功返回 True，否则返回 False。
    """
    log_file_path = log_dir / _LOG_FILENAME

    try:
        logger.add(
            str(log_file_path),
            rotation=_MAX_LOG_FILE_SIZE,  # 单文件达到指定大小后自动分割
            encoding="utf-8",              # 统一 UTF-8 编码
            enqueue=True,                  # 多进程/多线程安全队列写入
            backtrace=True,                # 异常时显示完整调用栈
            diagnose=True,                 # 调试模式下显示局部变量值
            level="DEBUG",                 # 文件记录更详细的 DEBUG 级别日志
        )
        return True
    except PermissionError:
        print(f"[警告] 日志文件权限不足，无法写入: {log_file_path}")
        return False
    except OSError as exc:
        # 磁盘满 (ENOSPC) 等系统级错误
        print(f"[警告] 日志文件写入失败 {log_file_path}，系统错误: {exc}")
        return False
    except Exception as exc:
        print(f"[警告] 文件日志配置失败 {log_file_path}: {exc}")
        return False


def setup_logger() -> Logger:
    """初始化并配置全局日志对象。

    该函数执行以下操作：
    1. 验证日志目录的可用性（存在、可写）；
    2. 移除 loguru 默认处理器，避免重复输出；
    3. 添加控制台处理器，实时打印日志到 stdout；
    4. 添加文件处理器，将日志写入 LOG_DIR/alpha101.log，支持自动轮转。

    若日志目录不可用或文件写入失败，仅输出警告信息到控制台，程序继续运行。

    Returns:
        Logger: 配置完成的 loguru Logger 对象，可直接用于日志记录。

    Notes:
        - 控制台日志级别为 INFO，文件日志级别为 DEBUG；
        - 文件日志单文件最大 5 MB，超过后自动分割；
        - 文件日志启用多线程安全队列 (enqueue=True)；
        - 若日志目录不可用，仅输出到控制台，不中断程序。
    """
    # 1. 验证日志目录可用性
    is_log_dir_valid = _validate_log_directory(LOG_DIR)

    if is_log_dir_valid:
        print(f"[日志] 日志目录已就绪: {LOG_DIR}")
        print(f"[日志] 日志轮转策略: 单文件最大 {_MAX_LOG_FILE_SIZE}")
    else:
        print(f"[日志] 日志目录不可用，将仅输出到控制台: {LOG_DIR}")

    # 2. 移除 loguru 默认处理器，避免重复输出
    logger.remove()

    # 3. 添加控制台日志处理器
    _add_console_handler()
    print("[日志] 控制台日志已启用")

    # 4. 添加文件日志处理器（仅当目录可用时）
    if is_log_dir_valid:
        file_handler_added = _add_file_handler(LOG_DIR)
        if file_handler_added:
            log_file_path = LOG_DIR / _LOG_FILENAME
            print(f"[日志] 文件日志已启用: {log_file_path}")
        else:
            print("[日志] 文件日志启用失败，仅使用控制台输出")
    else:
        print("[日志] 跳过文件日志配置（目录不可用）")

    print("[日志] 日志系统初始化完成")

    return logger
