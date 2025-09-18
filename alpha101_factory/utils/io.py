# -*- coding: utf-8 -*-
"""
Parquet 文件读写工具模块

本模块封装了对 Parquet 文件的常用读写操作：
1. 提供 `read_parquet` 函数安全读取文件（若文件不存在则返回空 DataFrame）；
2. 提供 `write_parquet` 函数安全写入文件（自动创建父目录）。

适用于量化研究与数据处理中间结果的存取，增强了 I/O 操作的健壮性。
"""

from pathlib import Path
import pandas as pd
from loguru import logger


def read_parquet(path: Path) -> pd.DataFrame:
    """安全读取 Parquet 文件。

    Args:
        path (Path): 待读取的 Parquet 文件路径。

    Returns:
        pd.DataFrame: 若文件存在且读取成功则返回 DataFrame，否则返回空 DataFrame。
    """
    if not path.exists():
        logger.warning(f"文件不存在: {path}")
        return pd.DataFrame()

    try:
        return pd.read_parquet(path)
    except Exception as e:
        logger.error(f"读取 Parquet 文件失败: {path}, 错误: {e}")
        return pd.DataFrame()


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """安全写入 DataFrame 至 Parquet 文件。

    Args:
        df (pd.DataFrame): 待写入的数据表。
        path (Path): 输出文件路径。

    Notes:
        - 自动创建父目录；
        - 若写入失败会捕获异常并记录日志。
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info(f"成功写入 Parquet 文件: {path}")
    except Exception as e:
        logger.error(f"写入 Parquet 文件失败: {path}, 错误: {e}")
