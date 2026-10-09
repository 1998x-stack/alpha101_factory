# -*- coding: utf-8 -*-
"""JSONL 文件 I/O 工具模块。

提供基于 JSON Lines 格式的数据读写功能，支持可选的元数据行。

核心特性:
    - 每行一个 JSON 对象，代表一条独立记录
    - 读取时自动解析日期列，写入时将 datetime 序列化为 ISO 8601 字符串
    - 可选的 ``_meta`` 首行，用于存储文件级元数据（如生成时间、记录数等）
    - 健壮的异常处理：编码错误、权限错误、畸形 JSON 行均不会导致程序崩溃
    - 详细的日志输出：文件大小、记录数、错误信息

典型用法::

    from pathlib import Path
    import pandas as pd
    from alpha101_factory.utils.io import read_jsonl, write_jsonl

    # 读取 JSONL 文件
    df = read_jsonl(Path("data/quotes/daily/600000.jsonl"))

    # 写入 JSONL 文件（带元数据）
    write_jsonl(
        df,
        Path("data/factors/Alpha101.jsonl"),
        meta={"factor_name": "Alpha101", "generated_at": "2024-01-01"},
    )

注意:
    本模块是项目的核心 I/O 层，被所有其他模块依赖。
    任何修改都必须保持向后兼容的公共 API。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


def read_jsonl(
    path: Path,
    parse_dates: list[str] | None = None,
    skip_meta: bool = True,
) -> pd.DataFrame:
    """读取 JSONL 文件并返回 pandas DataFrame。

    逐行解析 JSON 对象，跳过空行和元数据行（可选），最终合并为 DataFrame。
    支持自动日期解析和完善的错误处理。

    Args:
        path: JSONL 文件路径。如果文件不存在，记录警告日志并返回空 DataFrame。
        parse_dates: 需要解析为 datetime 类型的列名列表。
            默认为 ``["datetime"]``。如果设为空列表 ``[]``，则不解析任何日期列。
        skip_meta: 是否跳过包含 ``_meta`` 键的元数据行。
            默认为 ``True``。设为 ``False`` 可保留元数据行（通常用于调试）。

    Returns:
        pd.DataFrame: 包含所有有效记录的 DataFrame。
            如果文件不存在、为空或解析失败，返回空 DataFrame。

    Raises:
        不抛出异常，所有错误均通过日志记录并返回空 DataFrame。

    Examples:
        基本用法::

            >>> df = read_jsonl(Path("data/quotes/daily/600000.jsonl"))

        自定义日期列::

            >>> df = read_jsonl(
            ...     Path("data/features/600000.jsonl"),
            ...     parse_dates=["datetime", "trade_date"],
            ... )

        保留元数据行::

            >>> df = read_jsonl(
            ...     Path("data/features/600000.jsonl"),
            ...     skip_meta=False,
            ... )
    """
    # 参数校验：路径不能为 None
    if path is None:
        logger.error("读取 JSONL 失败：path 参数为 None")
        return pd.DataFrame()

    # 检查文件是否存在
    if not path.exists():
        logger.warning(f"JSONL 文件不存在: {path}")
        return pd.DataFrame()

    # 检查是否为文件（而非目录）
    if not path.is_file():
        logger.warning(f"路径不是文件: {path}")
        return pd.DataFrame()

    # 设置默认日期解析列
    if parse_dates is None:
        parse_dates = ["datetime"]

    try:
        # 获取文件大小用于日志
        file_size_bytes: int = path.stat().st_size
        logger.debug(f"开始读取 JSONL 文件: {path} ({_format_file_size(file_size_bytes)})")

        valid_records: list[dict[str, Any]] = []
        skipped_meta_count: int = 0
        skipped_malformed_count: int = 0
        line_number: int = 0

        with open(path, "r", encoding="utf-8", errors="replace") as file_handle:
            for raw_line in file_handle:
                line_number += 1
                stripped_line = raw_line.strip()

                # 跳过空行
                if not stripped_line:
                    continue

                try:
                    parsed_object: dict[str, Any] = json.loads(stripped_line)
                except json.JSONDecodeError as json_error:
                    # 记录畸形 JSON 行，跳过但不崩溃
                    logger.warning(
                        f"JSONL 第 {line_number} 行格式错误，已跳过: {path}, "
                        f"错误: {json_error}"
                    )
                    skipped_malformed_count += 1
                    continue

                # 跳过元数据行
                if skip_meta and parsed_object.get("_meta"):
                    skipped_meta_count += 1
                    continue

                valid_records.append(parsed_object)

        # 无有效记录时返回空 DataFrame
        if not valid_records:
            logger.info(f"JSONL 文件无有效记录: {path}")
            return pd.DataFrame()

        # 构建 DataFrame
        data_frame = pd.DataFrame(valid_records)

        # 解析日期列
        for date_column in parse_dates:
            if date_column in data_frame.columns:
                data_frame[date_column] = pd.to_datetime(data_frame[date_column])

        # 输出读取统计信息
        logger.info(
            f"成功读取 {len(valid_records)} 条记录 ← {path} "
            f"(跳过元数据行 {skipped_meta_count} 条, 畸形行 {skipped_malformed_count} 条)"
        )

        return data_frame

    except PermissionError as permission_error:
        logger.error(f"JSONL 文件权限不足，无法读取: {path}, 错误: {permission_error}")
        return pd.DataFrame()
    except OSError as os_error:
        logger.error(f"JSONL 文件读取失败 (OS 错误): {path}, 错误: {os_error}")
        return pd.DataFrame()
    except Exception as unexpected_error:
        logger.error(
            f"JSONL 文件读取时发生未知错误: {path}, 错误: {unexpected_error}",
            exc_info=True,
        )
        return pd.DataFrame()


def write_jsonl(
    df: pd.DataFrame,
    path: Path,
    meta: dict[str, Any] | None = None,
) -> None:
    """将 pandas DataFrame 写入 JSONL 文件。

    逐行序列化 DataFrame 记录为 JSON 对象，支持可选的元数据首行。
    自动创建父目录，datetime 列自动序列化为 ISO 8601 日期字符串。

    Args:
        df: 待写入的 DataFrame。如果为空，记录警告日志并跳过写入。
        path: 输出文件路径。父目录不存在时自动创建。
        meta: 可选的元数据字典，将作为 ``_meta`` 行写入文件首行。
            常用于记录生成时间、记录数、数据来源等信息。

    Raises:
        不抛出异常，所有错误均通过日志记录。

    Examples:
        基本用法::

            >>> write_jsonl(df, Path("data/factors/Alpha101.jsonl"))

        带元数据写入::

            >>> write_jsonl(
            ...     df,
            ...     Path("data/factors/Alpha101.jsonl"),
            ...     meta={
            ...         "factor_name": "Alpha101",
            ...         "generated_at": "2024-01-01",
            ...         "source": "akshare",
            ...     },
            ... )
    """
    # 参数校验：DataFrame 不能为 None
    if df is None:
        logger.error("写入 JSONL 失败：df 参数为 None")
        return

    # 参数校验：路径不能为 None
    if path is None:
        logger.error("写入 JSONL 失败：path 参数为 None")
        return

    # 空 DataFrame 跳过写入
    if df.empty:
        logger.warning(f"跳过写入 — DataFrame 为空: {path}")
        return

    try:
        # 自动创建父目录
        path.parent.mkdir(parents=True, exist_ok=True)

        # 深拷贝以避免修改原始 DataFrame
        output_frame = df.copy()

        # 将 datetime 列序列化为 ISO 8601 日期字符串 (YYYY-MM-DD)
        for column_name in output_frame.columns:
            if pd.api.types.is_datetime64_any_dtype(output_frame[column_name]):
                output_frame[column_name] = output_frame[column_name].dt.strftime("%Y-%m-%d")

        # 转换为记录列表（每行一个字典）
        record_list: list[dict[str, Any]] = output_frame.to_dict(orient="records")
        total_records: int = len(record_list)

        written_count: int = 0
        with open(path, "w", encoding="utf-8") as file_handle:
            # 写入元数据行（如果提供）
            if meta is not None:
                meta_record: dict[str, Any] = {"_meta": True, **meta}
                file_handle.write(json.dumps(meta_record, ensure_ascii=False) + "\n")
                written_count += 1

            # 逐行写入数据记录
            for record in record_list:
                file_handle.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False, default=str) + "\n"
                )
                written_count += 1

        # 获取写入后的文件大小
        final_file_size: int = path.stat().st_size

        # 输出写入统计信息
        logger.info(
            f"成功写入 {total_records} 条记录 → {path} "
            f"(文件大小 {_format_file_size(final_file_size)})"
        )

    except PermissionError as permission_error:
        logger.error(f"JSONL 文件权限不足，无法写入: {path}, 错误: {permission_error}")
    except OSError as os_error:
        logger.error(f"JSONL 文件写入失败 (OS 错误): {path}, 错误: {os_error}")
    except Exception as unexpected_error:
        logger.error(
            f"JSONL 文件写入时发生未知错误: {path}, 错误: {unexpected_error}",
            exc_info=True,
        )


def _format_file_size(size_bytes: int) -> str:
    """将字节数格式化为人类可读的文件大小字符串。

    Args:
        size_bytes: 文件大小（字节）。

    Returns:
        str: 格式化后的大小字符串，如 "1.23 MB"、"456 B"。

    Examples:
        >>> _format_file_size(1024)
        '1.00 KB'
        >>> _format_file_size(1500000)
        '1.43 MB'
    """
    if size_bytes < 0:
        return f"{size_bytes} B"

    if size_bytes < 1024:
        return f"{size_bytes} B"

    units: list[str] = ["KB", "MB", "GB", "TB"]
    size_float: float = float(size_bytes) / 1024

    for unit in units:
        if size_float < 1024 or unit == units[-1]:
            return f"{size_float:.2f} {unit}"
        size_float /= 1024

    return f"{size_float:.2f} {units[-1]}"


def read_parquet(path: Path) -> pd.DataFrame:
    """安全读取 Parquet 文件；文件不存在或读取失败时返回空 DataFrame。"""
    if not path.exists():
        logger.warning(f"文件不存在: {path}")
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as e:  # pragma: no cover - defensive
        logger.error(f"读取 Parquet 文件失败: {path}, 错误: {e}")
        return pd.DataFrame()


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """安全写入 DataFrame 至 Parquet 文件，自动创建父目录。"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info(f"成功写入 Parquet 文件: {path}")
    except Exception as e:  # pragma: no cover - defensive
        logger.error(f"写入 Parquet 文件失败: {path}, 错误: {e}")
