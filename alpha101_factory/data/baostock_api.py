# -*- coding: utf-8 -*-
"""BaoStock API 模块 — 封装 BaoStock 数据库操作。

本模块封装了 BaoStock 库的核心功能，提供了统一的登录/登出生命周期管理、
数据查询接口和异常处理机制。避免在多个地方重复编写相同的 BaoStock
连接代码，提高代码复用性和维护性。

注意：
- BaoStock 需要在每次请求前调用 ``login()``，请求后调用 ``logout()``。
- 登录状态是进程级别的，需谨慎处理并发访问。
"""

from __future__ import annotations

import atexit
import os
import threading
from typing import Optional, Dict, Any

import baostock as bs
import pandas as pd
from loguru import logger

# 全局登录锁，防止多线程并发登录
_login_lock = threading.Lock()

# 登录状态跟踪
_logged_in = False


def _ensure_logout():
    """确保程序退出时登出 BaoStock 连接，释放资源。"""
    global _logged_in
    if _logged_in:
        try:
            logout_result = bs.logout()
            if logout_result.error_code != "0":
                logger.warning(f"BaoStock 自动登出失败: {logout_result.error_msg}")
            else:
                logger.info("BaoStock 连接已自动登出")
        except Exception as e:
            logger.warning(f"BaoStock 自动登出异常: {e}")


# 注册程序退出清理钩子
atexit.register(_ensure_logout)


def login() -> bool:
    """登录 BaoStock 服务。

    Returns:
        bool: 登录成功返回 True，失败返回 False
    """
    global _logged_in

    with _login_lock:
        if _logged_in:
            logger.debug("BaoStock 已登录，跳过重复登录")
            return True

        # 尝试从环境变量获取账户信息
        user_id = os.getenv('BAOSTOCK_USER_ID', '')
        password = os.getenv('BAOSTOCK_PASSWORD', '123456')

        # 如果没有设置环境变量，尝试使用默认空凭据（BaoStock 允许有限的匿名访问）
        if not user_id:
            logger.info("BAOSTOCK_USER_ID 未设置，将尝试匿名访问模式（功能受限）")
            # For some BaoStock functions, even without login some basic functionality may work
            # But login is generally required for most data access
            user_id = ''  # Empty user ID for anonymous
            password = '123456'  # Default password

        try:
            login_result = bs.login(user_id=user_id, password=password)
            if login_result.error_code == "0":
                _logged_in = True
                logger.success("BaoStock 登录成功")
                return True
            else:
                logger.warning(f"BaoStock 登录失败: {login_result.error_msg} (错误码: {login_result.error_code})")
                logger.info("BaoStock 登录失败，将继续使用其他数据源")
                return False
        except Exception as e:
            logger.warning(f"BaoStock 登录异常: {e}")
            logger.info("BaoStock 登录失败，将继续使用其他数据源")
            return False


def logout() -> bool:
    """登出 BaoStock 服务。

    Returns:
        bool: 登出成功返回 True，失败返回 False
    """
    global _logged_in

    with _login_lock:
        if not _logged_in:
            logger.debug("BaoStock 未登录，跳过登出")
            return True

        try:
            logout_result = bs.logout()
            if logout_result.error_code == "0":
                _logged_in = False
                logger.info("BaoStock 登出成功")
                return True
            else:
                logger.error(f"BaoStock 登出失败: {logout_result.error_msg}")
                return False
        except Exception as e:
            logger.error(f"BaoStock 登出异常: {e}")
            _logged_in = False  # 假设登出失败但仍标记为未登录
            return False


def query_history_k_data_plus(
    code: str,
    fields: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    frequency: str = "d",
    adjustflag: str = "3"
) -> Optional[Dict[str, Any]]:
    """查询历史 K 线数据的包装函数，自动处理登录登出。

    Args:
        code: 股票代码，格式如 "sh.600000" 或 "sz.000001"
        fields: 查询字段，如 "date,open,high,low,close,volume,amount"
        start_date: 开始日期，格式 "YYYY-MM-DD"，可选
        end_date: 结束日期，格式 "YYYY-MM-DD"，可选
        frequency: 频率，"d"日k线，默认为 "d"
        adjustflag: 复权标志，"3"不复权，"2"前复权，"1"后复权，默认为 "3"

    Returns:
        包含查询结果和状态的字典，格式与 baostock.query_history_k_data_plus 一致
    """
    # Try login first, but don't fail immediately if login fails
    login_success = login()

    if not login_success:
        logger.warning("BaoStock 登录失败，跳过查询")
        return {
            'error_code': '10001008',
            'error_msg': '登录失败',
            'get_row_data': lambda: [],
            'next': lambda: False
        }

    try:
        query_result = bs.query_history_k_data_plus(
            code=code,
            fields=fields,
            start_date=start_date or "",
            end_date=end_date or "",
            frequency=frequency,
            adjustflag=adjustflag
        )

        return {
            'error_code': query_result.error_code,
            'error_msg': query_result.error_msg,
            'get_row_data': lambda: query_result.get_row_data(),
            'next': lambda: query_result.next()
        }
    except Exception as e:
        logger.error(f"BaoStock 查询异常: {e}")
        return None
    finally:
        # Note: Don't logout here since there might be more queries
        pass


def fetch_stock_data(
    code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    adjustflag: str = "3"
) -> Optional[pd.DataFrame]:
    """获取股票历史数据。

    Args:
        code: 股票代码，如 "sh.600000"
        start_date: 开始日期，格式 "YYYY-MM-DD"，可选
        end_date: 结束日期，格式 "YYYY-MM-DD"，可选
        adjustflag: 复权标志，默认 "3"（不复权）

    Returns:
        DataFrame 包含股票历史数据，失败时返回 None
    """
    fields = "date,open,high,low,close,volume,amount"

    query_result = query_history_k_data_plus(
        code=code,
        fields=fields,
        start_date=start_date,
        end_date=end_date,
        adjustflag=adjustflag
    )

    if query_result is None:
        logger.warning(f"BaoStock 查询结果为空: {code}")
        return pd.DataFrame()

    if query_result.get('error_code') != "0":
        error_msg = query_result.get('error_msg', '未知错误')
        logger.warning(f"BaoStock 查询失败: {code}, 错误: {error_msg}")
        return pd.DataFrame()

    # 逐行读取查询结果
    data_rows = []
    while query_result['next']():
        data_rows.append(query_result['get_row_data']())

    if not data_rows:
        logger.debug(f"BaoStock {code} 无数据返回")
        return pd.DataFrame()

    # 构建 DataFrame
    field_names = fields.split(",")
    df = pd.DataFrame(data_rows, columns=field_names)

    # 数据类型转换
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 日期转换
    if 'date' in df.columns:
        df.rename(columns={'date': 'datetime'}, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'])

    return df


def is_logged_in() -> bool:
    """检查当前是否已登录 BaoStock。

    Returns:
        bool: 已登录返回 True，否则返回 False
    """
    return _logged_in