# -*- coding: utf-8 -*-
"""Alpha101 Factory — 可插拔 Alpha101 因子工厂.

Usage:
    from alpha101_factory import list_factors, get_factor
    from alpha101_factory.data.yfinance_api import fetch_kline_yf
    from alpha101_factory.backtest.metrics import ic_rankic
"""

from alpha101_factory.factors.registry import list_factors, get_factor

__all__ = ["list_factors", "get_factor", "config"]
