# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict

class Factor(ABC):
    """Factor base class."""
    name: str = "BaseFactor"
    # 需要的列（K线/中间变量）
    requires: List[str] = []

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """输入为长表（索引：datetime，列含 symbol），返回因子值（Series，MultiIndex: [datetime, symbol]）"""
        raise NotImplementedError

    @staticmethod
    def as_cs_series(df: pd.DataFrame, values: pd.Series) -> pd.Series:
        # 统一 MultiIndex 截面索引
        idx = pd.MultiIndex.from_frame(df[["datetime","symbol"]], names=["datetime","symbol"])
        out = pd.Series(values.values, index=idx, name="value")
        return out
