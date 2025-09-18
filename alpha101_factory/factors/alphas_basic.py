# -*- coding: utf-8 -*-
"""
Alpha101因子库基础实现模块。

本模块包含Alpha101因子库中基础因子的实现，包括工具函数和各个Alpha因子的具体计算逻辑。
所有因子都继承自Factor基类，并通过装饰器注册到因子注册表中。
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from alpha101_factory.factors.base import Factor
from alpha101_factory.factors.registry import register
from alpha101_factory.utils import ops


def _cs_rank(df: pd.DataFrame, s: pd.Series) -> pd.Series:
    """
    计算截面排名。
    
    对给定的Series按时间截面进行排名计算，返回百分位排名。
    
    Args:
        df: 包含datetime和symbol列的DataFrame
        s: 需要计算排名的Series
        
    Returns:
        按时间截面排名的Series，值为0-1之间的百分位排名
        
    Raises:
        KeyError: 当DataFrame中缺少必要的datetime或symbol列时
        ValueError: 当输入数据格式不正确时
    """
    try:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df必须是pandas DataFrame")
        if not isinstance(s, pd.Series):
            raise ValueError("s必须是pandas Series")
            
        # 检查必要的列是否存在
        required_cols = ["datetime", "symbol"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"DataFrame缺少必要的列: {missing_cols}")
            
        idx = pd.MultiIndex.from_frame(df[required_cols], names=required_cols)
        s_indexed = pd.Series(s.values, index=idx)
        return s_indexed.groupby(level=0).rank(pct=True)
        
    except Exception as e:
        raise RuntimeError(f"计算截面排名时发生错误: {str(e)}") from e


def _g(df: pd.DataFrame, col: str, fn, *args):
    """
    按股票代码分组应用滚动函数。
    
    对指定列按symbol分组，然后对每个分组应用指定的函数。
    这是一个便捷函数，用于简化按股票分组的滚动计算。
    
    Args:
        df: 包含symbol列的DataFrame
        col: 要处理的列名
        fn: 要应用的函数
        *args: 传递给函数的额外参数
        
    Returns:
        应用函数后的结果Series
        
    Raises:
        KeyError: 当指定的列不存在时
        ValueError: 当输入参数格式不正确时
    """
    try:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df必须是pandas DataFrame")
        if col is not None and col not in df.columns:
            raise KeyError(f"列 '{col}' 不存在于DataFrame中")
        if "symbol" not in df.columns:
            raise KeyError("DataFrame必须包含'symbol'列")
            
        if col is None:
            # 当col为None时，对整个DataFrame的每个分组应用函数
            return df.groupby("symbol", group_keys=False).apply(lambda x: fn(x, *args))
        else:
            # 对指定列按symbol分组应用函数
            return df.groupby("symbol", group_keys=False)[col].apply(lambda x: fn(x, *args))
            
    except Exception as e:
        raise RuntimeError(f"按分组应用函数时发生错误: {str(e)}") from e

# ===== Alpha因子实现 =====

@register
class Alpha001(Factor):
    """
    Alpha001因子实现。
    
    该因子基于收益率和收盘价计算，当收益率为负时使用收益率的滚动标准差，
    否则使用收盘价，然后计算时间序列排名。
    """
    name = "Alpha001"
    requires = ["returns", "close"]
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha001因子值。
        
        因子计算逻辑：
        1. 当收益率为负时，使用收益率的20期滚动标准差
        2. 当收益率为正时，使用收盘价
        3. 对结果进行平方后计算5期时间序列排名
        4. 计算截面排名并减去0.5
        
        Args:
            df: 包含returns和close列的DataFrame
            
        Returns:
            计算得到的因子值Series
            
        Raises:
            KeyError: 当DataFrame中缺少必要的列时
            ValueError: 当数据格式不正确时
        """
        try:
            # 检查必要的列是否存在
            required_cols = ["returns", "close"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"DataFrame缺少必要的列: {missing_cols}")
            
            # 复制收益率数据
            x = df["returns"].copy()
            
            # 根据收益率正负选择不同的计算方式
            part = np.where(x < 0, 
                          _g(df, "returns", ops.rolling_std, 20), 
                          df["close"])
            
            # 计算时间序列排名
            val = _g(df.assign(part=part), "part", 
                    lambda s: ops.ts_rank(s**2, 5))
            
            # 计算截面排名并调整
            out = _cs_rank(df, val) - 0.5
            
            return Factor.as_cs_series(df, out)
            
        except Exception as e:
            raise RuntimeError(f"计算Alpha001因子时发生错误: {str(e)}") from e

@register
class Alpha003(Factor):
    """
    Alpha003因子实现。
    
    该因子基于开盘价和成交量的相关性计算，通过计算开盘价和成交量
    截面排名的10期滚动相关系数的负值来构建因子。
    """
    name = "Alpha003"
    requires = ["open", "volume"]
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha003因子值。
        
        因子计算逻辑：
        1. 计算开盘价的截面排名
        2. 计算成交量的截面排名
        3. 计算两者10期滚动相关系数
        4. 取负值作为因子值
        
        Args:
            df: 包含open和volume列的DataFrame
            
        Returns:
            计算得到的因子值Series
        """
        try:
            required_cols = ["open", "volume"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"DataFrame缺少必要的列: {missing_cols}")
            
            val = -_g(df, "open", 
                     lambda s: ops.rolling_corr(
                         ops.cs_rank(s), 
                         ops.cs_rank(df.loc[s.index, "volume"]), 
                         10))
            
            return Factor.as_cs_series(df, val)
            
        except Exception as e:
            raise RuntimeError(f"计算Alpha003因子时发生错误: {str(e)}") from e


@register
class Alpha004(Factor):
    """
    Alpha004因子实现。
    
    该因子基于最低价的时间序列排名计算，通过计算最低价截面排名
    的9期时间序列排名的负值来构建因子。
    """
    name = "Alpha004"
    requires = ["low"]
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha004因子值。
        
        因子计算逻辑：
        1. 计算最低价的截面排名
        2. 计算截面排名的9期时间序列排名
        3. 取负值作为因子值
        
        Args:
            df: 包含low列的DataFrame
            
        Returns:
            计算得到的因子值Series
        """
        try:
            required_cols = ["low"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"DataFrame缺少必要的列: {missing_cols}")
            
            val = -_g(df, "low", 
                     lambda s: ops.ts_rank(ops.cs_rank(s), 9))
            
            return Factor.as_cs_series(df, val)
            
        except Exception as e:
            raise RuntimeError(f"计算Alpha004因子时发生错误: {str(e)}") from e


@register
class Alpha005(Factor):
    """
    Alpha005因子实现。
    
    该因子基于开盘价、VWAP和收盘价计算，通过比较开盘价与10期均值的差异
    以及收盘价与VWAP的差异来构建因子。
    """
    name = "Alpha005"
    requires = ["open", "vwap", "close"]
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha005因子值。
        
        因子计算逻辑：
        1. 计算开盘价的10期移动平均
        2. 计算开盘价与均值的差异的截面排名
        3. 计算收盘价与VWAP差异的截面排名的绝对值负值
        4. 两者相乘得到因子值
        
        Args:
            df: 包含open、vwap和close列的DataFrame
            
        Returns:
            计算得到的因子值Series
        """
        try:
            required_cols = ["open", "vwap", "close"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"DataFrame缺少必要的列: {missing_cols}")
            
            # 计算开盘价的10期移动平均
            a = df.groupby("symbol")["open"].apply(
                lambda s: ops.rolling_sum(s, 10) / 10)
            
            # 计算开盘价与均值的差异的截面排名
            b = ops.cs_rank(df["open"] - a.values)
            
            # 计算收盘价与VWAP差异的截面排名的绝对值负值
            c = -np.abs(ops.cs_rank(df["close"] - df["vwap"]))
            
            val = b * c
            
            return Factor.as_cs_series(df, val)
            
        except Exception as e:
            raise RuntimeError(f"计算Alpha005因子时发生错误: {str(e)}") from e

@register
class Alpha006(Factor):
    """
    Alpha006因子实现。
    
    该因子基于开盘价和成交量的相关性计算，通过计算开盘价和成交量
    10期滚动相关系数的负值来构建因子。
    """
    name = "Alpha006"
    requires = ["open", "volume"]
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha006因子值。
        
        因子计算逻辑：
        1. 计算开盘价和成交量的10期滚动相关系数
        2. 取负值作为因子值
        
        Args:
            df: 包含open和volume列的DataFrame
            
        Returns:
            计算得到的因子值Series
        """
        try:
            required_cols = ["open", "volume"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"DataFrame缺少必要的列: {missing_cols}")
            
            val = -_g(df, "open", 
                     lambda s: ops.rolling_corr(s, df.loc[s.index, "volume"], 10))
            
            return Factor.as_cs_series(df, val)
            
        except Exception as e:
            raise RuntimeError(f"计算Alpha006因子时发生错误: {str(e)}") from e

@register
class Alpha009(Factor):
    """
    Alpha009因子实现。
    
    该因子基于收盘价的变化趋势计算，通过判断价格变化的方向
    来决定因子的正负值。
    """
    name = "Alpha009"
    requires = ["close"]
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha009因子值。
        
        因子计算逻辑：
        1. 计算收盘价的1期变化
        2. 判断5期内最小变化是否大于0（上升趋势）
        3. 判断5期内最大变化是否小于0（下降趋势）
        4. 根据趋势方向决定因子值的正负
        
        Args:
            df: 包含close列的DataFrame
            
        Returns:
            计算得到的因子值Series
        """
        try:
            required_cols = ["close"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"DataFrame缺少必要的列: {missing_cols}")
            
            # 计算收盘价的1期变化
            d1 = _g(df, "close", ops.delta, 1)
            
            # 判断上升趋势：5期内最小变化大于0
            cond1 = _g(df, "close", 
                       lambda s: ops.rolling_min(ops.delta(s, 1), 5)) > 0
            
            # 判断下降趋势：5期内最大变化小于0
            cond2 = _g(df, "close", 
                       lambda s: ops.rolling_max(ops.delta(s, 1), 5)) < 0
            
            # 根据趋势方向决定因子值
            val = np.where(cond1, d1, np.where(cond2, d1, -d1))
            
            return Factor.as_cs_series(df, pd.Series(val))
            
        except Exception as e:
            raise RuntimeError(f"计算Alpha009因子时发生错误: {str(e)}") from e

@register
class Alpha010(Factor):
    """
    Alpha010因子实现。
    
    该因子基于收盘价的变化趋势计算，与Alpha009类似但使用4期窗口
    并对结果进行截面排名。
    """
    name = "Alpha010"
    requires = ["close"]
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha010因子值。
        
        因子计算逻辑：
        1. 计算收盘价的1期变化
        2. 判断4期内最小变化是否大于0（上升趋势）
        3. 判断4期内最大变化是否小于0（下降趋势）
        4. 根据趋势方向决定因子值的正负
        5. 对结果进行截面排名
        
        Args:
            df: 包含close列的DataFrame
            
        Returns:
            计算得到的因子值Series
        """
        try:
            required_cols = ["close"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"DataFrame缺少必要的列: {missing_cols}")
            
            # 计算收盘价的1期变化
            d1 = _g(df, "close", ops.delta, 1)
            
            # 判断上升趋势：4期内最小变化大于0
            cond1 = _g(df, "close", 
                       lambda s: ops.rolling_min(ops.delta(s, 1), 4)) > 0
            
            # 判断下降趋势：4期内最大变化小于0
            cond2 = _g(df, "close", 
                       lambda s: ops.rolling_max(ops.delta(s, 1), 4)) < 0
            
            # 根据趋势方向决定因子值并进行截面排名
            val = ops.cs_rank(np.where(cond1, d1, np.where(cond2, d1, -d1)))
            
            return Factor.as_cs_series(df, val)
            
        except Exception as e:
            raise RuntimeError(f"计算Alpha010因子时发生错误: {str(e)}") from e

@register
class Alpha011(Factor):
    """
    Alpha011因子实现。
    
    该因子基于VWAP、收盘价和成交量计算，通过比较VWAP与收盘价的关系
    以及成交量的变化来构建因子。
    """
    name = "Alpha011"
    requires = ["vwap", "close", "volume"]
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha011因子值。
        
        因子计算逻辑：
        1. 计算VWAP与收盘价差异的3期时间序列排名
        2. 计算收盘价与VWAP差异的3期时间序列排名
        3. 计算成交量3期变化的3期时间序列排名
        4. 组合三个排名计算最终因子值
        
        Args:
            df: 包含vwap、close和volume列的DataFrame
            
        Returns:
            计算得到的因子值Series
        """
        try:
            required_cols = ["vwap", "close", "volume"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"DataFrame缺少必要的列: {missing_cols}")
            
            # VWAP与收盘价差异的3期时间序列排名
            a = _g(df, "vwap", 
                   lambda s: ops.ts_rank(s - df.loc[s.index, "close"], 3))
            
            # 收盘价与VWAP差异的3期时间序列排名
            b = _g(df, "vwap", 
                   lambda s: ops.ts_rank(df.loc[s.index, "close"] - s, 3))
            
            # 成交量3期变化的3期时间序列排名
            c = _g(df, "volume", 
                   lambda s: ops.ts_rank(ops.delta(s, 3), 3))
            
            # 组合三个排名
            val = (ops.cs_rank(a) + ops.cs_rank(b)) * ops.cs_rank(c)
            
            return Factor.as_cs_series(df, val)
            
        except Exception as e:
            raise RuntimeError(f"计算Alpha011因子时发生错误: {str(e)}") from e

@register
class Alpha012(Factor):
    """
    Alpha012因子实现。
    
    该因子基于收盘价和成交量的变化计算，通过成交量变化的方向
    与收盘价变化的负值相乘来构建因子。
    """
    name = "Alpha012"
    requires = ["close", "volume"]
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha012因子值。
        
        因子计算逻辑：
        1. 计算成交量的1期变化
        2. 计算收盘价的1期变化
        3. 取成交量变化的方向（正负号）
        4. 与收盘价变化的负值相乘
        
        Args:
            df: 包含close和volume列的DataFrame
            
        Returns:
            计算得到的因子值Series
        """
        try:
            required_cols = ["close", "volume"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"DataFrame缺少必要的列: {missing_cols}")
            
            # 成交量变化的方向与收盘价变化负值的乘积
            val = (np.sign(_g(df, "volume", ops.delta, 1)) * 
                   (-_g(df, "close", ops.delta, 1)))
            
            return Factor.as_cs_series(df, pd.Series(val))
            
        except Exception as e:
            raise RuntimeError(f"计算Alpha012因子时发生错误: {str(e)}") from e

@register
class Alpha013(Factor):
    """
    Alpha013因子实现。
    
    该因子基于收盘价和成交量的协方差计算，通过计算收盘价和成交量
    截面排名的5期滚动协方差的负值来构建因子。
    """
    name = "Alpha013"
    requires = ["close", "volume"]
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha013因子值。
        
        因子计算逻辑：
        1. 计算收盘价的截面排名
        2. 计算成交量的截面排名
        3. 计算两者5期滚动协方差
        4. 取负值作为因子值
        
        Args:
            df: 包含close和volume列的DataFrame
            
        Returns:
            计算得到的因子值Series
        """
        try:
            required_cols = ["close", "volume"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"DataFrame缺少必要的列: {missing_cols}")
            
            val = -_g(df, "close", 
                      lambda s: ops.rolling_cov(
                          ops.cs_rank(s), 
                          ops.cs_rank(df.loc[s.index, "volume"]), 
                          5))
            
            return Factor.as_cs_series(df, val)
            
        except Exception as e:
            raise RuntimeError(f"计算Alpha013因子时发生错误: {str(e)}") from e

@register
class Alpha014(Factor):
    """
    Alpha014因子实现。
    
    该因子基于开盘价、成交量和收益率计算，通过收益率3期变化的截面排名
    与开盘价成交量相关性的乘积来构建因子。
    """
    name = "Alpha014"
    requires = ["open", "volume", "returns"]
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha014因子值。
        
        因子计算逻辑：
        1. 计算收益率3期变化的截面排名
        2. 计算开盘价与成交量的10期滚动相关系数
        3. 两者相乘得到因子值
        
        Args:
            df: 包含open、volume和returns列的DataFrame
            
        Returns:
            计算得到的因子值Series
        """
        try:
            required_cols = ["open", "volume", "returns"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"DataFrame缺少必要的列: {missing_cols}")
            
            # 收益率3期变化的截面排名负值
            returns_rank = -ops.cs_rank(_g(df, "returns", ops.delta, 3))
            
            # 开盘价与成交量的10期滚动相关系数
            open_volume_corr = _g(df, "open", 
                                 lambda s: ops.rolling_corr(s, df.loc[s.index, "volume"], 10))
            
            val = returns_rank * open_volume_corr
            
            return Factor.as_cs_series(df, val)
            
        except Exception as e:
            raise RuntimeError(f"计算Alpha014因子时发生错误: {str(e)}") from e

@register
class Alpha016(Factor):
    """
    Alpha016因子实现。
    
    该因子基于最高价和成交量的协方差计算，通过计算最高价和成交量
    截面排名的5期滚动协方差的负值来构建因子。
    """
    name = "Alpha016"
    requires = ["high", "volume"]
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha016因子值。
        
        因子计算逻辑：
        1. 计算最高价的截面排名
        2. 计算成交量的截面排名
        3. 计算两者5期滚动协方差
        4. 取负值作为因子值
        
        Args:
            df: 包含high和volume列的DataFrame
            
        Returns:
            计算得到的因子值Series
        """
        try:
            required_cols = ["high", "volume"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"DataFrame缺少必要的列: {missing_cols}")
            
            val = -_g(df, "high", 
                      lambda s: ops.rolling_cov(
                          ops.cs_rank(s), 
                          ops.cs_rank(df.loc[s.index, "volume"]), 
                          5))
            
            return Factor.as_cs_series(df, val)
            
        except Exception as e:
            raise RuntimeError(f"计算Alpha016因子时发生错误: {str(e)}") from e

@register
class Alpha018(Factor):
    """
    Alpha018因子实现。
    
    该因子基于收盘价和开盘价计算，通过收盘价与开盘价差异的波动率、
    差异值和相关性来构建因子。
    """
    name = "Alpha018"
    requires = ["close", "open"]
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha018因子值。
        
        因子计算逻辑：
        1. 计算收盘价与开盘价差异绝对值的5期滚动标准差
        2. 计算收盘价与开盘价的差异
        3. 计算收盘价与开盘价的10期滚动相关系数
        4. 将三个值相加后进行截面排名并取负值
        
        Args:
            df: 包含close和open列的DataFrame
            
        Returns:
            计算得到的因子值Series
        """
        try:
            required_cols = ["close", "open"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"DataFrame缺少必要的列: {missing_cols}")
            
            # 收盘价与开盘价差异绝对值的5期滚动标准差
            a = _g(df, "close", 
                   lambda s: ops.rolling_std(np.abs(s - df.loc[s.index, "open"]), 5))
            
            # 收盘价与开盘价的差异
            b = df["close"] - df["open"]
            
            # 收盘价与开盘价的10期滚动相关系数
            c = _g(df, "close", 
                   lambda s: ops.rolling_corr(s, df.loc[s.index, "open"], 10))
            
            # 组合三个值并进行截面排名
            val = -ops.cs_rank(a + b + c)
            
            return Factor.as_cs_series(df, val)
            
        except Exception as e:
            raise RuntimeError(f"计算Alpha018因子时发生错误: {str(e)}") from e

@register
class Alpha019(Factor):
    """
    Alpha019因子实现。
    
    该因子基于收盘价和收益率计算，通过比较当前收盘价与7天前收盘价
    以及250期收益率累计和的排名来构建因子。
    """
    name = "Alpha019"
    requires = ["close", "returns"]
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha019因子值。
        
        因子计算逻辑：
        1. 计算收盘价与7天前收盘价的差异
        2. 计算收盘价7期变化
        3. 计算250期收益率累计和的排名
        4. 根据价格变化方向调整因子值
        
        Args:
            df: 包含close和returns列的DataFrame
            
        Returns:
            计算得到的因子值Series
        """
        try:
            required_cols = ["close", "returns"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"DataFrame缺少必要的列: {missing_cols}")
            
            # 计算价格变化的方向
            price_change = df["close"] - _g(df, "close", ops.delay, 7) + _g(df, "close", ops.delta, 7)
            
            # 计算250期收益率累计和的排名
            returns_sum_rank = ops.cs_rank(1 + _g(df, "returns", ops.rolling_sum, 250))
            
            # 组合计算因子值
            val = -np.sign(price_change) * (1 + returns_sum_rank)
            
            return Factor.as_cs_series(df, pd.Series(val))
            
        except Exception as e:
            raise RuntimeError(f"计算Alpha019因子时发生错误: {str(e)}") from e

@register
class Alpha020(Factor):
    """
    Alpha020因子实现。
    
    该因子基于开盘价、最高价、最低价和收盘价计算，通过比较当前开盘价
    与前一天的最高价、收盘价、最低价的差异来构建因子。
    """
    name = "Alpha020"
    requires = ["open", "high", "low", "close"]
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha020因子值。
        
        因子计算逻辑：
        1. 计算开盘价与前一天最高价差异的截面排名
        2. 计算开盘价与前一天收盘价差异的截面排名
        3. 计算开盘价与前一天最低价差异的截面排名
        4. 三个排名相乘得到因子值
        
        Args:
            df: 包含open、high、low和close列的DataFrame
            
        Returns:
            计算得到的因子值Series
        """
        try:
            required_cols = ["open", "high", "low", "close"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"DataFrame缺少必要的列: {missing_cols}")
            
            # 开盘价与前一天最高价差异的截面排名
            rank1 = -ops.cs_rank(df["open"] - _g(df, "high", ops.delay, 1))
            
            # 开盘价与前一天收盘价差异的截面排名
            rank2 = ops.cs_rank(df["open"] - _g(df, "close", ops.delay, 1))
            
            # 开盘价与前一天最低价差异的截面排名
            rank3 = ops.cs_rank(df["open"] - _g(df, "low", ops.delay, 1))
            
            val = rank1 * rank2 * rank3
            
            return Factor.as_cs_series(df, val)
            
        except Exception as e:
            raise RuntimeError(f"计算Alpha020因子时发生错误: {str(e)}") from e

@register
class Alpha021(Factor):
    """
    Alpha021因子实现。
    
    该因子基于收盘价和成交量计算，通过比较不同周期的移动平均
    和成交量相对平均成交量的关系来构建因子。
    """
    name = "Alpha021"
    requires = ["close", "volume"]
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha021因子值。
        
        因子计算逻辑：
        1. 计算收盘价8期移动平均和标准差
        2. 计算收盘价2期移动平均
        3. 计算成交量20期移动平均
        4. 根据价格趋势和成交量相对强度决定因子值
        
        Args:
            df: 包含close和volume列的DataFrame
            
        Returns:
            计算得到的因子值Series
        """
        try:
            required_cols = ["close", "volume"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"DataFrame缺少必要的列: {missing_cols}")
            
            # 收盘价8期移动平均
            s8 = _g(df, "close", lambda s: ops.rolling_sum(s, 8) / 8)
            
            # 收盘价8期标准差
            sd8 = _g(df, "close", lambda s: ops.rolling_std(s, 8))
            
            # 收盘价2期移动平均
            s2 = _g(df, "close", lambda s: ops.rolling_sum(s, 2) / 2)
            
            # 成交量20期移动平均
            adv20 = _g(df, "volume", lambda s: ops.adv(s, 20))
            
            # 价格趋势条件
            cond = ((s8 + sd8) < s2) * (-1) + ((s2 < (s8 - sd8)) * 1)
            
            # 成交量相对强度条件
            cond2 = (((df["volume"] / adv20) >= 1) * 1) + (((df["volume"] / adv20) < 1) * (-1))
            
            # 根据条件决定因子值
            val = np.where(cond != 0, cond, cond2)
            
            return Factor.as_cs_series(df, pd.Series(val))
            
        except Exception as e:
            raise RuntimeError(f"计算Alpha021因子时发生错误: {str(e)}") from e

@register
class Alpha022(Factor):
    """
    Alpha022因子实现。
    
    该因子基于最高价、成交量和收盘价计算，通过最高价与成交量的相关性变化
    和收盘价波动率的乘积来构建因子。
    """
    name = "Alpha022"
    requires = ["high", "volume", "close"]
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha022因子值。
        
        因子计算逻辑：
        1. 计算最高价与成交量的5期滚动相关系数
        2. 计算相关系数的5期变化
        3. 计算收盘价20期滚动标准差的截面排名
        4. 两者相乘并取负值
        
        Args:
            df: 包含high、volume和close列的DataFrame
            
        Returns:
            计算得到的因子值Series
        """
        try:
            required_cols = ["high", "volume", "close"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"DataFrame缺少必要的列: {missing_cols}")
            
            # 最高价与成交量的5期滚动相关系数
            corr = _g(df, "high", 
                      lambda s: ops.rolling_corr(s, df.loc[s.index, "volume"], 5))
            
            # 相关系数的5期变化
            corr_delta = _g(df, None, lambda *_: ops.delta(corr, 5))
            
            # 收盘价20期滚动标准差的截面排名
            close_std_rank = ops.cs_rank(_g(df, "close", ops.rolling_std, 20))
            
            val = -corr_delta * close_std_rank
            
            return Factor.as_cs_series(df, val)
            
        except Exception as e:
            raise RuntimeError(f"计算Alpha022因子时发生错误: {str(e)}") from e

@register
class Alpha023(Factor):
    """
    Alpha023因子实现。
    
    该因子基于最高价和收盘价计算，通过比较收盘价20期移动平均
    与当前最高价的关系来构建因子。
    """
    name = "Alpha023"
    requires = ["high", "close"]
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha023因子值。
        
        因子计算逻辑：
        1. 计算收盘价20期移动平均
        2. 判断移动平均是否小于当前最高价
        3. 如果条件满足，取最高价2期变化的负值
        4. 否则因子值为0
        
        Args:
            df: 包含high和close列的DataFrame
            
        Returns:
            计算得到的因子值Series
        """
        try:
            required_cols = ["high", "close"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"DataFrame缺少必要的列: {missing_cols}")
            
            # 收盘价20期移动平均
            close_ma20 = _g(df, "close", ops.rolling_sum, 20) / 20
            
            # 判断条件：移动平均小于当前最高价
            cond = close_ma20 < df["high"]
            
            # 根据条件决定因子值
            val = np.where(cond, -_g(df, "high", ops.delta, 2), 0)
            
            return Factor.as_cs_series(df, pd.Series(val))
            
        except Exception as e:
            raise RuntimeError(f"计算Alpha023因子时发生错误: {str(e)}") from e

@register
class Alpha025(Factor):
    """
    Alpha025因子实现。
    
    该因子基于收益率、VWAP、最高价、收盘价和成交量计算，通过
    收益率、平均成交量、VWAP和价格区间的组合来构建因子。
    """
    name = "Alpha025"
    requires = ["returns", "vwap", "high", "close", "volume"]
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha025因子值。
        
        因子计算逻辑：
        1. 计算成交量20期移动平均
        2. 计算收益率负值、平均成交量、VWAP和价格区间的乘积
        3. 对结果进行截面排名
        
        Args:
            df: 包含returns、vwap、high、close和volume列的DataFrame
            
        Returns:
            计算得到的因子值Series
        """
        try:
            required_cols = ["returns", "vwap", "high", "close", "volume"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"DataFrame缺少必要的列: {missing_cols}")
            
            # 成交量20期移动平均
            adv20 = _g(df, "volume", lambda s: ops.adv(s, 20))
            
            # 组合计算：收益率负值 * 平均成交量 * VWAP * 价格区间
            val = ops.cs_rank((-df["returns"]) * adv20 * df["vwap"] * (df["high"] - df["close"]))
            
            return Factor.as_cs_series(df, val)
            
        except Exception as e:
            raise RuntimeError(f"计算Alpha025因子时发生错误: {str(e)}") from e

@register
class Alpha026(Factor):
    """
    Alpha026因子实现。
    
    该因子基于成交量和最高价计算，通过成交量与最高价的时间序列排名
    相关性的最大值来构建因子。
    """
    name = "Alpha026"
    requires = ["volume", "high"]
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha026因子值。
        
        因子计算逻辑：
        1. 计算成交量5期时间序列排名
        2. 计算最高价5期时间序列排名
        3. 计算两者5期滚动相关系数
        4. 计算相关系数的3期滚动最大值
        5. 取负值作为因子值
        
        Args:
            df: 包含volume和high列的DataFrame
            
        Returns:
            计算得到的因子值Series
        """
        try:
            required_cols = ["volume", "high"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"DataFrame缺少必要的列: {missing_cols}")
            
            # 成交量5期时间序列排名
            a = _g(df, "volume", lambda s: ops.ts_rank(s, 5))
            
            # 最高价5期时间序列排名
            b = _g(df, "high", lambda s: ops.ts_rank(s, 5))
            
            # 计算相关系数的3期滚动最大值
            val = -_g(df, None, 
                      lambda *_: ops.rolling_max(ops.rolling_corr(a, b, 5), 3))
            
            return Factor.as_cs_series(df, val)
            
        except Exception as e:
            raise RuntimeError(f"计算Alpha026因子时发生错误: {str(e)}") from e

@register
class Alpha033(Factor):
    """
    Alpha033因子实现。
    
    该因子基于开盘价和收盘价计算，通过开盘价与收盘价比率的
    倒数进行截面排名来构建因子。
    """
    name = "Alpha033"
    requires = ["open", "close"]
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha033因子值。
        
        因子计算逻辑：
        1. 计算开盘价与收盘价的比率
        2. 计算1减去比率的负值
        3. 对结果进行截面排名
        
        Args:
            df: 包含open和close列的DataFrame
            
        Returns:
            计算得到的因子值Series
        """
        try:
            required_cols = ["open", "close"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"DataFrame缺少必要的列: {missing_cols}")
            
            val = ops.cs_rank(-(1 - (df["open"] / df["close"])))
            
            return Factor.as_cs_series(df, val)
            
        except Exception as e:
            raise RuntimeError(f"计算Alpha033因子时发生错误: {str(e)}") from e

@register
class Alpha034(Factor):
    """
    Alpha034因子实现。
    
    该因子基于收益率和收盘价计算，通过收益率短期与长期波动率比率
    和收盘价变化的排名来构建因子。
    """
    name = "Alpha034"
    requires = ["returns", "close"]
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha034因子值。
        
        因子计算逻辑：
        1. 计算收益率2期与5期标准差比率的截面排名
        2. 计算收盘价1期变化的截面排名
        3. 两者相加得到因子值
        
        Args:
            df: 包含returns和close列的DataFrame
            
        Returns:
            计算得到的因子值Series
        """
        try:
            required_cols = ["returns", "close"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"DataFrame缺少必要的列: {missing_cols}")
            
            # 收益率短期与长期波动率比率的排名
            a = 1 - ops.cs_rank(_g(df, "returns", 
                                   lambda s: ops.rolling_std(s, 2) / ops.rolling_std(s, 5)))
            
            # 收盘价1期变化的排名
            b = 1 - ops.cs_rank(_g(df, "close", ops.delta, 1))
            
            val = a + b
            
            return Factor.as_cs_series(df, val)
            
        except Exception as e:
            raise RuntimeError(f"计算Alpha034因子时发生错误: {str(e)}") from e

@register
class Alpha035(Factor):
    """
    Alpha035因子实现。
    
    该因子基于成交量、收盘价、最高价、最低价和收益率计算，通过
    成交量排名、价格区间排名和收益率排名的组合来构建因子。
    """
    name = "Alpha035"
    requires = ["volume", "close", "high", "low", "returns"]
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha035因子值。
        
        因子计算逻辑：
        1. 计算成交量32期时间序列排名
        2. 计算价格区间（收盘价+最高价-最低价）的16期时间序列排名
        3. 计算收益率32期时间序列排名
        4. 三个排名相乘得到因子值
        
        Args:
            df: 包含volume、close、high、low和returns列的DataFrame
            
        Returns:
            计算得到的因子值Series
        """
        try:
            required_cols = ["volume", "close", "high", "low", "returns"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"DataFrame缺少必要的列: {missing_cols}")
            
            # 成交量32期时间序列排名
            a = _g(df, "volume", lambda s: ops.ts_rank(s, 32))
            
            # 价格区间16期时间序列排名
            b = 1 - _g(df, None, 
                       lambda *_: ops.ts_rank(((df["close"] + df["high"]) - df["low"]), 16))
            
            # 收益率32期时间序列排名
            c = 1 - _g(df, "returns", lambda s: ops.ts_rank(s, 32))
            
            val = a * b * c
            
            return Factor.as_cs_series(df, val)
            
        except Exception as e:
            raise RuntimeError(f"计算Alpha035因子时发生错误: {str(e)}") from e

@register
class Alpha038(Factor):
    """
    Alpha038因子实现。
    
    该因子基于收盘价和开盘价计算，通过收盘价10期时间序列排名
    与收盘价开盘价比率的截面排名相乘来构建因子。
    """
    name = "Alpha038"
    requires = ["close", "open"]
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha038因子值。
        
        因子计算逻辑：
        1. 计算收盘价10期时间序列排名
        2. 计算收盘价与开盘价比率的截面排名
        3. 两者相乘并取负值
        
        Args:
            df: 包含close和open列的DataFrame
            
        Returns:
            计算得到的因子值Series
        """
        try:
            required_cols = ["close", "open"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"DataFrame缺少必要的列: {missing_cols}")
            
            # 收盘价10期时间序列排名
            close_ts_rank = _g(df, "close", lambda s: ops.ts_rank(s, 10))
            
            # 收盘价与开盘价比率的截面排名
            close_open_ratio_rank = ops.cs_rank(df["close"] / df["open"])
            
            val = -close_ts_rank * close_open_ratio_rank
            
            return Factor.as_cs_series(df, val)
            
        except Exception as e:
            raise RuntimeError(f"计算Alpha038因子时发生错误: {str(e)}") from e

@register
class Alpha040(Factor):
    """
    Alpha040因子实现。

    该因子使用最高价的10期滚动标准差的截面排名与
    最高价和成交量的10期滚动相关系数相乘并取负号。
    """
    name = "Alpha040"
    requires = ["high","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha040因子值。

        Args:
            df: 包含high与volume列的DataFrame

        Returns:
            因子值的Series
        """
        try:
            val = (- ops.cs_rank(_g(df,"high", ops.rolling_std, 10))) * _g(df,"high", lambda s: ops.rolling_corr(s, df.loc[s.index,"volume"], 10))
            return Factor.as_cs_series(df, val)
        except Exception as e:
            raise RuntimeError(f"计算Alpha040因子时发生错误: {str(e)}") from e

@register
class Alpha041(Factor):
    """
    Alpha041因子实现。

    该因子计算几何均价减去VWAP的差值。
    """
    name = "Alpha041"
    requires = ["high","low","vwap"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha041因子值。

        Args:
            df: 包含high、low、vwap列的DataFrame

        Returns:
            因子值的Series
        """
        try:
            val = (df["high"]*df["low"])**0.5 - df["vwap"]
            return Factor.as_cs_series(df, val)
        except Exception as e:
            raise RuntimeError(f"计算Alpha041因子时发生错误: {str(e)}") from e

@register
class Alpha042(Factor):
    """
    Alpha042因子实现。

    该因子为VWAP与收盘价之差的截面排名除以两者之和的截面排名。
    """
    name = "Alpha042"
    requires = ["vwap","close"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha042因子值。

        Args:
            df: 包含vwap、close列的DataFrame

        Returns:
            因子值的Series
        """
        try:
            val = ops.cs_rank(df["vwap"] - df["close"]) / ops.cs_rank(df["vwap"] + df["close"])
            return Factor.as_cs_series(df, val)
        except Exception as e:
            raise RuntimeError(f"计算Alpha042因子时发生错误: {str(e)}") from e

@register
class Alpha043(Factor):
    """
    Alpha043因子实现。

    该因子将成交量相对ADV20的20期时间序列排名与
    收盘价7期负变化的8期时间序列排名相乘。
    """
    name = "Alpha043"
    requires = ["volume","close","returns","vwap"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha043因子值。

        Args:
            df: 包含volume、close列的DataFrame

        Returns:
            因子值的Series
        """
        try:
            adv20 = _g(df,"volume", lambda s: ops.adv(s, 20))
            val = _g(df, None, lambda *_: ops.ts_rank(df["volume"]/adv20, 20)) * _g(df,"close", lambda s: ops.ts_rank(- ops.delta(s,7), 8))
            return Factor.as_cs_series(df, val)
        except Exception as e:
            raise RuntimeError(f"计算Alpha043因子时发生错误: {str(e)}") from e

@register
class Alpha045(Factor):
    """
    Alpha045因子实现。

    该因子由收盘价延迟均值、收盘价与成交量的相关性、
    以及收盘价不同窗口滚动和的相关性组合而成并取负号。
    """
    name = "Alpha045"
    requires = ["close","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha045因子值。

        Args:
            df: 包含close、volume列的DataFrame

        Returns:
            因子值的Series
        """
        try:
            a = ops.cs_rank(_g(df,"close", lambda s: ops.rolling_sum(ops.delay(s,5), 20)/20))
            b = _g(df,"close", lambda s: ops.rolling_corr(s, df.loc[s.index,"volume"], 2))
            c = _g(df,"close", lambda s: ops.rolling_corr(_g(df,"close", lambda s2: ops.rolling_sum(s2,5)), _g(df,"close", lambda s2: ops.rolling_sum(s2,20)), 2))
            val = - (a * b * ops.cs_rank(c))
            return Factor.as_cs_series(df, val)
        except Exception as e:
            raise RuntimeError(f"计算Alpha045因子时发生错误: {str(e)}") from e

@register
class Alpha046(Factor):
    """
    Alpha046因子实现。

    该因子基于收盘价不同延迟的差分构造门限信号。
    """
    name = "Alpha046"
    requires = ["close"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha046因子值。

        Args:
            df: 包含close列的DataFrame

        Returns:
            因子值的Series
        """
        try:
            a = (_g(df,"close", ops.delay,20) - _g(df,"close", ops.delay,10))/10 - (_g(df,"close", ops.delay,10) - df["close"])/10
            val = np.where(a > 0.25, -1, np.where(a < 0, 1, -1*(df["close"] - _g(df,"close", ops.delay,1))))
            return Factor.as_cs_series(df, pd.Series(val))
        except Exception as e:
            raise RuntimeError(f"计算Alpha046因子时发生错误: {str(e)}") from e

@register
class Alpha049(Factor):
    """
    Alpha049因子实现。

    该因子与Alpha046类似，但使用不同阈值规则。
    """
    name = "Alpha049"
    requires = ["close"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha049因子值。

        Args:
            df: 包含close列的DataFrame

        Returns:
            因子值的Series
        """
        try:
            a = (_g(df,"close", ops.delay,20) - _g(df,"close", ops.delay,10))/10 - (_g(df,"close", ops.delay,10) - df["close"])/10
            val = np.where(a < -0.1, 1, -1*(df["close"] - _g(df,"close", ops.delay,1)))
            return Factor.as_cs_series(df, pd.Series(val))
        except Exception as e:
            raise RuntimeError(f"计算Alpha049因子时发生错误: {str(e)}") from e

@register
class Alpha050(Factor):
    """
    Alpha050因子实现。

    该因子为成交量与VWAP截面排名的5期滚动相关的5期滚动最大值的负号。
    """
    name = "Alpha050"
    requires = ["volume","vwap"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha050因子值。

        Args:
            df: 包含volume、vwap列的DataFrame

        Returns:
            因子值的Series
        """
        try:
            val = - _g(df, None, lambda *_: ops.rolling_max(
                ops.cs_rank(_g(df,"volume", lambda s: ops.rolling_corr(ops.cs_rank(s), ops.cs_rank(df.loc[s.index,"vwap"]), 5))), 5))
            return Factor.as_cs_series(df, val)
        except Exception as e:
            raise RuntimeError(f"计算Alpha050因子时发生错误: {str(e)}") from e

@register
class Alpha051(Factor):
    """
    Alpha051因子实现。

    该因子基于收盘价延迟差分构造阈值信号，阈值不同于Alpha046/049。
    """
    name = "Alpha051"
    requires = ["close"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha051因子值。

        Args:
            df: 包含close列的DataFrame

        Returns:
            因子值的Series
        """
        try:
            a = (_g(df,"close", ops.delay,20) - _g(df,"close", ops.delay,10))/10 - (_g(df,"close", ops.delay,10) - df["close"])/10
            val = np.where(a < -0.05, 1, -1*(df["close"] - _g(df,"close", ops.delay,1)))
            return Factor.as_cs_series(df, pd.Series(val))
        except Exception as e:
            raise RuntimeError(f"计算Alpha051因子时发生错误: {str(e)}") from e

@register
class Alpha052(Factor):
    """
    Alpha052因子实现。

    该因子由最低价区间突破项、收益率长短期累加差的排名、
    以及成交量的时间序列排名相乘得到。
    """
    name = "Alpha052"
    requires = ["low","returns","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha052因子值。

        Args:
            df: 包含low、returns、volume列的DataFrame

        Returns:
            因子值的Series
        """
        try:
            part = (- _g(df,"low", lambda s: ops.rolling_min(s,5)) + _g(df,"low", lambda s: ops.delay(ops.rolling_min(s,5),5))) * ops.cs_rank((_g(df,"returns", ops.rolling_sum,240) - _g(df,"returns", ops.rolling_sum,20))/220) * _g(df,"volume", lambda s: ops.ts_rank(s,5))
            return Factor.as_cs_series(df, part)
        except Exception as e:
            raise RuntimeError(f"计算Alpha052因子时发生错误: {str(e)}") from e

@register
class Alpha053(Factor):
    """
    Alpha053因子实现。

    该因子基于价格位置不平衡度的9期变化并取负号。
    """
    name = "Alpha053"
    requires = ["close","low","high"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha053因子值。

        Args:
            df: 包含close、low、high列的DataFrame

        Returns:
            因子值的Series
        """
        try:
            x = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["close"] - df["low"]).replace(0,np.nan)
            val = - _g(df, None, lambda *_: ops.delta(x, 9))
            return Factor.as_cs_series(df, val)
        except Exception as e:
            raise RuntimeError(f"计算Alpha053因子时发生错误: {str(e)}") from e

@register
class Alpha055(Factor):
    """
    Alpha055因子实现。

    该因子基于价格在区间中的位置与成交量的滚动相关并取负号。
    """
    name = "Alpha055"
    requires = ["close","high","low","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha055因子值。

        Args:
            df: 包含close、high、low、volume列的DataFrame

        Returns:
            因子值的Series
        """
        try:
            num = (df["close"] - _g(df,"low", lambda s: ops.rolling_min(s,12))) / (_g(df,"high", lambda s: ops.rolling_max(s,12)) - _g(df,"low", lambda s: ops.rolling_min(s,12))).replace(0,np.nan)
            val = - _g(df, None, lambda *_: ops.rolling_corr(ops.cs_rank(num), ops.cs_rank(df["volume"]), 6))
            return Factor.as_cs_series(df, val)
        except Exception as e:
            raise RuntimeError(f"计算Alpha055因子时发生错误: {str(e)}") from e

@register
class Alpha060(Factor):
    """
    Alpha060因子实现。

    该因子综合价格位置与成交量，经衰减和时间序列排名后构造。
    """
    name = "Alpha060"
    requires = ["high","low","close","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha060因子值。

        Args:
            df: 包含high、low、close、volume列的DataFrame

        Returns:
            因子值的Series
        """
        try:
            x = (((df["close"]-df["low"]) - (df["high"]-df["close"])) / (df["high"]-df["low"]).replace(0,np.nan)) * df["volume"]
            val = - ( 2*ops.cs_rank(ops.decay_linear(x, 10)) - ops.cs_rank(_g(df,"close", lambda s: ops.ts_rank(s,10))) )
            return Factor.as_cs_series(df, val)
        except Exception as e:
            raise RuntimeError(f"计算Alpha060因子时发生错误: {str(e)}") from e

@register
class Alpha101(Factor):
    """
    Alpha101因子实现。

    该因子为K线实体长度与当日价格区间之比。
    """
    name = "Alpha101"
    requires = ["open","high","low","close"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha101因子值。

        Args:
            df: 包含open、high、low、close列的DataFrame

        Returns:
            因子值的Series
        """
        try:
            val = (df["close"] - df["open"]) / ((df["high"] - df["low"]) + 0.001)
            return Factor.as_cs_series(df, val)
        except Exception as e:
            raise RuntimeError(f"计算Alpha101因子时发生错误: {str(e)}") from e


@register
class Alpha024(Factor):
    """
    Alpha024因子实现。

    该因子根据100期均值变化的条件，选择不同的价格动量项。
    """
    name = "Alpha024"
    requires = ["close"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha024因子值。

        Args:
            df: 包含close列的DataFrame

        Returns:
            因子值的Series
        """
        try:
            s100 = _g(df,"close", lambda s: ops.rolling_sum(s,100) / 100)
            d = _g(df,"close", lambda s: ops.delta(ops.rolling_sum(s,100)/100, 100)) / _g(df,"close", lambda s: ops.delay(s,100))
            cond = (d <= 0.05)
            val = np.where(cond, - (df["close"] - _g(df,"close", ops.rolling_min, 100)),
                                  - _g(df,"close", ops.delta, 3))
            return Factor.as_cs_series(df, pd.Series(val))
        except Exception as e:
            raise RuntimeError(f"计算Alpha024因子时发生错误: {str(e)}") from e

@register
class Alpha030(Factor):
    """
    Alpha030因子实现。

    该因子以收盘价近三天方向信号加权5期与20期成交量滚动和之比。
    """
    name = "Alpha030"
    requires = ["close","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha030因子值。

        Args:
            df: 包含close、volume列的DataFrame

        Returns:
            因子值的Series
        """
        try:
            s5 = _g(df,"volume", lambda s: ops.rolling_sum(s,5))
            s20 = _g(df,"volume", lambda s: ops.rolling_sum(s,20))
            sig = (1.0 - _cs_rank(df, (np.sign(ops.delta(_g(df,"close", ops.delay,1),1)) +
                                       np.sign(ops.delta(_g(df,"close", ops.delay,2),1)) +
                                       np.sign(ops.delta(_g(df,"close", ops.delay,3),1)))))
            val = (sig * s5) / s20.replace(0,np.nan)
            return Factor.as_cs_series(df, val)
        except Exception as e:
            raise RuntimeError(f"计算Alpha030因子时发生错误: {str(e)}") from e

@register
class Alpha031(Factor):
    """
    Alpha031因子实现。

    该因子由三个子项相加：价格动量、短期跌幅、以及成交量与低价的相关性方向。
    """
    name = "Alpha031"
    requires = ["close","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha031因子值。

        Args:
            df: 包含close、volume（可选low）列的DataFrame

        Returns:
            因子值的Series
        """
        try:
            a = ops.cs_rank(ops.decay_linear(- _cs_rank(_g(df,"close", lambda s: ops.delta(s,10))), 10))
            b = ops.cs_rank(- _g(df,"close", ops.delta, 3))
            adv20 = _g(df,"volume", lambda s: ops.adv(s,20))
            c = np.sign(ops.cs_rank(_g(df,"volume", lambda s: ops.rolling_corr(adv20, df.loc[s.index,"low"] if "low" in df.columns else s, 12))))
            val = a + b + c
            return Factor.as_cs_series(df, val)
        except Exception as e:
            raise RuntimeError(f"计算Alpha031因子时发生错误: {str(e)}") from e

@register
class Alpha032(Factor):
    """
    Alpha032因子实现。

    该因子为7期均价相对现价的排名与VWAP和滞后收盘价的相关排名组合。
    """
    name = "Alpha032"
    requires = ["close","vwap"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha032因子值。

        Args:
            df: 包含close、vwap列的DataFrame

        Returns:
            因子值的Series
        """
        try:
            part1 = ops.cs_rank((_g(df,"close", lambda s: ops.rolling_sum(s,7)/7) - df["close"]))
            part2 = 20 * ops.cs_rank(_g(df,"close", lambda s: ops.rolling_corr(df.loc[s.index,"vwap"], _g(df,"close", ops.delay,5), 230)))
            return Factor.as_cs_series(df, part1 + part2)
        except Exception as e:
            raise RuntimeError(f"计算Alpha032因子时发生错误: {str(e)}") from e

@register
class Alpha036(Factor):
    name = "Alpha036"
    requires = ["close","open","volume","vwap","returns"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        a = 2.21 * ops.cs_rank(_g(df, None, lambda *_: ops.rolling_corr(df["close"]-df["open"], _g(df,"volume", ops.delay,1), 15)))
        b = 0.7 * ops.cs_rank(df["open"] - df["close"])
        c = 0.73 * ops.cs_rank(_g(df, None, lambda *_: ops.ts_rank(ops.delay(-df["returns"],6), 5)))
        d = ops.cs_rank(np.abs(_g(df, None, lambda *_: ops.rolling_corr(df["vwap"], _g(df,"volume", lambda s: ops.adv(s,20)), 6))))
        e = 0.6 * ops.cs_rank((_g(df,"close", lambda s: ops.rolling_sum(s,200)/200) - df["open"]) * (df["close"] - df["open"]))
        val = a + b + c + d + e
        return Factor.as_cs_series(df, val)

@register
class Alpha037(Factor):
    name = "Alpha037"
    requires = ["open","close"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        a = ops.cs_rank(_g(df, None, lambda *_: ops.rolling_corr(ops.delay(df["open"]-df["close"],1), df["close"], 200)))
        b = ops.cs_rank(df["open"] - df["close"])
        return Factor.as_cs_series(df, a + b)

@register
class Alpha039(Factor):
    name = "Alpha039"
    requires = ["close","volume","returns"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        adv20 = _g(df,"volume", lambda s: ops.adv(s,20))
        part = - ops.cs_rank(_g(df,"close", lambda s: ops.delta(s,7)) * (1 - ops.cs_rank(ops.decay_linear(df["volume"]/adv20, 9))))
        val = part * (1 + ops.cs_rank(_g(df,"returns", lambda s: ops.rolling_sum(s,250))))
        return Factor.as_cs_series(df, val)

@register
class Alpha044(Factor):
    name = "Alpha044"
    requires = ["high","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        val = - _g(df,"high", lambda s: ops.rolling_corr(s, ops.cs_rank(df.loc[s.index,"volume"]), 5))
        return Factor.as_cs_series(df, val)

@register
class Alpha047(Factor):
    name = "Alpha047"
    requires = ["close","high","vwap","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        adv20 = _g(df,"volume", lambda s: ops.adv(s,20))
        part1 = (ops.cs_rank(1/df["close"]) * df["volume"]) / adv20
        part2 = (df["high"] * ops.cs_rank(df["high"]-df["close"])) / (_g(df,"high", lambda s: ops.rolling_sum(s,5))/5)
        val = part1 * part2 - ops.cs_rank(df["vwap"] - _g(df,"vwap", ops.delay,5))
        return Factor.as_cs_series(df, val)

@register
class Alpha054(Factor):
    name = "Alpha054"
    requires = ["low","close","open","high"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        val = - ((df["low"] - df["close"]) * (df["open"]**5)) / ((df["low"] - df["high"]) * (df["close"]**5))
        return Factor.as_cs_series(df, val)

@register
class Alpha061(Factor):
    name = "Alpha061"
    requires = ["vwap","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        adv180 = _g(df,"volume", lambda s: ops.adv(s,180))
        a = ops.cs_rank(df["vwap"] - _g(df,"vwap", lambda s: ops.rolling_min(s, int(16.1219))))
        b = ops.cs_rank(_g(df, None, lambda *_: ops.rolling_corr(df["vwap"], adv180, int(17.9282))))
        val = (a < b).astype(float)
        return Factor.as_cs_series(df, val)

@register
class Alpha064(Factor):
    name = "Alpha064"
    requires = ["open","low","vwap","close"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        a = ops.cs_rank(_g(df, None, lambda *_: ops.rolling_corr((_g(df,"open", lambda s: 0.178404*s) + (df["low"]*(1-0.178404))), _g(df,"volume", lambda s: ops.adv(s,120)), int(16.6208))))
        b = ops.cs_rank(_g(df, None, lambda *_: ops.delta((((df["high"]+df["low"])/2)*0.178404 + df["vwap"]*(1-0.178404)), int(3.69741))))
        val = (a < b).astype(float) * -1
        return Factor.as_cs_series(df, val)

@register
class Alpha065(Factor):
    """
    Alpha065因子实现。

    比较开盘价与VWAP线性组合和ADV60的相关性排名，
    与开盘价相对近期低点的排名大小关系，取小于关系的负号。
    """
    name = "Alpha065"
    requires = ["open","vwap","low"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha065因子值。

        Args:
            df: 包含open、vwap、low（可选volume）列的DataFrame

        Returns:
            因子值的Series
        """
        try:
            a = ops.cs_rank(_g(df, None, lambda *_: ops.rolling_corr(0.00817205*df["open"] + (1-0.00817205)*df["vwap"], _g(df,"volume", lambda s: ops.adv(s,60)), int(6.40374))))
            b = ops.cs_rank(df["open"] - _g(df,"open", lambda s: ops.rolling_min(s, int(13.635))))
            val = (a < b).astype(float) * -1
            return Factor.as_cs_series(df, val)
        except Exception as e:
            raise RuntimeError(f"计算Alpha065因子时发生错误: {str(e)}") from e

@register
class Alpha071(Factor):
    """
    Alpha071因子实现。

    取两个复杂衰减与排名项的逐点最大值。
    """
    name = "Alpha071"
    requires = ["close","volume","low","open","vwap"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha071因子值。

        Args:
            df: 包含close、volume、low、open、vwap列的DataFrame

        Returns:
            因子值的Series
        """
        try:
            a = _g(df, None, lambda *_: ops.ts_rank(ops.decay_linear(_g(df,"close", lambda s: ops.ts_rank(s, int(3.43976))), int(4.20501)), int(15.6948)))
            b = _g(df, None, lambda *_: ops.ts_rank(ops.decay_linear(ops.cs_rank(((df["low"]+df["open"])-(df["vwap"]+df["vwap"]))**2), int(16.4662)), int(4.4388)))
            val = np.maximum(a, b)
            return Factor.as_cs_series(df, val)
        except Exception as e:
            raise RuntimeError(f"计算Alpha071因子时发生错误: {str(e)}") from e

@register
class Alpha083(Factor):
    """
    Alpha083因子实现。

    价差比例的滞后排名与成交量排名之积，
    除以价差比例与VWAP-收盘价的比值。
    """
    name = "Alpha083"
    requires = ["high","low","close","vwap","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha083因子值。

        Args:
            df: 包含high、low、close、vwap、volume列的DataFrame

        Returns:
            因子值的Series
        """
        try:
            num = ops.cs_rank(ops.delay(((df["high"]-df["low"]) / (_g(df,"close", lambda s: ops.rolling_sum(s,5))/5)), 2)) * ops.cs_rank(ops.cs_rank(df["volume"]))
            den = ((df["high"]-df["low"]) / (_g(df,"close", lambda s: ops.rolling_sum(s,5))/5)) / (df["vwap"] - df["close"]).replace(0,np.nan)
            val = num / den.replace(0,np.nan)
            return Factor.as_cs_series(df, val)
        except Exception as e:
            raise RuntimeError(f"计算Alpha083因子时发生错误: {str(e)}") from e

@register
class Alpha084(Factor):
    """
    Alpha084因子实现。

    收盘价约5期变化方向与VWAP相对近期最高值的时间序列排名之积。
    """
    name = "Alpha084"
    requires = ["vwap","close"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha084因子值。

        Args:
            df: 包含vwap、close列的DataFrame

        Returns:
            因子值的Series
        """
        try:
            val = np.sign(_g(df,"close", ops.delta, int(4.96796))) * _g(df,"vwap", lambda s: ops.ts_rank(s - _g(df,"vwap", ops.rolling_max, int(15.3217)), int(20.7127)))
            return Factor.as_cs_series(df, val)
        except Exception as e:
            raise RuntimeError(f"计算Alpha084因子时发生错误: {str(e)}") from e

@register
class Alpha085(Factor):
    """
    Alpha085因子实现。

    两个相关性项的截面排名做幂运算。
    """
    name = "Alpha085"
    requires = ["high","close","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha085因子值。

        Args:
            df: 包含high、close、volume（可选low）列的DataFrame

        Returns:
            因子值的Series
        """
        try:
            a = _g(df, None, lambda *_: ops.rolling_corr(0.876703*df["high"] + (1-0.876703)*df["close"], _g(df,"volume", lambda s: ops.adv(s,30)), int(9.61331)))
            b = _g(df, None, lambda *_: ops.rolling_corr(_g(df,"close", lambda s: ops.ts_rank((df["high"]+df["low"])/2, int(3.70596))), _g(df,"volume", lambda s: ops.ts_rank(df["volume"], int(10.1595))), int(7.11408)))
            val = ops.cs_rank(a) ** ops.cs_rank(b)
            return Factor.as_cs_series(df, val)
        except Exception as e:
            raise RuntimeError(f"计算Alpha085因子时发生错误: {str(e)}") from e

@register
class Alpha086(Factor):
    """
    Alpha086因子实现。

    比较收盘价与ADV20相关的排名项与开收VWAP差的排名大小，
    小于关系取负号。
    """
    name = "Alpha086"
    requires = ["close","open","vwap","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha086因子值。

        Args:
            df: 包含close、open、vwap、volume列的DataFrame

        Returns:
            因子值的Series
        """
        try:
            a = _g(df,"close", lambda s: ops.ts_rank(ops.rolling_corr(s, _g(df,"volume", lambda s2: ops.adv(s2,20)) .groupby(level=0, group_keys=False) if False else _g(df,"volume", lambda s2: ops.adv(s2,20)), int(6.00049)), int(20.4195)))
            b = ops.cs_rank((df["open"] + df["close"]) - (df["vwap"] + df["open"]))
            val = (a < b).astype(float) * -1
            return Factor.as_cs_series(df, val)
        except Exception as e:
            raise RuntimeError(f"计算Alpha086因子时发生错误: {str(e)}") from e

@register
class Alpha094(Factor):
    """
    Alpha094因子实现。

    将VWAP相对近期最小值的排名与一个相关性组合项的时间序列排名做幂后取负。
    """
    name = "Alpha094"
    requires = ["vwap","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha094因子值。

        Args:
            df: 包含vwap、volume列的DataFrame

        Returns:
            因子值的Series
        """
        try:
            adv60 = _g(df,"volume", lambda s: ops.adv(s,60))
            a = ops.cs_rank(df["vwap"] - _g(df,"vwap", lambda s: ops.rolling_min(s, int(11.5783))))
            b = _g(df, None, lambda *_: ops.ts_rank(ops.rolling_corr(_g(df,"vwap", lambda s: ops.ts_rank(s, int(19.6462))),
                                                                     _g(df,"volume", lambda s: ops.ts_rank(adv60, int(4.02992))), int(18.0926)), int(2.70756)))
            val = (a ** b) * -1
            return Factor.as_cs_series(df, val)
        except Exception as e:
            raise RuntimeError(f"计算Alpha094因子时发生错误: {str(e)}") from e

@register
class Alpha095(Factor):
    """
    Alpha095因子实现。

    比较开盘价相对近期低点的排名与一个价格/量相关性项的五次幂排名。
    """
    name = "Alpha095"
    requires = ["open","high","low","close","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha095因子值。

        Args:
            df: 包含open、high、low、close、volume列的DataFrame

        Returns:
            因子值的Series
        """
        try:
            a = ops.cs_rank(_g(df, None, lambda *_: ops.rolling_corr(_g(df,"close", lambda s: ops.rolling_sum((df["high"]+df["low"])/2, int(19.1351))),
                                                                     _g(df,"volume", lambda s: ops.adv(s,40)), int(12.8742))) ** 5)
            b = _g(df,"open", lambda s: ops.ts_rank(s - _g(df,"open", lambda s2: ops.rolling_min(s2, int(12.4105))), 1))
            val = (ops.cs_rank(df["open"] - _g(df,"open", lambda s: ops.rolling_min(s, int(12.4105)))) < a).astype(float)
            return Factor.as_cs_series(df, val)
        except Exception as e:
            raise RuntimeError(f"计算Alpha095因子时发生错误: {str(e)}") from e

@register
class Alpha096(Factor):
    """
    Alpha096因子实现。

    取两个与VWAP、成交量、收盘价相关的复杂排名项的负的最大值。
    """
    name = "Alpha096"
    requires = ["vwap","volume","close"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha096因子值。

        Args:
            df: 包含vwap、volume、close列的DataFrame

        Returns:
            因子值的Series
        """
        try:
            a = _g(df, None, lambda *_: ops.ts_rank(ops.decay_linear(ops.rolling_corr(ops.cs_rank(df["vwap"]), ops.cs_rank(df["volume"]), int(3.83878)), int(4.16783)), int(8.38151)))
            b = _g(df, None, lambda *_: ops.ts_rank(ops.decay_linear(ops.ts_rank(_g(df,"close", lambda s: ops.rolling_corr(ops.cs_rank(s), _g(df,"volume", lambda s2: ops.adv(s2,60)), int(4.13242))), int(7.45404)), int(14.0365)), int(13.4143)))
            val = - np.maximum(a, b)
            return Factor.as_cs_series(df, val)
        except Exception as e:
            raise RuntimeError(f"计算Alpha096因子时发生错误: {str(e)}") from e

@register
class Alpha098(Factor):
    """
    Alpha098因子实现。

    VWAP与成交量组合项的相关排名，减去一个基于开盘价与成交量相关的
    复杂时间序列排名项。
    """
    name = "Alpha098"
    requires = ["vwap","volume","open"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha098因子值。

        Args:
            df: 包含vwap、volume、open列的DataFrame

        Returns:
            因子值的Series
        """
        try:
            adv5 = _g(df,"volume", lambda s: ops.adv(s,5))
            a = ops.cs_rank(_g(df, None, lambda *_: ops.rolling_corr(df["vwap"], _g(df,"volume", lambda s: ops.rolling_sum(adv5, int(26.4719))), int(4.58418))))
            b = _g(df, None, lambda *_: ops.ts_rank(ops.ts_rank(ops.argmin(ops.rolling_corr(ops.cs_rank(df["open"]), _g(df,"volume", lambda s: ops.adv(s,15)), int(20.8187))), int(6.95668)), int(8.07206))) if hasattr(np, "argmin") else a*0
            val = a - b
            return Factor.as_cs_series(df, val)
        except Exception as e:
            raise RuntimeError(f"计算Alpha098因子时发生错误: {str(e)}") from e

@register
class Alpha099(Factor):
    """
    Alpha099因子实现。

    比较两个价格与成交量的相关排名大小并取小于关系的负号。
    """
    name = "Alpha099"
    requires = ["high","low","volume"]
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha099因子值。

        Args:
            df: 包含high、low、volume（可选close）列的DataFrame

        Returns:
            因子值的Series
        """
        try:
            a = _g(df, None, lambda *_: ops.rolling_corr(_g(df,"close", lambda s: ops.rolling_sum((df["high"]+df["low"])/2, int(19.8975))), _g(df,"volume", lambda s: ops.adv(s,60)), int(8.8136)))
            b = _g(df, None, lambda *_: ops.rolling_corr(df["low"], df["volume"], int(6.28259)))
            val = (ops.cs_rank(a) < ops.cs_rank(b)).astype(float) * -1
            return Factor.as_cs_series(df, val)
        except Exception as e:
            raise RuntimeError(f"计算Alpha099因子时发生错误: {str(e)}") from e
