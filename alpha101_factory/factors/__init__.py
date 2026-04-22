# -*- coding: utf-8 -*-
"""Alpha101 Factory 因子模块。

提供 Alpha 因子的定义、注册、发现与计算功能，包括：

- **因子基类**: ``Factor`` 抽象基类定义统一的因子接口，提供
  列校验、截面排名、分组计算等工具方法。
- **因子注册表**: 通过 ``@register`` 装饰器自动发现并注册 ``alphas_*.py``
  模块中定义的因子类，支持动态扩展。
- **因子工厂**: ``FactorFactory`` 提供因子的创建、校验、单例/批量计算。
- **内置因子**: ``alphas_basic`` 与 ``alphas_more`` 模块提供预定义的 Alpha 因子。

因子注册机制::

    1. 在 factors/ 目录下创建 alphas_*.py 文件
    2. 继承 Factor 基类，设置 name 和 requires 属性
    3. 使用 @register 装饰器注册因子类
    4. 因子自动被发现并加入注册表，无需手动配置

典型用法::

    from alpha101_factory.factors import (
        Factor, FactorFactory, register, list_factors, get_factor,
    )
    from alpha101_factory.utils.ops import returns, decay_linear

    # 查看已注册因子
    print(list_factors())

    # 自定义因子
    @register
    class MyAlpha(Factor):
        name = "MyAlpha"
        requires = ["close", "volume"]

        def compute(self, df):
            ret = returns(df["close"])
            return self.as_cs_series(df, decay_linear(ret, 10))

    # 通过工厂计算因子
    factory = FactorFactory()
    result = factory.compute("MyAlpha", df)
"""
from __future__ import annotations

from alpha101_factory.factors.base import Factor
from alpha101_factory.factors.registry import (
    register,
    get_factor,
    list_factors,
    FactorFactory,
)

# 显式导入因子模块以触发 @register 装饰器
# 若新增 alphas_*.py 文件，需在此处添加导入或依赖 registry.ensure_loaded()
from alpha101_factory.factors import alphas_basic  # noqa: F401

try:
    from alpha101_factory.factors import alphas_more  # noqa: F401
except Exception:
    pass

__all__ = [
    # 因子基类
    "Factor",
    # 因子注册与查询
    "register",
    "get_factor",
    "list_factors",
    # 因子工厂
    "FactorFactory",
]
