# -*- coding: utf-8 -*-
"""
因子注册与动态发现模块 (Factor Registry)

本模块主要提供以下功能：
1. 通过装饰器 `@register` 注册自定义因子类；
2. 自动发现并加载 `alpha101_factory.factors` 包下的所有因子模块
   （排除 `base.py` 与 `registry.py`，避免循环依赖）；
3. 提供统一接口 `get_factor` 与 `list_factors` 来获取已注册的因子。

该设计有助于实现因子库的模块化与可扩展性，便于在回测框架中动态调用。
"""

import sys
from pathlib import Path
from typing import Dict, Type
import importlib
import pkgutil

try:
    # 将项目根目录添加到 sys.path，确保跨目录调用时能正确导入
    sys.path.append(str(Path(__file__).resolve().parents[2]))
except Exception as e:
    raise RuntimeError("无法设置项目路径，请检查目录结构是否正确") from e

try:
    from alpha101_factory.factors.base import Factor
except ImportError as e:
    raise ImportError("无法导入 Factor 基类，请确认 alpha101_factory.factors.base 是否存在") from e


# ===== 全局注册表 =====
_REGISTRY: Dict[str, Type[Factor]] = {}   # 存放因子名称 -> 因子类 的映射
_LOADED: bool = False                    # 标记因子模块是否已加载


def register(cls: Type[Factor]) -> Type[Factor]:
    """注册因子类到全局注册表。

    Args:
        cls (Type[Factor]): 继承自 Factor 的因子类。

    Returns:
        Type[Factor]: 原因子类本身，便于装饰器链式调用。

    Raises:
        ValueError: 若因子类未定义 `name` 属性或该属性已存在于注册表中。
    """
    if not hasattr(cls, "name"):
        raise ValueError(f"因子类 {cls.__name__} 缺少 'name' 属性，无法注册")

    if cls.name in _REGISTRY:
        raise ValueError(f"因子名称重复: {cls.name} 已存在，请检查命名冲突")

    _REGISTRY[cls.name] = cls
    return cls


def _ensure_loaded() -> None:
    """确保因子模块已加载。

    使用 `pkgutil.iter_modules` 动态发现并导入 `alpha101_factory.factors`
    包下的所有模块（除 base 与 registry），以便自动完成因子注册。

    Raises:
        RuntimeError: 若包加载或模块导入过程中出现错误。
    """
    global _LOADED
    if _LOADED:
        return

    try:
        # 导入父包 alpha101_factory.factors
        package = importlib.import_module(__package__)
    except Exception as e:
        raise RuntimeError(f"无法导入因子包 {__package__}") from e

    try:
        for _, mod_name, ispkg in pkgutil.iter_modules(package.__path__):
            if ispkg:
                continue
            if mod_name in ("base", "registry"):
                continue
            try:
                # 动态导入模块，例如 alpha101_factory.factors.alphas_basic
                importlib.import_module(f"{__package__}.{mod_name}")
            except Exception as inner_e:
                # 出现错误时不中断整体流程，仅打印警告
                print(f"[警告] 无法加载模块 {mod_name}: {inner_e}")
    except Exception as e:
        raise RuntimeError("扫描因子包时出错，请检查包路径与模块定义") from e

    _LOADED = True


def get_factor(name: str) -> Type[Factor]:
    """根据因子名称获取对应的因子类。

    Args:
        name (str): 因子名称（需在注册表中已注册）。

    Returns:
        Type[Factor]: 对应的因子类。

    Raises:
        KeyError: 若因子名称不存在于注册表。
    """
    _ensure_loaded()
    if name not in _REGISTRY:
        raise KeyError(f"未找到因子: {name}")
    return _REGISTRY[name]


def list_factors() -> list[str]:
    """列出所有已注册的因子名称。

    Returns:
        list[str]: 按字母排序的因子名称列表。
    """
    _ensure_loaded()
    return sorted(_REGISTRY.keys())
