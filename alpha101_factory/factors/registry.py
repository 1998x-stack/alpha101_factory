# -*- coding: utf-8 -*-
"""因子注册与发现模块 (Factor Registration and Discovery Module).

本模块提供 Alpha 因子的自动发现、注册、查询与批量计算功能。核心机制如下:

1. **自动发现**: 通过 ``pkgutil.iter_modules`` 扫描本包下所有 ``alphas_*.py`` 模块,
   触发其中的 ``@register`` 装饰器将因子类写入全局注册表。
2. **装饰器注册**: 使用 ``@register`` 装饰因子类, 自动校验合法性并登记。
3. **工厂模式**: ``FactorFactory`` 提供因子的创建、校验、单例/批量计算与信息查询。

所有已注册因子可通过 ``list_factors()`` 获取名称列表, 通过 ``get_factor(name)``
获取因子类, 或通过 ``FactorFactory`` 进行实例化与计算。

典型用法::

    from alpha101_factory.factors.registry import register, get_factor, list_factors, FactorFactory
    from alpha101_factory.factors.base import Factor

    @register
    class MyAlpha(Factor):
        name = "MyAlpha"
        requires = ["close", "volume"]

        def compute(self, df):
            return self.as_cs_series(df, df["close"] - df["volume"])

    # 查询已注册因子
    print(list_factors())  # ['Alpha101', 'MyAlpha', ...]

    # 通过工厂计算因子
    factory = FactorFactory()
    result = factory.compute("MyAlpha", df)

注意事项:
    - 因子名称必须全局唯一, 重复注册将抛出 ``ValueError``。
    - 因子类必须继承 ``Factor`` 基类并设置 ``name`` 类属性。
    - 单股票场景下横截面 IC 为 NaN, 请使用 TS-IC 指标。
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Type

import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# 项目路径设置 (Project Path Setup)
# ---------------------------------------------------------------------------
# 将项目根目录加入 sys.path, 确保跨目录导入时模块解析正确。
# 目录结构: alpha101_factory/factors/registry.py → 上溯 2 级为项目根。
try:
    _project_root = Path(__file__).resolve().parents[2]
    _project_root_str = str(_project_root)
    if _project_root_str not in sys.path:
        sys.path.append(_project_root_str)
except Exception as exc:
    raise RuntimeError(
        f"无法设置项目路径, 请检查目录结构是否正确。"
        f"当前文件: {__file__}"
    ) from exc

# ---------------------------------------------------------------------------
# 基类导入 (Base Class Import)
# ---------------------------------------------------------------------------
# 导入 Factor 基类, 用于注册时的类型校验。
try:
    from alpha101_factory.factors.base import Factor
except ImportError as exc:
    raise ImportError(
        f"无法导入 Factor 基类, 请确认 alpha101_factory.factors.base 模块是否存在。"
        f"原始错误: {exc}"
    ) from exc


# ===================================================================
# 全局注册表 (Global Registry)
# ===================================================================
# _REGISTRY: 因子名称 → 因子类的映射字典, 键为 str, 值为 Factor 子类。
# _LOADED:   懒加载标志位, 确保模块扫描仅执行一次。
_REGISTRY: Dict[str, Type[Factor]] = {}
_LOADED: bool = False


# ===================================================================
# 注册装饰器 (Registration Decorator)
# ===================================================================
def register(factor_class: Type[Factor]) -> Type[Factor]:
    """注册因子类的装饰器 (Decorator for Registering Factor Classes)。

    将因子类登记到全局注册表中, 注册前执行以下校验:
        1. 因子类必须是 ``Factor`` 的子类。
        2. 因子类必须定义 ``name`` 类属性且非空字符串。
        3. 因子名称在全局注册表中必须唯一。

    用法::

        @register
        class MyAlpha(Factor):
            name = "MyAlpha"
            requires = ["close", "volume"]

            def compute(self, df):
                ...

    Args:
        factor_class: 待注册的因子类, 必须继承自 ``Factor`` 基类。

    Returns:
        原样返回输入的因子类, 以便装饰器语法正常工作。

    Raises:
        TypeError: 当 ``factor_class`` 不是 ``Factor`` 的子类时抛出。
        ValueError: 当因子类缺少 ``name`` 属性、``name`` 为空字符串,
            或因子名称已存在于注册表中时抛出。
    """
    # 校验 1: 必须是 Factor 的子类 (Must be a subclass of Factor)
    if not isinstance(factor_class, type) or not issubclass(factor_class, Factor):
        raise TypeError(
            f"注册失败: 类 '{factor_class.__name__}' 不是 Factor 的子类, "
            f"无法注册为因子。请确保继承自 alpha101_factory.factors.base.Factor。"
        )

    # 校验 2: 必须定义 name 属性且为非空字符串 (Must have non-empty 'name' attribute)
    if not hasattr(factor_class, "name"):
        raise ValueError(
            f"注册失败: 因子类 '{factor_class.__name__}' 缺少 'name' 属性。"
            f"请在类定义中设置 name = 'YourFactorName'。"
        )

    factor_name: str = factor_class.name  # type: ignore[assignment]

    if not isinstance(factor_name, str) or not factor_name.strip():
        raise ValueError(
            f"注册失败: 因子类 '{factor_class.__name__}' 的 'name' 属性必须为非空字符串, "
            f"当前值为 {factor_name!r}。"
        )

    # 校验 3: 因子名称必须唯一 (Factor name must be globally unique)
    if factor_name in _REGISTRY:
        existing_class = _REGISTRY[factor_name]
        raise ValueError(
            f"注册失败: 因子名称 '{factor_name}' 已被占用。"
            f"冲突类: {existing_class.__module__}.{existing_class.__qualname__}, "
            f"当前类: {factor_class.__module__}.{factor_class.__qualname__}。"
            f"请修改 name 属性以确保全局唯一。"
        )

    # 登记到全局注册表 (Register in global registry)
    _REGISTRY[factor_name] = factor_class
    logger.debug(f"因子 '{factor_name}' 已成功注册: {factor_class.__module__}.{factor_class.__qualname__}")

    return factor_class


# ===================================================================
# 懒加载机制 (Lazy Loading Mechanism)
# ===================================================================
def _ensure_loaded() -> None:
    """确保所有因子模块已被扫描并加载 (Ensure All Factor Modules Are Scanned and Loaded)。

    通过 ``pkgutil.iter_modules`` 遍历本包下的所有模块, 跳过 ``base`` 和
    ``registry`` 自身以及子包, 仅导入 ``alphas_*.py`` 等因子定义模块。
    加载完成后设置 ``_LOADED`` 标志, 避免重复扫描。

    导入过程中若某个因子模块出错, 仅记录警告日志而不中断整体流程,
    确保其他正常因子仍可被加载。

    Raises:
        RuntimeError: 当无法导入因子包或扫描过程发生致命错误时抛出。
    """
    global _LOADED  # noqa: PLW0603

    if _LOADED:
        return

    # 步骤 1: 导入因子包 (Import the factors package)
    try:
        factors_package = importlib.import_module(__package__)
    except Exception as exc:
        raise RuntimeError(
            f"无法导入因子包 '{__package__}', 请检查包路径与 __init__.py 是否正确。"
        ) from exc

    # 步骤 2: 遍历包内模块并逐个导入 (Iterate and import each module in the package)
    try:
        module_path = getattr(factors_package, "__path__", None)
        if module_path is None:
            raise RuntimeError(
                f"因子包 '{__package__}' 缺少 __path__ 属性, 无法进行模块扫描。"
                f"请确认该包为合法的 Python 包 (包含 __init__.py)。"
            )

        skipped_modules = ("base", "registry")
        discovered_count = 0
        loaded_count = 0
        error_count = 0

        for _module_finder, module_name, is_package in pkgutil.iter_modules(module_path):
            # 跳过子包和核心模块 (Skip sub-packages and core modules)
            if is_package or module_name in skipped_modules:
                continue

            discovered_count += 1

            try:
                importlib.import_module(f"{__package__}.{module_name}")
                loaded_count += 1
                logger.info(f"因子模块 '{module_name}' 加载成功")
            except Exception as import_exc:
                error_count += 1
                logger.warning(
                    f"因子模块 '{module_name}' 加载失败: {import_exc!r}。"
                    f"该模块中的因子将不可用, 但不影响其他因子。"
                )

        _LOADED = True

        # 打印发现与加载统计信息 (Print discovery and loading statistics)
        logger.info(
            f"因子模块扫描完成: 发现 {discovered_count} 个模块, "
            f"成功加载 {loaded_count} 个, 失败 {error_count} 个。"
        )
        logger.info(f"当前已注册因子数量: {len(_REGISTRY)}, 名称列表: {sorted(_REGISTRY.keys())}")

    except Exception as exc:
        raise RuntimeError(
            f"扫描因子包时发生致命错误, 请检查包路径与模块定义。"
            f"原始错误: {exc}"
        ) from exc


# ===================================================================
# 公开查询接口 (Public Query Interfaces)
# ===================================================================
def get_factor(factor_name: str) -> Type[Factor]:
    """根据因子名称获取已注册的因子类 (Get Registered Factor Class by Name)。

    自动触发模块扫描 (若尚未加载), 然后从注册表中查找并返回对应的因子类。

    Args:
        factor_name: 因子的唯一标识名称, 区分大小写。

    Returns:
        与 ``factor_name`` 对应的因子类 (Factor 子类)。

    Raises:
        KeyError: 当注册表中不存在指定名称的因子时抛出。
    """
    _ensure_loaded()

    if factor_name not in _REGISTRY:
        available_factors = sorted(_REGISTRY.keys())
        raise KeyError(
            f"未找到因子 '{factor_name}'。"
            f"当前已注册的因子有 {len(_REGISTRY)} 个: {available_factors}。"
        )

    return _REGISTRY[factor_name]


def list_factors() -> List[str]:
    """返回所有已注册因子的名称列表 (Return Names of All Registered Factors)。

    自动触发模块扫描 (若尚未加载), 返回按字母顺序排序的因子名称列表。

    Returns:
        已注册因子名称的排序列表, 每个元素为 str 类型。

    示例::

        >>> list_factors()
        ['Alpha101', 'Alpha101_001', 'Alpha101_002', ...]
    """
    _ensure_loaded()
    return sorted(_REGISTRY.keys())


# ===================================================================
# FactorFactory — 因子工厂类 (Factor Factory Class)
# ===================================================================
class FactorFactory:
    """可插拔因子工厂, 提供因子的创建、校验、计算与批量操作 (Pluggable Factor Factory)。

    封装因子的实例化、输入校验、单例计算、批量计算与信息查询功能。
    所有方法均通过注册表动态查找因子, 无需硬编码因子名称。

    典型用法::

        factory = FactorFactory()

        # 查看因子信息
        print(factory.info("Alpha101"))
        # {'name': 'Alpha101', 'requires': ['close', 'volume', ...]}

        # 校验输入数据是否满足因子需求
        factory.validate("Alpha101", df)

        # 计算单个因子
        result = factory.compute("Alpha101", df)

        # 批量计算多个因子
        results = factory.compute_batch(["Alpha101", "MyAlpha"], df)

    属性:
        无公开属性, 所有状态通过注册表全局管理。
    """

    def create(self, factor_name: str) -> Factor:
        """创建指定因子的实例 (Create an Instance of the Specified Factor)。

        Args:
            factor_name: 因子的唯一标识名称。

        Returns:
            新创建的因子实例。

        Raises:
            KeyError: 当因子名称不存在于注册表中时抛出。
        """
        factor_class: Type[Factor] = get_factor(factor_name)
        return factor_class()

    def validate(self, factor_name: str, data_frame: pd.DataFrame) -> List[str]:
        """校验输入 DataFrame 是否满足因子的列需求 (Validate DataFrame Columns for Factor)。

        创建因子实例后调用其 ``validate_requires`` 方法, 检查输入数据
        是否包含因子计算所需的全部列。

        Args:
            factor_name: 因子的唯一标识名称。
            data_frame: 待校验的行情数据 DataFrame。

        Returns:
            因子声明的所需列名列表 (即 ``factor.requires``)。

        Raises:
            KeyError: 当因子不存在或 DataFrame 缺少必需列时抛出。
        """
        factor_instance: Factor = self.create(factor_name)
        factor_instance.validate_requires(data_frame)
        return factor_instance.requires

    def compute(self, factor_name: str, data_frame: pd.DataFrame) -> pd.Series:
        """计算指定因子的因子值 (Compute Factor Values for the Specified Factor)。

        创建因子实例, 校验输入数据后执行 ``compute`` 方法。

        Args:
            factor_name: 因子的唯一标识名称。
            data_frame: 行情数据 DataFrame, 必须包含因子所需的全部列。

        Returns:
            因子值 Series, 索引为 MultiIndex[datetime, symbol], name 为 "value"。

        Raises:
            KeyError: 当因子不存在或 DataFrame 缺少必需列时抛出。
            Exception: 因子计算过程中发生的其他异常将向上传播。
        """
        factor_instance: Factor = self.create(factor_name)
        factor_instance.validate_requires(data_frame)
        return factor_instance.compute(data_frame)

    def compute_batch(
        self,
        factor_names: List[str],
        data_frame: pd.DataFrame,
        symbols: Optional[List[str]] = None,  # noqa: ARG002 — 预留接口, 供未来按股票子集过滤使用
    ) -> Dict[str, pd.Series]:
        """批量计算多个因子的因子值 (Batch Compute Multiple Factors)。

        遍历因子名称列表, 逐个计算并收集结果。单个因子计算失败不会中断
        整体流程, 仅记录错误日志并跳过该因子。

        Args:
            factor_names: 待计算的因子名称列表。
            data_frame: 行情数据 DataFrame。
            symbols: 预留参数, 当前未使用。未来可用于限制计算的股票范围。

        Returns:
            字典, 键为因子名称, 值为对应的因子值 Series。
            仅包含计算成功的因子, 失败的因子不会出现在结果中。
        """
        computed_results: Dict[str, pd.Series] = {}

        for current_factor_name in factor_names:
            try:
                computed_results[current_factor_name] = self.compute(
                    current_factor_name, data_frame
                )
                result_length = len(computed_results[current_factor_name])
                logger.info(f"因子 '{current_factor_name}' 计算完成: {result_length} 行数据")
            except Exception as compute_exc:
                logger.error(
                    f"因子 '{current_factor_name}' 计算失败: {compute_exc!r}。"
                    f"已跳过该因子, 继续处理后续因子。"
                )

        success_count = len(computed_results)
        total_count = len(factor_names)
        logger.info(
            f"批量计算完成: 成功 {success_count}/{total_count} 个因子。"
        )

        return computed_results

    def info(self, factor_name: str) -> Dict[str, object]:
        """获取指定因子的元信息 (Get Metadata for the Specified Factor)。

        Args:
            factor_name: 因子的唯一标识名称。

        Returns:
            包含因子元信息的字典, 键包括:
                - ``name`` (str): 因子名称。
                - ``requires`` (List[str]): 因子所需的 DataFrame 列名列表。

        Raises:
            KeyError: 当因子不存在时抛出。
        """
        factor_instance: Factor = self.create(factor_name)
        return {
            "name": factor_instance.name,
            "requires": factor_instance.requires,
        }

    def info_all(self) -> Dict[str, Dict[str, object]]:
        """获取所有已注册因子的元信息 (Get Metadata for All Registered Factors)。

        Returns:
            嵌套字典, 外层键为因子名称, 值为该因子的元信息字典
            (格式同 ``info()`` 方法的返回值)。
        """
        return {
            current_name: self.info(current_name)
            for current_name in list_factors()
        }
