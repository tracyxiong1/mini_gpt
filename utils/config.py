import os
from typing import Any, Dict

import yaml

"""配置读取与深度合并。"""


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """深度合并字典：用 override 覆盖 base 对应字段。
    
    Args:
        base: 基础字典
        override: 覆盖字典
    
    Returns:
        Dict[str, Any]: 合并后的字典
    """
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str, defaults: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """加载配置：从 YAML 文件读取并合并默认配置。
    
    Args:
        path: 配置文件路径
        defaults: 默认配置字典（可选）
    
    Returns:
        Dict[str, Any]: 加载并合并后的配置字典
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if defaults is None:
        return data
    return deep_merge(defaults, data)
