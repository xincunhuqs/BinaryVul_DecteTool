"""配置加载与访问。

设计原则:
    - 所有配置收敛于 ``config/config.yaml``；
    - 相对路径基于项目根目录（``src/bvsc/`` 上三级）解析；
    - 敏感信息（API Key 等）只从环境变量读取，禁止写入配置文件；
    - 环境变量可覆盖部分运行时选项（见 :meth:`Settings.load`）。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

from bvsc.exceptions import ConfigError

# src/bvsc/config.py -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "config" / "config.yaml"


class Settings:
    """应用配置对象。"""

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, config_file: Path | str | None = None) -> "Settings":
        """从 YAML 文件加载配置。

        Args:
            config_file: 配置文件路径；为 None 时使用项目默认配置。

        Raises:
            ConfigError: 文件不存在、解析失败或结构非法时抛出。
        """
        path = Path(config_file) if config_file else DEFAULT_CONFIG_FILE
        if not path.exists():
            raise ConfigError(f"配置文件不存在: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"配置文件解析失败: {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"配置文件结构非法（应为顶层映射）: {path}")
        return cls(data)

    # ------------------------------------------------------------------
    # 通用访问
    # ------------------------------------------------------------------
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """读取 ``section.key``，不存在时返回 default。"""
        try:
            return self._data[section][key]
        except (KeyError, TypeError):
            return default

    @property
    def raw(self) -> Dict[str, Any]:
        return self._data

    # ------------------------------------------------------------------
    # 路径解析（相对路径 -> 项目根目录下的绝对路径）
    # ------------------------------------------------------------------
    def resolve_path(self, section: str, key: str) -> Path:
        """将配置中的路径解析为绝对路径。

        相对路径基于项目根目录；绝对路径原样返回。
        """
        value = self.get(section, key)
        if value is None:
            raise ConfigError(f"配置缺失: {section}.{key}")
        path = Path(str(value))
        return path if path.is_absolute() else PROJECT_ROOT / path

    # ------------------------------------------------------------------
    # 常用配置项（类型化访问）
    # ------------------------------------------------------------------
    @property
    def model_checkpoint(self) -> Path:
        return self.resolve_path("model", "checkpoint")

    @property
    def tokenizer_path(self) -> Path:
        return self.resolve_path("model", "tokenizer_path")

    @property
    def device(self) -> str:
        """计算设备：auto -> 有 CUDA 用 cuda，否则 cpu。"""
        device = str(self.get("model", "device", "auto")).lower()
        if device == "auto":
            try:
                import torch  # 延迟导入，避免无 torch 环境启动 CLI 失败

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    @property
    def llm_api_key(self) -> str | None:
        """从环境变量读取 API Key（禁止硬编码）。"""
        env_name = str(self.get("llm", "api_key_env", "DEEPSEEK_API_KEY"))
        return os.environ.get(env_name)
