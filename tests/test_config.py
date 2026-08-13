"""配置模块单元测试。"""
from pathlib import Path

import pytest

from bvsc.config import PROJECT_ROOT, Settings
from bvsc.exceptions import ConfigError


def test_default_config_exists_and_loads():
    settings = Settings.load()
    assert settings.get("model", "d_model") == 512
    assert settings.get("model", "n_heads") == 8
    assert settings.get("dataset", "expand_samples_per_template") == 20


def test_load_missing_file_raises():
    with pytest.raises(ConfigError):
        Settings.load(Path("/nonexistent/config.yaml"))


def test_load_invalid_yaml_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(": : : invalid\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        Settings.load(bad)


def test_load_non_mapping_raises(tmp_path):
    bad = tmp_path / "list.yaml"
    bad.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        Settings.load(bad)


def test_get_default_when_missing():
    settings = Settings.load()
    assert settings.get("no_such_section", "key", "fallback") == "fallback"
    assert settings.get("model", "no_such_key", 42) == 42


def test_resolve_path_relative_to_project_root():
    settings = Settings.load()
    checkpoint = settings.model_checkpoint
    assert checkpoint.is_absolute()
    assert checkpoint.name == "transformer.pth"
    assert PROJECT_ROOT in checkpoint.parents


def test_device_auto_falls_back_to_cpu():
    settings = Settings.load()
    assert settings.device in ("cpu", "cuda")


def test_llm_api_key_from_env(monkeypatch):
    settings = Settings.load()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123")
    assert settings.llm_api_key == "sk-test-123"
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    assert settings.llm_api_key is None
