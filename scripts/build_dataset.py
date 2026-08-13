#!/usr/bin/env python3
"""一键构建缺陷汇编切片数据集（论文六步管线 ①-④）。

用法:
    python scripts/build_dataset.py [--config config/config.yaml]
"""
from __future__ import annotations  # 修复(B9): X|Y 注解在 Python<3.10 需未来导入

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import click  # noqa: E402

from bvsc.config import Settings  # noqa: E402
from bvsc.dataset.pipeline import GenerateVulSliceData  # noqa: E402
from bvsc.logging_setup import setup_logging  # noqa: E402


@click.command()
@click.option("-config", "--config", "config_file",
              type=click.Path(path_type=Path), default=None,
              help="配置文件路径（默认 config/config.yaml）")
def build(config_file: Path | None) -> None:
    """构建缺陷切片数据集（Juliet 预处理 -> 内联汇编 -> 模板 -> 扩样 -> 编译 -> 切片）。"""
    setup_logging()
    settings = Settings.load(config_file)
    pipeline = GenerateVulSliceData(settings)
    total_file = pipeline.generater_vulexefile()
    click.echo(f"[*] 数据集构建完成: {total_file}")


if __name__ == "__main__":
    build()
