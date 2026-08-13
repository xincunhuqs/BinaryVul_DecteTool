#!/usr/bin/env python3
"""训练 Transformer 漏洞检测模型（论文第五步）。

用法:
    python scripts/train_model.py [--config config/config.yaml]
        [--data data/total_defect_slicing.txt] [--epochs 6]

流程: 原始缺陷切片集合 -> 模型格式数据 -> 划分训练/测试集 -> 训练 -> 保存权重。
"""
from __future__ import annotations  # 修复(B9): X|Y 注解在 Python<3.10 需未来导入

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import click  # noqa: E402

from bvsc.config import Settings  # noqa: E402
from bvsc.logging_setup import setup_logging  # noqa: E402


@click.command()
@click.option("-config", "--config", "config_file",
              type=click.Path(path_type=Path), default=None,
              help="配置文件路径（默认 config/config.yaml）")
@click.option("-data", "--data", "data_file",
              type=click.Path(path_type=Path), default=None,
              help="缺陷切片集合文件（每行: 切片<TAB><CWE类型>）")
@click.option("-epochs", "--epochs", type=int, default=None,
              help="训练轮数（覆盖配置）")
@click.option("-benign", "--benign", "benign_file",
              type=click.Path(path_type=Path), default=None,
              help="良性切片负样本文件（每行: 切片<TAB>NO_VULN；由 -rsd 检测收集，可选）")
def train(config_file: Path | None, data_file: Path | None, epochs: int | None,
          benign_file: Path | None) -> None:
    """训练模型。"""
    from bvsc.config import PROJECT_ROOT
    from bvsc.model.dataset import generate_model_data, split_dataset
    from bvsc.model.tokenizer import build_vocabulary, save_tokenizer
    from bvsc.model.trainer import train_model
    from bvsc.model.transformer import config_from_settings

    setup_logging()
    settings = Settings.load(config_file)
    work_dir = PROJECT_ROOT / "data" / "train_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1) 原始切片 -> 模型格式
    source = data_file or work_dir.parent / "total_defect_slicing.txt"
    if not source.exists():
        raise click.BadParameter(f"缺陷切片集合文件不存在: {source}")
    model_data = work_dir / "trainsformer_datast.txt"
    generate_model_data(source, model_data)

    # 修复(B11): 合并良性负样本（检测阶段用 -rsd 收集，标 NO_VULN）
    if benign_file is not None:
        if not Path(benign_file).exists():
            raise click.BadParameter(f"良性样本文件不存在: {benign_file}")
        benign_model_data = work_dir / "benign_model_data.txt"
        generate_model_data(benign_file, benign_model_data)
        with open(model_data, "a", encoding="utf-8") as fout:
            fout.write(benign_model_data.read_text(encoding="utf-8"))
        click.echo(f"[*] 已合并良性负样本: {benign_file}")

    # 2) 构建/加载词表
    tokenizer_path = settings.tokenizer_path
    if not tokenizer_path.exists():
        vocab = build_vocabulary(model_data)
        save_tokenizer(vocab, tokenizer_path)
        click.echo(f"[*] 已构建词表: {tokenizer_path} ({len(vocab)} 词)")
    else:
        click.echo(f"[*] 使用已有词表: {tokenizer_path}")

    # 3) 划分训练/测试集
    ratio = float(settings.get("training", "train_ratio", 0.95))
    seed = int(settings.get("training", "seed", 42))
    train_path, test_path = split_dataset(model_data, work_dir / "data", ratio, seed)
    click.echo(f"[*] 数据集划分: train={train_path.stat().st_size}行源文件 -> 见 {work_dir / 'data'}")

    # 4) 训练
    cfg = config_from_settings(settings)
    stats = train_model(
        tokenizer_dict_path=tokenizer_path,
        train_data_path=train_path,
        checkpoint_path=settings.model_checkpoint,
        cfg=cfg,
        device=settings.device,
        epochs=epochs or int(settings.get("training", "epochs", 6)),
        batch_size=int(settings.get("training", "batch_size", 15)),
        learning_rate=float(settings.get("training", "learning_rate", 0.008)),
        momentum=float(settings.get("training", "momentum", 0.75)),
        seed=seed,
    )
    # 美化输出: 训练完成统计
    click.echo("=" * 56)
    click.echo("训练完成!")
    click.echo(f"  总准确率 : {stats['accuracy']:.2f}%")
    click.echo(f"  平均 loss: {stats['loss']:.6f}")
    click.echo(f"  训练轮数 : {stats['epochs']}")
    click.echo(f"  总耗时   : {stats['elapsed_sec']}s")
    click.echo(f"  模型保存 : {settings.model_checkpoint}")
    click.echo("=" * 56)


if __name__ == "__main__":
    train()
