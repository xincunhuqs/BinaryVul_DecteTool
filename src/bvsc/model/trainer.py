"""模型训练脚本（对应论文第五步：用缺陷汇编数据集训练 Transformer）。

原实现(transformer_v3.py)中训练/测试/预测混于同一文件，且每轮训练
打印全部张量；本模块将训练逻辑独立，日志收敛、指标随训练记录。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

from bvsc._compat import tqdm
from bvsc.logging_setup import get_logger
from bvsc.model.dataset import DefectcodeDataset
from bvsc.model.transformer import TransformerConfig, build_transformer

logger = get_logger(__name__)


def train_model(
    tokenizer_dict_path: str | Path,
    train_data_path: str | Path,
    checkpoint_path: str | Path,
    cfg: TransformerConfig,
    device: str = "cpu",
    epochs: int = 6,
    batch_size: int = 15,
    learning_rate: float = 0.008,
    momentum: float = 0.75,
    seed: Optional[int] = 42,
) -> dict:
    """训练 Transformer 模型并保存权重。

    Args:
        tokenizer_dict_path: 词表文件路径。
        train_data_path: 训练数据文件（模型格式，见 model.dataset）。
        checkpoint_path: 权重保存路径。
        cfg: 模型超参数。
        device: 计算设备。
        epochs / batch_size / learning_rate / momentum: 训练超参。
        seed: 随机种子（None 则不固定）。

    Returns:
        训练统计: ``{"accuracy": float, "loss": float, "epochs": int, "elapsed_sec": float}``
    """
    if seed is not None:
        torch.manual_seed(seed)

    model = build_transformer(cfg, device)
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.pad_id)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    loader = Data.DataLoader(
        DefectcodeDataset(tokenizer_dict_path, train_data_path, cfg.max_len),
        batch_size=batch_size,
        shuffle=True,
    )

    start = time.time()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    total_batches = len(loader)

    logger.info("=" * 56)
    logger.info("开始训练: 样本=%d, epochs=%d, batch_size=%d, device=%s",
                len(loader.dataset), epochs, batch_size, device)
    logger.info("=" * 56)

    for epoch in range(1, epochs + 1):
        epoch_loss, epoch_correct, epoch_samples = 0.0, 0, 0
        # 优化: 每个 epoch 用进度条展示 batch 训练进度，postfix 实时显示 loss/acc
        pbar = tqdm(enumerate(loader, 1), total=total_batches,
                    desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for batch_index, (enc_inputs, dec_inputs, dec_outputs) in pbar:
            enc_inputs = enc_inputs.to(device)
            dec_inputs = dec_inputs.to(device)
            dec_outputs = dec_outputs.to(device)

            outputs, _, _, _ = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, dec_outputs.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=-1)
            # 仅统计非 pad 位置的正确 token 数
            mask = dec_outputs.view(-1) != cfg.pad_id
            correct = (preds[mask] == dec_outputs.view(-1)[mask]).sum().item()
            n = mask.sum().item()

            epoch_loss += loss.item()
            epoch_correct += correct
            epoch_samples += n

            # 进度条实时指标
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg_loss=f"{epoch_loss / batch_index:.4f}",
                acc=f"{100.0 * epoch_correct / max(epoch_samples, 1):.2f}%",
            )
            # 每 10 个 batch 输出一次日志（供日志文件追溯）
            if batch_index % 10 == 0 or batch_index == total_batches:
                logger.info(
                    "Epoch %d/%d batch %d/%d loss=%.6f acc=%.2f%%",
                    epoch, epochs, batch_index, total_batches,
                    loss.item(), 100.0 * epoch_correct / max(epoch_samples, 1),
                )

        total_loss += epoch_loss
        total_correct += epoch_correct
        total_samples += epoch_samples
        logger.info(
            "Epoch %d/%d 完成: avg_loss=%.6f acc=%.2f%% 耗时=%.1fs",
            epoch, epochs,
            epoch_loss / max(total_batches, 1),
            100.0 * epoch_correct / max(epoch_samples, 1),
            time.time() - start,
        )

    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    elapsed = time.time() - start

    # 美化输出: 训练完成汇总
    final_acc = 100.0 * total_correct / max(total_samples, 1)
    final_loss = total_loss / max(epochs * total_batches, 1)
    logger.info("=" * 56)
    logger.info("训练完成!")
    logger.info("  总准确率 : %.2f%%", final_acc)
    logger.info("  平均 loss: %.6f", final_loss)
    logger.info("  训练轮数 : %d", epochs)
    logger.info("  总耗时   : %.1fs (%.1fmin)", elapsed, elapsed / 60)
    logger.info("  模型保存 : %s", checkpoint)
    logger.info("=" * 56)

    return {
        "accuracy": final_acc,
        "loss": final_loss,
        "epochs": epochs,
        "elapsed_sec": round(elapsed, 2),
    }
