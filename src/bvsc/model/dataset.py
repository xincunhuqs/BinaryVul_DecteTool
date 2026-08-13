"""深度学习数据集构建（对应论文第五步：缺陷汇编数据集 -> 模型训练输入）。

数据格式说明（与训练/预测保持一致）:
    原始切片文件每行: ``<切片代码>\\t\\t<CWE类型>``
    转换后每行: ``<切片代码> ! S <CWE类型> ! <CWE类型> E``
    其中 ``S`` 为解码起始符、``E`` 为解码结束符（对应词表中的 S / E）。
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

import torch
import torch.utils.data as Data
from bvsc._compat import tqdm
from bvsc.logging_setup import get_logger
from bvsc.model.tokenizer import load_tokenizer, word2index

logger = get_logger(__name__)

_SEP = "\t\t"
_TEMPLATE = "{code} ! S {vul_type} ! {vul_type} E"


def generate_model_data(defectcode_setfile: str | Path, output_path: str | Path) -> None:
    """将缺陷切片集合文件转换为模型训练格式（每行带 S/E 标记）。"""
    defectcode_setfile = Path(defectcode_setfile)
    output_path = Path(output_path)
    if not defectcode_setfile.exists():
        raise FileNotFoundError(f"缺陷切片集合文件不存在: {defectcode_setfile}")

    lines = defectcode_setfile.read_text(encoding="utf-8").split("\n")
    converted: list[str] = []
    # 优化: 大批量转换时用进度条展示进度
    for codeline in tqdm(lines, desc="转换模型格式数据", unit="行"):
        if not codeline.strip():
            continue
        parts = codeline.split(_SEP)
        if len(parts) < 2:
            logger.warning("跳过无法解析的行: %s", codeline[:80])
            continue
        vul_type = parts[-1].strip()
        code = " ; ".join(parts[:-1]).strip()
        converted.append(_TEMPLATE.format(code=code, vul_type=vul_type))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(converted) + "\n", encoding="utf-8")
    logger.info("已生成训练格式数据 %d 条 -> %s", len(converted), output_path)


def split_dataset(
    data_path: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.95,
    seed: int = 42,
) -> tuple[Path, Path]:
    """按比例随机划分训练集/测试集。

    Returns:
        (train_path, test_path)
    """
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    lines = [ln for ln in data_path.read_text(encoding="utf-8").split("\n") if ln.strip()]
    rng = random.Random(seed)
    rng.shuffle(lines)

    split_idx = int(len(lines) * train_ratio)
    train_data, test_data = lines[:split_idx], lines[split_idx:]

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.txt"
    test_path = output_dir / "test.txt"
    train_path.write_text("\n".join(train_data) + "\n", encoding="utf-8")
    test_path.write_text("\n".join(test_data) + "\n", encoding="utf-8")

    logger.info("数据集划分完成：train=%d, test=%d", len(train_data), len(test_data))
    return train_path, test_path


def count_max_seq_len(data_path: str | Path, slack: int = 100) -> int:
    """统计数据集中序列最大长度（分词后的词数），并附加冗余量。"""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    max_len = 0
    for line in data_path.read_text(encoding="utf-8").split("\n"):
        max_len = max(max_len, len(line.split(" ")))
    return max_len + slack


class DefectcodeDataset(Data.Dataset):
    """缺陷切片数据集（PyTorch Dataset）。

    每行格式: ``<code> ! S <CWE> ! <CWE> E``
    按 ``!`` 拆分为 enc_input / dec_input / dec_output 三段。
    """

    def __init__(
        self,
        tokenizer_dict_path: str | Path,
        data_path: str | Path,
        max_seq_len: int = 200,
    ) -> None:
        super().__init__()
        self.tokenizer_dict = load_tokenizer(tokenizer_dict_path)
        self.datas = [
            ln
            for ln in Path(data_path).read_text(encoding="utf-8").split("\n")
            if ln.strip()
        ]
        self.max_seq_len = max_seq_len
        self.data_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.datas)

    def __getitem__(self, index: int):
        if index in self.data_cache:
            return self.data_cache[index]

        enc_part, dec_part, out_part = self.datas[index].strip().split("!")
        enc_input = word2index(enc_part, self.tokenizer_dict, self.max_seq_len)
        dec_input = word2index(dec_part.strip(), self.tokenizer_dict, self.max_seq_len)
        dec_output = word2index(out_part.strip(), self.tokenizer_dict, self.max_seq_len)

        item = (
            torch.LongTensor(enc_input),
            torch.LongTensor(dec_input),
            torch.LongTensor(dec_output),
        )
        self.data_cache[index] = item
        return item
