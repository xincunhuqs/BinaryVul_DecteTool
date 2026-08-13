"""本地汇编分词器（对应论文第五步：缺陷汇编数据集的特征化输入）。

实现说明:
    - 词表为 ``{token: index}`` 字典，保持「键顺序 == 值」不变量，
      便于预测阶段按索引反查 token；
    - ``pad`` 固定为 0，未知词（非十六进制）映射到保留的
      ``unknow_key*`` 槽位，推理过程**不修改词表文件**（消除原实现的副作用）。
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, List

from bvsc.exceptions import ModelError
from bvsc.logging_setup import get_logger

logger = get_logger(__name__)

PAD_TOKEN = "pad"
HEX_TOKEN = "0x_word"
_UNKNOWN_PREFIX = "unknow_key"
_UNKNOWN_PATTERN = re.compile(_UNKNOWN_PREFIX)


def build_vocabulary(defectfile_path: str | Path, max_vocab_size: int = 1000) -> Dict[str, int]:
    """从缺陷切片文件构建初始词表。

    Args:
        defectfile_path: 缺陷汇编代码文件（每行一个切片，可含 CWE 标注）。
        max_vocab_size: 词表目标规模，不足部分以 ``unknow_key*`` 填充。

    Returns:
        词表字典 ``{token: index}``（键顺序与索引一致）。
    """
    vocabulary: Dict[str, int] = {PAD_TOKEN: 0}
    index = len(vocabulary)  # 1
    seen_hex = False

    with open(defectfile_path, "r", encoding="utf-8") as f:
        content = f.read()

    for line in content.split("\n"):
        # 与训练数据构建保持一致的分词预处理
        normalized = (
            line.replace(",", " ")
            .replace(":", " ")
            .replace(";", "; ")
            .replace("\t\t", " ")
            .replace("[", " [ ")
            .replace("]", " ] ")
        )
        for word in normalized.split(" "):
            if "0x" in word:
                if not seen_hex:
                    vocabulary[HEX_TOKEN] = index
                    index += 1
                    seen_hex = True
                continue
            if word and word not in vocabulary:
                vocabulary[word] = index
                index += 1

    # 用 unknow_key 槽位补齐到 max_vocab_size，供推理期未知词映射
    for i in range(max_vocab_size - len(vocabulary)):
        vocabulary[f"{_UNKNOWN_PREFIX}{i}"] = index
        index += 1

    logger.info("词表构建完成：%d 个词（目标 %d）", len(vocabulary), max_vocab_size)
    return vocabulary


def word2index(
    sentence: str,
    tokenizer_dict: Dict[str, int],
    max_sentence_len: int = 350,
) -> List[int]:
    """将汇编语句转换为索引序列，不足长度以 ``pad``(0) 填充。

    Args:
        sentence: 汇编语句文本。
        tokenizer_dict: 词表。
        max_sentence_len: 序列最大长度（超出截断，不足填充）。

    Returns:
        索引列表，长度恒为 ``max_sentence_len``。
    """
    words = sentence.split(" ")
    if max_sentence_len:
        words = words[:max_sentence_len]
        words += [PAD_TOKEN] * (max_sentence_len - len(words))

    unknown_index = _first_unknown_index(tokenizer_dict)
    indices: List[int] = []
    for word in words:
        if word in tokenizer_dict:
            indices.append(tokenizer_dict[word])
        elif "0x" in word and HEX_TOKEN in tokenizer_dict:
            indices.append(tokenizer_dict[HEX_TOKEN])
        elif unknown_index is not None:
            indices.append(unknown_index)  # 未知词映射到保留槽位，不改写词表
        else:  # 词表无未知槽位（理论上不会发生）
            indices.append(tokenizer_dict.get(PAD_TOKEN, 0))
    return indices


def load_tokenizer(token_path: str | Path) -> Dict[str, int]:
    """从文件加载词表（文件内容为 Python 字面量字典）。"""
    path = Path(token_path)
    if not path.exists():
        raise ModelError(f"词表文件不存在: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.readlines()[-1].strip("\n")
        tokenizer = ast.literal_eval(content)
    except (OSError, SyntaxError, ValueError) as exc:
        raise ModelError(f"词表文件解析失败: {path}") from exc
    if not isinstance(tokenizer, dict):
        raise ModelError(f"词表文件结构非法: {path}")
    return tokenizer


def save_tokenizer(tokenizer_dict: Dict[str, int], token_path: str | Path) -> None:
    """保存词表到文件。"""
    path = Path(token_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(tokenizer_dict) + "\n")


def _first_unknown_index(tokenizer_dict: Dict[str, int]) -> int | None:
    """返回第一个 ``unknow_key*`` 槽位索引（词表内建未知词）。

    优化(P4): 原实现对每个词全表扫描 O(vocab)；词表实例不变时结果固定，
    按词表对象 id 缓存，将分词复杂度降为 O(1)。
    """
    key = id(tokenizer_dict)
    if key in _UNKNOWN_INDEX_CACHE:
        return _UNKNOWN_INDEX_CACHE[key]
    idx = None
    for k, v in tokenizer_dict.items():
        if _UNKNOWN_PATTERN.search(k):
            idx = v
            break
    _UNKNOWN_INDEX_CACHE[key] = idx
    return idx


_UNKNOWN_INDEX_CACHE: Dict[int, int | None] = {}
