"""汇编代码切片模块（论文第六步：切片后送入模型）。

将反汇编文本按固定指令条数切块，每个切片内的指令丢弃地址、
以 ``;`` 连接，与训练数据中缺陷切片的文本形态保持一致。
"""
from __future__ import annotations

import random
from typing import Sequence

from bvsc.exceptions import SlicingError
from bvsc.logging_setup import get_logger

logger = get_logger(__name__)

_SLICE_HEADER = "===========slicing number:{number}==========="


def slice_disassembly(
    disassembly_text: str,
    block_size: int = 100,
    block_size_range: tuple[int, int] | None = None,
) -> list[str]:
    """将反汇编文本切分为定长指令块。

    Args:
        disassembly_text: ``disassembler.Disassembler.disassemble`` 的输出。
        block_size: 每个切片包含的指令条数；<=0 时在
            ``block_size_range`` 区间内随机（用于检测时增加鲁棒性）。
        block_size_range: 随机区间 (min, max)。

    Returns:
        切片列表，每项为 ``;`` 连接的指令串（无地址）。

    Raises:
        SlicingError: 输入为空或 block_size 非法。
    """
    if not disassembly_text or not disassembly_text.strip():
        raise SlicingError("反汇编文本为空，无法切片")

    lines = [ln for ln in disassembly_text.split("\n") if ln.strip()]
    if not lines:
        raise SlicingError("反汇编文本无有效指令行")

    size = block_size
    if size <= 0:
        low, high = block_size_range or (80, 130)
        if low <= 0 or high <= low:
            raise SlicingError(f"非法切片随机区间: ({low}, {high})")
        size = random.randint(low, high)

    slices: list[str] = []
    for index in range(0, len(lines), size):
        block = lines[index : index + size]
        # 修复(B13): 仅当行首为十六进制地址时才剥离；否则保留整行（健壮性）。
        # 丢弃地址，保留 "mnemonic op_str;"，与训练切片形态一致
        parts = []
        for line in block:
            stripped = line.strip()
            if not stripped:
                continue
            tokens = stripped.split(" ")
            if len(tokens) > 1 and tokens[0].startswith("0x"):
                parts.append(" ".join(tokens[1:]).strip())
            else:
                parts.append(stripped)
        slice_text = "".join(p + ";" for p in parts if p)
        if slice_text:
            slices.append(slice_text)
    logger.debug("共生成 %d 个切片，block_size=%d", len(slices), size)
    return slices


def format_slices_with_headers(slices: Sequence[str]) -> str:
    """生成带 ``slicing number`` 分隔头的可读切片文本（供调试/落盘）。"""
    return "\n".join(
        f"{_SLICE_HEADER.format(number=i + 1)}\n{sl}" for i, sl in enumerate(slices)
    )
