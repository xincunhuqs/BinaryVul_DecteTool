"""切片模块单元测试（纯 Python）。"""
import pytest

from bvsc.exceptions import SlicingError
from bvsc.slicer import format_slices_with_headers, slice_disassembly


def _fake_disassembly(n: int = 20) -> str:
    return "\n".join(
        f"{hex(0x401000 + i)} mov eax, {i}" for i in range(n)
    ) + "\n"


def test_slice_block_size():
    dis = _fake_disassembly(20)
    slices = slice_disassembly(dis, block_size=8)
    assert len(slices) == 3  # 20 -> 8/8/4
    assert slices[0].count(";") == 8
    # 丢弃地址，只保留 mnemonic + 操作数
    assert "mov eax, 0;" in slices[0]
    assert "0x401000" not in slices[0]


def test_slice_headers_format():
    dis = _fake_disassembly(12)
    slices = slice_disassembly(dis, block_size=6)
    text = format_slices_with_headers(slices)
    assert "slicing number:1" in text
    assert "slicing number:2" in text


def test_slice_random_range():
    dis = _fake_disassembly(500)
    slices = slice_disassembly(dis, block_size=0, block_size_range=(80, 130))
    # 随机块大小应落在区间内：每切片 ; 数 ∈ [80, 130]
    assert 80 <= slices[0].count(";") <= 130


def test_slice_empty_raises():
    with pytest.raises(SlicingError):
        slice_disassembly("")


def test_slice_invalid_range_raises():
    with pytest.raises(SlicingError):
        slice_disassembly("mov eax, 1", block_size=0, block_size_range=(5, 2))
