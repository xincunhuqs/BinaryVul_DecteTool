"""PE 可执行文件反汇编模块（论文第六步：对待检二进制进行反汇编）。

职责:
    - 解析 PE 文件，定位并读取 ``.text`` 节原始字节；
    - 使用 Capstone 反汇编为汇编指令序列；
    - 同时被检测管线（disassembly -> slice -> predict）与
      数据集构建管线（缺陷可执行文件切片提取）复用，避免代码重复。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bvsc.exceptions import DisassemblyError
from bvsc.logging_setup import get_logger

logger = get_logger(__name__)

# 修复(B3): capstone Cs() 需要整数常量而非字符串（原实现用字符串导致初始化崩溃）
try:
    from capstone import CS_ARCH_X86, CS_MODE_32, CS_MODE_64
    _ARCH_MAP = {"x86": CS_ARCH_X86}
    _MODE_MAP = {"32": CS_MODE_32, "64": CS_MODE_64}
except ImportError:  # 缺 capstone 时延迟到 __init__ 再报错
    _ARCH_MAP = {}
    _MODE_MAP = {}


@dataclass(frozen=True)
class Instruction:
    """一条反汇编指令。"""

    address: int
    mnemonic: str
    op_str: str

    def to_text(self, with_address: bool = True) -> str:
        """格式化指令文本，如 ``0x401000 mov eax, 1`` 或 ``mov eax, 1``。"""
        if with_address:
            return f"{hex(self.address)} {self.mnemonic} {self.op_str}".rstrip()
        return f"{self.mnemonic} {self.op_str}".rstrip()


class Disassembler:
    """基于 Capstone 的 PE ``.text`` 节反汇编器。"""

    def __init__(self, arch: str = "x86", mode: str = "32") -> None:
        """初始化 Capstone 引擎。

        Args:
            arch: 指令架构，当前支持 ``x86``。
            mode: 指令模式，``32`` 或 ``64``。

        Raises:
            DisassemblyError: 引擎初始化失败或缺少 capstone 依赖。
        """
        try:
            from capstone import Cs  # 惰性导入，避免缺依赖时模块级崩溃
        except ImportError as exc:
            raise DisassemblyError(
                "缺少依赖 capstone，请先执行 pip install -r requirements.txt"
            ) from exc

        try:
            if arch not in _ARCH_MAP or str(mode) not in _MODE_MAP:
                raise KeyError(f"arch={arch}, mode={mode}")
            self._md = Cs(_ARCH_MAP[arch], _MODE_MAP[str(mode)])
            self._md.detail = False
        except (KeyError, ValueError) as exc:
            raise DisassemblyError(
                f"不支持的反汇编架构/模式: arch={arch}, mode={mode}"
            ) from exc

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def disassemble(self, executable_path: str | Path) -> str:
        """反汇编可执行文件的 ``.text`` 节，返回纯文本汇编（含地址）。

        Args:
            executable_path: PE 可执行文件路径。

        Raises:
            DisassemblyError: 文件不可读、非 PE、无 ``.text`` 节或反汇编失败。
        """
        instructions = self.disassemble_instructions(executable_path)
        return "\n".join(ins.to_text(with_address=True) for ins in instructions)

    def disassemble_instructions(
        self, executable_path: str | Path
    ) -> list[Instruction]:
        """反汇编 ``.text`` 节，返回指令对象列表。"""
        path = Path(executable_path)
        if not path.exists():
            raise DisassemblyError(f"文件不存在: {path}")
        if not path.is_file():
            raise DisassemblyError(f"路径不是文件: {path}")

        text_bytes, start_va = self._read_text_section(path)

        instructions: list[Instruction] = []
        try:
            for ins in self._md.disasm(text_bytes, start_va):
                instructions.append(
                    Instruction(ins.address, ins.mnemonic, ins.op_str)
                )
        except Exception as exc:  # Capstone 内部异常统一包装
            raise DisassemblyError(
                f"反汇编失败: {path}: {exc}"
            ) from exc

        if not instructions:
            logger.warning("文件反汇编结果为空: %s", path)
        return instructions

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    @staticmethod
    def _read_text_section(path: Path) -> tuple[bytes, int]:
        """读取 PE ``.text`` 节字节与起始虚拟地址。

        Returns:
            (节原始字节, 起始虚拟地址 = ImageBase + VirtualAddress)

        Raises:
            DisassemblyError: 非 PE 文件或缺少 ``.text`` 节。
        """
        try:
            import pefile
        except ImportError as exc:  # pragma: no cover
            raise DisassemblyError("缺少依赖 pefile，请先执行 pip install -r requirements.txt") from exc

        try:
            pe = pefile.PE(str(path))
        except Exception as exc:
            raise DisassemblyError(f"无法解析 PE 文件（可能不是有效 PE）: {path}") from exc

        text_section = None
        for section in pe.sections:
            name = section.Name.rstrip(b"\x00").decode("utf-8", errors="ignore")
            if name == ".text":
                text_section = section
                break
        if text_section is None:
            raise DisassemblyError(f"PE 文件中未找到 .text 节: {path}")

        virtual_address = text_section.VirtualAddress
        raw_size = text_section.SizeOfRawData
        raw_offset = text_section.PointerToRawData
        image_base = pe.OPTIONAL_HEADER.ImageBase

        with open(path, "rb") as f:
            f.seek(raw_offset, 0)
            text_bytes = f.read(raw_size)

        return text_bytes, image_base + virtual_address
