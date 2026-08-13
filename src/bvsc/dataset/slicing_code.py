"""缺陷汇编切片提取（论文第四步后半：差异比较提取缺陷反汇编代码）。

原理（论文 Algorithm 2 / Table 1）:
    缺陷样例编译后，内联汇编标识以 ``nop; nop; jmp defact_flage; ...; nop``
    形态保留在 ``.text`` 节中。反汇编后按该模式定位掩码区间：
    ``[第一个 nop, 第二个 nop 之后的 jmp, ...]`` 提取缺陷语句及其
    相关数据流构成的汇编切片。

与原实现相比:
    - 复用 :class:`bvsc.disassembler.Disassembler`，消除反汇编代码重复；
    - 掩码定位改为显式状态机，失败样例统一记录日志而非静默丢弃。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from bvsc._compat import tqdm

from bvsc.disassembler import Disassembler
from bvsc.logging_setup import get_logger

logger = get_logger(__name__)


class GenerateDissassemblySlicing:
    """缺陷可执行文件 -> 缺陷汇编切片。"""

    def __init__(self, arch: str = "x86", mode: str = "32") -> None:
        self._disassembler = Disassembler(arch, mode)

    # ------------------------------------------------------------------
    # 内联汇编掩码定位
    # ------------------------------------------------------------------
    @staticmethod
    def find_mask_begin_indices(disassembly_lines: list[str]) -> list[int]:
        """定位内联汇编掩码起始索引。

        理论掩码形态: ``nop -> nop -> jmp -> nop``；实测 MSVC 会把连续 ``nop; nop;``
        合并为一个 nop（即使 /Od），因此同时兼容 ``nop -> jmp -> nop``（修复#18）。
        返回每处掩码块第一个 nop 的行索引。
        """
        mask_list: list[int] = []
        state = 0       # 0=找首 nop, 1=已见 1~2 个 nop, 2=已见 jmp, 3=已见 jmp 后 nop
        mask_begin = 0
        for index, code in enumerate(disassembly_lines):
            if state == 0:
                if "nop" in code:
                    state = 1
                    mask_begin = index
            elif state == 1:
                if "nop" in code:
                    continue          # 允许第二个 nop（容忍编译器合并/保留）
                if "jmp" in code:
                    state = 2
                else:
                    state, mask_begin = 0, 0
            elif state == 2:
                if "nop" in code:
                    mask_list.append(mask_begin)
                    state, mask_begin = 0, 0
                else:
                    state, mask_begin = 0, 0
        return mask_list

    # ------------------------------------------------------------------
    # 单文件切片提取
    # ------------------------------------------------------------------
    def gather_defect_slicing(self, exefile_path: str | Path) -> Optional[str]:
        """提取单个缺陷可执行文件的缺陷汇编切片。

        Args:
            exefile_path: 缺陷可执行文件路径。

        Returns:
            以 ``;`` 连接的缺陷汇编切片；定位失败返回 None。
        """
        disassembly_lines = self._disassembler.disassemble(str(exefile_path)).split("\n")
        mask_list = self.find_mask_begin_indices(disassembly_lines)
        if len(mask_list) != 2:
            logger.warning("掩码定位失败（期望 2 处，实际 %d 处）: %s",
                           len(mask_list), exefile_path)
            return None

        slicing_begin = mask_list[0] + 3   # 跳过首个掩码块(nop;jmp;nop，兼容双nop)；原 +4 按 4 指令块假设，实测 MSVC 合并 nop 后为 3 条(修复#18)
        slicing_end = mask_list[1]
        if slicing_begin >= slicing_end:
            logger.warning("切片区间非法: %s", exefile_path)
            return None

        defect_slicing = disassembly_lines[slicing_begin:slicing_end]
        # 修复(B13): 仅当行首为十六进制地址才剥离，否则保留整行
        parts = []
        for line in defect_slicing:
            stripped = line.strip()
            if not stripped:
                continue
            tokens = stripped.split(" ")
            if len(tokens) > 1 and tokens[0].startswith("0x"):
                parts.append(" ".join(tokens[1:]).strip())
            else:
                parts.append(stripped)
        return ";".join(parts)

    # ------------------------------------------------------------------
    # 批量提取
    # ------------------------------------------------------------------
    def acquired_defect_slicing(
        self,
        folder_exefile: str | Path,
        vul_type: str,
        save_folder: str | Path,
        fail_log: str | Path | None = None,
    ) -> None:
        """提取文件夹内全部缺陷可执行文件的切片，写入 ``<vul_type>.txt``。"""
        folder_exefile = Path(folder_exefile)
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = save_folder / f"{vul_type}.txt"

        written = 0
        with open(save_path, "w", encoding="utf-8") as fdefect:
            for exefile in sorted(os.listdir(folder_exefile)):
                if not exefile.lower().endswith(".exe"):
                    continue
                exefile_path = folder_exefile / exefile
                defect_slicing = self.gather_defect_slicing(exefile_path)
                if defect_slicing:
                    fdefect.write(f"{defect_slicing}\t\t{vul_type}\n")
                    written += 1
                elif fail_log is not None:
                    Path(fail_log).parent.mkdir(parents=True, exist_ok=True)
                    with open(fail_log, "a", encoding="utf-8") as ffail:
                        ffail.write(f"{exefile_path}\n")
        exe_count = sum(1 for f in folder_exefile.iterdir() if f.suffix.lower() == ".exe")
        logger.info("%s: 提取成功 %d/%d", vul_type, written, exe_count)

    def auto_gather_disassemcode_slicing_from_fatherfolder(
        self,
        father_folder: str | Path,
        save_folder: str | Path,
        fail_log: str | Path | None = None,
    ) -> None:
        """批量提取缺陷可执行文件根目录下全部缺陷类型的切片。"""
        father_folder = Path(father_folder)
        for vul_type in tqdm(sorted(os.listdir(father_folder)), desc="提取缺陷切片中..."):
            folder_path = father_folder / vul_type
            if not folder_path.is_dir():
                continue
            self.acquired_defect_slicing(folder_path, vul_type, save_folder, fail_log)

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    def get_total_defect_slicing(self, slicing_folder: str | Path, output_file: str | Path) -> None:
        """汇总所有缺陷类型的切片文件为单一数据集文件。"""
        slicing_folder = Path(slicing_folder)
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        total = 0
        with open(output_file, "w", encoding="utf-8") as ftotal:
            for filename in sorted(os.listdir(slicing_folder)):
                file_path = slicing_folder / filename
                if not file_path.is_file():
                    continue
                content = file_path.read_text(encoding="utf-8")
                ftotal.write(content)
                total += len([ln for ln in content.split("\n") if ln.strip()])
        logger.info("缺陷切片汇总完成：%d 条 -> %s", total, output_file)
