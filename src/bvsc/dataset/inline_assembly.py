"""内联汇编标识插入（论文第二步：在缺陷源码处插入内联汇编）。

在 Juliet 缺陷样例中:
    - 缺陷语句（``POTENTIAL FLAW`` 注释）之后插入缺陷标识块，
      编译后以 ``nop; nop; jmp defact_flage; defact_flage: nop`` 形态出现；
    - main 函数体开头插入变量标识块（``variable_flage``）。

内联汇编编译后保持不变，从而在反汇编文件中可被定位，
实现「源码层面 ↔ 汇编层面」的联动（论文 Fig.2）。
"""
from __future__ import annotations

import os
from pathlib import Path

from bvsc._compat import tqdm

from bvsc.logging_setup import get_logger

logger = get_logger(__name__)

# 修复(B7): MSVC 内联汇编块内语句必须换行分隔，单行分号写法会把闭合 } 吞入汇编块导致 C1004。
# 多行写法编译后仍产出 nop;nop;jmp label;label:nop 掩码序列，不影响后续反汇编掩码定位。
_FLAW_ASM = "__asm {\nnop; nop;\njmp defact_flage;\ndefact_flage: nop;\n}"
_VARIABLE_ASM = "__asm {\nnop; nop;\njmp variable_flage;\nvariable_flage: nop;\n}"


class AddInlineAssembly:
    """向精简后的 Juliet 缺陷样例插入内联汇编标识。"""

    # ------------------------------------------------------------------
    # 定位
    # ------------------------------------------------------------------
    def get_marked_index(self, content_lines: list[str]) -> int:
        """返回缺陷标识（POTENTIAL FLAW 注释结束）之后的行索引。

        支持单行注释 ``/* POTENTIAL FLAW: ... */`` 与多行注释两种形态。
        """
        flage = False
        for index, line in enumerate(content_lines):
            if "FLAW" in line:
                if "/*" in line and "*/" in line:
                    if "* FLAW" in line:
                        return index + 1
                    continue
                if "*/" not in line:
                    flage = True
            if flage and "*/" in line:
                return index + 1
        return 0

    def get_main_masked_index(self, content_lines: list[str]) -> int:
        """返回 main 函数体左花括号所在行索引。"""
        flage = False
        for index, line in enumerate(content_lines):
            if line.strip() == "int main()":
                flage = True
                continue
            if flage and line.strip() == "{":
                return index
        return 0

    # ------------------------------------------------------------------
    # 插入
    # ------------------------------------------------------------------
    def add_inline_assembly(self, file_path: str | Path, save_path: str | Path) -> None:
        """在缺陷标识处与 main 函数体插入内联汇编，写入 save_path。

        Args:
            file_path: 精简后的缺陷样例源文件。
            save_path: 插入内联汇编后的输出文件。
        """
        content_lines = Path(file_path).read_text(encoding="utf-8").split("\n")

        # 兼容原实现：若 main 位于 #ifndef _WIN32 未闭合段内，先补 #endif
        content_lines = self._fix_win32_guard(content_lines)

        variable_index = self.get_main_masked_index(content_lines) + 1
        flaw_index = self.get_marked_index(content_lines)

        content_lines.insert(variable_index, _VARIABLE_ASM)
        if variable_index < flaw_index:
            flaw_index += 1
        content_lines.insert(flaw_index, _FLAW_ASM)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text("\n".join(content_lines), encoding="utf-8")
        logger.debug("已插入内联汇编: %s -> %s", file_path, save_path)

    @staticmethod
    def _fix_win32_guard(content_lines: list[str]) -> list[str]:
        """若 ``int main()`` 处于未闭合的 ``#ifndef _WIN32`` 段内，补插 ``#endif``。"""
        lines = content_lines[:]
        flage_ifndef = False
        for index, codeline in enumerate(lines):
            if codeline.strip() == "#ifndef _WIN32":
                flage_ifndef = True
            if codeline.strip() == "#endif":
                break
            if codeline.strip() == "int main()" and flage_ifndef:
                lines.insert(index, "#endif")
                break
        return lines

    # ------------------------------------------------------------------
    # 批量处理
    # ------------------------------------------------------------------
    def gather_inline_assembly_example(
        self, folder_path: str | Path, save_folder: str | Path
    ) -> None:
        """对所有缺陷样例（bad 文件）批量插入内联汇编。

        Args:
            folder_path: 精简后缺陷样例所在目录（含 CWE 子目录）。
            save_folder: 输出目录。
        """
        folder_path = Path(folder_path)
        save_folder = Path(save_folder)
        father_folders = sorted(os.listdir(folder_path))

        for father_folder in tqdm(father_folders, desc="插入内联汇编中..."):
            incoming_folder = folder_path / father_folder
            if not incoming_folder.is_dir():
                continue
            out_folder = save_folder / father_folder
            out_folder.mkdir(parents=True, exist_ok=True)

            for file in sorted(incoming_folder.iterdir()):
                if "bad" in file.name and "good" not in file.name:
                    self.add_inline_assembly(file, out_folder / file.name)
        logger.info("内联汇编插入完成：%s", save_folder)
