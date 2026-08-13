"""Juliet 数据集精简预处理（论文第一步：数据集精简）。

Juliet 测试套件（NSA CAS 出品）中每个缺陷样例同时包含恶意代码(bad)
与修复代码(good)，并混入 Windows/Linux 双环境宏。本模块:
    1. 收集缺陷源文件（文件名以 1-9 结尾的 bad 样例）；
    2. 去除空白行、去除 Windows/Linux 条件编译分支（保留 _WIN32 分支）；
    3. 按 ``#ifndef OMITBAD / OMITGOOD`` 提取恶意/修复代码段，分别生成
       独立的缺陷源文件与修复源文件，使每个缺陷样例只包含与缺陷相关的代码。
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from bvsc._compat import tqdm

from bvsc.logging_setup import get_logger

logger = get_logger(__name__)

_BLANK_LINE = re.compile(r"^\s*\n")
_SOURCE_SUFFIXES = (".c", ".cpp")
_BAD_SAMPLE_SUFFIXES = ("1", "2", "3", "4", "5", "6", "7", "8", "9")


def _is_defect_sample(path: Path) -> bool:
    """判断是否为缺陷样例文件。

    Juliet 实际命名两种形态：
        - ``..._01.c`` / ``..._02.c``（数字结尾，含 bad+good 函数）；
        - ``..._bad.cpp`` / ``..._bad.c``（bad 结尾，仅恶意代码）。
    两者均排除 good 修复样例。
    """
    stem = path.stem.lower()
    if stem.endswith("good"):
        return False
    return stem.endswith("bad") or stem[-1] in _BAD_SAMPLE_SUFFIXES


class JulietPretreatment:
    """Juliet 缺陷样例精简器。"""

    # ------------------------------------------------------------------
    # 1. 源文件收集
    # ------------------------------------------------------------------
    def collect_source_paths(self, file_dir: str | Path) -> list[Path]:
        """收集 Juliet 中全部缺陷源文件路径。

        Juliet 目录结构为 ``testcases/<CWE类型>/s01/<文件>``（两层），
        本方法递归遍历；缺陷样例判定见 :func:`_is_defect_sample`。
        """
        file_dir = Path(file_dir)
        if not file_dir.exists():
            raise FileNotFoundError(f"Juliet 缺陷用例目录不存在: {file_dir}")

        source_paths: list[Path] = []
        for root, _, files in os.walk(file_dir):
            for name in sorted(files):
                path = Path(root) / name
                if path.suffix.lower() in _SOURCE_SUFFIXES and _is_defect_sample(path):
                    source_paths.append(path)
        logger.info("收集到 %d 个缺陷源文件", len(source_paths))
        return source_paths

    # ------------------------------------------------------------------
    # 2. 内容提取与清洗
    # ------------------------------------------------------------------
    @staticmethod
    def filecontent_extract(
        source_path: str | Path, file_dir: str | Path | None = None
    ) -> tuple[str, str, str]:
        """读取源文件，去除空白行。

        Args:
            source_path: 缺陷源文件路径。
            file_dir: Juliet 用例根目录（用于可靠推导缺陷类型文件夹名）；
                      为 None 时按目录层级兜底推断。

        Returns:
            (去除空白行后的代码, 文件名, 缺陷类型文件夹名)

        修复(#21): 原实现固定取 ``parent.parent.name``，当源文件直接位于 CWE 目录
        （无 s01 等批次子目录，如 CWE15 部分样例）或 juliet_testcases_dir 指向
        具体 CWE 目录时，会错误归档为 ``testcases``。传入 file_dir 后按相对路径
        第一段可靠推导。
        """
        source_path = Path(source_path)
        original = source_path.read_text(encoding="utf-8", errors="ignore")
        new_content = _BLANK_LINE.sub("", original)
        if file_dir is not None:
            rel = source_path.resolve().relative_to(Path(file_dir).resolve())
            # 相对路径第一段为 CWE 目录名；无子目录（文件直接在根下）时取根目录名
            cwe_folder = rel.parts[0] if len(rel.parts) > 1 else Path(file_dir).name
        else:
            # 兼容旧调用（无 file_dir）：按 Juliet 约定从目录段中定位
            # ``CWE&lt;编号&gt;_&lt;名称&gt;`` 目录（比固定层级更健壮）
            cwe_folder = next(
                (p.name for p in source_path.parents if p.name.startswith("CWE")),
                "",
            )
            if not cwe_folder:
                cwe_folder = source_path.parent.name
        return new_content, source_path.name, cwe_folder

    @staticmethod
    def remove_win32_guards(source_code: str) -> list[str]:
        """去除 Windows/Linux 条件编译分支。

        目标平台为 Windows（_WIN32 已定义），因此：
            - ``#ifdef _WIN32 ... #endif``：删除指令行，保留 _WIN32 分支内容，跳过 #else 分支；
            - ``#ifndef _WIN32 ... #endif``：删除整个非 Windows 分支（含 #else 前的代码）；
              ``#ifndef _WIN32 ... #else ... #endif`` 则保留 #else 分支（Windows 代码）。

        修复(#17): 原实现未处理 ``#ifndef _WIN32``，导致 Windows 下整块代码
        （如 typedef/宏定义）被预处理器跳过，编译报未定义标识符。
        """
        filtered: list[str] = []
        skip_else_branch = False   # #ifdef _WIN32 的 #else 分支（Linux）
        skip_ifndef = False        # #ifndef _WIN32 区间（非 Windows）
        keep_else_of_ifndef = False  # #ifndef..#else..#endif 中 #else 之后为 Windows 分支
        for line in source_code.split("\n"):
            if "#ifndef _WIN32" in line:
                skip_ifndef = True
                continue
            if skip_ifndef:
                if "#else" in line:
                    keep_else_of_ifndef = True
                    skip_ifndef = False
                    continue
                if "#endif" in line:
                    skip_ifndef = False
                    keep_else_of_ifndef = False
                continue
            if keep_else_of_ifndef:
                if "#endif" in line:
                    keep_else_of_ifndef = False
                    continue
                filtered.append(line)
                continue
            if "#else" in line or skip_else_branch:
                skip_else_branch = True
                if "#endif" in line:
                    skip_else_branch = False
                continue
            if "#ifdef _WIN32" in line or "#endif" in line:
                continue
            filtered.append(line)
        return filtered

    # ------------------------------------------------------------------
    # 3. 恶意/修复代码段提取
    # ------------------------------------------------------------------
    @staticmethod
    def extract_code(clean_lines: list[str]) -> tuple[list[str], list[str]]:
        """从清理后的代码行中分别提取恶意代码与修复代码。

        Returns:
            (bad_lines, good_lines)
        """
        flage_mac = True   # 宏定义区（头文件/公共部分）
        flage_bad = False
        flage_good = False
        over_flage = True

        bad_lines: list[str] = []
        good_lines: list[str] = []
        for code in clean_lines:
            if "#ifndef OMITBAD" in code:
                flage_mac = False
            # 公共部分（库文件等）
            if flage_mac:
                if "#endif" not in code:
                    bad_lines.append(code)
                    good_lines.append(code)
            # 恶意代码段
            if ("#ifndef OMITBAD" in code or flage_bad) and over_flage:
                if "#ifndef OMITGOOD" not in code and code:
                    if "bad()" in code:
                        code = "int main()"
                    bad_lines.append(code.strip(" "))
                flage_bad = True
                if "#ifndef OMITGOOD" in code:
                    over_flage = False
            # 修复代码段
            if "#ifndef OMITGOOD" in code or flage_good:
                if "#ifdef INCLUDEMAIN" not in code and "main()." not in code and code:
                    if "good()" in code:
                        code = "int main()"
                    good_lines.append(code)
                flage_good = True
                if "main()." in code:
                    break
        return bad_lines, good_lines

    # ------------------------------------------------------------------
    # 4. 生成独立缺陷/修复源文件
    # ------------------------------------------------------------------
    @staticmethod
    def generate_code_file(
        bad_lines: list[str],
        good_lines: list[str],
        source_name: str,
        flow_folder: str,
        save_dir: str | Path,
    ) -> None:
        """按提取结果生成 ``<name>bad.c`` 与 ``<name>good.c`` 两个源文件。"""
        save_dir = Path(save_dir)
        save_folder = save_dir / flow_folder
        save_folder.mkdir(parents=True, exist_ok=True)

        stem, suffix = os.path.splitext(source_name)
        bad_path = save_folder / f"{stem}bad{suffix}"
        good_path = save_folder / f"{stem}good{suffix}"

        if not (any("#ifndef OMITGOOD" in ln for ln in good_lines) or
                any("#ifndef OMITBAD" in ln for ln in bad_lines)):
            return  # 同时缺少恶意/修复代码，跳过

        # ---- 恶意文件 ----
        if bad_lines:
            bad_lines[-1] = "return 0; }"
        flage_bracket = False
        with open(bad_path, "w", encoding="utf-8") as badfile:
            for code in bad_lines:
                flage_write = True
                if "namespace" in code:
                    flage_write = False
                    flage_bracket = True
                if "{" in code and flage_bracket:
                    flage_write = False
                    flage_bracket = False
                if "#ifndef OMITBAD" in code:
                    flage_write = False
                if flage_write:
                    badfile.write(code + "\n")

        # ---- 修复文件 ----
        flage_namespace = False
        flage_bracket = False
        count = 0
        with open(good_path, "w", encoding="utf-8") as goodfile:
            for i, code in enumerate(good_lines):
                count += 1
                flage_write = True
                if "namespace" in code:
                    flage_namespace = True
                    flage_write = False
                    flage_bracket = True
                if "{" in code and flage_bracket:
                    flage_write = False
                    flage_bracket = False
                if flage_namespace and count == len(good_lines) - 1:
                    goodfile.write("return 0; }\n")
                    return
                if not flage_namespace and count == len(good_lines):
                    code = "return 0; }"
                if "#ifndef OMITGOOD" in code:
                    flage_write = False
                if flage_write:
                    goodfile.write(code + "\n")

    # ------------------------------------------------------------------
    # 5. 批量预处理入口
    # ------------------------------------------------------------------
    def julietdata_pretreatment(
        self, file_dir: str | Path, save_dir: str | Path, only_cwe: str | None = None
    ) -> None:
        """批量精简 Juliet 数据集。

        Args:
            file_dir: Juliet 缺陷用例根目录（.../C/testcases）。
            save_dir: 精简后源文件保存目录。
            only_cwe: 仅处理指定缺陷类型（如 CWE121_Stack_Based_Buffer_Overflow）；
                      None 时处理全部。增强: 支持构建单缺陷类型数据集。
        """
        save_dir = Path(save_dir)
        if save_dir.exists():
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        source_paths = self.collect_source_paths(file_dir)
        if only_cwe:
            target = Path(file_dir) / only_cwe
            source_paths = [p for p in source_paths if target in p.parents]
            if not source_paths:
                logger.warning("未找到缺陷类型 %s 的源文件（目录: %s）", only_cwe, target)
        for path in tqdm(source_paths, desc="Juliet 缺陷样例精简中..."):
            source_content, source_name, flow_folder = self.filecontent_extract(path, file_dir)
            clean_lines = self.remove_win32_guards(source_content)
            bad_lines, good_lines = self.extract_code(clean_lines)
            self.generate_code_file(
                bad_lines, good_lines, source_name, flow_folder, save_dir
            )
        logger.info("Juliet 精简完成：%s（%d 个源文件）", save_dir, len(source_paths))
