"""MSVC 编译链接（论文第四步前半：缺陷样本编译为可执行文件）。

在 Windows 环境下使用 Visual Studio 的 ``cl.exe`` 编译缺陷样本：
    - 通过 ``vcvarsall.bat`` 初始化编译环境；
    - 使用 ``subst`` 虚拟盘符规避 Windows 路径过长问题；
    - 编译错误记录到日志文件，单个文件夹连续失败超阈值时跳过。
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from bvsc._compat import tqdm

from bvsc.exceptions import CompilationError
from bvsc.logging_setup import get_logger

logger = get_logger(__name__)

# 连续编译失败达到该次数后跳过当前文件夹（原实现为 index+=20 的等价行为）
_ERROR_TOLERANCE = 20


def compile_folder(
    folder_path: str | Path,
    save_folder: str | Path,
    vcvarsall_path: str,
    support_dir: str | Path,
    subst_drive: str = "K:",
    error_log: str | Path | None = None,
) -> None:
    """编译单个缺陷类型文件夹下的全部源文件。

    Args:
        folder_path: 缺陷源文件所在文件夹（如 .../CWE416_Use_After_Free）。
        save_folder: 可执行文件保存根目录（其下按缺陷类型分子目录）。
        vcvarsall_path: vcvarsall.bat 绝对路径。
        support_dir: 编译支持头文件目录（conf/testcasesupport）。
        subst_drive: subst 虚拟盘符，如 ``K:``。
        error_log: 编译错误日志文件路径。

    Raises:
        CompilationError: 编译环境缺失（vcvarsall 不存在）。
    """
    if not Path(vcvarsall_path).exists():
        raise CompilationError(f"vcvarsall.bat 不存在: {vcvarsall_path}")

    vul_type = Path(folder_path).name
    source_files = sorted(
        f for f in os.listdir(folder_path) if f.lower().endswith((".c", ".cpp"))
    )
    if not source_files:
        logger.warning("文件夹无源文件，跳过: %s", folder_path)
        return 0, 0

    temp_folder = Path(save_folder) / vul_type
    temp_folder.mkdir(parents=True, exist_ok=True)

    support_dir = Path(support_dir).resolve()
    folder_path = Path(folder_path).resolve()

    # 虚拟盘符映射（规避 Windows 路径过长问题）
    subprocess.run(f"subst {subst_drive} /D", shell=True, capture_output=True)
    subprocess.run(f"subst {subst_drive} {folder_path}", shell=True, capture_output=True)

    failure_count = 0
    processed = 0       # 实际处理的源文件数（含失败/超时）
    success_count = 0   # 编译成功数
    total = len(source_files)
    for source_file in source_files:
        processed += 1
        _t0 = time.time()
        # 修复(B6): 原 /Fo"目录" 无尾斜杠触发 D8036；带尾斜杠在 cmd 引号解析下仍不稳定。
        # 改为不传 /Fo，以 cwd=temp_folder 运行，.obj 默认输出到目标目录。
        command = (
            f'call "{vcvarsall_path}" x86 && cl /I "{support_dir}" '
            f'/Fe"{temp_folder / Path(source_file).stem}" '
            f'"{support_dir}/io.c" "{support_dir}/std_thread.c" '
            f'"{subst_drive}\\{source_file}"'
        )
        logger.debug("编译命令: %s", command)

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=temp_folder,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=300,
            )
            output = (result.stdout or "") + (result.stderr or "")
        except subprocess.TimeoutExpired:
            logger.info("[编译 %d/%d] ⏱ %s 超时(300s)", processed, total, source_file)
            failure_count += 1
            if failure_count >= _ERROR_TOLERANCE:
                logger.warning("[编译] ✗✗ 连续 %d 次失败，跳过文件夹: %s", _ERROR_TOLERANCE, folder_path)
                break
            continue

        if result.returncode != 0 or "error" in output.lower():
            failure_count += 1
            _log_compile_error(command, source_file, output, error_log)
            logger.info("[编译 %d/%d] ✗ %s 失败(%.1fs)",
                        processed, total, source_file, time.time() - _t0)
            if failure_count >= _ERROR_TOLERANCE:
                logger.warning("[编译] ✗✗ 连续 %d 次失败，跳过文件夹: %s", _ERROR_TOLERANCE, folder_path)
                break
        else:
            failure_count = 0
            success_count += 1
            logger.info("[编译 %d/%d] ✓ %s 成功(%.1fs)",
                        processed, total, source_file, time.time() - _t0)

    subprocess.run(f"subst {subst_drive} /D", shell=True, capture_output=True)
    logger.info("编译完成: %s -> %s（成功 %d/%d）",
                folder_path, temp_folder, success_count, processed)
    return processed, success_count


def _log_compile_error(command: str, source_file: str, output: str, error_log) -> None:
    """记录编译错误到日志文件。"""
    if error_log is None:
        logger.error("编译失败 %s: %s", source_file, output[-500:])
        return
    path = Path(error_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"命令: {command}\n时间: {timestamp}\n源文件: {source_file}\n{output}\n\n")
    logger.error("编译失败 %s，已记录到 %s", source_file, path)


def automatically_acquired_exefile(
    folder_father: str | Path,
    save_folder: str | Path,
    vcvarsall_path: str,
    support_dir: str | Path,
    subst_drive: str = "K:",
    error_log: str | Path | None = None,
) -> None:
    """批量编译缺陷源文件根目录下的全部缺陷类型文件夹。

    Args:
        folder_father: 缺陷源文件根目录（含 CWE 子目录）。
        save_folder: 可执行文件保存根目录。
        vcvarsall_path: vcvarsall.bat 绝对路径。
        support_dir: 编译支持目录。
        subst_drive: subst 虚拟盘符。
        error_log: 编译错误日志路径。
    """
    folder_father = Path(folder_father)
    subfolders = sorted(
        p for p in folder_father.iterdir() if p.is_dir()
    )
    if not subfolders:
        logger.warning("无缺陷类型文件夹，跳过编译: %s", folder_father)
        return

    # 优化: 进度条按「源代码文件总数」展示编译进度（原实现按缺陷类型文件夹数，
    # 单类型构建时进度条恒为 0/1，无法反映真实编译进度）。
    folder_sources: dict[str, int] = {}
    total_sources = 0
    for subfolder in subfolders:
        n = sum(
            1 for f in os.listdir(subfolder)
            if f.lower().endswith((".c", ".cpp"))
        )
        folder_sources[subfolder.name] = n
        total_sources += n
    logger.info("开始批量编译 %d 个缺陷类型文件夹，共 %d 个源文件...",
                len(subfolders), total_sources)
    if total_sources == 0:
        logger.warning("未找到任何 .c/.cpp 源文件，跳过编译: %s", folder_father)
        return

    total_success = 0
    with tqdm(total=total_sources, desc="批量编译可执行文件中") as pbar:
        for subfolder in subfolders:
            try:
                processed, success = compile_folder(
                    subfolder, save_folder, vcvarsall_path,
                    support_dir, subst_drive, error_log,
                )
            except CompilationError as exc:
                logger.error("跳过文件夹 %s: %s", subfolder, exc)
                processed = folder_sources.get(subfolder.name, 0)
                success = 0
            total_success += success
            pbar.update(processed)
            pbar.set_postfix(文件夹=subfolder.name, 成功=total_success)
    logger.info("批量编译完成：成功 %d / %d 个源文件", total_success, total_sources)
