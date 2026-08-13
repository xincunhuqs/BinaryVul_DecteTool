"""BVSC 命令行入口（参照 subfinder 等安全工具的开发标准）。

约定:
    - 日志输出到 **stderr**，检测结果输出到 **stdout**（Unix 哲学）；
    - ``-silent`` 静默模式：stdout 仅输出检测结果，便于管道/脚本消费；
    - ``-json`` 结构化输出，支持与 ``-o`` 组合落盘；
    - 退出码：0=正常完成，1=参数/配置错误，2=执行异常；
    - 长/短选项并存，``--help`` 自带完整用法说明。

用法示例:
    bvsc -efp target.exe
    bvsc -efdp ./bin_folder -acsc -o result.txt
    bvsc -efp target.exe -json -silent
    bvsc -efp target.exe -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from bvsc import __version__
from bvsc.config import Settings
from bvsc.exceptions import BvscError
from bvsc.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_ERROR = 2

# 检测运行所需的重依赖（缺失时给出明确提示，避免堆栈泄漏）
_RUNTIME_DEPS = {
    "torch": "深度学习模型推理",
    "capstone": "反汇编",
    "pefile": "PE 解析",
}


def _check_runtime_deps() -> str | None:
    """检查运行依赖，返回缺失项说明（None 表示全部就绪）。"""
    import importlib.util

    missing = [
        f"{name}（{purpose}）"
        for name, purpose in _RUNTIME_DEPS.items()
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        return "缺少运行依赖: " + ", ".join(missing) + "；请执行 pip install -r requirements.txt"
    return None


def _target_paths(exefile_path: str | None, exefile_folder_path: str | None) -> list[Path]:
    """解析检测目标列表（单文件或文件夹内全部 .exe）。"""
    if exefile_folder_path:
        folder = Path(exefile_folder_path)
        if not folder.is_dir():
            raise click.BadParameter(f"文件夹不存在: {folder}", param_hint="-efdp")
        return sorted(folder.glob("*.exe"))
    if exefile_path:
        path = Path(exefile_path)
        if not path.is_file():
            raise click.BadParameter(f"文件不存在: {path}", param_hint="-efp")
        return [path]
    raise click.UsageError("必须提供 -efp（单文件）或 -efdp（文件夹）之一")


# ----------------------------------------------------------------------
# 自定义命令：专业分组帮助页（含 Banner / 分组 Options / 示例 / 退出码）
# ----------------------------------------------------------------------
class BvscCommand(click.Command):
    """自定义 click Command，重写帮助渲染：Banner + 分组排版。"""

    # 参数名 -> 分组
    _GROUPS = [
        ("Input 输入", ["exefile_path", "exefile_folder_path"]),
        ("Scan Modes 扫描模式", ["normal_scan", "accurate_scan", "record_secure"]),
        ("Output 输出", ["silent", "json_output", "output_file", "verbose"]),
        ("Other 其他", ["config_file", "show_version"]),
    ]

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """渲染美观的分组帮助页（banner 下方内容使用绿色主题）。"""
        from bvsc import __version__
        from bvsc.banner import get_banner

        formatter.width = 100  # 避免长选项换行
        formatter.write(get_banner(version=__version__, color=ctx.color) + "\n\n")
        formatter.write_usage("BVSC.py", "[OPTIONS]")
        formatter.write("\n\n")

        def _title(text: str) -> str:
            """分组标题：绿色加粗。"""
            return click.style(f"  {text}", fg="green", bold=True) if ctx.color else f"  {text}"

        def _item(term: str, desc: str) -> str:
            """选项行：term 绿色加粗 + 对齐，desc 浅绿。"""
            t = click.style(f"{term:<34}", fg="bright_green", bold=True) if ctx.color else f"{term:<34}"
            d = click.style(desc, fg="green") if ctx.color else desc
            return f"    {t}{d}"

        # Description 简介
        formatter.write(_title("Description 简介") + "\n")
        desc = (
            "    基于深度学习结合内联汇编比较的二进制漏洞检测工具\n"
            "    对应论文: Binary vulnerability detection based on deep "
            "learning combined with inline assembly comparison\n\n"
        )
        formatter.write(click.style(desc, fg="green") if ctx.color else desc)

        # 分组选项
        params = {p.name: p for p in self.get_params(ctx)}
        for title, names in self._GROUPS:
            formatter.write(_title(title) + "\n")
            for name in names:
                param = params.get(name)
                if param is None:
                    continue
                term = ", ".join(param.opts)
                formatter.write(_item(term, param.help or "") + "\n")
            formatter.write("\n")

        # Examples 示例
        formatter.write(_title("Examples 示例") + "\n")
        examples = [
            ("单文件检测", "python BVSC.py -efp target.exe"),
            ("批量检测 + 结果落盘", "python BVSC.py -efdp ./bin_folder -o result.txt"),
            ("精确扫描（DeepSeek 研判）", "python BVSC.py -efp target.exe -acsc"),
            ("JSON 结构化输出", "python BVSC.py -efp target.exe -json -silent"),
        ]
        for label, cmd in examples:
            formatter.write(_item(label, cmd) + "\n")


@click.command(name="bvsc", cls=BvscCommand,
               context_settings={"help_option_names": ["-h", "--help"]})
@click.option("-efp", "--exefile-path", "exefile_path", type=str, default=None,
              help="待检测的二进制文件路径（PE 格式）")
@click.option("-efdp", "--exefile-folder-path", "exefile_folder_path", type=str, default=None,
              help="待检测的二进制文件夹路径（批量检测其中全部 .exe）")
@click.option("-nrsc", "--normal-scan/--no-normal-scan", "normal_scan",
              is_flag=True, default=True, show_default=True,
              help="普通扫描：仅使用本地 Transformer 模型（--no-normal-scan 关闭）")
@click.option("-acsc", "--accurate-scan/--no-accurate-scan", "accurate_scan",
              is_flag=True, default=False, show_default=True,
              help="精确扫描：调用 DeepSeek 大模型对结果降噪研判（需 DEEPSEEK_API_KEY）")
@click.option("-rsd", "--record-secure/--no-record-secure", "record_secure",
              is_flag=True, default=False, show_default=True,
              help="记录检测中判定为安全的反汇编切片（供再训练）")
@click.option("-silent", "--silent", "silent", is_flag=True, default=False,
              help="静默模式：stdout 仅输出检测结果，日志写入 stderr")
@click.option("-json", "--json", "json_output", is_flag=True, default=False,
              help="以 JSON 格式输出检测结果")
@click.option("-o", "--output", "output_file", type=click.Path(path_type=Path), default=None,
              help="结果输出文件路径（与 stdout 同时输出）")
@click.option("-config", "--config", "config_file", type=click.Path(path_type=Path), default=None,
              help="配置文件路径（默认 config/config.yaml）")
@click.option("-v", "--verbose", "verbose", is_flag=True, default=False,
              help="详细模式：输出 DEBUG 级日志")
@click.option("-version", "--version", "show_version", is_flag=True, default=False,
              help="显示版本号并退出")
def main(
    exefile_path: str | None,
    exefile_folder_path: str | None,
    normal_scan: bool,
    accurate_scan: bool,
    record_secure: bool,
    silent: bool,
    json_output: bool,
    output_file: Path | None,
    config_file: Path | None,
    verbose: bool,
    show_version: bool,
) -> int:
    """BVSC —— 基于深度学习结合内联汇编比较的二进制漏洞检测工具。

    对应论文: Binary vulnerability detection based on deep learning combined
              with inline assembly comparison
    """
    if show_version:
        # 美化: 版本展示带 Banner
        from bvsc.banner import print_banner
        print_banner(version=__version__)
        return EXIT_OK

    # 日志初始化：verbose -> DEBUG；silent 时仍输出到 stderr
    setup_logging(verbose=verbose, log_file=None)

    # ------- 配置加载 -------
    try:
        settings = Settings.load(config_file)
        # CLI 参数覆盖配置文件
        if accurate_scan:
            settings.raw.setdefault("mode", {})["accurate_scan"] = True
            settings.raw.setdefault("llm", {})["enabled"] = True
        if record_secure:
            settings.raw.setdefault("mode", {})["record_secure_disassembly"] = True
        if not normal_scan:
            settings.raw.setdefault("mode", {})["normal_scan"] = False
    except BvscError as exc:
        click.echo(f"[!] 配置加载失败: {exc}", err=True)
        return EXIT_USAGE

    # ------- 目标解析 -------
    try:
        targets = _target_paths(exefile_path, exefile_folder_path)
    except (click.BadParameter, click.UsageError) as exc:
        click.echo(f"[!] {exc}", err=True)
        return EXIT_USAGE
    if not targets:
        click.echo("[!] 未找到待检测的 .exe 文件", err=True)
        return EXIT_USAGE

    # ------- 运行模式校验：至少需要一种扫描模式 -------
    if not normal_scan and not accurate_scan:
        click.echo("[!] 请至少开启一种扫描模式（-nrsc 或 -acsc）", err=True)
        return EXIT_USAGE

    # ------- 运行依赖预检（subfinder 风格：环境错误明确提示） -------
    missing_deps = _check_runtime_deps()
    if missing_deps:
        click.echo(f"[!] {missing_deps}", err=True)
        return EXIT_USAGE

    # ------- 执行检测 -------
    from bvsc.detector import BinaryVulnerabilityDetector

    # 美化: 非静默模式下打印启动 Banner（stdout），静默模式保持输出纯净
    if not silent:
        from bvsc.banner import print_banner
        print_banner(version=__version__)

    exit_code = EXIT_OK
    try:
        # 修复(B4): detector 构造（含 Disassembler 初始化）纳入 try/except，
        # 避免异常时向终端泄漏完整 Python 堆栈（subfinder 风格要求明确错误信息）。
        detector = BinaryVulnerabilityDetector(settings)
        results = []
        for target in targets:
            logger.info(">>> 检测目标: %s", target)
            report = detector.detect_file(target)
            results.append(report)
            _emit(report, silent=silent, json_output=json_output)
        if output_file:
            _write_output_file(results, output_file)
    except BvscError as exc:
        click.echo(f"[!] 检测执行失败: {exc}", err=True)
        exit_code = EXIT_ERROR
    except Exception as exc:  # 兜底异常，避免堆栈泄漏到终端
        logger.exception("未预期异常")
        click.echo(f"[!] 检测执行异常: {exc}", err=True)
        exit_code = EXIT_ERROR
    return exit_code


# ----------------------------------------------------------------------
# 输出
# ----------------------------------------------------------------------
def _emit(report, silent: bool, json_output: bool) -> None:
    """按模式输出单文件检测结果到 stdout。

    - silent=False 且非 json：输出完整文本报告；
    - silent=True：仅输出命中行（``文件路径:漏洞类型:切片#n``）；
    - json：输出 JSON 数组元素。
    """
    confirmed = report.confirmed_items()
    # 修复(B12): 反汇编/处理失败时向用户明确提示（stdout 结果、stderr 日志分离）
    if report.error:
        logger.warning("文件处理失败: %s: %s", report.target_file, report.error)
    if json_output:
        payload = {
            "target_file": report.target_file,
            "detect_time": report.detect_time,
            "total_slices": report.total_slices,
            "defect_count": len(confirmed),
            "defects": [
                {
                    "index": it.index,
                    "vul_type": it.vul_type,
                    "verdict": it.verdict,
                    "analysis": it.analysis,
                }
                for it in confirmed
            ],
        }
        click.echo(json.dumps(payload, ensure_ascii=False))
        return

    if silent:
        for it in confirmed:
            click.echo(f"{report.target_file}:{it.vul_type}:slice#{it.index}")
        return

    # 默认模式：人类可读摘要
    click.echo(f"[*] {report.target_file}  切片总数={report.total_slices} "
               f"确认缺陷={len(confirmed)}")
    if report.error:
        click.echo(f"    [!] 处理失败: {report.error}")
    for it in confirmed:
        click.echo(f"    slice#{it.index}  {it.vul_type}"
                   + (f"  [verdict={it.verdict}]" if it.verdict else ""))


def _write_output_file(results, output_file: Path) -> None:
    """将全部结果写入 -o 指定的文件（文本模式）。"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for report in results:
        lines.append(f"=== {report.target_file} ===")
        for it in report.confirmed_items():
            lines.append(f"{report.detect_time}  {report.target_file}  "
                         f"切片#{it.index}  {it.vul_type}")
            lines.append(f"缺陷汇编代码: {it.slice_code}")
            if it.analysis:
                lines.append(f"分析: {it.analysis}")
        lines.append("")
    output_file.write_text("\n".join(lines), encoding="utf-8")
    logger.info("结果已写入: %s", output_file)


if __name__ == "__main__":
    sys.exit(main(standalone_mode=False))  # type: ignore[call-arg]
