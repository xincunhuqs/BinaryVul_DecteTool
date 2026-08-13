"""BVSC 启动横幅与终端美化。

风格参考 dirfinder / ffuf / gobuster 等安全工具：
    - 粗体 block Logo（S/C 字母区分度高，避免与 BVCC 混淆）
    - 居中版本行 + 对齐的描述/仓库信息 + 双线分隔符
    - ANSI 配色（红 = Logo 品牌色，绿 = 信息区）
"""
from __future__ import annotations

import click

# BVSC ASCII Art（figlet block 粗体字体，字母间隔 2 空格）
_LOGO = r"""██████╗  ██╗   ██╗  ███████╗   ██████╗
██╔══██╗  ██║   ██║  ██╔════╝  ██╔════╝
██████╔╝  ██║   ██║  ███████╗  ██║     
██╔══██╗  ╚██╗ ██╔╝  ╚════██║  ██║     
██████╔╝   ╚████╔╝  ███████║  ╚██████╗
╚═════╝    ╚═══╝  ╚══════╝   ╚═════╝"""

_TAGLINE = "Binary Vulnerability detection System Combined"
_SUBTITLE = "Deep learning + Inline Assembly Comparison"
_REPO = "https://github.com/xincunhuqs/BinaryVul_DecteTool"
_SEP = "═" * 58


def get_banner(version: str | None = None, color: bool = True) -> str:
    """生成 BVSC 启动横幅。

    Args:
        version: 版本号（如 "1.0.0"），非 None 时显示。
        color: 是否启用 ANSI 颜色。

    Returns:
        多行横幅字符串。
    """
    if color:
        logo = click.style(_LOGO, fg="bright_red", bold=True)
        title = (
            click.style(f"              BVSC v{version}", fg="bright_green", bold=True)
            if version else click.style("              BVSC", fg="bright_green", bold=True)
        )
        tagline = click.style(f"    {_TAGLINE}", fg="green")
        subtitle = click.style(f"       {_SUBTITLE}", fg="green")
        repo = click.style(f"       {_REPO}", fg="green", dim=True)
        sep = click.style(_SEP, fg="green", dim=True)
    else:
        logo = _LOGO
        title = f"              BVSC v{version}" if version else "              BVSC"
        tagline = f"    {_TAGLINE}"
        subtitle = f"       {_SUBTITLE}"
        repo = f"       {_REPO}"
        sep = _SEP
    lines = [logo, "", title, tagline, subtitle, repo, sep]
    return "\n".join(lines)


def print_banner(version: str | None = None, color: bool = True) -> None:
    """向 stdout 打印启动横幅（含版本号与分隔线）。"""
    click.echo(get_banner(version=version, color=color))


__all__ = ["get_banner", "print_banner"]
