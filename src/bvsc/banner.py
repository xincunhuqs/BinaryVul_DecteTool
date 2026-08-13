"""BVSC 启动横幅与终端美化。

风格参考 dirfinder / ffuf / gobuster 等安全工具：
    - 粗体 block Logo（S/C 等字母区分度高，避免与 BVCC 混淆）
    - 版本 / 工具全名 / 仓库信息
    - ANSI 配色（红 = Logo，绿 = 版本，白 = 工具名，青 = 描述）
"""
from __future__ import annotations

import click

# BVSC ASCII Art（figlet block 粗体字体）
# 说明：block 字体的 S（实心锯齿块）与 C（开口矩形）区分明显，
# 避免 Standard 字体下 S/C 形似导致误读为 BVCC。
_LOGO = r"""██████╗  ██╗   ██╗  ███████╗   ██████╗
██╔══██╗  ██║   ██║  ██╔════╝  ██╔════╝
██████╔╝  ██║   ██║  ███████╗  ██║     
██╔══██╗  ╚██╗ ██╔╝  ╚════██║  ██║     
██████╔╝   ╚████╔╝  ███████║  ╚██████╗
╚═════╝    ╚═══╝  ╚══════╝   ╚═════╝"""

_TAGLINE = "Binary Vulnerability detection System Combined"
_SUBTITLE = "Deep learning + Inline Assembly Comparison"
_REPO = "https://github.com/xincunhuqs/BinaryVul_DecteTool"
_SEP = "=" * 56


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
        head = click.style(f"BVSC v{version}", fg="bright_green", bold=True) if version \
            else click.style("BVSC", fg="bright_green", bold=True)
        tagline = click.style(f"  {_TAGLINE}", fg="bright_white")
        subtitle = click.style(f"  {_SUBTITLE}", fg="cyan")
        repo = click.style(f"  {_REPO}", fg="blue", dim=True)
    else:
        logo, head, tagline, subtitle, repo = (
            _LOGO,
            f"BVSC v{version}" if version else "BVSC",
            f"  {_TAGLINE}",
            f"  {_SUBTITLE}",
            f"  {_REPO}",
        )
    lines = [logo, "", head + tagline, subtitle, repo, _SEP]
    return "\n".join(lines)


def print_banner(version: str | None = None, color: bool = True) -> None:
    """向 stdout 打印启动横幅（含版本号与分隔线）。"""
    click.echo(get_banner(version=version, color=color))


__all__ = ["get_banner", "print_banner"]
