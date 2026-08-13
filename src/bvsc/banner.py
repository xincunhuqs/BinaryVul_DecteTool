"""BVSC 启动横幅与终端美化。

风格：带装饰框的启动 Banner（参考 nuclei / gobuster 等安全工具），
框线保证所有行完美对齐，Logo 使用 ANSI Shadow 标准字形。

配色：Logo 红（品牌色）· 版本/标题亮绿 · 描述绿 · 框线绿。
"""
from __future__ import annotations

import click

# BVSC ASCII Art（figlet ANSI Shadow 字体，S 锯齿块 / C 开口 / V 收拢，字形标准）
_LOGO = r"""██████╗ ██╗   ██╗███████╗ ██████╗
██╔══██╗██║   ██║██╔════╝██╔════╝
██████╔╝██║   ██║███████╗██║     
██╔══██╗╚██╗ ██╔╝╚════██║██║     
██████╔╝ ╚████╔╝ ███████║╚██████╗
╚═════╝   ╚═══╝  ╚══════╝ ╚═════╝"""

_TAGLINE = "Binary Vulnerability detection System Combined"
_SUBTITLE = "Deep learning + Inline Assembly Comparison"
_REPO = "https://github.com/xincunhuqs/BinaryVul_DecteTool"

# 框线规格（框内宽 58，保证 Logo/文本/分隔全部对齐）
_FRAME_W = 60
_INNER = _FRAME_W - 2


def _frame(content: str, fg: str | None = None, bold: bool = False,
           dim: bool = False, color: bool = True) -> str:
    """生成框内一行：左侧内容 + 右侧空格补齐到 _INNER 宽。"""
    pad = max(_INNER - len(content), 0)
    styled = content
    if fg and color:
        styled = click.style(content, fg=fg, bold=bold, dim=dim)
    return "║" + styled + " " * pad + "║"


def get_banner(version: str | None = None, color: bool = True) -> str:
    """生成 BVSC 启动横幅（装饰框布局）。

    Args:
        version: 版本号（如 "1.0.0"），非 None 时显示。
        color: 是否启用 ANSI 颜色。

    Returns:
        多行横幅字符串。
    """
    top = "╔" + "═" * _INNER + "╗"
    bottom = "╚" + "═" * _INNER + "╝"
    if color:
        top = click.style(top, fg="green", dim=True)
        bottom = click.style(bottom, fg="green", dim=True)

    blank = "║" + " " * _INNER + "║"
    lines = [top, blank]
    for logo_line in _LOGO.split("\n"):
        lines.append(_frame("  " + logo_line, fg="bright_red", bold=True, color=color))
    lines.append(blank)
    lines.append(_frame(f"  BVSC v{version}", fg="bright_green", bold=True, color=color))
    lines.append(_frame(f"  {_TAGLINE}", fg="green", color=color))
    lines.append(_frame(f"  {_SUBTITLE}", fg="green", color=color))
    lines.append(_frame(f"  {_REPO}", fg="green", dim=True, color=color))
    lines.append(blank)
    lines.append(bottom)
    return "\n".join(lines)


def print_banner(version: str | None = None, color: bool = True) -> None:
    """向 stdout 打印启动横幅。"""
    click.echo(get_banner(version=version, color=color))


__all__ = ["get_banner", "print_banner"]
