"""日志初始化：统一日志格式、级别与输出通道（控制台 + 文件）。"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | str | None = None,
    verbose: bool = False,
) -> None:
    """初始化全局日志。

    Args:
        level: 日志级别（文件日志使用）。
        log_file: 可选日志文件路径（自动轮转 5MB x 3 份）。
        verbose: 为 True 时控制台输出 DEBUG 级信息（对应 CLI -v 参数）。
    """
    global _initialized
    if _initialized:
        return

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    console_level = logging.DEBUG if verbose else logging.INFO
    # 修复(#19): 日志统一输出到 stderr（Unix 哲学），stdout 仅承载检测结果，
    # 保证 -silent / 管道消费场景 stdout 纯净（原实现误用 sys.stdout）。
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))
    root.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))
        root.addHandler(file_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """获取带模块名命名空间的 logger。"""
    return logging.getLogger(name)
