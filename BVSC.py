#!/usr/bin/env python3
"""BVSC 命令行入口包装器（保持原工具名，兼容 ``python BVSC.py ...`` 用法）。

等价于安装后执行 ``bvsc`` 命令。用法:
    python BVSC.py -efp target.exe [-v] [-acsc] [-silent] [-json] [-o result.txt]
    python BVSC.py -efdp ./bin_folder ...
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from bvsc.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(standalone_mode=False))  # type: ignore[call-arg]
