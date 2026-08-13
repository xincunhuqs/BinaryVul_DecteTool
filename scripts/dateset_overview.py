#!/usr/bin/env python3
"""统计缺陷可执行数据集基本信息（缺陷类型 / 缺陷个数 / 缺陷成因种类）。

用法:
    python scripts/dateset_overview.py <缺陷可执行文件根目录>
"""
from __future__ import annotations  # 修复(B9): X|Y 注解在 Python<3.10 需未来导入

import sys
from pathlib import Path


def get_baseinfo(folder_path: str | Path) -> dict[str, list[int]]:
    """统计每个缺陷类型文件夹的可执行文件数量与成因种类数。

    成因种类：以文件名 ``_<id>_`` 倒数第二段去重计数。
    """
    folder_path = Path(folder_path)
    baseinfo: dict[str, list[int]] = {}
    for folder in sorted(folder_path.iterdir()):
        if not folder.is_dir():
            continue
        exefiles = [f for f in folder.iterdir() if f.suffix.lower() == ".exe"]
        causes = {f.stem.split("_")[-2] for f in exefiles if len(f.stem.split("_")) >= 2}
        baseinfo[folder.name] = [len(exefiles), len(causes)]
    return baseinfo


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python dateset_overview.py <缺陷可执行文件根目录>")
        sys.exit(1)
    baseinfo = get_baseinfo(sys.argv[1])
    for vul_type, (count, causes) in baseinfo.items():
        print("=" * 60)
        print(f"缺陷类型: {vul_type}")
        print(f"缺陷个数: {count}")
        print(f"缺陷成因种类: {causes}")
    print("=" * 60)


if __name__ == "__main__":
    main()
