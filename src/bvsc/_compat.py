"""第三方库兼容层：可选依赖缺失时优雅降级。

企业级实践中，纯逻辑模块不应因进度条等非核心依赖缺失而整体不可用。
若 ``tqdm`` 未安装，退化为无操作进度条（``iter`` 原样返回）。
"""
from __future__ import annotations

from typing import Any, Iterable, Iterator

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover

    class tqdm:  # type: ignore[no-redef]
        """tqdm 的 no-op 替代品。"""

        def __init__(self, iterable: Iterable[Any] | None = None, *args, **kwargs) -> None:
            self._iterable = iterable
            self.n = 0
            self.total = kwargs.get("total", None)

        def __iter__(self) -> Iterator[Any]:
            return iter(self._iterable or ())

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            return None

        def update(self, n: int = 1) -> None:
            """进度 +n（no-op 实现，仅记录计数）。"""
            self.n += n

        def set_postfix(self, **kwargs) -> None:
            """设置尾部附加信息（no-op）。"""
            return None


__all__ = ["tqdm"]
