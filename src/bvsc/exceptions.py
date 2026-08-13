"""BVSC 自定义异常层次。

所有异常均继承自 :class:`BvscError`，便于上层统一捕获与日志记录。
"""


class BvscError(Exception):
    """BVSC 全局异常基类。"""


class ConfigError(BvscError):
    """配置加载/校验失败。"""


class DisassemblyError(BvscError):
    """反汇编失败（非 PE 文件、缺少 .text 节、引擎初始化失败等）。"""


class SlicingError(BvscError):
    """汇编切片失败。"""


class ModelError(BvscError):
    """模型加载/推理失败。"""


class LlmError(BvscError):
    """大模型调用失败。"""


class DatasetBuildError(BvscError):
    """缺陷数据集构建失败。"""


class CompilationError(BvscError):
    """MSVC 编译链接失败。"""
