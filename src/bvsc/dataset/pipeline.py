"""缺陷数据集构建管线（论文六步管线 ①-④ 步的编排）。

完整流程:
    ① Juliet 精简预处理   -> 缺陷/修复源文件
    ② 内联汇编插入        -> 缺陷标识（nop/jmp 掩码）
    ③ 变量替换成模板      -> 缺陷模板
    ④ 模板随机实例化扩样  -> 缺陷样本集
    ⑤ MSVC 编译          -> 缺陷可执行文件
    ⑥ 反汇编+差异比较切片 -> 缺陷汇编切片数据集

所有路径与参数均来自 :class:`bvsc.config.Settings`。
"""
from __future__ import annotations

import shutil
from pathlib import Path

from bvsc.config import PROJECT_ROOT, Settings
from bvsc.dataset.augment import VulnerabilityFileExtend
from bvsc.dataset.compiler import automatically_acquired_exefile
from bvsc.dataset.inline_assembly import AddInlineAssembly
from bvsc.dataset.juliet import JulietPretreatment
from bvsc.dataset.slicing_code import GenerateDissassemblySlicing
from bvsc.dataset.template import VariableSubstitution
from bvsc.logging_setup import get_logger

logger = get_logger(__name__)


class GenerateVulSliceData:
    """缺陷汇编切片数据集构建器。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.work_dir = Path(settings.get("dataset", "work_dir", "data/temp"))
        self.work_dir = (
            self.work_dir if self.work_dir.is_absolute()
            else PROJECT_ROOT / self.work_dir
        )
        # 汇总切片数据集输出位置（注意：严禁写入 Juliet 原始数据目录）
        self.total_slicing_file = PROJECT_ROOT / "data" / "total_defect_slicing.txt"

        self.pretreatment_dir = self.work_dir / "pretreatment"
        self.inline_source_dir = self.work_dir / "windowsourcefile_inline_template"
        self.template_dir = self.work_dir / "windowsource_file_template"
        self.instantiation_dir = self.work_dir / "windowsource_file_instantiation"
        self.exefile_dir = self.work_dir / "windowsource_VUL_exefile"
        self.slicing_dir = self.work_dir / "windowsource_VUL_slicingcode"

    # ------------------------------------------------------------------
    # 各阶段（对应论文步骤 ①-④）
    # ------------------------------------------------------------------
    def stage_juliet_pretreatment(self) -> None:
        """① Juliet 精简：提取缺陷/修复独立源文件。"""
        juliet_dir = self.settings.resolve_path("dataset", "juliet_testcases_dir")
        # 增强: 支持仅构建单一缺陷类型（config: dataset.only_cwe）
        only_cwe = self.settings.get("dataset", "only_cwe")
        JulietPretreatment().julietdata_pretreatment(
            juliet_dir, self.pretreatment_dir, only_cwe=only_cwe
        )

    def stage_inline_assembly(self) -> None:
        """② 内联汇编插入。"""
        AddInlineAssembly().gather_inline_assembly_example(
            self.pretreatment_dir, self.inline_source_dir
        )

    def stage_template(self) -> None:
        """③ 变量替换 -> 缺陷模板。"""
        VariableSubstitution().gather_tempalte_variable(
            self.inline_source_dir, self.template_dir
        )

    def stage_augment(self) -> None:
        """④ 模板实例化扩样。"""
        expand = int(self.settings.get("dataset", "expand_samples_per_template", 20))
        VulnerabilityFileExtend(expand_number_samples=expand).gather_instantiation_file(
            self.template_dir, self.instantiation_dir
        )

    def stage_compile(self) -> None:
        """⑤ MSVC 编译为可执行文件。"""
        vcvarsall = self.settings.get("dataset", "vcvarsall_path")
        support_dir = self.settings.resolve_path("dataset", "support_dir")
        subst_drive = str(self.settings.get("dataset", "subst_drive", "K:"))
        error_log = self.work_dir / "fild_compilation_cfile.txt"
        automatically_acquired_exefile(
            self.instantiation_dir, self.exefile_dir,
            vcvarsall, support_dir, subst_drive, error_log,
        )

    def stage_slicing(self) -> None:
        """⑥ 反汇编 + 差异比较提取缺陷切片。"""
        gather = GenerateDissassemblySlicing()
        fail_log = self.work_dir / "faild_exefile.txt"
        gather.auto_gather_disassemcode_slicing_from_fatherfolder(
            self.exefile_dir, self.slicing_dir, fail_log
        )
        gather.get_total_defect_slicing(self.slicing_dir, self.total_slicing_file)

    # ------------------------------------------------------------------
    # 一键构建
    # ------------------------------------------------------------------
    def generater_vulexefile(self) -> Path:
        """执行完整数据集构建管线。

        Returns:
            汇总后的缺陷切片数据集文件路径。
        """
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        logger.info("========== 阶段① Juliet 精简 ==========")
        self.stage_juliet_pretreatment()
        logger.info("========== 阶段② 内联汇编插入 ==========")
        self.stage_inline_assembly()
        logger.info("========== 阶段③ 缺陷模板生成 ==========")
        self.stage_template()
        logger.info("========== 阶段④ 缺陷样本扩展 ==========")
        self.stage_augment()
        logger.info("========== 阶段⑤ 编译可执行文件 ==========")
        self.stage_compile()
        logger.info("========== 阶段⑥ 反汇编切片提取 ==========")
        self.stage_slicing()

        logger.info("数据集构建完成：%s", self.total_slicing_file)
        return self.total_slicing_file
