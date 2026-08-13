"""检测流程编排（论文第六步：反汇编 -> 切片 -> 模型识别 -> 可选 LLM 降噪）。

流程:
    1. 反汇编待检 PE 的 ``.text`` 节；
    2. 按块大小切片；
    3. 逐切片送入本地 Transformer 预测漏洞类型；
    4. 命中 CWE 的候选切片，在 accurate_scan 模式下调用 DeepSeek 研判降噪；
    5. 结果汇总为 :class:`DetectionReport` 并落盘（文本 + JSON）。

原实现问题修复:
    - ``-efdp`` 文件夹批量检测真正接线；
    - 安全切片记录（record_secure_disassembly）逻辑独立成方法；
    - 大模型调用失败不再导致整个检测中断。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from bvsc.config import Settings
from bvsc.disassembler import Disassembler
from bvsc.exceptions import DisassemblyError, LlmError, ModelError
from bvsc.llm_client import (
    VERDICT_CONFIRMED,
    VERDICT_FALSE_POSITIVE,
    DeepSeekClient,
)
from bvsc.logging_setup import get_logger
from bvsc.reporter import DetectionItem, DetectionReport, Reporter, now_str
from bvsc.slicer import slice_disassembly

logger = get_logger(__name__)

_CWE_PREFIX = "CWE"


class BinaryVulnerabilityDetector:
    """二进制漏洞检测器。"""

    def __init__(
        self,
        settings: Settings,
        predictor: Optional[VulnerabilityPredictor] = None,
        llm_client: Optional[DeepSeekClient] = None,
    ) -> None:
        """Args:
            settings: 应用配置。
            predictor: 本地模型预测器（惰性加载）。
            llm_client: DeepSeek 客户端（accurate_scan 模式下使用）。
        """
        self.settings = settings
        self._predictor = predictor
        self._llm_client = llm_client

        arch = str(settings.get("detection", "arch", "x86"))
        mode = str(settings.get("detection", "mode", 32))
        self._disassembler = Disassembler(arch, mode)

        output_dir = settings.resolve_path("detection", "output_dir")
        self._reporter = Reporter(output_dir)
        self._accurate_scan = bool(settings.get("mode", "accurate_scan", False))
        self._record_secure = bool(
            settings.get("mode", "record_secure_disassembly", False)
        )
        self._discovery_dir = Path(
            str(settings.get("detection", "discovery_dir", "DefectDiscoveryTrainDate"))
        )

    # ------------------------------------------------------------------
    # 惰性依赖
    # ------------------------------------------------------------------
    @property
    def predictor(self) -> "VulnerabilityPredictor":
        """延迟加载本地模型预测器（torch 依赖在首次使用时才导入）。"""
        if self._predictor is None:
            try:
                from bvsc.model.predictor import VulnerabilityPredictor
                from bvsc.model.transformer import config_from_settings
            except ImportError as exc:  # 缺 torch 等重依赖
                raise ModelError(
                    "缺少深度学习依赖（torch），请先执行 pip install -r requirements.txt"
                ) from exc

            self._predictor = VulnerabilityPredictor(
                checkpoint=self.settings.model_checkpoint,
                tokenizer_path=self.settings.tokenizer_path,
                device=self.settings.device,
                max_seq_len=int(self.settings.get("model", "max_seq_len", 700)),
                model_config=config_from_settings(self.settings),
            )
        return self._predictor

    @property
    def llm_client(self) -> DeepSeekClient | None:
        if self._llm_client is None and self._accurate_scan:
            api_key = self.settings.llm_api_key
            if api_key:
                self._llm_client = DeepSeekClient(
                    api_key=api_key,
                    base_url=str(self.settings.get("llm", "base_url", "https://api.deepseek.com")),
                    model=str(self.settings.get("llm", "model", "deepseek-chat")),
                    timeout=int(self.settings.get("llm", "timeout", 120)),
                )
            else:
                logger.warning("accurate_scan 已开启但未配置 API Key，将跳过 LLM 降噪")
        return self._llm_client

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def detect_file(self, exefile: str | Path) -> DetectionReport:
        """检测单个 PE 可执行文件。"""
        exefile = Path(exefile)
        if not exefile.exists():
            raise FileNotFoundError(f"待检文件不存在: {exefile}")

        logger.info("开始检测: %s", exefile)
        report = DetectionReport(target_file=str(exefile), detect_time=now_str())

        # 1. 反汇编
        try:
            disassembly = self._disassembler.disassemble(exefile)
        except DisassemblyError as exc:
            # 修复(B12): 失败信息写入报告，CLI 可向用户明确提示（原实现仅打日志，静默返回空报告）
            logger.error("反汇编失败，跳过: %s", exc)
            report.error = str(exc)
            return report

        # 2. 切片
        block_size = int(self.settings.get("detection", "slice_block_size", 100))
        block_range = None
        if block_size <= 0:
            block_range = (
                int(self.settings.get("detection", "slice_block_size_min", 80)),
                int(self.settings.get("detection", "slice_block_size_max", 130)),
            )
        slices = slice_disassembly(disassembly, block_size, block_range)
        report.total_slices = len(slices)
        logger.info("切片完成: %d 个", len(slices))

        # 显式触发模型加载：加载失败（缺依赖/权重缺失）立即冒泡，不吞异常
        self.predictor

        # 3-4. 逐切片预测 + 可选 LLM 降噪
        for idx, slice_text in enumerate(slices, 1):
            try:
                _, vul_type = self.predictor.predict(slice_text)
            except Exception as exc:  # 单个切片失败不影响整体
                logger.warning("切片 %d 预测失败: %s", idx, exc)
                continue

            if _CWE_PREFIX in vul_type:
                item = DetectionItem(index=idx, slice_code=slice_text, vul_type=vul_type)
                if self._accurate_scan:
                    self._llm_verify(item)
                else:
                    # 普通扫描：模型判定即为最终结果（进入 JSON/静默输出）
                    item.is_confirmed = True
                report.candidates.append(item)
                logger.info("发现候选缺陷: 切片#%d -> %s", idx, vul_type)
            elif self._record_secure:
                report.secure_slices.append(slice_text)

        # 5. 报告落盘
        self._write_reports(report)
        if self._record_secure and report.secure_slices:
            self._save_secure_slices(report)
        return report

    def detect_folder(self, folder_path: str | Path) -> list[DetectionReport]:
        """批量检测文件夹内全部 ``.exe`` 文件。"""
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            raise FileNotFoundError(f"待检文件夹不存在: {folder_path}")

        reports: list[DetectionReport] = []
        exe_files = sorted(folder_path.glob("*.exe"))
        logger.info("文件夹内共 %d 个可执行文件: %s", len(exe_files), folder_path)
        for exe_file in exe_files:
            try:
                reports.append(self.detect_file(exe_file))
            except Exception as exc:
                logger.error("检测失败，跳过 %s: %s", exe_file, exc)
        return reports

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _llm_verify(self, item: DetectionItem) -> None:
        """调用大模型研判候选缺陷，按结论过滤。"""
        client = self.llm_client
        if client is None:
            return
        try:
            analysis = client.analyze(item.slice_code, item.vul_type)
            verdict = client.verdict(analysis)
        except LlmError as exc:
            logger.warning("LLM 研判失败，保留候选: %s", exc)
            return
        item.analysis = analysis
        item.verdict = verdict
        item.is_confirmed = verdict == VERDICT_CONFIRMED
        if verdict == VERDICT_FALSE_POSITIVE:
            logger.info("LLM 判定为误报: 切片#%d", item.index)

    def _write_reports(self, report: DetectionReport) -> None:
        stem = Path(report.target_file).stem
        self._reporter.write_text_report(report, f"{stem}_result.txt")
        self._reporter.write_json_report(report, f"{stem}_result.json")

    def _save_secure_slices(self, report: DetectionReport) -> None:
        """保存判定为安全的切片（用于后续再训练，论文提及的降噪手段）。

        修复(B11): 输出带 ``NO_VULN`` 标签的训练格式行（切片\t\tNO_VULN），
        可直接作为负样本并入训练数据，弥补训练集无良性样本导致的 100% 误报问题。
        """
        path = self._discovery_dir / f"{Path(report.target_file).stem}_secure.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{sl}\t\tNO_VULN" for sl in report.secure_slices]
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("安全切片已记录: %s (%d 条)", path, len(report.secure_slices))
