"""检测结果模型与报告输出。

报告形态:
    - 文本报告（与原 ``single_deteresult.txt`` 格式兼容，便于人工阅读）；
    - JSON 报告（结构化，便于下游系统集成）。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bvsc.logging_setup import get_logger

logger = get_logger(__name__)

_SEP_LINE = "-" * 100


@dataclass
class DetectionItem:
    """单条缺陷检测结果。"""

    index: int
    slice_code: str
    vul_type: str
    probability: float | None = None
    analysis: str | None = None          # 大模型研判文本
    verdict: str | None = None           # 研判结论（confirmed/not_exploitable/false_positive）
    is_confirmed: bool = False           # 是否确认可利用（进入最终报告）


@dataclass
class DetectionReport:
    """一次检测任务（单文件）的完整报告。"""

    target_file: str
    detect_time: str
    total_slices: int = 0
    candidates: list[DetectionItem] = field(default_factory=list)
    secure_slices: list[str] = field(default_factory=list)
    error: str | None = None  # 修复(B12): 记录反汇编/检测过程中的错误信息，供上层提示用户

    def confirmed_items(self) -> list[DetectionItem]:
        return [it for it in self.candidates if it.is_confirmed]


class Reporter:
    """检测结果落盘。"""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def write_text_report(
        self, report: DetectionReport, filename: str | None = None
    ) -> Path:
        """写入文本报告（与原 single_deteresult.txt 格式兼容）。"""
        filename = filename or f"{Path(report.target_file).stem}_result.txt"
        path = self.output_dir / filename

        lines: list[str] = []
        lines.append(_SEP_LINE)
        lines.append(f"检测时间：{report.detect_time}")
        lines.append(f"检测文件：{report.target_file}")
        lines.append(f"切片总数：{report.total_slices}，缺陷候选：{len(report.candidates)}")
        lines.append(_SEP_LINE)

        for item in report.candidates:
            lines.append(
                f"\n--------------------------------------缺陷代码块索引：{item.index}"
                f"---------------------------------------"
            )
            lines.append(f"检测时间：{report.detect_time}")
            lines.append(f"检测文件：{report.target_file}")
            lines.append(f"检测结果：{item.vul_type}")
            pretty_code = ";\n".join(item.slice_code.split(";"))
            lines.append(f"可疑缺陷汇编代码块：\n{pretty_code}")
            if item.analysis:
                lines.append(f"分析结果：\n{item.analysis}")
            lines.append(_SEP_LINE)

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("文本报告已生成: %s", path)
        return path

    # ------------------------------------------------------------------
    def write_json_report(
        self, report: DetectionReport, filename: str | None = None
    ) -> Path:
        """写入 JSON 结构化报告。"""
        filename = filename or f"{Path(report.target_file).stem}_result.json"
        path = self.output_dir / filename

        payload: dict[str, Any] = {
            "target_file": report.target_file,
            "detect_time": report.detect_time,
            "total_slices": report.total_slices,
            "defect_count": len(report.confirmed_items()),
            "defects": [
                {
                    "index": it.index,
                    "vul_type": it.vul_type,
                    "probability": it.probability,
                    "verdict": it.verdict,
                    "analysis": it.analysis,
                    "slice_code": it.slice_code,
                }
                for it in report.confirmed_items()
            ],
            "secure_slice_count": len(report.secure_slices),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("JSON 报告已生成: %s", path)
        return path


def now_str() -> str:
    """当前时间字符串（报告时间戳）。"""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
