"""缺陷样本扩展（论文第四步：模板随机实例化）。

基于同一缺陷模板，随机生成变量名与数字常量并替换特殊字段，
批量构造同类型缺陷样本，解决 Juliet 数据集缺陷样本数量不足的问题
（论文 Algorithm 1：Defect Sample Expansion Algorithm Based on Defect Templates）。
"""
from __future__ import annotations

import os
import random
import re
import string
from pathlib import Path

from bvsc._compat import tqdm

from bvsc.logging_setup import get_logger

logger = get_logger(__name__)

_VAR_PLACEHOLDER = re.compile(r"!+\w+!+")
_DIGIT_PLACEHOLDER = re.compile(r"!!digit_\d+!")


class VulnerabilityFileExtend:
    """缺陷模板实例化器。"""

    def __init__(self, expand_number_samples: int = 20, seed: int | None = None) -> None:
        """Args:
            expand_number_samples: 每个模板生成的样本数。
            seed: 随机种子（None 则每次随机，与论文实验一致）。
        """
        self.expand_number_samples = expand_number_samples
        if seed is not None:
            random.seed(seed)

    # ------------------------------------------------------------------
    # 随机标识符
    # ------------------------------------------------------------------
    @staticmethod
    def generate_character_constants() -> str:
        """随机生成合法标识符（首字母 + 5~8 位字母数字）。"""
        length = random.randint(5, 8)
        characters = string.ascii_letters + string.digits
        start = random.choice(string.ascii_letters)
        return start + "".join(random.sample(characters, length))

    @staticmethod
    def generate_digital_constants() -> int:
        """随机生成数字常量（8~500）。"""
        return random.randint(8, 500)

    # ------------------------------------------------------------------
    # 单模板实例化
    # ------------------------------------------------------------------
    def defect_sample_instantiation(
        self,
        file_path: str | Path,
        folder: str,
        file_number: int,
        save_folder: str | Path,
    ) -> None:
        """按模板生成 ``file_number`` 个缺陷样本。

        Args:
            file_path: 缺陷模板文件路径。
            folder: 缺陷类型（子目录名）。
            file_number: 生成样本数量。
            save_folder: 样本保存根目录。
        """
        code = Path(file_path).read_text(encoding="utf-8")

        # 收集模板中的占位符（变量与数字）
        variables = sorted(set(_VAR_PLACEHOLDER.findall(code)))
        digit_variables = sorted(set(_DIGIT_PLACEHOLDER.findall(code)))
        variables = [v for v in variables if v not in digit_variables]

        save_dir = Path(save_folder) / folder
        save_dir.mkdir(parents=True, exist_ok=True)

        stem = os.path.basename(file_path).split(".")[0]
        suffix = os.path.basename(file_path).split(".")[1]

        for index in range(1, file_number + 1):
            var_map = {
                v: ("main" if v.strip("!") == "main" else self.generate_character_constants())
                for v in variables
            }
            digit_map = {
                d: str(self.generate_digital_constants()) for d in digit_variables
            }

            out_lines: list[str] = []
            for codeline in code.split("\n"):
                for variable, value in var_map.items():
                    codeline = re.sub(re.escape(variable), value, codeline)
                for digit, value in digit_map.items():
                    codeline = re.sub(re.escape(digit), value, codeline)
                out_lines.append(codeline)

            save_path = save_dir / f"{stem}_instantiation{index}.{suffix}"
            save_path.write_text("\n".join(out_lines), encoding="utf-8")

    # ------------------------------------------------------------------
    # 批量实例化
    # ------------------------------------------------------------------
    def gather_instantiation_file(
        self, folder_path: str | Path, save_folder: str | Path
    ) -> None:
        """批量按缺陷模板扩展缺陷样本。

        Args:
            folder_path: 缺陷模板目录（含 CWE 子目录）。
            save_folder: 扩展样本保存目录。
        """
        folder_path = Path(folder_path)
        for folder in tqdm(sorted(os.listdir(folder_path)), desc="扩展缺陷样本中..."):
            vul_folder = folder_path / folder
            if not vul_folder.is_dir():
                continue
            for filename in sorted(vul_folder.iterdir()):
                if filename.suffix.lower() in (".c", ".cpp") and "good" not in filename.name:
                    self.defect_sample_instantiation(
                        filename, folder, self.expand_number_samples, save_folder
                    )
        logger.info(
            "缺陷样本扩展完成：每个模板扩展 %d 个 -> %s",
            self.expand_number_samples, save_folder,
        )
