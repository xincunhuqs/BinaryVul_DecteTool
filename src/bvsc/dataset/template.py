"""缺陷模板生成（论文第三步：变量替换为特殊字段）。

将内联汇编缺陷样例中的变量名与数字常量替换为规范特殊字段
（``!!variable_i!`` / ``!!digit_i!``），消除不同样例间变量命名与
取值差异，构造缺陷模板，为后续随机实例化扩展样本做准备。
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Tuple

from bvsc._compat import tqdm

from bvsc.logging_setup import get_logger

logger = get_logger(__name__)

# 变量声明捕获：char/int/float/... 后的标识符
_VAR_PATTERN = re.compile(
    r"\b(?:char|int|float|double|long|short|unsigned|signed|void|struct|union)"
    r"\s*\**\s*(\w+)\s*[\[;=]"
)
_DIGIT_PATTERN = re.compile(r"\d+")
_COMMENT_STRIP = re.compile(r"(?s)/\*.*?\*/|//.*?")
_PREPROCESSOR_STRIP = re.compile(r"^\s*#.*?$", re.MULTILINE)


def _replace_tokens(text: str, mapping: dict[str, str]) -> str:
    """按映射做词边界替换；键按长度降序处理，长名优先避免短名先被替换。

    替换分两步：先写入临时占位符（保证不互相干扰），再统一还原。
    """
    if not mapping:
        return text
    # 1) 长名优先，全部替换为临时占位符
    temp_text = text
    for i, (token, _) in enumerate(sorted(mapping.items(), key=lambda kv: -len(kv[0]))):
        temp_text = re.sub(
            r"\b" + re.escape(token) + r"\b",
            f"__BVSC_TMP_{i}__",
            temp_text,
        )
    # 2) 统一还原为目标值
    for i, (_, value) in enumerate(sorted(mapping.items(), key=lambda kv: -len(kv[0]))):
        temp_text = temp_text.replace(f"__BVSC_TMP_{i}__", value)
    return temp_text


def extract_template_variables(code: str) -> Tuple[list[str], list[str]]:
    """从源代码中提取变量名与数字常量（去重、过滤过短变量）。

    Returns:
        (variables, digits)
    """
    code_no_comments = _COMMENT_STRIP.sub("", code)
    variables = [
        v for v in _VAR_PATTERN.findall(code_no_comments) if len(v) > 2
    ]

    code_no_pp = _PREPROCESSOR_STRIP.sub("", code_no_comments)
    digits = [d for d in _DIGIT_PATTERN.findall(code_no_pp) if d != "0"]

    return list(set(variables)), list(set(digits))


class VariableSubstitution:
    """缺陷样例 -> 缺陷模板。"""

    def extract_variables(
        self,
        file_path: str | Path,
        vul_type: str,
        template_index: int,
        save_folder: str | Path,
    ) -> None:
        """将单个缺陷样例转换为缺陷模板文件。

        Args:
            file_path: 缺陷样例（已含内联汇编）。
            vul_type: 缺陷类型（如 CWE416_Use_After_Free）。
            template_index: 模板编号（写入文件名）。
            save_folder: 模板保存目录（其下按 vul_type 分子目录）。
        """
        code = Path(file_path).read_text(encoding="utf-8")
        variables, digits = extract_template_variables(code)

        var_placeholders = {v: f"!!variable_{i}!" for i, v in enumerate(variables)}
        digit_placeholders = {d: f"!!digit_{i}!" for i, d in enumerate(digits)}

        stem = os.path.basename(file_path).split(".")[0]
        suffix = os.path.basename(file_path).split(".")[1]
        file_name = f"{stem}_template{template_index}.{suffix}"

        save_dir = Path(save_folder) / vul_type
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / file_name

        out_lines: list[str] = []
        for codeline in code.split("\n"):
            # 修复(B8): 原实现用子串匹配判断跳过行（"#"/"return"/"\\" 命中即整行不替换），
            # 导致字符串转义('\\0')、宏定义(#define sizeof(data))、含 return 的行漏替换。
            # 改为精确判定：仅跳过内联汇编续行（行尾反斜杠）。
            if codeline.rstrip().endswith("\\"):
                out_lines.append(codeline)
                continue
            # 预处理指令行：保留指令关键字本身，但宏体中的变量/常量仍需替换
            stripped = codeline.lstrip()
            if stripped.startswith("#"):
                directive_end = len(stripped) - len(stripped.lstrip("#"))
                head, tail = codeline[: directive_end], codeline[directive_end:]
                # 宏体替换（如 #define CHAR_ARRAY_SIZE (3 * sizeof(data) + 2) 中的 data）
                if "define" in stripped:
                    tail = _replace_tokens(tail, digit_placeholders)
                    tail = _replace_tokens(tail, var_placeholders)
                out_lines.append(head + tail)
                continue
            # 数字常量（词边界）与变量（词边界 + 长名优先）分别替换
            codeline = _replace_tokens(codeline, digit_placeholders)
            codeline = _replace_tokens(codeline, var_placeholders)
            out_lines.append(codeline)

        save_path.write_text("\n".join(out_lines), encoding="utf-8")

    def gather_tempalte_variable(
        self, folder_path: str | Path, save_folder: str | Path
    ) -> None:
        """批量将缺陷样例文件夹转换为缺陷模板文件夹。

        Args:
            folder_path: 缺陷样例目录（含 CWE 子目录）。
            save_folder: 模板保存目录。
        """
        folder_path = Path(folder_path)
        for vultype in tqdm(sorted(os.listdir(folder_path)), desc="生成缺陷模板中..."):
            vul_folder = folder_path / vultype
            if not vul_folder.is_dir():
                continue
            index = 0
            for file in sorted(vul_folder.iterdir()):
                if file.suffix.lower() in (".c", ".cpp") and "good" not in file.name:
                    index += 1
                    self.extract_variables(file, vultype, index, save_folder)
        logger.info("缺陷模板生成完成：%s", save_folder)
