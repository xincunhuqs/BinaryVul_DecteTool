"""DeepSeek 大模型降噪客户端（论文第六步可选环节：对模型结果研判降噪）。

设计原则:
    - API Key 只从环境变量读取（见 config.llm.api_key_env），禁止硬编码；
    - 外部依赖（openai SDK）延迟导入，未安装时调用方应捕获 :class:`LlmError`。
"""
from __future__ import annotations


from bvsc.exceptions import LlmError
from bvsc.logging_setup import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = "你是一个资深的二进制文件逆向分析漏洞检测安全专家"

# 判定关键词（与论文实验口径一致）
VERDICT_CONFIRMED = "准确且可利用"
VERDICT_NOT_EXPLOITABLE = "准确但不可利用"
VERDICT_FALSE_POSITIVE = "不准确"


class DeepSeekClient:
    """DeepSeek chat 客户端（OpenAI 兼容协议）。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout: int = 120,
    ) -> None:
        """Args:
            api_key: API Key（从环境变量获取后传入）。
            base_url: 服务地址。
            model: 模型名。
            timeout: 请求超时（秒）。
        """
        if not api_key:
            raise LlmError(
                "未配置 DeepSeek API Key，请设置环境变量 DEEPSEEK_API_KEY "
                "或关闭 accurate_scan 模式"
            )
        self._base_url = base_url
        self._model = model
        self._timeout = timeout
        self._client = self._build_client(api_key)

    def _build_client(self, api_key: str):
        """延迟导入 openai SDK 构建客户端。修复(B10): 原实现 @staticmethod 误带 self 形参导致实例化必崩。"""
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LlmError(
                "缺少依赖 openai，请先执行 pip install -r requirements.txt"
            ) from exc
        return OpenAI(api_key=api_key, base_url=self._base_url)

    def analyze(self, defective_code: str, vul_type: str) -> str:
        """让大模型研判汇编代码中漏洞的真实性。

        Args:
            defective_code: 缺陷汇编切片。
            vul_type: 模型预测的漏洞类型（如 CWE416_Use_After_Free）。

        Returns:
            大模型分析文本（通常以 准确且可利用/准确但不可利用/不准确 开头）。

        Raises:
            LlmError: 调用失败或响应异常。
        """
        prompt = (
            f"请作为二进制逆向分析专家，分析下面这段汇编代码中是否存在 {vul_type}"
            f"漏洞，并说明判断依据。\n"
            f"汇编代码：\n{defective_code}\n\n"
            f"回答格式（仅选一个结论开头）：\n"
            f"{VERDICT_CONFIRMED}（确认存在该漏洞）/ "
            f"{VERDICT_NOT_EXPLOITABLE}（代码相关但不可利用）/ "
            f"{VERDICT_FALSE_POSITIVE}（不存在，属误报）\n"
            f"原因如下：..."
        )
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                timeout=self._timeout,
            )
            content = response.choices[0].message.content or ""
        except Exception as exc:
            raise LlmError(f"DeepSeek 调用失败: {exc}") from exc

        lines = [ln.replace("###", "") for ln in content.split("\n") if ln.strip()]
        return "\n".join(lines)

    def verdict(self, analysis: str) -> str | None:
        """从分析文本中提取研判结论。

        Returns:
            VERDICT_CONFIRMED / VERDICT_NOT_EXPLOITABLE / VERDICT_FALSE_POSITIVE，
            无法识别时返回 None。
        """
        for keyword in (VERDICT_CONFIRMED, VERDICT_NOT_EXPLOITABLE, VERDICT_FALSE_POSITIVE):
            if keyword in analysis:
                return keyword
        return None
