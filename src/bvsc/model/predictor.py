"""模型加载与缺陷预测（对应论文第六步：切片送入本地 Transformer 识别）。

预测方式（与论文/原实现一致）:
    以切片代码为 Encoder 输入，从起始符 ``S`` 开始贪心解码，
    取解码序列的首个 token 反查词表，得到漏洞类型（如 ``CWE416_Use_After_Free``）。
"""
from __future__ import annotations

from pathlib import Path

import torch

from bvsc.exceptions import ModelError
from bvsc.logging_setup import get_logger
from bvsc.model.tokenizer import load_tokenizer, word2index
from bvsc.model.transformer import Transformer, TransformerConfig, build_transformer

logger = get_logger(__name__)

_START_TOKEN = "S"
_END_TOKEN = "E"
_CWE_PREFIX = "CWE"


@torch.no_grad()
def greedy_decoder(
    model: Transformer,
    enc_input: torch.Tensor,
    start_symbol: int,
    eos_id: int,
    max_decode_len: int = 10,
) -> torch.Tensor:
    """贪心解码：逐 token 生成解码序列。

    Args:
        model: 已加载权重的 Transformer 模型。
        enc_input: [1, src_len] 编码输入。
        start_symbol: 起始符索引（词表中的 S）。
        eos_id: 结束符索引（词表中的 E）。
        max_decode_len: 最大解码长度（防死循环）。

    Returns:
        [1, tgt_len] 解码输入序列。
    """
    enc_outputs, _ = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0, dtype=enc_input.dtype, device=enc_input.device)
    next_symbol = start_symbol
    for _ in range(max_decode_len):
        dec_input = torch.cat(
            [dec_input, torch.tensor([[next_symbol]], dtype=enc_input.dtype, device=enc_input.device)],
            dim=-1,
        )
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_symbol = prob.data[-1].item()
        if next_symbol == eos_id:
            break
    return dec_input


class VulnerabilityPredictor:
    """本地 Transformer 漏洞类型预测器。"""

    def __init__(
        self,
        checkpoint: str | Path,
        tokenizer_path: str | Path,
        device: str = "cpu",
        max_seq_len: int = 700,
        model_config: TransformerConfig | None = None,
    ) -> None:
        """加载模型权重与词表。

        Raises:
            ModelError: 权重/词表缺失或加载失败。
        """
        checkpoint = Path(checkpoint)
        tokenizer_path = Path(tokenizer_path)
        if not checkpoint.exists():
            raise ModelError(f"模型权重不存在: {checkpoint}")
        if not tokenizer_path.exists():
            raise ModelError(f"词表文件不存在: {tokenizer_path}")

        self.device = device
        self.tokenizer_dict = load_tokenizer(tokenizer_path)
        self.max_seq_len = max_seq_len

        cfg = model_config or TransformerConfig(vocab_size=len(self.tokenizer_dict))
        cfg.vocab_size = len(self.tokenizer_dict)  # 以实际词表大小为准
        cfg.max_len = max(cfg.max_len, max_seq_len)

        self.model = build_transformer(cfg, device)
        try:
            # 修复(B1): weights_only 为 torch>=2.0 新增参数，低版本会 TypeError；
            # 先尝试安全模式，失败回退到传统加载，保证 torch 1.8~2.x 均可用。
            try:
                state_dict = torch.load(checkpoint, map_location=device, weights_only=False)
            except TypeError:  # torch < 2.0 不支持 weights_only 关键字
                state_dict = torch.load(checkpoint, map_location=device)
            self.model.load_state_dict(state_dict)
        except Exception as exc:
            raise ModelError(f"模型权重加载失败: {checkpoint}: {exc}") from exc
        self.model.eval()
        logger.info("模型加载完成: %s (device=%s)", checkpoint, device)

        self._start_id = self.tokenizer_dict.get(_START_TOKEN)
        self._eos_id = self.tokenizer_dict.get(_END_TOKEN)
        if self._start_id is None or self._eos_id is None:
            raise ModelError("词表中缺少起始符 S / 结束符 E")

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def predict(self, slice_text: str) -> tuple[str, str]:
        """预测单个汇编切片的漏洞类型。

        Returns:
            (切片原文, 漏洞类型)；漏洞类型以 ``CWE`` 开头，否则为未知。
        """
        token_ids = word2index(slice_text, self.tokenizer_dict, self.max_seq_len)
        enc_input = torch.LongTensor(token_ids).unsqueeze(0).to(self.device)

        dec_input = greedy_decoder(
            self.model, enc_input, self._start_id, self._eos_id
        )
        predict, _, _, _ = self.model(enc_input, dec_input)
        predict = predict.data.max(1, keepdim=True)[1]

        # 修复: 解码序列仅 1 个 token 时 squeeze() 会得到 0-dim 张量，索引 [0] 报错；
        # 统一用 flatten 后取首元素。
        first_idx = predict.flatten()[0].item()
        keys = list(self.tokenizer_dict.keys())
        if first_idx >= len(keys):
            vul_type = ""
        else:
            vul_type = keys[first_idx]
        return slice_text, vul_type

    def is_vulnerable(self, slice_text: str) -> bool:
        """判断切片是否被判为缺陷（漏洞类型含 CWE 前缀）。"""
        _, vul_type = self.predict(slice_text)
        return _CWE_PREFIX in vul_type
