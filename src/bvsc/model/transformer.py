"""Transformer 模型定义（对应论文第五步：Encoder-Decoder 结构）。

结构说明（与论文 Fig.4 一致）:
    - Encoder: 词嵌入 -> 三角函数位置编码 -> 多头自注意力 -> 残差+层归一化
               -> 前馈网络 -> 残差+层归一化；
    - Decoder: 词嵌入+位置编码 -> 自注意力(mask) -> 与 Encoder 输出的交叉注意力
               -> 前馈网络 -> 线性映射 + Softmax 分类。

与原实现(transformer_v3.py)相比的规范化改动:
    - 超参由 :class:`TransformerConfig` 构造注入，不再依赖模块级全局变量；
    - 移除 forward 中的 ``.cuda()`` 硬编码，设备在构建时统一设置；
    - 移除未使用的 sympy 依赖。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TransformerConfig:
    """Transformer 超参数。"""

    d_model: int = 512          # Embedding Size
    d_ff: int = 2048            # FeedForward dimension
    d_k: int = 64               # dimension of K(=Q)
    d_v: int = 64               # dimension of V
    n_layers: int = 6           # Encoder/Decoder 层数
    n_heads: int = 8            # 多头注意力头数
    vocab_size: int = 1000      # 词表大小
    max_len: int = 700          # 位置编码最大长度
    dropout: float = 0.1
    pad_id: int = 0


class PositionalEncoding(nn.Module):
    """三角函数位置编码。"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 700) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [seq_len, batch_size, d_model] -> 叠加位置编码并 dropout。"""
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(
    seq_q: torch.Tensor, seq_k: torch.Tensor
) -> torch.Tensor:
    """padding 掩码：seq_k 中值为 pad_id 的位置置 True（被 mask）。

    Args:
        seq_q: [batch_size, len_q]
        seq_k: [batch_size, len_k]

    Returns:
        [batch_size, len_q, len_k] 布尔掩码。
    """
    batch_size, len_q = seq_q.size()
    _, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [B, 1, len_k]
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequence_mask(seq: torch.Tensor) -> torch.Tensor:
    """上三角掩码（Decoder 自注意力屏蔽未来信息）。

    Args:
        seq: [batch_size, tgt_len]

    Returns:
        [batch_size, tgt_len, tgt_len] 布尔掩码。
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    return torch.from_numpy(subsequence_mask).byte().bool()


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力。"""

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Q/K/V: [B, n_heads, len, d_k/d_v]；attn_mask: [B, n_heads, seq, seq]"""
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    """多头自/交叉注意力，含残差连接与层归一化。"""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Linear(cfg.d_model, cfg.d_k * cfg.n_heads, bias=False)
        self.W_K = nn.Linear(cfg.d_model, cfg.d_k * cfg.n_heads, bias=False)
        self.W_V = nn.Linear(cfg.d_model, cfg.d_v * cfg.n_heads, bias=False)
        self.fc = nn.Linear(cfg.n_heads * cfg.d_v, cfg.d_model, bias=False)
        # 修复(B2): 原实现(transformer_v3.py)与发布权重均无 layer_norm 参数；
        # 重构时误新增 LayerNorm 导致官方权重无法加载（缺 60 个键），此处移除对齐。
        self.layer_norm = None  # 占位，保持接口一致（不再参与计算）

    def forward(
        self,
        input_Q: torch.Tensor,
        input_K: torch.Tensor,
        input_V: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """输入: [B, len, d_model]；返回 (残差输出, 注意力矩阵)。"""
        residual, batch_size = input_Q, input_Q.size(0)
        n_heads, d_k, d_v = self.cfg.n_heads, self.cfg.d_k, self.cfg.d_v

        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        return output + residual, attn


class PoswiseFeedForwardNet(nn.Module):
    """前馈网络（两层线性 + ReLU），含残差连接与层归一化。"""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(cfg.d_ff, cfg.d_model, bias=False),
        )
        # 修复(B2): 与发布权重架构对齐，移除 layer_norm
        self.layer_norm = None  # 占位，保持接口一致（不再参与计算）

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs) + inputs


class EncoderLayer(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.enc_self_attn = MultiHeadAttention(cfg)
        self.pos_ffn = PoswiseFeedForwardNet(cfg)

    def forward(
        self, enc_inputs: torch.Tensor, enc_self_attn_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_outputs, attn = self.enc_self_attn(
            enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask
        )
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    """Transformer Encoder。"""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.src_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = PositionalEncoding(cfg.d_model, cfg.dropout, cfg.max_len)
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_layers)])

    def forward(
        self, enc_inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """enc_inputs: [B, src_len] -> (enc_outputs, attns列表)"""
        enc_outputs = self.pos_emb(self.src_emb(enc_inputs))
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns: List[torch.Tensor] = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.dec_self_attn = MultiHeadAttention(cfg)
        self.dec_enc_attn = MultiHeadAttention(cfg)
        self.pos_ffn = PoswiseFeedForwardNet(cfg)

    def forward(
        self,
        dec_inputs: torch.Tensor,
        enc_outputs: torch.Tensor,
        dec_self_attn_mask: torch.Tensor,
        dec_enc_attn_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dec_outputs, dec_self_attn = self.dec_self_attn(
            dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask
        )
        dec_outputs, dec_enc_attn = self.dec_enc_attn(
            dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask
        )
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    """Transformer Decoder。"""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tgt_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = PositionalEncoding(cfg.d_model, cfg.dropout, cfg.max_len)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_layers)])

    def forward(
        self,
        dec_inputs: torch.Tensor,
        enc_inputs: torch.Tensor,
        enc_outputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """dec_inputs: [B, tgt_len] -> (dec_outputs, 自注意力列表, 交叉注意力列表)"""
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1)

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)
        dec_self_attn_mask = torch.gt(
            dec_self_attn_pad_mask + dec_self_attn_subsequence_mask, 0
        )
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns: List[torch.Tensor] = []
        dec_enc_attns: List[torch.Tensor] = []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(
                dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask
            )
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    """Encoder-Decoder Transformer（序列到序列，用于缺陷切片分类）。"""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.projection = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(
        self, enc_inputs: torch.Tensor, dec_inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """返回 (dec_logits.view(-1, vocab), enc_attns, dec_self_attns, dec_enc_attns)"""
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(
            dec_inputs, enc_inputs, enc_outputs
        )
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


def build_transformer(cfg: TransformerConfig, device: str = "cpu") -> Transformer:
    """构建模型并迁移到指定设备。"""
    model = Transformer(cfg)
    model.to(device)
    return model


def config_from_settings(settings, vocab_size: Optional[int] = None) -> TransformerConfig:
    """从 :class:`bvsc.config.Settings` 构建模型配置。

    Args:
        settings: 应用配置对象。
        vocab_size: 词表大小；为 None 时取配置值（默认 1000）。
    """
    get = lambda key, default: settings.get("model", key, default)  # noqa: E731
    return TransformerConfig(
        d_model=int(get("d_model", 512)),
        d_ff=int(get("d_ff", 2048)),
        d_k=int(get("d_k", 64)),
        d_v=int(get("d_v", 64)),
        n_layers=int(get("n_layers", 6)),
        n_heads=int(get("n_heads", 8)),
        vocab_size=int(vocab_size or get("vocab_size", 1000)),
        max_len=int(get("max_seq_len", 700)),
        dropout=float(get("dropout", 0.1)),
        pad_id=int(get("pad_id", 0)),
    )
