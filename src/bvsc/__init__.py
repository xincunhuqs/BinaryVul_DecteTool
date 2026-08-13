"""BVSC —— 基于深度学习结合内联汇编比较的二进制漏洞检测系统。

对应论文: Binary vulnerability detection based on deep learning combined
          with inline assembly comparison (Huqinsong, Chenxiaoquan)

包结构:
    bvsc.cli            命令行入口
    bvsc.detector       检测流程编排
    bvsc.disassembler   PE 反汇编
    bvsc.slicer         汇编切片
    bvsc.reporter       检测报告输出
    bvsc.llm_client     DeepSeek 大模型降噪客户端
    bvsc.model          深度学习模型（Transformer/分词/训练/预测）
    bvsc.dataset        缺陷数据集构建管线（Juliet 预处理→切片提取）
"""

__version__ = "1.0.0"
