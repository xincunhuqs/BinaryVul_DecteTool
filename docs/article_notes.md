# 📖 论文思路与代码模块映射

> 本文档说明《Binary vulnerability detection based on deep learning combined with inline assembly comparison》
> 的核心思路与本工程各代码模块的对应关系，便于快速定位论文中各环节的具体实现。

**快速导航**：[论文六步管线与模块映射](#-论文六步管线与模块映射) · [关键算法实现位置](#-关键算法实现位置) · [核心模块职责](#-核心模块职责) · [模型文件](#-模型文件)

---

## 🧭 论文六步管线与模块映射

| 步骤 | 论文内容 | 对应模块 |
|---|---|---|
| ① | Juliet 数据集精简预处理（去除修复代码、优化 main、去除双环境宏） | `src/bvsc/dataset/juliet.py` |
| ② | 缺陷源码插入内联汇编标识（nop/jmp 掩码，完成缺陷代码标注） | `src/bvsc/dataset/inline_assembly.py` |
| ③ | 变量/常量替换为特殊字段，构造缺陷模板 | `src/bvsc/dataset/template.py` |
| ④ | 模板随机实例化，扩展缺陷样本 | `src/bvsc/dataset/augment.py` |
| ⑤ | 编译链接 + 反汇编差异比较提取缺陷切片 | `src/bvsc/dataset/compiler.py` + `src/bvsc/dataset/slicing_code.py` |
| 训练 | 训练数据生成 / 本地分词器 / Transformer 模型 | `src/bvsc/model/dataset.py` + `tokenizer.py` + `transformer.py` + `trainer.py` |
| ⑥ | 检测应用：反汇编 → 切片 → 模型识别 →（可选）大模型研判 | `src/bvsc/disassembler.py` + `slicer.py` + `detector.py` + `llm_client.py` + `cli.py` |

---

## 🎯 关键算法实现位置

### 缺陷模板扩展算法

| 环节 | 实现 |
|---|---|
| 模板构造：变量 → `!!variable_i!`、常量 → `!!digit_i!` | `dataset/template.py` |
| 随机实例化：每个模板扩展 N 个样本（默认 20） | `dataset/augment.py` |

### 内联汇编差异比较算法

| 环节 | 实现 |
|---|---|
| 内联汇编标识：`nop; nop; jmp defact_flage; defact_flage: nop` | `dataset/inline_assembly.py` |
| 反汇编掩码定位（nop→(nop)→jmp→nop 状态机，兼容 MSVC 合并连续 nop） | `dataset/slicing_code.py::find_mask_begin_indices` |
| 切片提取区间 `[mask_begin+4, mask_end]` | `dataset/slicing_code.py::gather_defect_slicing` |

### Transformer 模型

- **Encoder**：词嵌入 → 三角位置编码 → 多头自注意力 → 残差 + LayerNorm → FFN → LayerNorm
- **Decoder**：自注意力（mask）→ 交叉注意力 → FFN → 线性映射 + Softmax
- **超参**：d_model=512, d_ff=2048, d_k=d_v=64, layers=6, heads=8（`config/config.yaml`）

---

## 🧩 核心模块职责

| 模块 | 职责 |
|---|---|
| `cli.py` | 命令行入口（subfinder 风格：`-silent` / `-json` / `-o` / `-v`） |
| `detector.py` | 检测编排：反汇编 → 切片 → 预测 → 大模型研判 → 报告 |
| `disassembler.py` | PE `.text` 节反汇编（capstone + pefile） |
| `slicer.py` | 定长汇编切片 |
| `reporter.py` | 文本 / JSON 检测报告 |
| `llm_client.py` | DeepSeek 大模型研判客户端（API Key 环境变量注入） |
| `model/` | transformer（模型定义）/ tokenizer（分词）/ dataset（训练数据）/ predictor（预测）/ trainer（训练） |
| `dataset/` | juliet（预处理）/ inline_assembly（内联汇编）/ template（模板）/ augment（扩样）/ compiler（编译）/ slicing_code（切片）/ pipeline（管线编排） |
| `scripts/` | build_dataset.py（构建数据集）/ train_model.py（训练）/ dateset_overview.py（数据集统计） |

---

## 📦 模型文件

| 文件 | 说明 |
|---|---|
| `models/transformer.pth` | Transformer 模型权重 |
| `models/tokenize_dict.txt` | 本地汇编词表 |
