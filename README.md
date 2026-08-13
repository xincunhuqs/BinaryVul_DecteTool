<p align="center">
  <h1 align="center">BVSC</h1>
  <p align="center">
    基于深度学习结合内联汇编比较的二进制漏洞检测工具
    <br/>
    论文《Binary vulnerability detection based on deep learning combined with inline assembly comparison》工程实践
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.8+-blue" alt="Python 3.8+"/>
    <img src="https://img.shields.io/badge/Platform-Windows%20%2F%20Linux-lightgrey" alt="Platform"/>
    <img src="https://img.shields.io/badge/Format-PE%20(x86%2F32)-orange" alt="PE x86/32"/>
    <img src="https://img.shields.io/badge/Model-Transformer-green" alt="Transformer"/>
    <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License MIT"/>
  </p>
</p>

---

> **BVSC**（Binary Vulnerability detection System Combined）是论文 *Binary vulnerability detection based on deep learning combined with inline assembly comparison* 的完整实现：以 **Juliet 缺陷数据集** 为原始数据源，通过 **内联汇编差异比较** 与 **缺陷模板扩展** 两项技术提取缺陷汇编切片，使用 **Transformer** 深度学习模型训练识别，最终实现 PE 二进制文件的漏洞检测与 **代码级缺陷定位**。

**快速导航**：[论文方法](#-论文方法) · [特性](#-特性) · [安装](#-安装) · [使用](#-使用) · [配置](#-配置) · [构造自己的缺陷数据集](#-构造自己的缺陷数据集) · [目录结构](#-目录结构) · [测试](#-测试) · [文档](#-文档) · [已知限制](#-已知限制)

---

## 📖 论文方法

### 背景与痛点

二进制漏洞检测相比源码检测的难点在于**可读性差、语义割裂**。主流缺陷定位方法（AST/CFG/DFG 切片、Pin 插桩 + add2line、敏感 API 调用切片等）实现较为复杂，且多适用于特定场景；同时深度学习检测用于二进制漏洞检测时，主要面临**缺陷定位粒度粗、缺陷样本少**两大问题。

### 两项关键技术

| 技术 | 解决的问题 | 做法 |
|---|---|---|
| **① 内联汇编差异比较算法** | 缺陷定位粒度粗（无法定位到代码级） | 大多数缺陷数据集（如 Juliet）都会在源码处用注释标注出缺陷产生代码（`POTENTIAL FLAW`），因此只需在缺陷源码处插入对应的内联汇编标识（`nop; nop; jmp defact_flage; nop`）即可完成缺陷代码的标注；内联汇编编译后保持不变，编译为可执行文件后反汇编，按标识即可在汇编层面定位缺陷语句，提取包含数据流的完整缺陷切片，实现「源码层 ↔ 汇编层」的联动 |
| **② 缺陷模板扩展算法** | 缺陷样本少（Juliet 每类仅数百个） | 将缺陷样本中的变量/常量替换为特殊字段（`!!variable_i!` / `!!digit_i!`）构成**缺陷模板**，随机实例化批量扩样（每模板 20 个），让模型充分学习同类型缺陷特征 |

### 六步检测流程

```
① 插入内联汇编  →  ② 变量替换成模板  →  ③ 模板实例化扩样
      ↓
④ 编译 + 反汇编 + 差异比较提取缺陷切片
      ↓
⑤ Transformer（Encoder-Decoder）训练学习缺陷特征
      ↓
⑥ 待检二进制 → 反汇编 + 切片 → 本地模型识别缺陷 →（可选）大模型研判
```

### 工具定位

论文提供了一种二进制漏洞检测思路——以内联汇编差异比较与缺陷模板扩展为核心的代码级缺陷定位方案。本工具是对该思路的一次工程实践：尝试将**内联汇编差异比较**与**缺陷模板扩展**两项技术落地为可复用的缺陷数据集构建管线（`scripts/build_dataset.py`），集成 Transformer 深度学习模型的训练（`src/bvsc/model/`）与检测流程，通过命令行工具（`BVSC.py`）完成「反汇编 → 切片 → 模型识别 →（可选）大模型研判」的检测链路，对论文所述的代码级缺陷定位进行探索。

> 📄 论文思路与代码模块的完整映射见 [`docs/article_notes.md`](docs/article_notes.md)

---

## ✨ 特性

- 🔍 **缺陷检测与定位**：基于 Transformer 模型识别多种 CWE 缺陷类型，通过内联汇编标识联动源码与汇编，输出精确到切片
- 🧠 **Transformer 深度模型**：Encoder-Decoder 结构，位置编码 + 多头注意力 + 残差归一化
- 🤖 **大模型研判**（可选）：DeepSeek 对候选缺陷进行验证分析
- 📦 **缺陷数据集构建管线**：Juliet 精简 → 内联汇编 → 模板 → 扩样 → 编译 → 切片，一键构造自己的缺陷汇编数据集
- 🛠 **subfinder 风格 CLI**：`-silent` / `-json` / `-o` / `-v`，结果走 stdout、日志走 stderr，退出码语义化
- 🔒 **安全默认**：API Key 环境变量注入、配置与代码分离、官方数据源只读

---

## 📦 安装

```bash
# 环境要求：Python >= 3.8（Windows / Linux）
#  注：代码使用 PEP 604 联合类型注解（str | None），已通过 from __future__ import annotations
#      兼容 Python 3.8+；运行时仅需 torch/capstone 等依赖可用。
pip install -r requirements.txt

# 可选：安装为 bvsc 命令（需已安装 torch 等依赖）
pip install -e .
```

> ⚠️ **Windows 下编译数据集**需安装 Visual Studio 2019+（MSVC），并在
> `config/config.yaml` 中配置 `dataset.vcvarsall_path`。

---

## 🚀 使用

### 快速开始

```bash
# 查看帮助（-h / --help 均可）
python BVSC.py -h

# 检测单个二进制文件
python BVSC.py -efp target.exe

# 批量检测文件夹内全部 .exe
python BVSC.py -efdp ./bin_folder
```

### 参数一览

**输入**

| 参数 | 说明 | 默认 |
|---|---|---|
| `-efp, --exefile-path` | 待检测二进制文件路径（PE 格式） | - |
| `-efdp, --exefile-folder-path` | 待检测二进制文件夹路径 | - |

**扫描模式**

| 参数 | 说明 | 默认 |
|---|---|---|
| `-nrsc, --normal-scan` | 普通扫描：仅本地 Transformer 模型 | `True` |
| `-acsc, --accurate-scan` | 精确扫描：DeepSeek 大模型研判 | `False` |
| `-rsd, --record-secure` | 记录安全反汇编切片（供再训练） | `False` |

**输出**

| 参数 | 说明 | 默认 |
|---|---|---|
| `-silent, --silent` | 静默模式：stdout 仅输出结果（日志走 stderr） | `False` |
| `-json, --json` | JSON 结构化输出 | `False` |
| `-o, --output` | 结果输出文件路径 | - |
| `-v, --verbose` | 详细模式（DEBUG 日志） | `False` |

**其他**

| 参数 | 说明 | 默认 |
|---|---|---|
| `-config, --config` | 配置文件路径 | `config/config.yaml` |
| `-version, --version` | 显示版本号 | - |

### 示例

```bash
# 普通扫描 + 详细日志
python BVSC.py -efp target.exe -v

# 精确扫描（DeepSeek 降噪，需配置 DEEPSEEK_API_KEY）
python BVSC.py -efp target.exe -acsc

# 批量检测 + 结果写入文件
python BVSC.py -efdp ./bin_folder -o result.txt

# 管道输出（仅命中行，供脚本消费）
python BVSC.py -efp target.exe -silent

# JSON 结构化输出
python BVSC.py -efp target.exe -json -silent > hits.json

# 自定义配置
python BVSC.py -config ./my_config.yaml -efp target.exe
```

### 退出码

| 码 | 含义 |
|---|---|
| `0` | 检测完成（无论是否发现漏洞） |
| `1` | 参数 / 配置 / 运行依赖错误 |
| `2` | 检测执行异常 |

---

## ⚙️ 配置

全部配置收敛于 [`config/config.yaml`](config/config.yaml)：模型路径与超参、切片大小、LLM、训练参数、数据集构建路径等。

敏感信息**只从环境变量读取**，禁止写入配置文件：

```bash
export DEEPSEEK_API_KEY=sk-xxxx
```

---

## 📘 完整使用指南

> 📖 图文完整版使用手册见 [`docs/usage_guide.html`](docs/usage_guide.html)

### 端到端工作流

```
① 构建数据集 → ② 训练模型 → ③ 检测二进制 →（可选）④ 大模型研判
```

### ① 构建缺陷汇编数据集

```bash
# 构建全部缺陷类型（或见上文「构建单一缺陷类型的数据集」只构建单类型）
python scripts/build_dataset.py

# 查看构建产物统计（缺陷类型/个数/成因种类）
python scripts/dateset_overview.py data/temp/windowsource_VUL_exefile
```

构建过程带**实时进度展示**：
- 阶段⑤编译的进度条按「源文件总数」展示（如 `3/620`），并逐文件输出日志：`[编译 3/620] ✓ xxx.c 成功(4.2s)`；
- 编译失败文件的完整 cl 命令与错误详情记录在 `data/temp/fild_compilation_cfile.txt`。

### ② 训练模型

```bash
# 基本训练（数据默认取 data/total_defect_slicing.txt，权重输出到 models/transformer.pth）
python scripts/train_model.py

# 指定数据 / 轮数 / 合并良性负样本
python scripts/train_model.py -data data/total_defect_slicing.txt -epochs 6 \
    -benign DefectDiscoveryTrainDate/xxx_secure.txt
```

训练过程带**进度条**（epoch 内 batch 进度 + 实时 loss / acc）与完成汇总（总准确率 / 平均 loss / 耗时 / 模型路径）。

训练超参（`config/config.yaml` → `training`）：`epochs` / `batch_size` / `learning_rate` / `momentum` / `train_ratio` / `seed`。

### ③ 检测二进制文件

```bash
# 单文件 / 批量
python BVSC.py -efp target.exe
python BVSC.py -efdp ./bin_folder

# 精确扫描（DeepSeek 研判，需 DEEPSEEK_API_KEY 环境变量）
python BVSC.py -efp target.exe -acsc

# 输出控制
python BVSC.py -efp target.exe -silent        # stdout 仅命中行（适合管道）
python BVSC.py -efp target.exe -json -silent  # JSON 结构化输出
python BVSC.py -efp target.exe -o result.txt  # 结果落盘（与 stdout 同时输出）
python BVSC.py -efp target.exe -v             # DEBUG 级日志
```

检测后自动在 `output/` 目录生成 `<文件名>_result.txt`（文本报告）与 `<文件名>_result.json`（结构化报告）。

### ④ 常见问题速查

| 问题 | 解决 |
|---|---|
| 良性程序误报多 | 用 `-rsd` 收集安全切片 + `-benign` 重新训练（见上文「训练时合并良性负样本」） |
| 数据集构建编译失败 | 确认 `config.yaml` 的 `vcvarsall_path` 指向本机 VS 2019+；构建仅在 Windows 支持 |
| 支持的文件格式 | x86/32 位 PE（架构在 `config.yaml` → `detection.arch/mode`） |
| 精确扫描无效果 | 设置环境变量 `DEEPSEEK_API_KEY`（`export DEEPSEEK_API_KEY=sk-xxx`） |
| 训练很慢 | CPU 上 10 万切片 × 6 epochs 需数天，建议 GPU 或先用 `only_cwe` 构建单类型小数据集验证 |

完整参数与配置说明见上文「参数一览」「⚙️ 配置」，更多细节见 [`docs/usage_guide.html`](docs/usage_guide.html)。

---

## 🧪 构造自己的缺陷数据集

项目内置完整的数据集构造管线（`scripts/build_dataset.py`），基于 Juliet 缺陷样例可一键产出缺陷汇编切片数据集：

```bash
# 一键执行：Juliet 精简 → 内联汇编插入 → 缺陷模板生成 → 样本扩展 →
#           MSVC 编译 → 反汇编差异比较提取缺陷切片
python scripts/build_dataset.py
```

构造流程与产物：

| 阶段 | 说明 | 产物目录 |
|---|---|---|
| ① Juliet 精简 | 提取缺陷/修复独立源文件，去除双环境宏 | `data/temp/pretreatment/` |
| ② 内联汇编插入 | 缺陷标识 nop/jmp 掩码 | `data/temp/windowsourcefile_inline_template/` |
| ③ 缺陷模板生成 | 变量/常量替换为特殊字段 | `data/temp/windowsource_file_template/` |
| ④ 样本扩展 | 模板随机实例化（每模板 N 个，默认 20，见 `config` 的 `expand_samples_per_template`） | `data/temp/windowsource_file_instantiation/` |
| ⑤ MSVC 编译 | 编译为 PE 可执行文件 | `data/temp/windowsource_VUL_exefile/` |
| ⑥ 切片提取 | 反汇编 + 差异比较提取缺陷切片（按缺陷类型生成 `<类型>.txt`） | `data/temp/windowsource_VUL_slicingcode/` → 汇总 `data/total_defect_slicing.txt` |

其他工具：

```bash
# 统计已构造的缺陷可执行数据集（缺陷类型/个数/成因种类）
python scripts/dateset_overview.py <缺陷可执行文件根目录>
```

> 📌 说明：
> - **上述产物目录均为构建过程中自动生成**的中间产物，构建前并不存在（已加入 `.gitignore` 忽略）；
> - 各阶段参数（扩样数量、编译环境、路径等）均在 `config/config.yaml` 的 `dataset` 节配置；
> - 构造产物全部写入 `data/temp/`，**Juliet 官方数据源保持只读**；
> - 编译环节需在 Windows + Visual Studio（MSVC）环境下执行。

### 构建单一缺陷类型的数据集

如需仅针对某一种 Juliet 缺陷构建数据集（例如 CWE121 栈缓冲区溢出），在 `config/config.yaml` 中设置：

```yaml
dataset:
  only_cwe: CWE121_Stack_Based_Buffer_Overflow   # 置空(null)则构建全部缺陷类型
```

然后照常执行 `python scripts/build_dataset.py` 即可，六步管线只处理该缺陷类型，大幅缩短构建时间。

### 训练时合并良性负样本（降低误报）

模型训练数据仅含缺陷切片时，模型对良性代码也会输出某一 CWE 类型（高误报）。
推荐先用 `-rsd` 对一批正常程序做检测收集安全切片，再训练时合并为负样本：

```bash
# 1. 收集安全切片（输出到 DefectDiscoveryTrainDate/，每行带 NO_VULN 标签）
python BVSC.py -efdp ./normal_bin_folder -rsd

# 2. 训练时合并良性负样本
python scripts/train_model.py -data data/total_defect_slicing.txt -benign DefectDiscoveryTrainDate/xxx_secure.txt
```

---

## 📁 目录结构

```
bvsc/
├── BVSC.py                  # CLI 入口（python BVSC.py -efp ...）
├── config/config.yaml       # 统一配置
├── models/                  # 模型权重 transformer.pth + 词表 tokenize_dict.txt
├── src/bvsc/                # 源码包
│   ├── cli.py               # subfinder 风格 CLI
│   ├── detector.py          # 检测编排：反汇编→切片→预测→LLM 降噪→报告
│   ├── disassembler.py      # PE .text 节反汇编（capstone + pefile）
│   ├── slicer.py            # 定长汇编切片
│   ├── reporter.py          # 文本 / JSON 报告
│   ├── llm_client.py        # DeepSeek 降噪客户端
│   ├── model/               # transformer / tokenizer / dataset / predictor / trainer
│   └── dataset/             # juliet / inline_assembly / template / augment
│                            # compiler / slicing_code / pipeline（论文六步管线）
├── scripts/                 # build_dataset.py / train_model.py / dateset_overview.py
├── docs/                    # 论文映射文档、Windows 长路径说明
├── data/                    # Juliet 官方数据源（只读）
├── conf/                    # MSVC 编译支持文件
├── pyproject.toml / requirements.txt / .gitignore
```

---

## ✅ 测试

```bash
pip install pytest
pytest tests/ -v        # 已配置自动加载 src 路径，无需额外设置 PYTHONPATH
```

---

## 📚 文档

| 文档 | 说明 |
|---|---|
| [`docs/usage_guide.html`](docs/usage_guide.html) | **完整使用手册**：数据集构建 / 模型训练 / 漏洞检测 / 配置 / FAQ |
| [`docs/article_notes.md`](docs/article_notes.md) | 论文六步管线 ↔ 代码模块映射、关键算法实现位置、核心模块职责 |
| [`docs/test_report.html`](docs/test_report.html) | 软件测试报告（测试过程、缺陷清单、复现命令） |
| [`docs/optimization_report.html`](docs/optimization_report.html) | 修复与优化报告（修复问题清单、功能验收结果） |
| [`docs/windows_long_path.txt`](docs/windows_long_path.txt) | Windows 长路径支持配置说明 |

---

## ⚠️ 已知限制

- 仅支持 **x86 / 32 位 PE** 文件 —— 其实要支持 Linux / 64 位也并非难事，构造一个对应平台的缺陷数据集重新训练即可；奈何本人没有多余精力继续改进了，这个坑就留给有缘人来填吧 🫡
- 数据集构建与编译环节依赖 **Windows + Visual Studio**（vcvarsall.bat 路径可配置）
- 大模型降噪依赖网络与 DeepSeek API
- `data/2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3/` 为官方原始数据源，**只读使用**：构建产物全部写入 `data/temp/`，汇总数据集输出至 `data/total_defect_slicing.txt`
- 检测准确度由**数据集规模、模型训练效果与是否包含良性负样本**共同决定：仅用缺陷样本训练的模型对良性代码有较高误报率，建议按上文「训练时合并良性负样本」操作；本工具的效果也因此受限于本地 GPU 资源 —— 不过只要你数据更全、显卡更给力，随时可以重新训练自己的大模型，再让检测工具加载你训练好的权重即可；到时候准确度能到多少，就看你的显卡有多争气了 🤭

---

## 🙏 写在最后

本项目是作者的**本科毕业设计作品**，分享出来主要是为了**保存本科阶段做的一点东西**，同时希望能给研究二进制漏洞检测的同学带来一点点参考。

由于个人水平有限，论文提出的方法必然存在**局限性和不足**，代码实现也比较简单，还有很多可以优化的地方；如果代码中存在错误或不当之处，**欢迎大家留言指正**，非常感谢！如果这个小工具能对二进制漏洞检测起到一点点作用，那就再好不过了；如果没什么用，那也帮大家排掉了一个「此路不通」的选项，勉强算是一点小小的贡献 😂

---

## 📄 License

MIT

---

*本工具是论文《Binary vulnerability detection based on deep learning combined with inline assembly comparison》实现思路的完整工程体现：内联汇编差异比较、缺陷模板扩展、Transformer 缺陷识别与代码级定位等核心技术均已落地实现。*
