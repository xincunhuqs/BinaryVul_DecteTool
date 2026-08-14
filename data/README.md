# Juliet 测试套件数据源说明

## 为什么本目录没有包含 Juliet 原始数据集？

本目录下的 `2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3/` 是官方原始数据源（**只读**），
体积约 **707 MB**，远超 GitHub 单文件限制（100MB），因此**未随项目上传到 GitHub 仓库**。
克隆仓库后该目录不存在，属于正常现象。

## 如何获取与使用

### 1. 下载 Juliet 测试套件

官方下载地址（NIST SAMATE SARD，测试套件 #112）：

> **https://samate.nist.gov/SARD/test-suites/112**

选择 **"2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3"**（C/C++ 版本）下载，
解压后得到 `2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3` 目录。

> 💡 若官方站点较慢，可搜索镜像或使用学术网络加速下载。

### 2. 放置到正确位置

将解压后的目录放到本 `data/` 下，最终路径必须为：

```
data/2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3/C/testcases
```

即：`data/2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3/C/testcases/CWE121_Stack_Based_Buffer_Overflow/...`

### 3. 验证路径

```bash
# 应能看到大量 CWE 目录（CWE121_...、CWE416_... 等）
ls data/2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3/C/testcases
```

### 4. 配置与构建数据集

`config/config.yaml` 中 `dataset.juliet_testcases_dir` 默认指向上述路径（无需修改）：

```yaml
dataset:
  juliet_testcases_dir: data/2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3/C/testcases
  only_cwe: null          # 置空 = 构建全部缺陷类型；或指定如 CWE121_Stack_Based_Buffer_Overflow 只构建单类型
```

然后执行一键构建：

```bash
python scripts/build_dataset.py
```

构建产物写入 `data/temp/`（自动生成，已加入 .gitignore），汇总数据集输出至
`data/total_defect_slicing.txt`。

## 其他大文件说明

| 文件 | 大小 | 说明 | 获取方式 |
|---|---|---|---|
| `models/transformer.pth` | 176 MB | 预训练模型权重 | 见项目 README「下载与安装」：GitHub Release 附件 |
| `data/total_defect_slicing.txt` | 64 MB | 缺陷汇编切片训练数据集 | 见项目 README「下载与安装」：GitHub Release 附件，或自行构建 |

> 📌 约定：Juliet 官方数据源**只读使用**，构建产物全部写入 `data/temp/`，
> 修改/删除原始文件不会影响重新构建。
