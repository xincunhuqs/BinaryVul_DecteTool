"""缺陷模板提取与样本扩展单元测试（纯 Python）。"""
from bvsc.dataset.template import extract_template_variables


def test_extract_variables_and_digits():
    code = (
        "char dataBuffer[100];\n"
        "int data = 10;\n"
        "data = (int)malloc(952 * sizeof(char));\n"
        "if (dataBuffer == NULL) exit(-1);\n"
        "memset(dataBuffer, 's', 952 - 1);\n"
    )
    variables, digits = extract_template_variables(code)
    assert "dataBuffer" in variables
    assert "data" in variables
    # "952"、"100"、"10"、"1" 为数字常量（过滤掉 "0"）
    assert "952" in digits
    assert "0" not in digits


def test_comment_and_preprocessor_removal():
    code = (
        "#include <stdio.h>\n"
        "/* POTENTIAL FLAW: free data */\n"
        "int counter = 123;\n"
    )
    variables, digits = extract_template_variables(code)
    assert "counter" in variables
    assert "123" in digits
    # 注释中的内容不应被提取
    assert "POTENTIAL" not in variables


def test_short_variables_filtered():
    code = "char a;\nint bc;\nint long_name;\n"
    variables, _ = extract_template_variables(code)
    assert "long_name" in variables
    assert "a" not in variables  # 长度<=2 的变量被过滤（原实现行为）


def test_variable_replacement_no_partial_match():
    """词边界替换：data 不应误替换 dataBuffer 内的子串。"""
    from bvsc.dataset.template import _replace_tokens

    code = "char dataBuffer[100];\nint data = dataBuffer[0];\n"
    # 模拟 template 阶段的替换顺序：长名优先
    var_map = {"data": "!!variable_0!", "dataBuffer": "!!variable_1!"}
    digit_map = {"100": "!!digit_0!", "0": "!!digit_1!"}
    lines = []
    for codeline in code.split("\n"):
        codeline = _replace_tokens(codeline, digit_map)
        codeline = _replace_tokens(codeline, var_map)
        lines.append(codeline)
    result = "\n".join(lines)
    assert "!!variable_1!!variable_0!" not in result  # 无子串误替换
    assert "char !!variable_1![!!digit_0!];" in result
    assert "int !!variable_0! = !!variable_1![!!digit_1!];" in result
