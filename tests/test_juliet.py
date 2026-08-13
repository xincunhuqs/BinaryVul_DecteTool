"""Juliet 预处理核心逻辑单元测试（纯 Python）。"""
from bvsc.dataset.juliet import JulietPretreatment


def test_remove_win32_guards_keeps_win32_branch():
    code = (
        "#ifdef _WIN32\n"
        "char* data = (char*)malloc(100);\n"
        "#else\n"
        "char* data = (char*)malloc(200);\n"
        "#endif\n"
        "return 0;\n"
    )
    lines = JulietPretreatment.remove_win32_guards(code)
    result = "\n".join(lines)
    assert "#ifdef _WIN32" not in result
    assert "#else" not in result
    assert "#endif" not in result
    # 保留 _WIN32 分支（Windows 编译目标）
    assert "malloc(100)" in result
    assert "malloc(200)" not in result


def test_extract_code_separates_bad_and_good():
    # 真实 Juliet 结构：函数定义与函数体分行
    code = (
        "#include <stdio.h>\n"
        "#ifndef OMITBAD\n"
        "void bad()\n"
        "{\n"
        "    int x = 1;\n"
        "}\n"
        "#endif\n"
        "#ifndef OMITGOOD\n"
        "void good()\n"
        "{\n"
        "    int x = 2;\n"
        "}\n"
        "#endif\n"
    )
    bad, good = JulietPretreatment.extract_code(code.split("\n"))
    bad_text = "\n".join(bad)
    good_text = "\n".join(good)
    assert "int x = 1" in bad_text
    assert "int x = 2" in good_text
    # bad 函数入口被改写为 main（原实现行为）
    assert "int main()" in bad_text


def test_collect_source_paths(tmp_path):
    # 真实 Juliet 结构: CWE_X/s01/<file_01.c> + <file_bad.cpp>
    cwe_dir = tmp_path / "CWE416_Use_After_Free"
    s01 = cwe_dir / "s01"
    s01.mkdir(parents=True)
    (s01 / "sample_01.c").write_text("int main(){return 0;}", encoding="utf-8")
    (s01 / "sample_bad.cpp").write_text("int main(){return 0;}", encoding="utf-8")
    (s01 / "sample_good.c").write_text("int main(){return 0;}", encoding="utf-8")
    (s01 / "notes.txt").write_text("x", encoding="utf-8")

    paths = JulietPretreatment().collect_source_paths(tmp_path)
    names = sorted(p.name for p in paths)
    assert names == ["sample_01.c", "sample_bad.cpp"]


def test_filecontent_extract_returns_cwe_folder(tmp_path):
    cwe_dir = tmp_path / "CWE416_Use_After_Free" / "s01"
    cwe_dir.mkdir(parents=True)
    src = cwe_dir / "sample_01.c"
    src.write_text("\nint main() {\n    return 0;\n}\n", encoding="utf-8")

    content, name, flow_folder = JulietPretreatment.filecontent_extract(src)
    assert "\n\n" not in content  # 空白行已去除
    assert name == "sample_01.c"
    assert flow_folder == "CWE416_Use_After_Free"


def test_filecontent_extract_flat_cwe_dir(tmp_path):
    """回归(#21): 源文件直接位于 CWE 目录（无 s01 子目录）时归档正确。"""
    cwe_dir = tmp_path / "CWE15_External_Control"
    cwe_dir.mkdir(parents=True)
    src = cwe_dir / "sample_01bad.c"
    src.write_text("int main(){return 0;}\n", encoding="utf-8")

    content, name, flow_folder = JulietPretreatment.filecontent_extract(src, tmp_path)
    assert flow_folder == "CWE15_External_Control"
    # 兼容旧调用（无 file_dir）：兜底取父目录名
    _, _, folder2 = JulietPretreatment.filecontent_extract(src)
    assert folder2 == "CWE15_External_Control"


def test_filecontent_extract_file_dir_is_cwe_dir(tmp_path):
    """回归(#21): juliet_testcases_dir 指向具体 CWE 目录时归档为该目录名。"""
    cwe_dir = tmp_path / "CWE121_Stack_Based_Buffer_Overflow"
    cwe_dir.mkdir(parents=True)
    src = cwe_dir / "sample_01.c"
    src.write_text("int main(){return 0;}\n", encoding="utf-8")

    _, _, flow_folder = JulietPretreatment.filecontent_extract(src, cwe_dir)
    assert flow_folder == "CWE121_Stack_Based_Buffer_Overflow"
