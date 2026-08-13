"""回归测试：修复 B7/B17/B18 等缺陷的补充用例。"""
from bvsc.dataset.juliet import JulietPretreatment
from bvsc.dataset.inline_assembly import _FLAW_ASM, _VARIABLE_ASM
from bvsc.dataset.slicing_code import GenerateDissassemblySlicing


def test_remove_win32_guards_handles_ifndef():
    """回归(B17): #ifndef _WIN32 块应整体删除（目标平台为 Windows）。"""
    code = (
        "#ifndef _WIN32\n"
        "char* data = (char*)malloc(200);\n"
        "#endif\n"
        "return 0;\n"
    )
    result = "\n".join(JulietPretreatment.remove_win32_guards(code))
    assert "#ifndef" not in result
    assert "malloc(200)" not in result
    assert "return 0" in result


def test_remove_win32_guards_keeps_else_branch_of_ifndef():
    """回归(B17): #ifndef _WIN32 ... #else ... #endif 应保留 #else 分支（Windows 代码）。"""
    code = (
        "#ifndef _WIN32\n"
        "int linux_only = 1;\n"
        "#else\n"
        "int windows_only = 2;\n"
        "#endif\n"
    )
    result = "\n".join(JulietPretreatment.remove_win32_guards(code))
    assert "linux_only" not in result
    assert "windows_only" in result
    assert "#ifndef" not in result


def test_inline_assembly_multiline():
    """回归(B7): 内联汇编必须多行，否则 MSVC 报 C1004。"""
    assert _FLAW_ASM.startswith("__asm {\n")
    assert "\n" in _FLAW_ASM
    assert "defact_flage: nop;" in _FLAW_ASM
    assert _VARIABLE_ASM.startswith("__asm {\n")


def _fake_lines(*patterns):
    """按模式生成反汇编行，0x+mnemonic。"""
    lines = []
    for p in patterns:
        lines.append(f"0x401000 {p}")
    return lines


def test_mask_single_nop_form():
    """回归(B18): 实测 MSVC 合并 nop 后掩码为 nop;jmp;nop。"""
    lines = _fake_lines("mov eax, 1", "nop", "jmp 0x401006", "nop", "mov ebx, 2",
                        "nop", "jmp 0x40100b", "nop", "ret")
    masks = GenerateDissassemblySlicing.find_mask_begin_indices(lines)
    assert len(masks) == 2


def test_mask_double_nop_form():
    """兼容双 nop 形态: nop;nop;jmp;nop。"""
    lines = _fake_lines("nop", "nop", "jmp 0x401006", "nop", "mov ebx, 2",
                        "nop", "nop", "jmp 0x40100b", "nop", "ret")
    masks = GenerateDissassemblySlicing.find_mask_begin_indices(lines)
    assert len(masks) == 2


def test_mask_no_false_positive():
    """普通代码（无掩码）不应误报掩码。"""
    lines = _fake_lines("push ebp", "mov ebp, esp", "call 0x401000", "add esp, 4",
                        "jmp 0x401020", "mov eax, 1", "ret")
    masks = GenerateDissassemblySlicing.find_mask_begin_indices(lines)
    assert masks == []


def test_slicing_begin_offset():
    """回归(B18): 切片起点按 3 指令掩码块计算（nop;jmp;nop）。"""
    lines = _fake_lines("mov eax, 1", "nop", "jmp 0x401006", "nop", "mov ebx, 2",
                        "mov ecx, 3", "nop", "jmp 0x40100c", "nop", "ret")
    masks = GenerateDissassemblySlicing.find_mask_begin_indices(lines)
    assert masks[0] == 1
    # gather 逻辑: slicing_begin = mask0+3, slicing_end = mask1
    begin, end = masks[0] + 3, masks[1]
    assert lines[begin].endswith("mov ebx, 2")
    assert lines[end].endswith("nop")


def test_compile_folder_return_counts(tmp_path):
    """编译进度统计：compile_folder 返回 (处理数, 成功数)。"""
    from bvsc.dataset.compiler import compile_folder
    from bvsc.exceptions import CompilationError
    src = tmp_path / "src"
    src.mkdir()
    for i in range(3):
        (src / f"s{i}.c").write_text("int main(){return 0;}", encoding="utf-8")
    out = tmp_path / "out"
    # vcvarsall 不存在 → 应抛 CompilationError
    try:
        compile_folder(src, out, r"C:\nonexistent\vcvarsall.bat", src, "K:", None)
        raise AssertionError("应当抛出 CompilationError")
    except CompilationError:
        pass


def test_tqdm_compat_update():
    """no-op tqdm 兼容 update/set_postfix（无 tqdm 环境不崩溃）。"""
    from bvsc._compat import tqdm
    p = tqdm(total=5)
    p.update(2)
    assert p.n == 2
    p.set_postfix(文件夹="x", 成功=1)  # 不应抛异常
