"""检测编排集成测试（stub capstone/pefile，验证全流程）。

通过注入假的 capstone/pefile 模块与 mock 预测器，
验证 反汇编 -> 切片 -> 预测 -> 报告 的编排逻辑与结果落盘。
"""
import sys
import types

from bvsc.config import Settings


# ----------------------------------------------------------------------
# capstone / pefile stub（不依赖真实库）
# ----------------------------------------------------------------------
def _install_capstone_stub(n_instructions: int = 50):
    """注入 capstone 桩：disasm 生成 n_instructions 条假指令。"""
    mod = types.ModuleType("capstone")

    class _FakeIns:
        def __init__(self, i):
            self.address = 0x401000 + i
            self.mnemonic = "mov"
            self.op_str = f"eax, {i}"

    class _FakeCs:
        detail = False

        def __init__(self, arch, mode):
            self._n = n_instructions

        def disasm(self, data, va):
            return [_FakeIns(i) for i in range(self._n)]

    mod.Cs = _FakeCs
    mod.CS_ARCH_X86 = 1
    mod.CS_MODE_32 = 2
    mod.CS_MODE_64 = 4
    sys.modules["capstone"] = mod


def _install_pefile_stub():
    """注入 pefile 桩：返回带 .text 节的假 PE。"""
    mod = types.ModuleType("pefile")

    class _Section:
        Name = b".text\x00\x00\x00"
        VirtualAddress = 0x1000
        SizeOfRawData = 0x200
        PointerToRawData = 0x400

    class _OptHeader:
        ImageBase = 0x400000

    class _FakePE:
        sections = [_Section()]
        OPTIONAL_HEADER = _OptHeader()

    mod.PE = lambda path: _FakePE()
    sys.modules["pefile"] = mod


def _make_settings(tmp_path) -> Settings:
    """构造使用临时输出目录的 Settings。"""
    settings = Settings.load()
    settings.raw.setdefault("detection", {})["output_dir"] = str(tmp_path)
    settings.raw.setdefault("mode", {})["accurate_scan"] = False
    return settings


class _FakePredictor:
    """mock 预测器：固定返回 CWE 类型。"""

    def predict(self, slice_text):
        return slice_text, "CWE416_Use_After_Free"


class _NoVulPredictor:
    """mock 预测器：全部返回非 CWE。"""

    def predict(self, slice_text):
        return slice_text, "mov"


def _test_detector_flow(tmp_path, predictor, expect_defects: bool):
    _install_capstone_stub(50)
    _install_pefile_stub()

    # detector 在模块顶层 import disassembler（惰性 capstone），此处需先建 stub
    from bvsc.detector import BinaryVulnerabilityDetector

    settings = _make_settings(tmp_path)
    detector = BinaryVulnerabilityDetector(settings, predictor=predictor)

    exe = tmp_path / "target.exe"
    exe.write_bytes(b"MZ" + b"\x00" * 64)

    report = detector.detect_file(exe)
    assert report.total_slices >= 1
    if expect_defects:
        assert len(report.candidates) >= 1
        assert report.candidates[0].vul_type == "CWE416_Use_After_Free"
    else:
        assert len(report.candidates) == 0

    # 报告落盘检查
    result_txt = tmp_path / "target_result.txt"
    result_json = tmp_path / "target_result.json"
    assert result_txt.exists()
    assert result_json.exists()
    if expect_defects:
        assert "CWE416_Use_After_Free" in result_txt.read_text(encoding="utf-8")
        assert "CWE416_Use_After_Free" in result_json.read_text(encoding="utf-8")


def test_detector_finds_defects(tmp_path):
    _test_detector_flow(tmp_path, _FakePredictor(), expect_defects=True)


def test_detector_no_defects(tmp_path):
    _test_detector_flow(tmp_path, _NoVulPredictor(), expect_defects=False)


def test_detector_reports_missing_file(tmp_path):
    from bvsc.detector import BinaryVulnerabilityDetector

    settings = _make_settings(tmp_path)
    detector = BinaryVulnerabilityDetector(settings, predictor=_FakePredictor())
    try:
        detector.detect_file(tmp_path / "not_exists.exe")
        raise AssertionError("应当抛出 FileNotFoundError")
    except FileNotFoundError:
        pass
