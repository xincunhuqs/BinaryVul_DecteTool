"""本地分词器单元测试（纯 Python，无需 torch）。"""
from bvsc.model.tokenizer import (
    HEX_TOKEN,
    PAD_TOKEN,
    build_vocabulary,
    load_tokenizer,
    save_tokenizer,
    word2index,
)


def _small_vocab() -> dict:
    return {
        PAD_TOKEN: 0,
        "mov": 1,
        "eax": 2,
        "push": 3,
        HEX_TOKEN: 4,
        "unknow_key0": 5,
        "unknow_key1": 6,
    }


def test_word2index_known_words():
    vocab = _small_vocab()
    result = word2index("mov eax", vocab, max_sentence_len=4)
    assert result == [1, 2, 0, 0]  # pad 填充


def test_word2index_hex_normalization():
    vocab = _small_vocab()
    result = word2index("mov eax, 0x401000", vocab, max_sentence_len=4)
    # "0x401000" 含 0x -> 映射到 HEX_TOKEN(4)；"," 未知 -> unknow(5)
    assert 4 in result
    assert result[0] == 1


def test_word2index_truncation():
    vocab = _small_vocab()
    result = word2index("mov eax push mov eax", vocab, max_sentence_len=3)
    assert len(result) == 3
    assert result == [1, 2, 3]


def test_word2index_no_vocab_mutation():
    """推理过程不得修改词表（原实现副作用修复验证）。"""
    vocab = _small_vocab()
    snapshot = dict(vocab)
    word2index("strange_unknown_token eax", vocab, max_sentence_len=4)
    assert vocab == snapshot


def test_build_vocabulary():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "slices.txt"
        data.write_text("mov eax, 1; push 0x401000\nmov ebx; je 0x1000\n", encoding="utf-8")
        vocab = build_vocabulary(data, max_vocab_size=64)
        assert vocab[PAD_TOKEN] == 0
        assert HEX_TOKEN in vocab
        assert len(vocab) == 64  # 补齐到 max_vocab_size


def test_save_load_roundtrip(tmp_path):
    vocab = _small_vocab()
    path = tmp_path / "tokenize_dict.txt"
    save_tokenizer(vocab, path)
    loaded = load_tokenizer(path)
    assert loaded == vocab
