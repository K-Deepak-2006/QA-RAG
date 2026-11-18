import pytest

from waterrag.chunking import chunk_text
from waterrag.fusion import rrf_fuse, tokenize


def test_chunk_overlap_and_coverage():
    text = " ".join(str(i) for i in range(100))
    chunks = chunk_text(text, size=30, overlap=10)
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]
    assert chunks[-1].split()[-1] == "99"


def test_chunk_short_and_empty():
    assert chunk_text("", 30, 10) == []
    assert chunk_text("a b c", 30, 10) == ["a b c"]


def test_chunk_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("a b c", 10, 10)


def test_rrf_prefers_items_ranked_high_in_both_lists():
    fused = rrf_fuse([[1, 2, 3], [3, 1, 4]])
    assert fused[0] == 1
    assert set(fused) == {1, 2, 3, 4}


def test_tokenize():
    assert tokenize("Dew-point, 90%!") == ["dew", "point", "90"]
