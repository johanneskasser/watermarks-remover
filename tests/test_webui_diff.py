"""Word-level diff + chunking helpers behind the web UI."""

from __future__ import annotations

import time

import pytest
from webui_diff import diff_segments
from webui_rewrite import split_chunks, strip_think


def _roundtrip(a: str, b: str) -> list[dict[str, str]]:
    segs = diff_segments(a, b)
    assert "".join(s["a"] for s in segs) == a
    assert "".join(s["b"] for s in segs) == b
    for s in segs:
        assert s["op"] in {"equal", "delete", "insert", "replace"}
        if s["op"] == "equal":
            assert s["a"] == s["b"]
        elif s["op"] == "delete":
            assert s["a"] and s["b"] == ""
        elif s["op"] == "insert":
            assert s["a"] == "" and s["b"]
        else:
            assert s["a"] and s["b"] and s["a"] != s["b"]
    return segs


def test_identical_text_is_one_equal_segment() -> None:
    assert diff_segments("same text", "same text") == [
        {"op": "equal", "a": "same text", "b": "same text"}
    ]


def test_empty_inputs() -> None:
    assert diff_segments("", "") == []
    assert _roundtrip("", "new")[0]["op"] == "insert"
    assert _roundtrip("old", "")[0]["op"] == "delete"


def test_word_replace_and_insert() -> None:
    segs = _roundtrip("The quick brown fox", "The slow brown fox jumps")
    ops = [s["op"] for s in segs]
    assert "replace" in ops
    assert "insert" in ops
    replace = next(s for s in segs if s["op"] == "replace")
    assert (replace["a"], replace["b"]) == ("quick", "slow")


def test_invisible_characters_are_their_own_tokens() -> None:
    segs = _roundtrip("zero​width and nbsp", "zerowidth and nbsp")
    deleted = [s for s in segs if s["op"] in ("delete", "replace")]
    assert any("​" in s["a"] for s in deleted)
    # The equal context stays intact rather than being swallowed into one big replace.
    assert any(s["op"] == "equal" and "and" in s["a"] for s in segs)


def test_whitespace_between_changes_is_absorbed() -> None:
    segs = _roundtrip("alpha beta gamma", "ALPHA BETA gamma")
    non_equal = [s for s in segs if s["op"] != "equal"]
    assert len(non_equal) == 1
    assert (non_equal[0]["a"], non_equal[0]["b"]) == ("alpha beta", "ALPHA BETA")


def test_paragraph_breaks_are_not_absorbed() -> None:
    segs = _roundtrip("one\n\ntwo", "ONE\n\nTWO")
    assert any(s["op"] == "equal" and "\n\n" in s["a"] for s in segs)


def test_unicode_and_astral_roundtrip() -> None:
    _roundtrip("Grüße \U0001f600 world\U000e0041", "Gruesse \U0001f600 Welt")


def test_large_text_is_fast() -> None:
    para = "The committee reviewed the proposal and approved the budget for next year. " * 12
    a = "\n\n".join(para + f"Section {i}." for i in range(60))
    b = "\n\n".join(para.replace("approved", "endorsed") + f"Part {i}." for i in range(60))
    assert len(a) > 50_000
    start = time.monotonic()
    _roundtrip(a, b)
    assert time.monotonic() - start < 10


def test_single_giant_paragraph_is_fast() -> None:
    words = [f"word{i % 97}" for i in range(14_000)]
    a = " ".join(words)
    b = " ".join(w if i % 7 else w.upper() for i, w in enumerate(words))
    start = time.monotonic()
    _roundtrip(a, b)
    assert time.monotonic() - start < 15


@pytest.mark.parametrize(
    "text",
    [
        "",
        "short",
        "one\n\ntwo\n\n\nthree\n",
        "  leading and trailing  \n\n",
        "A. B. C. " * 800,
        "x" * 9000,
        "para\n\n" * 1500,
    ],
)
def test_split_chunks_roundtrip_and_limit(text: str) -> None:
    chunks = split_chunks(text, limit=3000)
    assert "".join(chunks) == text
    assert all(len(c) <= 3000 for c in chunks)


def test_split_chunks_prefers_paragraph_boundaries() -> None:
    paras = [f"Paragraph {i} " * 60 for i in range(8)]
    text = "\n\n".join(paras)
    chunks = split_chunks(text, limit=3000)
    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert chunk.endswith("\n\n")


def test_strip_think_variants() -> None:
    assert strip_think("<think>plan\nmore</think>\nAnswer") == "Answer"
    assert strip_think("<THINK>x</THINK>Answer") == "Answer"
    assert strip_think("dangling reasoning</think>Answer") == "Answer"
    assert strip_think("<think>never closed") == ""
    assert strip_think("plain") == "plain"
