"""Word-level diff for the text web UI.

``diff_segments(a, b)`` returns ``[{"op", "a", "b"}]`` where ``op`` is one of
``equal | delete | insert | replace``. Invariant: concatenating every segment's
``a`` gives the source and every ``b`` gives the target, so a UI can render
either side losslessly. Every invisible or control character is its own token,
so a zero-width space shows up as its own change instead of hiding in a word.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

_TOKEN_RE = re.compile(r"\w+|\s+|.", re.DOTALL)
_PARAGRAPH_RE = re.compile(r".*?(?:\n\s*\n+|\Z)", re.DOTALL)

Segment = dict[str, str]


def _paragraphs(text: str) -> list[str]:
    return [p for p in _PARAGRAPH_RE.findall(text) if p]


def _op_for(a: str, b: str) -> str:
    if a == b:
        return "equal"
    if not b:
        return "delete"
    if not a:
        return "insert"
    return "replace"


def _trim_shared_space(sa: str, sb: str) -> list[Segment]:
    """Split shared leading/trailing whitespace off a replace so it stays 'equal'."""
    lead = 0
    limit = min(len(sa), len(sb))
    while lead < limit and sa[lead] == sb[lead] and sa[lead].isspace():
        lead += 1
    trail = 0
    while trail < limit - lead and sa[-1 - trail] == sb[-1 - trail] and sa[-1 - trail].isspace():
        trail += 1
    parts = [
        ("equal", sa[:lead], sb[:lead]),
        (None, sa[lead : len(sa) - trail], sb[lead : len(sb) - trail]),
        ("equal", sa[len(sa) - trail :], sb[len(sb) - trail :]),
    ]
    return [{"op": op or _op_for(x, y), "a": x, "b": y} for op, x, y in parts if x or y]


def _token_diff(a: str, b: str) -> list[Segment]:
    """Diff two strings token by token (words, whitespace runs, single other chars)."""
    if a == b:
        return [{"op": "equal", "a": a, "b": b}] if a else []
    ta = _TOKEN_RE.findall(a)
    tb = _TOKEN_RE.findall(b)
    # Whitespace tokens are junk: they never anchor a match (they are far too
    # frequent), but still extend one. Keeps the matcher near-linear on prose.
    matcher = SequenceMatcher(lambda t: t.isspace(), ta, tb, autojunk=False)
    segs: list[Segment] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        sa, sb = "".join(ta[i1:i2]), "".join(tb[j1:j2])
        if tag == "equal":
            segs.append({"op": "equal", "a": sa, "b": sb})
        else:
            segs.extend(_trim_shared_space(sa, sb))
    return segs


def _diff_blocks(a: str, b: str) -> list[Segment]:
    """Diff paragraph lists first so only changed paragraphs get a token diff."""
    pa, pb = _paragraphs(a), _paragraphs(b)
    matcher = SequenceMatcher(None, pa, pb, autojunk=False)
    segs: list[Segment] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            text = "".join(pa[i1:i2])
            segs.append({"op": "equal", "a": text, "b": text})
        elif tag == "replace" and (i2 - i1) == (j2 - j1):
            for k in range(i2 - i1):
                segs.extend(_token_diff(pa[i1 + k], pb[j1 + k]))
        else:
            segs.extend(_token_diff("".join(pa[i1:i2]), "".join(pb[j1:j2])))
    return segs


def _join(x: Segment, y: Segment) -> Segment:
    a, b = x["a"] + y["a"], x["b"] + y["b"]
    both_equal = x["op"] == "equal" and y["op"] == "equal"
    return {"op": "equal" if both_equal else _op_for(a, b), "a": a, "b": b}


def _is_inline_space(seg: Segment) -> bool:
    return seg["op"] == "equal" and seg["a"].isspace() and "\n" not in seg["a"]


def _merge(segs: list[Segment]) -> list[Segment]:
    """Normalise in one pass: join like neighbours, absorb inline gaps between changes."""
    out: list[Segment] = []
    for seg in segs:
        if not seg["a"] and not seg["b"]:
            continue
        if out and (out[-1]["op"] == "equal") == (seg["op"] == "equal"):
            out[-1] = _join(out[-1], seg)
            continue
        if (
            seg["op"] != "equal"
            and len(out) >= 2
            and _is_inline_space(out[-1])
            and out[-2]["op"] != "equal"
        ):
            gap = out.pop()
            merged = _join(_join(out.pop(), gap), seg)
            if merged["op"] == "equal" and out and out[-1]["op"] == "equal":
                out[-1] = _join(out[-1], merged)
            else:
                out.append(merged)
            continue
        out.append(seg)
    return out


def diff_segments(a: str, b: str) -> list[Segment]:
    """Diff *a* (source) against *b* (target) into equal/delete/insert/replace segments."""
    if a == b:
        return [{"op": "equal", "a": a, "b": b}] if a else []
    return _merge(_diff_blocks(a, b))
