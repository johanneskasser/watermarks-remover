"""Layer B rewriting for the text web UI: paraphrase modes and strategy chains.

Long inputs are split on paragraph boundaries into chunks (default 3000 chars)
that are rewritten one after another, then re-joined with their original
whitespace. Prompts come from ``rewrite_text`` (``build_prompt``); this module
only adds the style table, chunking and think-block handling.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from rewrite_text import LLM_TACTICS, apply_strategy, build_prompt
from text_unicode import clean_text
from webui_llm import (
    DEFAULT_LLM_TIMEOUT,
    ApiError,
    LlmProfile,
    generate,
    llm_failure,
    resolve_model,
    strip_think,
)

__all__ = [
    "CHUNK_CHARS",
    "MODES",
    "apply_layer_b",
    "paraphrase",
    "split_chunks",
    "strip_think",
]

CHUNK_CHARS = 3000

_SAME_LANGUAGE = "written in the same language as the input"

# mode -> style description appended to the paraphrase prompt (None = plain paraphrase).
MODES: dict[str, str] = {
    "standard": "natural and idiomatic",
    "fluent": "fluent and natural, like an experienced native writer, with smooth transitions",
    "formal": "formal and professional in register, without contractions or slang",
    "simple": "simple and plain: short sentences, common words, no jargon",
    "creative": "creative and vivid, with expressive phrasing while keeping the meaning",
    "shorten": (
        "much more concise: roughly 70 percent of the original length, cutting redundancy "
        "and filler while keeping every key fact"
    ),
    "expand": (
        "somewhat more elaborate: roughly 130 percent of the original length, spelling out "
        "implicit connections and adding clarity without inventing new facts"
    ),
}

_PARAGRAPH_RE = re.compile(r".*?(?:\n\s*\n+|\Z)", re.DOTALL)
_SENTENCE_RE = re.compile(r".*?(?:[.!?]+[\"')\]]*\s+|\Z)", re.DOTALL)


def _hard_split(piece: str, limit: int) -> list[str]:
    out: list[str] = []
    while len(piece) > limit:
        cut = piece.rfind(" ", 0, limit)
        cut = cut + 1 if cut > limit // 2 else limit
        out.append(piece[:cut])
        piece = piece[cut:]
    if piece:
        out.append(piece)
    return out


def _pieces(text: str, limit: int) -> list[str]:
    """Paragraphs; oversized ones broken at sentences, then at spaces."""
    pieces: list[str] = []
    for para in _PARAGRAPH_RE.findall(text):
        if not para:
            continue
        if len(para) <= limit:
            pieces.append(para)
            continue
        for sentence in _SENTENCE_RE.findall(para):
            if len(sentence) <= limit:
                if sentence:
                    pieces.append(sentence)
            else:
                pieces.extend(_hard_split(sentence, limit))
    return pieces


def split_chunks(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Split *text* into chunks of at most *limit* chars; ``"".join(chunks) == text``."""
    chunks: list[str] = []
    cur = ""
    for piece in _pieces(text, limit):
        if cur and len(cur) + len(piece) > limit:
            chunks.append(cur)
            cur = ""
        cur += piece
    if cur:
        chunks.append(cur)
    return chunks


def _split_ws(chunk: str) -> tuple[str, str, str]:
    core = chunk.strip()
    if not core:
        return chunk, "", ""
    start = chunk.index(core)
    return chunk[:start], core, chunk[start + len(core) :]


def _rewrite_chunks(
    text: str, rewrite_one: Callable[[str], str], warnings: list[str]
) -> tuple[str, int]:
    """Rewrite every non-blank chunk sequentially; keep the original on an empty reply."""
    chunks = split_chunks(text)
    out: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        lead, core, trail = _split_ws(chunk)
        if not core:
            out.append(chunk)
            continue
        rewritten = rewrite_one(core).strip()
        if not rewritten:
            warnings.append(f"chunk {index}/{len(chunks)}: model returned no text; kept original")
            out.append(chunk)
        else:
            out.append(lead + rewritten + trail)
    return "".join(out), len(chunks)


def paraphrase(
    text: str,
    *,
    profile: LlmProfile,
    model: str | None,
    mode: str,
    intensity: float,
    layer_a_after: bool,
) -> dict[str, Any]:
    """Paraphrase *text* with the profile's model. Returns the response payload body."""
    resolved = resolve_model(profile, model)
    style = f"{MODES[mode]}, {_SAME_LANGUAGE}"
    warnings: list[str] = []

    def one(chunk: str) -> str:
        prompt = build_prompt("paraphrase", chunk, rewrite_level=intensity, style=style)
        return generate(profile, resolved, prompt, timeout=DEFAULT_LLM_TIMEOUT)

    rewritten, chunks = _rewrite_chunks(text, one, warnings)
    noop = rewritten.strip() == text.strip()
    removed = replaced = 0
    final = rewritten
    if layer_a_after:
        final, stats = clean_text(rewritten)
        removed, replaced = stats["removed_count"], stats["replaced_count"]
    return {
        "original": text,
        "final": final,
        "mode": mode,
        "intensity": intensity,
        "model": resolved,
        "profile": profile.id,
        "noop": noop,
        "chunks": chunks,
        "layer_a": {"removed_count": removed, "replaced_count": replaced},
        "warnings": warnings,
    }


def apply_layer_b(
    text: str,
    steps: list[tuple[str, float]],
    *,
    profile: LlmProfile | None,
    model: str | None,
    style: str | None,
) -> tuple[str, dict[str, Any]]:
    """Run a parsed strategy over *text* chunk by chunk. Returns (text, layer_b info)."""
    needs_llm = any(tactic in LLM_TACTICS for tactic, _ in steps)
    resolved: str | None = None
    if style:
        style = f"{style}, {_SAME_LANGUAGE}"
    if needs_llm:
        if profile is None:
            raise ApiError(503, "no LLM profiles configured", "no_profiles")
        resolved = resolve_model(profile, model)
    warnings: list[str] = []
    totals: dict[int, dict[str, Any]] = {}

    def one(chunk: str) -> str:
        cur = chunk
        for index, step in enumerate(steps):
            cur, stats = _run_step(cur, step, profile, resolved, style)
            if not cur.strip():
                return ""
            agg = totals.setdefault(
                index,
                {"tactic": step[0], "intensity": round(step[1], 4), "in_chars": 0, "out_chars": 0},
            )
            agg["in_chars"] += stats["input_chars"]
            agg["out_chars"] += stats["output_chars"]
        return cur

    final, chunks = _rewrite_chunks(text, one, warnings)
    return final, {
        "applied": True,
        "strategy": ",".join(f"{t}@{i:g}" for t, i in steps),
        "model": resolved,
        "profile": profile.id if needs_llm and profile else None,
        "steps": [totals[i] for i in sorted(totals)],
        "chunks": chunks,
        "noop": final.strip() == text.strip(),
        "warnings": warnings,
    }


def _run_step(
    text: str,
    step: tuple[str, float],
    profile: LlmProfile | None,
    model: str | None,
    style: str | None,
) -> tuple[str, dict[str, Any]]:
    """Apply one strategy step through ``apply_strategy`` (reasoning stripped)."""
    backend = profile.backend if profile else "print-prompt"
    try:
        out, stats = apply_strategy(
            text,
            [step],
            backend=backend,
            model=model,
            base_url=profile.base_url if profile else None,
            api_key=profile.api_key if profile else None,
            timeout=DEFAULT_LLM_TIMEOUT,
            temperature=profile.temperature if profile else 0.8,
            reasoning_effort=profile.reasoning_effort if profile else None,
            style=style,
            layer_a_after=False,
        )
    except ApiError:
        raise
    except RuntimeError as e:
        if " empty " in f" {e} ":
            return "", {"input_chars": len(text), "output_chars": 0}
        if profile is None:
            raise
        raise llm_failure(profile, e) from e
    except SystemExit as e:
        raise ApiError(400, str(e), "bad_strategy") from e
    except Exception as e:
        if profile is None:
            raise
        raise llm_failure(profile, e) from e
    return strip_think(out), stats
