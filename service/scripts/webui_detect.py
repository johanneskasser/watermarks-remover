"""Scan logic for the text web UI: Layer A hits, Layer B detectors, AI-content verdict.

AI-content detection here is evidence, not proof. Three independent methods can
contribute, each reporting ``score`` in 0..1 or ``None`` plus a plain-text
detail (and ``error`` when the method itself failed - never an exception):

* ``stylometry``  - the repo's heuristic (burstiness, stock AI phrases, lexical
  diversity). Always available, weak on its own.
* ``llm-judge``   - the selected LLM profile is asked for a strict-JSON opinion.
* ``sloptotal``   - an optional classifier sidecar (SlopTotal) configured by the
  operator via ``WATERMARKS_SLOPTOTAL_URL``; it is the only real AI-text
  classifier in the stack.

Combination rule: weighted mean of the methods that returned a score, where the
classifier counts 3x and each heuristic/judge 1x. Verdict from the combined
score: >= 0.65 ``likely_ai``, >= 0.40 ``mixed``, below ``likely_human``, and
``unknown`` when no method produced a score.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
from typing import Any

from rewrite_text import _http_json
from score_stylometry import score_text_stylometry
from text_detectors import all_detectors
from text_unicode import inspect_text
from webui_llm import (
    ApiError,
    LlmProfile,
    generate,
    normalize_base_url,
    resolve_model,
    strip_think,
)

DISCLAIMER = (
    "AI-content detection is heuristic, not proof. Scores can be wrong in both directions: "
    "formal, edited or non-native writing is often misread as AI (false positives), and "
    "paraphrased or humanized AI text often passes as human. Do not use this result alone "
    "to accuse anyone."
)

LIKELY_AI = 0.65
MIXED = 0.40
JUDGE_TIMEOUT = 180.0
JUDGE_MAX_CHARS = 8000
SLOPTOTAL_TIMEOUT = 15.0
SLOPTOTAL_MAX_CHARS = 12000
SLOPTOTAL_MIN_CHARS = 50  # the sidecar rejects shorter texts
METHOD_WEIGHTS = {"stylometry": 1.0, "llm-judge": 1.0, "sloptotal": 3.0}

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def utf16_offset(text: str, index: int) -> int:
    """Convert a code point index into a UTF-16 code unit offset (what JS strings use)."""
    return index + sum(1 for ch in text[:index] if ord(ch) > 0xFFFF)


def scan_layer_a(text: str) -> dict[str, Any]:
    """Invisible-Unicode / space-homoglyph hits with code point and UTF-16 offsets."""
    report = inspect_text(text).to_dict()
    hits = []
    for hit in report["hits"]:
        offsets = list(hit["sample_offsets"])
        hits.append(
            {
                "codepoint": hit["codepoint"],
                "label": hit["label"],
                "kind": hit["kind"],
                "confidence": hit["confidence"],
                "count": hit["count"],
                "offsets": offsets,
                "offsets_utf16": [utf16_offset(text, o) for o in offsets],
            }
        )
    return {
        "suspicious_total": report["suspicious_total"],
        "hits": hits,
        "notes": report["notes"],
    }


def scan_layer_b(text: str) -> dict[str, Any]:
    """Run every text watermark detector (unavailable ones report why) plus stylometry."""
    detectors: list[dict[str, Any]] = []
    for det in all_detectors():
        try:
            available = bool(det.available())
            detail = det.detect(text)
        except Exception as e:  # a detector must never break the scan
            available, detail = False, {"error": f"detector failed: {e}"}
        detectors.append({"name": det.name, "available": available, "detail": detail})
    sty = score_text_stylometry(text).to_dict()
    stylometry = {
        "score": sty["score"],
        "confidence_level": sty["confidence_level"],
        "density_tier": sty["density_tier"],
        "status": sty["status"],
        "word_count": sty["word_count"],
    }
    return {"detectors": detectors, "stylometry": stylometry}


def stylometry_method(text: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The heuristic method entry and its marker spans (code point offsets)."""
    report = score_text_stylometry(text).to_dict()
    spans = []
    for marker in report["matched_markers"]:
        for start, end in marker["spans"]:
            spans.append(
                {"start": start, "end": end, "reason": f"AI cadence phrase: {marker['phrase']}"}
            )
    if report["score"] is None:
        detail = report["notes"][0] if report["notes"] else "too little text to score"
    else:
        parts = [
            f"burstiness CV {report['burstiness_cv']}"
            if report["burstiness_cv"] is not None
            else "burstiness n/a"
        ]
        parts.append(f"{len(report['matched_markers'])} AI phrase pattern(s)")
        parts.append(f"lexical diversity {report['lexical_diversity']}")
        detail = "; ".join(parts + report["findings"][:3])
    return {"name": "stylometry", "score": report["score"], "detail": detail}, spans


def _clamp_probability(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number > 1.0 and number <= 100.0:
        number /= 100.0
    if number != number or not 0.0 <= number <= 1.0:  # NaN or out of range
        return None
    return number


def _str_list(value: Any, limit: int) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [v.strip() for v in value if isinstance(v, str) and v.strip()][:limit]


def parse_judge_reply(reply: str) -> dict[str, Any]:
    """Parse the judge's reply into {ai_probability, reasons, suspect_sentences}; ValueError if hopeless."""
    cleaned = strip_think(reply)
    candidates = [m for m in _FENCE_RE.findall(cleaned)]
    first, last = cleaned.find("{"), cleaned.rfind("}")
    if first != -1 and last > first:
        candidates.append(cleaned[first : last + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        prob = _clamp_probability(data.get("ai_probability"))
        if prob is None:
            continue
        return {
            "ai_probability": prob,
            "reasons": _str_list(data.get("reasons"), 5),
            "suspect_sentences": _str_list(data.get("suspect_sentences"), 8),
        }
    match = re.search(r'"?ai_probability"?\s*[:=]\s*"?([0-9]*\.?[0-9]+)', cleaned)
    if match:
        prob = _clamp_probability(match.group(1))
        if prob is not None:
            return {"ai_probability": prob, "reasons": [], "suspect_sentences": []}
    raise ValueError("the model did not return a usable JSON verdict")


def _judge_prompt(text: str) -> str:
    return (
        "You are a careful forensic reader estimating whether a text was written by an AI "
        "language model. Be calibrated: formal, polished or non-native human writing is NOT "
        "evidence of AI by itself, and you can be wrong. The text may be in any language.\n\n"
        "Reply with ONLY a JSON object, no prose and no code fences, exactly of this shape:\n"
        '{"ai_probability": <number 0..1>, "reasons": [<up to 4 short strings>], '
        '"suspect_sentences": [<up to 5 sentences copied verbatim from the text>]}\n\n'
        "TEXT TO JUDGE:\n---\n" + text
    )


def _locate(text: str, sentence: str) -> tuple[int, int] | None:
    index = text.find(sentence)
    if index == -1:
        return None
    return index, index + len(sentence)


def judge_method(
    text: str, profile: LlmProfile, model: str | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Ask the LLM for a verdict. Never raises: failures become ``error`` on the method."""
    method: dict[str, Any] = {"name": "llm-judge", "score": None, "detail": ""}
    sample = text[:JUDGE_MAX_CHARS]
    try:
        resolved = resolve_model(profile, model)
        reply = generate(
            profile, resolved, _judge_prompt(sample), timeout=JUDGE_TIMEOUT, temperature=0.1
        )
        verdict = parse_judge_reply(reply)
    except ApiError as e:
        method["error"] = str(e)
        return method, []
    except ValueError as e:
        method["error"] = str(e)
        return method, []
    method["score"] = verdict["ai_probability"]
    detail = "; ".join(verdict["reasons"]) or "no reasons given"
    if len(text) > JUDGE_MAX_CHARS:
        detail += f" (judged the first {JUDGE_MAX_CHARS} characters)"
    method["detail"] = f"{resolved}: {detail}"
    spans = []
    for sentence in verdict["suspect_sentences"]:
        found = _locate(text, sentence)
        if found:
            spans.append(
                {"start": found[0], "end": found[1], "reason": "LLM judge: suspect sentence"}
            )
    return method, spans


def sloptotal_url() -> str | None:
    """The operator-configured classifier URL, or None when unset."""
    raw = os.environ.get("WATERMARKS_SLOPTOTAL_URL", "").strip()
    return raw or None


def sloptotal_method(text: str) -> dict[str, Any]:
    """Score *text* with the SlopTotal sidecar (``POST /api/analyze``). Never raises."""
    method: dict[str, Any] = {"name": "sloptotal", "score": None, "detail": ""}
    raw_url = sloptotal_url()
    try:
        base = normalize_base_url(raw_url or "")
    except ValueError:
        method["error"] = "WATERMARKS_SLOPTOTAL_URL is not a valid http(s) URL"
        return method
    sample = text.strip()[:SLOPTOTAL_MAX_CHARS]
    if len(sample) < SLOPTOTAL_MIN_CHARS:
        method["detail"] = (
            f"skipped: the classifier needs at least {SLOPTOTAL_MIN_CHARS} characters"
        )
        return method
    try:
        data = _http_json(base + "/api/analyze", {"text": sample}, {}, SLOPTOTAL_TIMEOUT)
        score = float(data["overall_score"]) / 100.0
        if not 0.0 <= score <= 1.0:
            raise ValueError("overall_score out of range")
    except urllib.error.HTTPError as e:
        method["error"] = f"SlopTotal returned HTTP {e.code}"
        return method
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        reason = getattr(e, "reason", e)
        method["error"] = f"SlopTotal unreachable or timed out: {reason}"
        return method
    except (KeyError, TypeError, ValueError, AttributeError):
        method["error"] = "SlopTotal returned an unexpected response"
        return method
    method["score"] = score
    parts = [str(data.get("overall_verdict") or "").strip()]
    if data.get("engines_total"):
        parts.append(f"{data.get('engines_flagged')}/{data['engines_total']} engines flagged")
    language = data.get("language")
    if language:
        parts.append(f"language {language} ({data.get('language_support', 'unknown')} support)")
    if len(text.strip()) > SLOPTOTAL_MAX_CHARS:
        parts.append(f"first {SLOPTOTAL_MAX_CHARS} characters only")
    method["detail"] = "; ".join(p for p in parts if p)
    return method


def combine(methods: list[dict[str, Any]]) -> tuple[str, float | None]:
    """Weighted mean of scored methods -> (verdict, score)."""
    total = weight = 0.0
    for method in methods:
        if method["score"] is None:
            continue
        w = METHOD_WEIGHTS.get(method["name"], 1.0)
        total += w * method["score"]
        weight += w
    if not weight:
        return "unknown", None
    score = total / weight
    if score >= LIKELY_AI:
        return "likely_ai", score
    if score >= MIXED:
        return "mixed", score
    return "likely_human", score


def ai_content(
    text: str, mode: str, profile: LlmProfile | None, model: str | None
) -> dict[str, Any]:
    """The ``ai_content`` block for ``mode`` in off|heuristic|llm|both."""
    methods: list[dict[str, Any]] = []
    spans: list[dict[str, Any]] = []
    if mode in ("heuristic", "both"):
        method, found = stylometry_method(text)
        methods.append(method)
        spans.extend(found)
    if mode in ("llm", "both") and profile is not None:
        method, found = judge_method(text, profile, model)
        methods.append(method)
        spans.extend(found)
    if mode in ("heuristic", "both") and sloptotal_url():
        methods.append(sloptotal_method(text))
    verdict, score = combine(methods)
    spans.sort(key=lambda s: (s["start"], s["end"]))
    for span in spans:
        span["start_utf16"] = utf16_offset(text, span["start"])
        span["end_utf16"] = utf16_offset(text, span["end"])
    return {
        "verdict": verdict,
        "score": round(score, 4) if score is not None else None,
        "methods": methods,
        "spans": spans,
        "disclaimer": DISCLAIMER,
    }
