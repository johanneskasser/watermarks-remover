"""Backend for the text web UI: routes, validation and static file serving.

``server.py`` only routes: GET/POST handlers here take a decoded request and
return a JSON-able dict or raise ``ApiError`` (status + optional ``error_code``).

Endpoints (all JSON, ``{"ok": true, ...}`` or ``{"ok": false, "error": ...}``):

    GET  /api/llm/profiles
    GET  /api/llm/models?profile=ID
    POST /api/text/scan
    POST /api/text/clean
    POST /api/text/paraphrase

The page itself is served at ``GET /`` and ``GET /ui/<file>`` from ``web/``
(exempt from the bearer check, since a browser page load cannot send a header).
The browser never supplies an LLM URL: it can only name a server-side profile.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import webui_detect
import webui_llm
from rewrite_text import LLM_TACTICS, parse_strategy
from text_unicode import clean_text
from webui_diff import diff_segments
from webui_llm import ApiError
from webui_rewrite import MODES, apply_layer_b, paraphrase

__all__ = [
    "GET_ROUTES",
    "MAX_TEXT_CHARS",
    "POST_ROUTES",
    "WEB_DIR",
    "ApiError",
    "StaticResponse",
    "openapi_paths",
    "static_response",
]

MAX_TEXT_CHARS = 100_000
DEFAULT_STRATEGY = "paraphrase@0.8"
DEFAULT_INTENSITY = 0.8
AI_DETECT_MODES = ("off", "heuristic", "llm", "both")
LAYER_A_OPTIONS = (
    "nfkc",
    "aggressive_homoglyphs",
    "normalize_spaces",
    "strip_emoji_glue",
    "strip_bidi",
)
WEB_DIR = Path(__file__).resolve().parent / "web"


# --------------------------------------------------------------------------- validation


def _text(body: dict[str, Any]) -> str:
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ApiError(400, "'text' must be a non-empty string")
    if len(text) > MAX_TEXT_CHARS:
        raise ApiError(400, f"'text' exceeds the {MAX_TEXT_CHARS}-character limit")
    return text


def _optional_str(body: dict[str, Any], key: str, max_len: int = 200) -> str | None:
    value = body.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > max_len:
        raise ApiError(400, f"'{key}' must be a string of at most {max_len} characters")
    return value.strip() or None


def _optional_bool(body: dict[str, Any], key: str, default: bool, where: str = "") -> bool:
    value = body.get(key, default)
    if not isinstance(value, bool):
        raise ApiError(400, f"'{where}{key}' must be a boolean")
    return value


def _object(body: dict[str, Any], key: str) -> dict[str, Any]:
    value = body.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ApiError(400, f"'{key}' must be an object")
    return value


# --------------------------------------------------------------------------- handlers


def handle_profiles(_query: dict[str, str]) -> dict[str, Any]:
    reg = webui_llm.get_registry()
    if not reg.profiles:
        raise ApiError(503, "no LLM profiles configured", "no_profiles")
    return {
        "ok": True,
        "default": reg.default_id,
        "profiles": [p.public() for p in reg.profiles.values()],
        "sloptotal_configured": webui_detect.sloptotal_url() is not None,
    }


def handle_models(query: dict[str, str]) -> dict[str, Any]:
    profile = webui_llm.get_profile(query.get("profile") or None)
    models = webui_llm.list_models(profile)
    default = profile.model or next((m for m in models if "embed" not in m.lower()), None)
    return {"ok": True, "profile": profile.id, "models": models, "default_model": default}


def handle_scan(body: dict[str, Any]) -> dict[str, Any]:
    text = _text(body)
    mode = body.get("ai_detect", "heuristic")
    if mode not in AI_DETECT_MODES:
        raise ApiError(400, f"'ai_detect' must be one of {', '.join(AI_DETECT_MODES)}")
    model = _optional_str(body, "model")
    profile = None
    if mode in ("llm", "both"):
        profile = webui_llm.get_profile(_optional_str(body, "profile"))
    return {
        "ok": True,
        "length": len(text),
        "layer_a": webui_detect.scan_layer_a(text),
        "layer_b": webui_detect.scan_layer_b(text),
        "ai_content": webui_detect.ai_content(text, mode, profile, model),
    }


def _layer_a_options(body: dict[str, Any]) -> dict[str, bool]:
    raw = _object(body, "layer_a")
    unknown = sorted(set(raw) - set(LAYER_A_OPTIONS))
    if unknown:
        raise ApiError(400, f"unknown layer_a option(s): {', '.join(unknown)}")
    options = {"normalize_spaces": True}
    for key in LAYER_A_OPTIONS:
        if key in raw:
            options[key] = _optional_bool(raw, key, False, "layer_a.")
    return options


def handle_clean(body: dict[str, Any]) -> dict[str, Any]:
    text = _text(body)
    options = _layer_a_options(body)
    layer_b_req = _object(body, "layer_b")
    enabled = _optional_bool(layer_b_req, "enabled", False, "layer_b.")
    strategy = _optional_str(layer_b_req, "strategy") or DEFAULT_STRATEGY
    style = _optional_str(layer_b_req, "style", 300)
    model = _optional_str(body, "model")
    profile_id = _optional_str(body, "profile")

    steps: list[tuple[str, float]] = []
    profile = None
    if enabled:
        try:
            steps = parse_strategy(strategy)
        except ValueError as e:
            raise ApiError(400, str(e)) from e
        if any(t == "mlm" for t, _ in steps) and importlib.util.find_spec("transformers") is None:
            raise ApiError(400, "Layer B 'mlm' step requires the 'transformers' package")
        if any(t in LLM_TACTICS for t, _ in steps):
            profile = webui_llm.get_profile(profile_id)

    after_a, stats = clean_text(text, **options)
    layer_a = {
        "removed_count": stats["removed_count"],
        "replaced_count": stats["replaced_count"],
        "removed": stats["removed"],
        "replaced": stats["replaced"],
    }
    if enabled:
        final, layer_b = apply_layer_b(after_a, steps, profile=profile, model=model, style=style)
    else:
        final, layer_b = after_a, {"applied": False, "reason": "Layer B not enabled"}
    return {
        "ok": True,
        "original": text,
        "after_layer_a": after_a,
        "final": final,
        "layer_a": layer_a,
        "layer_b": layer_b,
        "diff_a": diff_segments(text, after_a),
        "diff_b": diff_segments(after_a, final),
        "diff": diff_segments(text, final),
    }


def handle_paraphrase(body: dict[str, Any]) -> dict[str, Any]:
    text = _text(body)
    mode = body.get("mode", "standard")
    if mode not in MODES:
        raise ApiError(400, f"'mode' must be one of {', '.join(MODES)}")
    intensity = body.get("intensity", DEFAULT_INTENSITY)
    if isinstance(intensity, bool) or not isinstance(intensity, (int, float)):
        raise ApiError(400, "'intensity' must be a number between 0 and 1")
    if not 0.0 <= intensity <= 1.0:
        raise ApiError(400, "'intensity' must be between 0 and 1")
    layer_a_after = _optional_bool(body, "layer_a_after", True)
    model = _optional_str(body, "model")
    profile = webui_llm.get_profile(_optional_str(body, "profile"))
    result = paraphrase(
        text,
        profile=profile,
        model=model,
        mode=mode,
        intensity=float(intensity),
        layer_a_after=layer_a_after,
    )
    result["diff"] = diff_segments(text, result["final"])
    return {"ok": True, **result}


GET_ROUTES: dict[str, Callable[[dict[str, str]], dict[str, Any]]] = {
    "/api/llm/profiles": handle_profiles,
    "/api/llm/models": handle_models,
}
POST_ROUTES: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "/api/text/scan": handle_scan,
    "/api/text/clean": handle_clean,
    "/api/text/paraphrase": handle_paraphrase,
}


# --------------------------------------------------------------------------- static files

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".json": "application/json; charset=utf-8",
}
SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
}
_PLACEHOLDER = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    "<title>watermarks-remover</title></head><body><h1>Web UI not installed</h1>"
    "<p>The service is running, but <code>service/scripts/web/index.html</code> is missing. "
    'The JSON API is available at <a href="/openapi.json">/openapi.json</a>.</p></body></html>'
)


@dataclass(frozen=True)
class StaticResponse:
    status: int
    content_type: str
    body: bytes


def is_static_path(path: str) -> bool:
    """True for the paths served without authentication: the page and its assets."""
    return path in ("/", "/favicon.ico") or path.startswith("/ui/") or path == "/ui"


def _read_asset(name: str) -> StaticResponse | None:
    if not name or name.startswith(".") or "/" in name or "\\" in name or "\x00" in name:
        return None
    content_type = _CONTENT_TYPES.get(Path(name).suffix.lower())
    if content_type is None:
        return None
    base = WEB_DIR.resolve()
    candidate = base / name
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if resolved.parent != base or not resolved.is_file():
        return None
    try:
        return StaticResponse(200, content_type, resolved.read_bytes())
    except OSError:
        return None


def static_response(path: str) -> StaticResponse:
    """Resolve a static path to a response (404 JSON-less body when unknown)."""
    not_found = StaticResponse(
        404, "application/json; charset=utf-8", b'{"ok":false,"error":"not found"}'
    )
    if path == "/":
        found = _read_asset("index.html")
        return found or StaticResponse(200, _CONTENT_TYPES[".html"], _PLACEHOLDER.encode("utf-8"))
    if path == "/favicon.ico":
        return _read_asset("favicon.ico") or not_found
    if path.startswith("/ui/"):
        return _read_asset(path[len("/ui/") :]) or not_found
    return not_found


# --------------------------------------------------------------------------- OpenAPI


def _obj(**props: Any) -> dict[str, Any]:
    return {"type": "object", "properties": props}


_STR: dict[str, Any] = {"type": "string"}
_ERR = {
    "description": "Error",
    "content": {
        "application/json": {
            "schema": _obj(
                ok={"type": "boolean", "enum": [False]},
                error=_STR,
                error_code={"type": "string"},
            )
        }
    },
}
_LLM_ERRORS = {
    "502": {
        **_ERR,
        "description": "LLM server unreachable or failed (llm_unreachable | llm_error)",
    },
    "503": {**_ERR, "description": "No LLM profiles configured"},
}
_SEG = _obj(op={"type": "string", "enum": ["equal", "delete", "insert", "replace"]}, a=_STR, b=_STR)
_DIFF = {"type": "array", "items": _SEG}


def _post(
    summary: str, props: dict[str, Any], response: dict[str, Any], llm: bool
) -> dict[str, Any]:
    return {
        "post": {
            "summary": summary,
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"type": "object", "required": ["text"], "properties": props}
                    }
                },
            },
            "responses": {"200": response, **(_LLM_ERRORS if llm else {})},
        }
    }


def openapi_paths() -> dict[str, Any]:
    """OpenAPI path items for the web UI endpoints (merged into the main spec)."""
    text = {"type": "string", "description": f"Up to {MAX_TEXT_CHARS} characters"}
    profile = {
        "type": "string",
        "description": "Server-side LLM profile id (default profile if omitted)",
    }
    model = {"type": "string", "description": "Model override (profile/default model if omitted)"}
    ok = {"type": "boolean"}
    return {
        "/api/llm/profiles": {
            "get": {
                "summary": "Server-side LLM profiles (no URLs or keys)",
                "responses": {
                    "200": _obj(
                        ok=ok,
                        default=_STR,
                        profiles={
                            "type": "array",
                            "items": _obj(
                                id=_STR, label=_STR, backend=_STR, model=_STR, base_url_host=_STR
                            ),
                        },
                        sloptotal_configured=ok,
                    ),
                    "503": _LLM_ERRORS["503"],
                },
            }
        },
        "/api/llm/models": {
            "get": {
                "summary": "Models offered by a profile's LLM server",
                "parameters": [
                    {"name": "profile", "in": "query", "required": False, "schema": _STR}
                ],
                "responses": {
                    "200": _obj(
                        ok=ok,
                        profile=_STR,
                        models={"type": "array", "items": _STR},
                        default_model=_STR,
                    ),
                    **_LLM_ERRORS,
                },
            }
        },
        "/api/text/scan": _post(
            "Scan text: invisible Unicode, watermark detectors, AI-content verdict",
            {
                "text": text,
                "profile": profile,
                "model": model,
                "ai_detect": {
                    "type": "string",
                    "enum": list(AI_DETECT_MODES),
                    "default": "heuristic",
                },
            },
            _obj(
                ok=ok,
                length={"type": "integer"},
                layer_a=_obj(
                    suspicious_total={"type": "integer"},
                    hits={"type": "array", "items": {"type": "object"}},
                    notes={"type": "array", "items": _STR},
                ),
                layer_b=_obj(
                    detectors={"type": "array", "items": {"type": "object"}},
                    stylometry={"type": "object"},
                ),
                ai_content=_obj(
                    verdict={
                        "type": "string",
                        "enum": ["likely_ai", "mixed", "likely_human", "unknown"],
                    },
                    score={"type": "number", "nullable": True},
                    methods={"type": "array", "items": {"type": "object"}},
                    spans={"type": "array", "items": {"type": "object"}},
                    disclaimer=_STR,
                ),
            ),
            llm=True,
        ),
        "/api/text/clean": _post(
            "Clean text: Layer A (invisible Unicode) then optional Layer B rewrite",
            {
                "text": text,
                "profile": profile,
                "model": model,
                "layer_a": {
                    "type": "object",
                    "properties": {k: {"type": "boolean"} for k in LAYER_A_OPTIONS},
                    "additionalProperties": False,
                },
                "layer_b": _obj(
                    enabled={"type": "boolean", "default": False},
                    strategy={"type": "string", "default": DEFAULT_STRATEGY},
                    style=_STR,
                ),
            },
            _obj(
                ok=ok,
                original=_STR,
                after_layer_a=_STR,
                final=_STR,
                layer_a={"type": "object"},
                layer_b={"type": "object"},
                diff_a=_DIFF,
                diff_b=_DIFF,
                diff=_DIFF,
            ),
            llm=True,
        ),
        "/api/text/paraphrase": _post(
            "Paraphrase text with a server-side LLM profile",
            {
                "text": text,
                "profile": profile,
                "model": model,
                "mode": {"type": "string", "enum": list(MODES), "default": "standard"},
                "intensity": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": DEFAULT_INTENSITY,
                },
                "layer_a_after": {"type": "boolean", "default": True},
            },
            _obj(
                ok=ok,
                original=_STR,
                final=_STR,
                mode=_STR,
                intensity={"type": "number"},
                model=_STR,
                profile=_STR,
                noop=ok,
                chunks={"type": "integer"},
                diff=_DIFF,
                layer_a=_obj(removed_count={"type": "integer"}, replaced_count={"type": "integer"}),
            ),
            llm=True,
        ),
    }
