"""Server-side LLM profiles and transport for the text web UI.

Profiles are configured by the operator (JSON file or environment), never by the
browser: a request may only *name* a profile and optionally a model, so the web
UI cannot be used to make the server call an arbitrary URL (SSRF). Profiles are
trusted operator configuration, so there is no allow-remote gate here; the
rewrite primitives in ``rewrite_text`` are used directly for the HTTP calls.

API keys are resolved from the environment (``api_key_env``) and never appear in
any response. OpenAI-compatible profiles without a key send the dummy bearer
``lm-studio`` (LM Studio ignores it).

Stdlib only.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rewrite_text import _NoRedirect, call_ollama, call_openai_compatible

BACKENDS = ("ollama", "openai-compatible")
DUMMY_API_KEY = "lm-studio"
DEFAULT_LLM_TIMEOUT = 300.0
MODELS_TIMEOUT = 15.0
DEFAULT_TEMPERATURE = 0.8

_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)


class ApiError(Exception):
    """An error with an HTTP status and an optional machine-readable code."""

    def __init__(self, status: int, message: str, error_code: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.error_code = error_code

    def payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": False, "error": str(self)}
        if self.error_code:
            out["error_code"] = self.error_code
        return out


def strip_think(text: str) -> str:
    """Remove reasoning blocks (``<think>...</think>``) from model output."""
    out = _THINK_BLOCK_RE.sub("", text)
    if _THINK_CLOSE_RE.search(out):
        # Opening tag swallowed by the server: everything up to the last close is reasoning.
        out = _THINK_CLOSE_RE.split(out)[-1]
    match = _THINK_OPEN_RE.search(out)
    if match:
        out = out[: match.start()]  # unterminated block: reasoning ate the whole budget
    return out.strip()


def normalize_base_url(raw: str) -> str:
    """Validate an http(s) base URL and strip a trailing ``/`` and ``/v1``.

    ``rewrite_text.call_openai_compatible`` appends ``/v1/chat/completions``
    itself, so the base URL must not carry the ``/v1`` suffix.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("base_url must be a non-empty string")
    parsed = urlparse(raw.strip())
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"base_url must be http(s), got {raw!r}")
    if not parsed.hostname:
        raise ValueError(f"base_url has no host: {raw!r}")
    if parsed.username or parsed.password:
        raise ValueError("base_url must not embed credentials; use api_key_env")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return f"{parsed.scheme}://{parsed.netloc}{path}"


@dataclass(frozen=True)
class LlmProfile:
    """One operator-configured LLM endpoint."""

    id: str
    label: str
    backend: str
    base_url: str
    model: str | None = None
    api_key: str | None = field(default=None, repr=False)
    temperature: float = DEFAULT_TEMPERATURE
    reasoning_effort: str | None = None

    def public(self) -> dict[str, Any]:
        """The browser-safe view: no URL path, no credentials."""
        parsed = urlparse(self.base_url)
        host = parsed.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        if parsed.port:
            host = f"{host}:{parsed.port}"
        return {
            "id": self.id,
            "label": self.label,
            "backend": self.backend,
            "model": self.model,
            "base_url_host": host,
        }


@dataclass(frozen=True)
class Registry:
    profiles: dict[str, LlmProfile]
    default_id: str | None


def _profile_from_config(pid: str, cfg: Any, env: Mapping[str, str]) -> LlmProfile:
    if not _ID_RE.match(pid):
        raise ValueError(f"invalid profile id {pid!r} (use letters, digits, '_', '-', '.')")
    if not isinstance(cfg, dict):
        raise ValueError(f"profile {pid!r} must be an object")
    backend = cfg.get("backend")
    if backend not in BACKENDS:
        raise ValueError(f"profile {pid!r}: backend must be one of {', '.join(BACKENDS)}")
    try:
        base_url = normalize_base_url(cfg.get("base_url", ""))
    except ValueError as e:
        raise ValueError(f"profile {pid!r}: {e}") from e
    model = cfg.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError(f"profile {pid!r}: model must be a non-empty string")
    label = cfg.get("label", pid)
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"profile {pid!r}: label must be a non-empty string")
    api_key: str | None = None
    key_env = cfg.get("api_key_env")
    if key_env is not None:
        if not isinstance(key_env, str) or not key_env:
            raise ValueError(f"profile {pid!r}: api_key_env must be a variable name")
        api_key = env.get(key_env) or None
    if api_key is None and backend == "openai-compatible":
        api_key = DUMMY_API_KEY
    temperature = cfg.get("temperature", DEFAULT_TEMPERATURE)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError(f"profile {pid!r}: temperature must be a number")
    # "none" is the repo-wide default (rewrite_text); "off" omits the parameter entirely.
    effort = cfg.get("reasoning_effort", "none" if backend == "openai-compatible" else None)
    if effort is not None and not isinstance(effort, str):
        raise ValueError(f"profile {pid!r}: reasoning_effort must be a string")
    return LlmProfile(
        id=pid,
        label=label.strip(),
        backend=backend,
        base_url=base_url,
        model=model.strip() if model else None,
        api_key=api_key,
        temperature=float(temperature),
        reasoning_effort=effort or None,
    )


def load_registry(path: Path | None, env: Mapping[str, str]) -> Registry:
    """Build the profile registry from a JSON file, or built-in defaults.

    Raises ValueError for a malformed file (the server turns that into a
    startup error, like the strategy config).
    """
    if path is not None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid LLM profiles file {path}: {e}") from e
        if not isinstance(data, dict) or not isinstance(data.get("profiles"), dict):
            raise ValueError(f"{path}: expected an object with a 'profiles' object")
        profiles = {
            pid: _profile_from_config(pid, cfg, env) for pid, cfg in data["profiles"].items()
        }
        default = data.get("default")
        if default is not None and default not in profiles:
            raise ValueError(f"{path}: default profile {default!r} is not defined")
        if default is None and profiles:
            default = next(iter(profiles))
        return Registry(profiles, default)

    profiles: dict[str, LlmProfile] = {
        "local": LlmProfile(
            id="local",
            label="Local Ollama",
            backend="ollama",
            base_url="http://127.0.0.1:11434",
        )
    }
    default = "local"
    remote_url = (env.get("WATERMARKS_LLM_REMOTE_URL") or "").strip()
    if remote_url:
        profiles["remote"] = _profile_from_config(
            "remote",
            {"label": "Remote LM Studio", "backend": "openai-compatible", "base_url": remote_url},
            env,
        )
        default = "remote"
    return Registry(profiles, default)


_REGISTRY: Registry | None = None


def set_registry(registry: Registry) -> None:
    """Install the registry (called once at server startup)."""
    global _REGISTRY  # noqa: PLW0603 — process-wide config set at startup
    _REGISTRY = registry


def get_registry() -> Registry:
    """The active registry; lazily the built-in defaults when none was installed."""
    global _REGISTRY  # noqa: PLW0603
    if _REGISTRY is None:
        _REGISTRY = load_registry(None, os.environ)
    return _REGISTRY


def get_profile(profile_id: str | None) -> LlmProfile:
    """Resolve a profile by id (None = default); ApiError 400/503 otherwise."""
    reg = get_registry()
    if not reg.profiles:
        raise ApiError(503, "no LLM profiles configured", "no_profiles")
    if profile_id is None:
        profile_id = reg.default_id
    if not isinstance(profile_id, str) or profile_id not in reg.profiles:
        raise ApiError(400, "unknown profile", "unknown_profile")
    return reg.profiles[profile_id]


def llm_failure(profile: LlmProfile, exc: BaseException) -> ApiError:
    where = profile.public()["base_url_host"]
    if isinstance(exc, urllib.error.HTTPError):
        detail = ""
        with contextlib.suppress(OSError):
            detail = exc.read(300).decode("utf-8", "replace").strip()
        msg = f"LLM server {where} returned HTTP {exc.code}"
        return ApiError(502, f"{msg}: {detail}" if detail else msg, "llm_error")
    if isinstance(exc, (urllib.error.URLError, TimeoutError, OSError)):
        reason = getattr(exc, "reason", exc)
        return ApiError(502, f"LLM server {where} unreachable: {reason}", "llm_unreachable")
    return ApiError(502, f"LLM server {where} error: {exc}", "llm_error")


def _http_get_json(url: str, headers: dict[str, str], timeout: float) -> Any:
    if urlparse(url).scheme not in ("http", "https"):
        raise ValueError(f"refusing non-http(s) endpoint: {url}")
    req = urllib.request.Request(url, headers=headers, method="GET")  # noqa: S310 — scheme checked
    opener = urllib.request.build_opener(_NoRedirect())
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _models_sort_key(name: str) -> int:
    return 1 if "embed" in name.lower() else 0


def list_models(profile: LlmProfile) -> list[str]:
    """Models the profile's server offers, embedding-only models last."""
    headers: dict[str, str] = {}
    if profile.backend == "ollama":
        url = profile.base_url + "/api/tags"
    else:
        url = profile.base_url + "/v1/models"
        if profile.api_key:
            headers["Authorization"] = f"Bearer {profile.api_key}"
    try:
        data = _http_get_json(url, headers, MODELS_TIMEOUT)
        if profile.backend == "ollama":
            names = [m.get("name") or m.get("model") for m in data.get("models", [])]
        else:
            names = [m.get("id") for m in data.get("data", [])]
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise llm_failure(profile, e) from e
    except (ValueError, AttributeError, TypeError) as e:
        raise ApiError(
            502,
            f"LLM server {profile.public()['base_url_host']} returned an unreadable model list",
            "llm_error",
        ) from e
    clean = [n for n in names if isinstance(n, str) and n]
    return sorted(clean, key=_models_sort_key)  # stable: embeddings last


def resolve_model(profile: LlmProfile, requested: str | None) -> str:
    """Pick the model: request override, profile pin, else first non-embedding."""
    if requested:
        return requested
    if profile.model:
        return profile.model
    models = list_models(profile)
    if not models:
        raise ApiError(502, "LLM server reports no models", "llm_error")
    return models[0]


def generate(
    profile: LlmProfile,
    model: str,
    prompt: str,
    *,
    timeout: float = DEFAULT_LLM_TIMEOUT,
    temperature: float | None = None,
) -> str:
    """One chat completion; reasoning blocks stripped. Empty string = empty reply."""
    temp = profile.temperature if temperature is None else temperature
    try:
        if profile.backend == "ollama":
            out = call_ollama(profile.base_url, model, prompt, timeout, temp)
        else:
            out = call_openai_compatible(
                profile.base_url,
                model,
                prompt,
                profile.api_key,
                timeout,
                temp,
                profile.reasoning_effort,
            )
    except RuntimeError as e:
        if " empty " in f" {e} ":
            return ""
        raise llm_failure(profile, e) from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise llm_failure(profile, e) from e
    except (ValueError, KeyError, TypeError, AttributeError) as e:  # bad JSON / shape
        raise llm_failure(profile, e) from e
    return strip_think(out)
