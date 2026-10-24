"""LLM profile loading and transport for the web UI (webui_llm.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import webui_llm
from fake_llm import FakeLlm
from webui_llm import ApiError, LlmProfile, load_registry, normalize_base_url


@pytest.fixture
def fake() -> FakeLlm:
    with FakeLlm() as f:
        yield f


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://127.0.0.1:1234", "http://127.0.0.1:1234"),
        ("http://127.0.0.1:1234/", "http://127.0.0.1:1234"),
        ("http://127.0.0.1:1234/v1", "http://127.0.0.1:1234"),
        ("http://host.lan:1234/v1/", "http://host.lan:1234"),
        ("https://llm.example.com/prefix/v1", "https://llm.example.com/prefix"),
    ],
)
def test_normalize_base_url(raw: str, expected: str) -> None:
    assert normalize_base_url(raw) == expected


@pytest.mark.parametrize(
    "raw", ["file:///etc/passwd", "ftp://x", "127.0.0.1:1234", "", "http://", "http://u:p@h:1/"]
)
def test_normalize_base_url_rejects(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_base_url(raw)


def test_load_registry_from_file(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {
            "default": "remote",
            "profiles": {
                "local": {
                    "label": "Local Ollama",
                    "backend": "ollama",
                    "base_url": "http://127.0.0.1:11434",
                    "model": "gemma4:12b",
                },
                "remote": {
                    "label": "Remote LM Studio",
                    "backend": "openai-compatible",
                    "base_url": "http://100.113.133.107:1234/v1",
                    "model": "qwen/qwen3.8-27b",
                    "api_key_env": "WM_TEST_KEY",
                },
            },
        },
    )
    reg = load_registry(path, {"WM_TEST_KEY": "s3cret"})
    assert reg.default_id == "remote"
    remote = reg.profiles["remote"]
    assert remote.base_url == "http://100.113.133.107:1234"
    assert remote.api_key == "s3cret"
    assert remote.public() == {
        "id": "remote",
        "label": "Remote LM Studio",
        "backend": "openai-compatible",
        "model": "qwen/qwen3.8-27b",
        "base_url_host": "100.113.133.107:1234",
    }
    assert "s3cret" not in json.dumps(remote.public())
    assert "s3cret" not in repr(remote)


def test_openai_profile_without_key_uses_dummy(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {"profiles": {"r": {"backend": "openai-compatible", "base_url": "http://h:1"}}},
    )
    reg = load_registry(path, {})
    assert reg.profiles["r"].api_key == "lm-studio"
    assert reg.default_id == "r"
    assert reg.profiles["r"].label == "r"


def test_openai_profile_defaults_reasoning_none_and_off_omits(tmp_path: Path) -> None:
    profiles = {
        "a": {"backend": "openai-compatible", "base_url": "http://h:1"},
        "b": {"backend": "openai-compatible", "base_url": "http://h:1", "reasoning_effort": "off"},
        "c": {"backend": "ollama", "base_url": "http://h:2"},
    }
    reg = load_registry(_write(tmp_path, {"profiles": profiles}), {})
    assert reg.profiles["a"].reasoning_effort == "none"
    assert reg.profiles["b"].reasoning_effort == "off"
    assert reg.profiles["c"].reasoning_effort is None


@pytest.mark.parametrize(
    "data",
    [
        {"profiles": []},
        {"profiles": {"a": {"backend": "nope", "base_url": "http://h:1"}}},
        {"profiles": {"a": {"backend": "ollama", "base_url": "file:///x"}}},
        {"profiles": {"a": {"backend": "ollama"}}},
        {"default": "ghost", "profiles": {"a": {"backend": "ollama", "base_url": "http://h:1"}}},
        {"profiles": {"bad id!": {"backend": "ollama", "base_url": "http://h:1"}}},
    ],
)
def test_load_registry_rejects_bad_files(tmp_path: Path, data: dict) -> None:
    with pytest.raises(ValueError):
        load_registry(_write(tmp_path, data), {})


def test_load_registry_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text("{nope", encoding="utf-8")
    with pytest.raises(ValueError):
        load_registry(path, {})


def test_builtin_registry_local_only() -> None:
    reg = load_registry(None, {})
    assert list(reg.profiles) == ["local"]
    assert reg.profiles["local"].backend == "ollama"
    assert reg.profiles["local"].model is None
    assert reg.default_id == "local"


def test_builtin_registry_with_remote_env() -> None:
    reg = load_registry(None, {"WATERMARKS_LLM_REMOTE_URL": "http://100.1.2.3:1234/v1"})
    assert list(reg.profiles) == ["local", "remote"]
    assert reg.profiles["remote"].backend == "openai-compatible"
    assert reg.profiles["remote"].base_url == "http://100.1.2.3:1234"
    assert reg.default_id == "remote"


def test_builtin_registry_rejects_bad_remote_env() -> None:
    with pytest.raises(ValueError):
        load_registry(None, {"WATERMARKS_LLM_REMOTE_URL": "file:///x"})


def test_get_profile_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(webui_llm, "_REGISTRY", webui_llm.Registry({}, None))
    with pytest.raises(ApiError) as exc:
        webui_llm.get_profile(None)
    assert exc.value.status == 503
    reg = load_registry(None, {})
    monkeypatch.setattr(webui_llm, "_REGISTRY", reg)
    with pytest.raises(ApiError) as exc:
        webui_llm.get_profile("ghost")
    assert exc.value.status == 400
    assert webui_llm.get_profile(None).id == "local"
    with pytest.raises(ApiError):
        webui_llm.get_profile(123)  # type: ignore[arg-type]


def _profile(fake: FakeLlm, backend: str, model: str | None = None) -> LlmProfile:
    return LlmProfile(
        id="t",
        label="t",
        backend=backend,
        base_url=fake.base_url,
        model=model,
        api_key="lm-studio" if backend == "openai-compatible" else None,
    )


def test_list_models_openai_puts_embeddings_last(fake: FakeLlm) -> None:
    models = webui_llm.list_models(_profile(fake, "openai-compatible"))
    assert models == ["qwen/qwen3.8-27b", "gemma-fake", "text-embedding-nomic-embed-text-v1.5"]
    assert fake.requests[0]["path"] == "/v1/models"


def test_list_models_ollama(fake: FakeLlm) -> None:
    models = webui_llm.list_models(_profile(fake, "ollama"))
    assert models == ["gemma4:12b", "nomic-embed-text:latest"]
    assert fake.requests[0]["path"] == "/api/tags"


def test_resolve_model_prefers_request_then_profile_then_first(fake: FakeLlm) -> None:
    prof = _profile(fake, "openai-compatible", model="pinned")
    assert webui_llm.resolve_model(prof, "asked") == "asked"
    assert webui_llm.resolve_model(prof, None) == "pinned"
    lazy = _profile(fake, "openai-compatible")
    assert webui_llm.resolve_model(lazy, None) == "qwen/qwen3.8-27b"


def test_generate_openai_sends_key_and_strips_think(fake: FakeLlm) -> None:
    fake.responder = lambda prompt: "<think>hmm</think>  Answer: " + prompt[-3:]
    out = webui_llm.generate(_profile(fake, "openai-compatible"), "m", "prompt-abc", timeout=5)
    assert out == "Answer: abc"
    req = fake.requests[0]
    assert req["path"] == "/v1/chat/completions"
    assert req["headers"]["Authorization"] == "Bearer lm-studio"
    assert req["body"]["model"] == "m"


def test_generate_ollama(fake: FakeLlm) -> None:
    out = webui_llm.generate(_profile(fake, "ollama"), "g", "x\n---\nbody", timeout=5)
    assert out == "REWRITTEN: body"
    assert fake.requests[0]["path"] == "/api/chat"


def test_generate_http_error_is_llm_error(fake: FakeLlm) -> None:
    fake.fail_status = 500
    with pytest.raises(ApiError) as exc:
        webui_llm.generate(_profile(fake, "openai-compatible"), "m", "p", timeout=5)
    assert exc.value.status == 502
    assert exc.value.error_code == "llm_error"
    assert "500" in str(exc.value)


def test_generate_unreachable() -> None:
    dead = LlmProfile(
        id="dead", label="d", backend="openai-compatible", base_url="http://127.0.0.1:9", model="m"
    )
    with pytest.raises(ApiError) as exc:
        webui_llm.generate(dead, "m", "p", timeout=3)
    assert exc.value.status == 502
    assert exc.value.error_code == "llm_unreachable"


def test_list_models_unreachable() -> None:
    dead = LlmProfile(id="dead", label="d", backend="ollama", base_url="http://127.0.0.1:9")
    with pytest.raises(ApiError) as exc:
        webui_llm.list_models(dead)
    assert exc.value.error_code == "llm_unreachable"


def test_generate_empty_reply_is_empty_string(fake: FakeLlm) -> None:
    fake.responder = lambda prompt: "<think>only thinking</think>"
    out = webui_llm.generate(_profile(fake, "openai-compatible"), "m", "p", timeout=5)
    assert out == ""


def test_shipped_example_profiles_file_loads() -> None:
    example = Path(__file__).resolve().parents[1] / "config" / "llm_profiles.example.json"
    reg = load_registry(example, {})
    assert reg.default_id == "remote"
    assert set(reg.profiles) == {"local", "remote"}
    assert reg.profiles["remote"].base_url.startswith("http")
