"""HTTP tests for the text web UI API and static routes (no real network)."""

from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import server
import webui_api
import webui_llm
from fake_llm import FakeLlm
from webui_llm import LlmProfile, Registry

AI_TEXT = (
    "Moreover, it is important to note that this approach plays a crucial role. "
    "Furthermore, it is worth noting that the landscape is rich and vibrant. "
    "In conclusion, this serves as a testament to the pivotal importance of the topic. "
) * 4
HUMAN_TEXT = "Ich bin gestern spät nach Hause gekommen, und der Hund hat natürlich gebellt."


@pytest.fixture
def fake() -> Iterator[FakeLlm]:
    with FakeLlm() as f:
        yield f


@pytest.fixture
def registry(fake: FakeLlm, monkeypatch: pytest.MonkeyPatch) -> Registry:
    reg = Registry(
        profiles={
            "remote": LlmProfile(
                id="remote",
                label="Remote LM Studio",
                backend="openai-compatible",
                base_url=fake.base_url,
                model="qwen/qwen3.8-27b",
                api_key="super-secret-key",
            ),
            "local": LlmProfile(
                id="local",
                label="Local Ollama",
                backend="ollama",
                base_url=fake.base_url,
                model=None,
            ),
        },
        default_id="remote",
    )
    monkeypatch.setattr(webui_llm, "_REGISTRY", reg)
    monkeypatch.delenv("WATERMARKS_SLOPTOTAL_URL", raising=False)
    return reg


@pytest.fixture
def conn(registry: Registry) -> Iterator[http.client.HTTPConnection]:
    srv = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=30)
    yield c
    c.close()
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def _post(conn: http.client.HTTPConnection, path: str, payload: Any) -> tuple[int, dict]:
    conn.request(
        "POST",
        path,
        body=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    data = resp.read()
    return resp.status, json.loads(data) if data else {}


def _get(conn: http.client.HTTPConnection, path: str, headers: dict | None = None):
    conn.request("GET", path, headers=headers or {})
    resp = conn.getresponse()
    return resp.status, resp, resp.read()


def _get_json(conn: http.client.HTTPConnection, path: str) -> tuple[int, dict]:
    status, _resp, data = _get(conn, path)
    return status, json.loads(data)


# ---------------------------------------------------------------- profiles / models


def test_profiles_endpoint_hides_keys_and_urls(conn, fake: FakeLlm) -> None:
    status, body = _get_json(conn, "/api/llm/profiles")
    assert status == 200
    assert body["ok"] is True
    assert body["default"] == "remote"
    by_id = {p["id"]: p for p in body["profiles"]}
    assert by_id["remote"] == {
        "id": "remote",
        "label": "Remote LM Studio",
        "backend": "openai-compatible",
        "model": "qwen/qwen3.8-27b",
        "base_url_host": f"127.0.0.1:{fake.port}",
    }
    raw = json.dumps(body)
    assert "super-secret-key" not in raw
    assert "http://" not in raw


def test_profiles_503_when_none_configured(conn, monkeypatch) -> None:
    monkeypatch.setattr(webui_llm, "_REGISTRY", Registry({}, None))
    status, body = _get_json(conn, "/api/llm/profiles")
    assert status == 503
    assert body["ok"] is False


def test_models_endpoint_openai(conn) -> None:
    status, body = _get_json(conn, "/api/llm/models?profile=remote")
    assert status == 200
    assert body["profile"] == "remote"
    assert body["models"][0] == "qwen/qwen3.8-27b"
    assert body["models"][-1].startswith("text-embedding")
    assert body["default_model"] == "qwen/qwen3.8-27b"


def test_models_endpoint_ollama_default_model_is_first_non_embed(conn) -> None:
    status, body = _get_json(conn, "/api/llm/models?profile=local")
    assert status == 200
    assert body["models"] == ["gemma4:12b", "nomic-embed-text:latest"]
    assert body["default_model"] == "gemma4:12b"


def test_models_endpoint_default_profile_and_unknown(conn) -> None:
    status, body = _get_json(conn, "/api/llm/models")
    assert status == 200
    assert body["profile"] == "remote"
    status, body = _get_json(conn, "/api/llm/models?profile=ghost")
    assert status == 400
    assert body["ok"] is False


def test_models_endpoint_unreachable_is_502(conn, registry, monkeypatch) -> None:
    dead = LlmProfile(id="dead", label="d", backend="ollama", base_url="http://127.0.0.1:9")
    monkeypatch.setattr(webui_llm, "_REGISTRY", Registry({"dead": dead}, "dead"))
    status, body = _get_json(conn, "/api/llm/models?profile=dead")
    assert status == 502
    assert body["error_code"] == "llm_unreachable"
    assert body["ok"] is False


# ---------------------------------------------------------------- scan


def test_scan_layer_a_hits_with_offsets(conn) -> None:
    text = "zero​width and tag\U000e0041 chars"
    status, body = _post(conn, "/api/text/scan", {"text": text})
    assert status == 200, body
    assert body["ok"] is True
    assert body["length"] == len(text)
    layer_a = body["layer_a"]
    assert layer_a["suspicious_total"] >= 2
    zwsp = next(h for h in layer_a["hits"] if h["codepoint"] == "U+200B")
    assert zwsp["count"] == 1
    assert zwsp["offsets"] == [4]
    assert zwsp["offsets_utf16"] == [4]
    assert {"label", "kind", "confidence"} <= set(zwsp)
    tag = next(h for h in layer_a["hits"] if h["codepoint"] == "U+E0041")
    # Astral char: code point offset 17 -> UTF-16 offset 18 (one surrogate pair earlier? none)
    assert tag["offsets"][0] == text.index("\U000e0041")
    assert tag["offsets_utf16"][0] == tag["offsets"][0]
    assert isinstance(layer_a["notes"], list)


def test_scan_utf16_offsets_after_astral_char(conn) -> None:
    text = "\U0001f600 a​b"
    _status, body = _post(conn, "/api/text/scan", {"text": text})
    hit = body["layer_a"]["hits"][0]
    assert hit["offsets"] == [3]
    assert hit["offsets_utf16"] == [4]


def test_scan_layer_b_and_stylometry_shape(conn) -> None:
    _status, body = _post(conn, "/api/text/scan", {"text": AI_TEXT})
    layer_b = body["layer_b"]
    assert isinstance(layer_b["detectors"], list)
    for det in layer_b["detectors"]:
        assert {"name", "available", "detail"} <= set(det)
        assert isinstance(det["available"], bool)
    assert {"score", "confidence_level", "density_tier"} <= set(layer_b["stylometry"])


def test_scan_heuristic_flags_ai_cadence_with_spans(conn) -> None:
    status, body = _post(conn, "/api/text/scan", {"text": AI_TEXT})
    assert status == 200
    ai = body["ai_content"]
    assert ai["verdict"] in {"likely_ai", "mixed"}
    assert 0.0 <= ai["score"] <= 1.0
    assert [m["name"] for m in ai["methods"]] == ["stylometry"]
    assert ai["methods"][0]["score"] is not None
    assert ai["spans"]
    span = ai["spans"][0]
    assert 0 <= span["start"] < span["end"] <= len(AI_TEXT)
    assert span["reason"]
    assert "not proof" in ai["disclaimer"].lower() or "heuristic" in ai["disclaimer"].lower()


def test_scan_short_text_is_unknown(conn) -> None:
    _status, body = _post(conn, "/api/text/scan", {"text": HUMAN_TEXT})
    ai = body["ai_content"]
    assert ai["verdict"] == "unknown"
    assert ai["score"] is None
    assert ai["disclaimer"]


def test_scan_ai_detect_off(conn, fake: FakeLlm) -> None:
    _status, body = _post(conn, "/api/text/scan", {"text": AI_TEXT, "ai_detect": "off"})
    ai = body["ai_content"]
    assert ai["methods"] == []
    assert ai["verdict"] == "unknown"
    assert ai["disclaimer"]
    assert fake.prompts() == []


def test_scan_llm_judge(conn, fake: FakeLlm) -> None:
    status, body = _post(
        conn,
        "/api/text/scan",
        {"text": AI_TEXT, "ai_detect": "llm", "profile": "remote", "model": "gemma-fake"},
    )
    assert status == 200, body
    ai = body["ai_content"]
    assert [m["name"] for m in ai["methods"]] == ["llm-judge"]
    assert ai["methods"][0]["score"] == pytest.approx(0.9)
    assert ai["verdict"] == "likely_ai"
    assert ai["score"] == pytest.approx(0.9)
    assert fake.requests[-1]["body"]["model"] == "gemma-fake"
    # suspect sentence is located in the text -> span.
    quoted = "Moreover, it is important to note this."
    assert quoted not in AI_TEXT  # judge quoted something not verbatim: no span, no crash
    assert isinstance(ai["spans"], list)


def test_scan_llm_judge_span_from_verbatim_sentence(conn, fake: FakeLlm) -> None:
    sentence = "Furthermore, it is worth noting that the landscape is rich and vibrant."
    fake.responder = lambda p: json.dumps(
        {"ai_probability": 0.3, "reasons": ["a"], "suspect_sentences": [sentence]}
    )
    _status, body = _post(conn, "/api/text/scan", {"text": AI_TEXT, "ai_detect": "llm"})
    spans = body["ai_content"]["spans"]
    assert any(AI_TEXT[s["start"] : s["end"]] == sentence for s in spans)
    assert body["ai_content"]["verdict"] == "likely_human"


def test_scan_both_combines_scores(conn, fake: FakeLlm) -> None:
    fake.responder = lambda p: '{"ai_probability": 0.1, "reasons": [], "suspect_sentences": []}'
    _status, body = _post(conn, "/api/text/scan", {"text": AI_TEXT, "ai_detect": "both"})
    ai = body["ai_content"]
    assert [m["name"] for m in ai["methods"]] == ["stylometry", "llm-judge"]
    scores = [m["score"] for m in ai["methods"]]
    assert ai["score"] == pytest.approx(sum(scores) / 2, abs=1e-3)


@pytest.mark.parametrize(
    "reply",
    [
        '```json\n{"ai_probability": 0.8, "reasons": ["x"], "suspect_sentences": []}\n```',
        '<think>let me think {not json}</think>\n{"ai_probability": 0.8, "reasons": []}',
        'Sure! Here you go: {"ai_probability": "0.8", "reasons": "single"} hope it helps',
        '{"ai_probability": 80, "reasons": []}',
    ],
)
def test_scan_llm_judge_parses_messy_replies(conn, fake: FakeLlm, reply: str) -> None:
    fake.responder = lambda p: reply
    _status, body = _post(conn, "/api/text/scan", {"text": AI_TEXT, "ai_detect": "llm"})
    method = body["ai_content"]["methods"][0]
    assert method["score"] == pytest.approx(0.8)
    assert "error" not in method


def test_scan_llm_judge_garbage_reply_is_soft_error(conn, fake: FakeLlm) -> None:
    fake.responder = lambda p: "I cannot determine that."
    status, body = _post(conn, "/api/text/scan", {"text": AI_TEXT, "ai_detect": "llm"})
    assert status == 200
    method = body["ai_content"]["methods"][0]
    assert method["name"] == "llm-judge"
    assert method["score"] is None
    assert method["error"]
    assert body["ai_content"]["verdict"] == "unknown"


def test_scan_llm_judge_unreachable_is_soft_error(conn, monkeypatch) -> None:
    dead = LlmProfile(
        id="dead", label="d", backend="openai-compatible", base_url="http://127.0.0.1:9", model="m"
    )
    monkeypatch.setattr(webui_llm, "_REGISTRY", Registry({"dead": dead}, "dead"))
    status, body = _post(conn, "/api/text/scan", {"text": AI_TEXT, "ai_detect": "llm"})
    assert status == 200
    method = body["ai_content"]["methods"][0]
    assert method["score"] is None
    assert method["error"]
    assert body["layer_a"]["suspicious_total"] == 0


def test_scan_validation(conn) -> None:
    for payload in (
        {},
        {"text": ""},
        {"text": 5},
        {"text": "x", "ai_detect": "bogus"},
        {"text": "x", "ai_detect": "llm", "profile": "ghost"},
        {"text": "x", "model": 5},
        {"text": "x" * (webui_api.MAX_TEXT_CHARS + 1)},
    ):
        status, body = _post(conn, "/api/text/scan", payload)
        assert status == 400, payload if len(str(payload)) < 200 else "big"
        assert body["ok"] is False
        assert body["error"]


# ---------------------------------------------------------------- SlopTotal classifier


def test_sloptotal_not_called_without_env(conn, fake: FakeLlm) -> None:
    _status, body = _post(conn, "/api/text/scan", {"text": AI_TEXT})
    assert [m["name"] for m in body["ai_content"]["methods"]] == ["stylometry"]
    assert not [r for r in fake.requests if r["path"] == "/api/analyze"]


def test_sloptotal_included_when_configured(conn, fake: FakeLlm, monkeypatch) -> None:
    monkeypatch.setenv("WATERMARKS_SLOPTOTAL_URL", fake.base_url + "/")
    status, body = _post(conn, "/api/text/scan", {"text": AI_TEXT})
    assert status == 200
    ai = body["ai_content"]
    names = [m["name"] for m in ai["methods"]]
    assert names == ["stylometry", "sloptotal"]
    slop = ai["methods"][1]
    assert slop["score"] == pytest.approx(0.825)
    assert slop["detail"]
    assert "error" not in slop
    sent = next(r for r in fake.requests if r["path"] == "/api/analyze")
    assert sent["body"]["text"].startswith("Moreover")
    # Classifier counts 3x as much as a heuristic in the combined score.
    sty = ai["methods"][0]["score"]
    assert ai["score"] == pytest.approx((3 * 0.825 + sty) / 4, abs=1e-3)


def test_sloptotal_skipped_for_off_and_llm(conn, fake: FakeLlm, monkeypatch) -> None:
    monkeypatch.setenv("WATERMARKS_SLOPTOTAL_URL", fake.base_url)
    for mode in ("off", "llm"):
        _status, body = _post(conn, "/api/text/scan", {"text": AI_TEXT, "ai_detect": mode})
        assert "sloptotal" not in [m["name"] for m in body["ai_content"]["methods"]]
    assert not [r for r in fake.requests if r["path"] == "/api/analyze"]


def test_sloptotal_in_both_mode(conn, fake: FakeLlm, monkeypatch) -> None:
    monkeypatch.setenv("WATERMARKS_SLOPTOTAL_URL", fake.base_url)
    _status, body = _post(conn, "/api/text/scan", {"text": AI_TEXT, "ai_detect": "both"})
    assert [m["name"] for m in body["ai_content"]["methods"]] == [
        "stylometry",
        "llm-judge",
        "sloptotal",
    ]


def test_sloptotal_unreachable_is_soft_error(conn, monkeypatch) -> None:
    monkeypatch.setenv("WATERMARKS_SLOPTOTAL_URL", "http://127.0.0.1:9")
    status, body = _post(conn, "/api/text/scan", {"text": AI_TEXT})
    assert status == 200
    methods = {m["name"]: m for m in body["ai_content"]["methods"]}
    assert methods["sloptotal"]["score"] is None
    assert methods["sloptotal"]["error"]
    assert methods["stylometry"]["score"] is not None
    assert body["ai_content"]["score"] == pytest.approx(methods["stylometry"]["score"], abs=1e-3)


def test_sloptotal_timeout_is_soft_error(conn, fake: FakeLlm, monkeypatch) -> None:
    monkeypatch.setenv("WATERMARKS_SLOPTOTAL_URL", fake.base_url)
    monkeypatch.setattr(webui_api.webui_detect, "SLOPTOTAL_TIMEOUT", 0.3)
    fake.analyze_delay = 1.5
    status, body = _post(conn, "/api/text/scan", {"text": AI_TEXT})
    assert status == 200
    slop = next(m for m in body["ai_content"]["methods"] if m["name"] == "sloptotal")
    assert slop["score"] is None
    assert "timed out" in slop["error"].lower() or "timeout" in slop["error"].lower()


def test_sloptotal_http_error_and_bad_payload(conn, fake: FakeLlm, monkeypatch) -> None:
    monkeypatch.setenv("WATERMARKS_SLOPTOTAL_URL", fake.base_url)
    fake.analyze_status = 429
    fake.analyze_response = {"error": "busy", "retry_after": 2}
    _status, body = _post(conn, "/api/text/scan", {"text": AI_TEXT})
    slop = next(m for m in body["ai_content"]["methods"] if m["name"] == "sloptotal")
    assert slop["score"] is None
    assert "429" in slop["error"]
    fake.analyze_status = 200
    fake.analyze_response = {"unexpected": True}
    _status, body = _post(conn, "/api/text/scan", {"text": AI_TEXT})
    slop = next(m for m in body["ai_content"]["methods"] if m["name"] == "sloptotal")
    assert slop["score"] is None
    assert slop["error"]


def test_sloptotal_invalid_env_url_is_soft_error(conn, monkeypatch) -> None:
    monkeypatch.setenv("WATERMARKS_SLOPTOTAL_URL", "file:///etc/passwd")
    status, body = _post(conn, "/api/text/scan", {"text": AI_TEXT})
    assert status == 200
    slop = next(m for m in body["ai_content"]["methods"] if m["name"] == "sloptotal")
    assert slop["score"] is None
    assert slop["error"]


def test_sloptotal_short_text_not_sent(conn, fake: FakeLlm, monkeypatch) -> None:
    monkeypatch.setenv("WATERMARKS_SLOPTOTAL_URL", fake.base_url)
    _status, body = _post(conn, "/api/text/scan", {"text": "Too short."})
    slop = next(m for m in body["ai_content"]["methods"] if m["name"] == "sloptotal")
    assert slop["score"] is None
    assert not [r for r in fake.requests if r["path"] == "/api/analyze"]


def test_profiles_endpoint_reports_classifier_flag(conn, fake: FakeLlm, monkeypatch) -> None:
    _status, body = _get_json(conn, "/api/llm/profiles")
    assert body["sloptotal_configured"] is False
    monkeypatch.setenv("WATERMARKS_SLOPTOTAL_URL", fake.base_url)
    _status, body = _get_json(conn, "/api/llm/profiles")
    assert body["sloptotal_configured"] is True
    assert fake.base_url not in json.dumps(body)


# ---------------------------------------------------------------- clean


def test_clean_layer_a_only(conn, fake: FakeLlm) -> None:
    text = "hello​ world again"
    status, body = _post(conn, "/api/text/clean", {"text": text, "layer_a": {}})
    assert status == 200, body
    assert body["original"] == text
    assert body["after_layer_a"] == "hello world again"
    assert body["final"] == "hello world again"
    assert body["layer_a"]["removed_count"] == 1
    assert body["layer_a"]["replaced_count"] == 1
    assert sum(body["layer_a"]["removed"].values()) == 1
    assert body["layer_b"]["applied"] is False
    assert body["layer_b"]["reason"]
    assert "".join(s["a"] for s in body["diff_a"]) == text
    assert "".join(s["b"] for s in body["diff_a"]) == body["after_layer_a"]
    assert "".join(s["b"] for s in body["diff_b"]) == body["final"]
    assert "".join(s["a"] for s in body["diff"]) == text
    assert fake.requests == []


def test_clean_layer_a_options(conn) -> None:
    text = "café ＡＢＣ x y"
    _s, body = _post(conn, "/api/text/clean", {"text": text, "layer_a": {"nfkc": True}})
    assert "ABC" in body["after_layer_a"]
    _s, body = _post(
        conn, "/api/text/clean", {"text": text, "layer_a": {"normalize_spaces": False}}
    )
    assert " " in body["after_layer_a"]


def test_clean_layer_a_option_validation(conn) -> None:
    for layer_a in ({"nfkc": "yes"}, {"bogus": True}, [], "x"):
        status, body = _post(conn, "/api/text/clean", {"text": "x", "layer_a": layer_a})
        assert status == 400, layer_a
        assert body["ok"] is False


def test_clean_with_layer_b_default_strategy(conn, fake: FakeLlm) -> None:
    status, body = _post(
        conn,
        "/api/text/clean",
        {
            "text": "Das ist​ ein Test.",
            "layer_a": {},
            "layer_b": {"enabled": True},
            "profile": "remote",
        },
    )
    assert status == 200, body
    assert body["after_layer_a"] == "Das ist ein Test."
    assert body["final"] == "REWRITTEN: Das ist ein Test."
    lb = body["layer_b"]
    assert lb["applied"] is True
    assert lb["strategy"] == "paraphrase@0.8"
    assert lb["profile"] == "remote"
    assert lb["model"] == "qwen/qwen3.8-27b"
    assert lb["steps"][0]["tactic"] == "paraphrase"
    assert "".join(s["a"] for s in body["diff_b"]) == body["after_layer_a"]
    assert "".join(s["b"] for s in body["diff"]) == body["final"]
    # The model never saw the zero-width space (Layer A ran first).
    assert "​" not in fake.prompts()[0]
    assert fake.requests[-1]["headers"]["Authorization"] == "Bearer super-secret-key"


def test_clean_with_style_and_strategy_chain(conn, fake: FakeLlm) -> None:
    _s, body = _post(
        conn,
        "/api/text/clean",
        {
            "text": "Plain input text.",
            "layer_b": {
                "enabled": True,
                "strategy": "paraphrase@0.5,chunk@0.3",
                "style": "terse",
            },
        },
    )
    assert _s == 200, body
    assert [s["tactic"] for s in body["layer_b"]["steps"]] == ["paraphrase", "chunk"]
    prompts = fake.prompts()
    assert len(prompts) == 2
    assert "terse" in prompts[0]
    assert body["final"] == "REWRITTEN: REWRITTEN: Plain input text."


def test_clean_layer_b_errors(conn, fake: FakeLlm, monkeypatch) -> None:
    status, body = _post(
        conn, "/api/text/clean", {"text": "x", "layer_b": {"enabled": True, "strategy": "bogus@1"}}
    )
    assert status == 400
    status, body = _post(
        conn,
        "/api/text/clean",
        {"text": "x", "layer_b": {"enabled": True, "strategy": "paraphrase@2"}},
    )
    assert status == 400
    status, body = _post(
        conn, "/api/text/clean", {"text": "x", "layer_b": {"enabled": True}, "profile": "ghost"}
    )
    assert status == 400
    status, body = _post(conn, "/api/text/clean", {"text": "x", "layer_b": {"enabled": "yes"}})
    assert status == 400
    fake.fail_status = 500
    status, body = _post(conn, "/api/text/clean", {"text": "abc", "layer_b": {"enabled": True}})
    assert status == 502
    assert body["error_code"] == "llm_error"
    assert body["ok"] is False


def test_clean_mlm_requires_transformers(conn, monkeypatch) -> None:
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "transformers" else real(name, *a, **k),
    )
    status, body = _post(
        conn, "/api/text/clean", {"text": "x", "layer_b": {"enabled": True, "strategy": "mlm@0.2"}}
    )
    assert status == 400
    assert "transformers" in body["error"]


def test_clean_text_too_long(conn) -> None:
    status, body = _post(conn, "/api/text/clean", {"text": "x" * 100_001})
    assert status == 400
    assert "100000" in body["error"] or "100_000" in body["error"]


def test_clean_long_text_is_chunked(conn, fake: FakeLlm) -> None:
    paras = [f"Paragraph {i} " + "word " * 120 for i in range(12)]
    text = "\n\n".join(paras)
    status, body = _post(conn, "/api/text/clean", {"text": text, "layer_b": {"enabled": True}})
    assert status == 200, body
    assert body["layer_b"]["chunks"] == len(fake.prompts()) > 1
    assert body["final"].count("REWRITTEN:") == len(fake.prompts())
    assert "\n\n" in body["final"]


# ---------------------------------------------------------------- paraphrase


def test_paraphrase_default(conn, fake: FakeLlm) -> None:
    status, body = _post(conn, "/api/text/paraphrase", {"text": "The cat sat on the mat."})
    assert status == 200, body
    assert body["original"] == "The cat sat on the mat."
    assert body["final"] == "REWRITTEN: The cat sat on the mat."
    assert body["mode"] == "standard"
    assert body["intensity"] == 0.8
    assert body["model"] == "qwen/qwen3.8-27b"
    assert body["profile"] == "remote"
    assert body["noop"] is False
    assert body["chunks"] == 1
    assert "".join(s["a"] for s in body["diff"]) == body["original"]
    assert "".join(s["b"] for s in body["diff"]) == body["final"]
    assert body["layer_a"] == {"removed_count": 0, "replaced_count": 0}
    prompt = fake.prompts()[0]
    assert "0.80" in prompt
    assert "same language" in prompt.lower()
    assert fake.requests[0]["body"]["model"] == "qwen/qwen3.8-27b"


@pytest.mark.parametrize(
    ("mode", "needle"),
    [
        ("fluent", "fluent"),
        ("formal", "formal"),
        ("simple", "simple"),
        ("creative", "creative"),
        ("shorten", "concise"),
        ("expand", "elaborate"),
    ],
)
def test_paraphrase_modes_add_style_clause(conn, fake: FakeLlm, mode: str, needle: str) -> None:
    status, body = _post(
        conn,
        "/api/text/paraphrase",
        {"text": "Some input text here.", "mode": mode, "intensity": 0.4},
    )
    assert status == 200, body
    assert body["mode"] == mode
    assert body["intensity"] == 0.4
    assert needle in fake.prompts()[0].lower()
    assert "0.40" in fake.prompts()[0]


def test_paraphrase_validation(conn) -> None:
    for payload in (
        {"text": "x", "mode": "weird"},
        {"text": "x", "intensity": 1.5},
        {"text": "x", "intensity": -0.1},
        {"text": "x", "intensity": "high"},
        {"text": "x", "intensity": True},
        {"text": "x", "layer_a_after": "yes"},
        {"text": "x", "profile": "ghost"},
        {"text": ""},
        {},
    ):
        status, body = _post(conn, "/api/text/paraphrase", payload)
        assert status == 400, payload
        assert body["ok"] is False


def test_paraphrase_layer_a_after(conn, fake: FakeLlm) -> None:
    fake.responder = lambda p: "Clean​ result here"
    _s, body = _post(conn, "/api/text/paraphrase", {"text": "Original text here."})
    assert body["final"] == "Clean result here"
    assert body["layer_a"] == {"removed_count": 1, "replaced_count": 1}
    _s, body = _post(
        conn, "/api/text/paraphrase", {"text": "Original text here.", "layer_a_after": False}
    )
    assert body["final"] == "Clean​ result here"
    assert body["layer_a"] == {"removed_count": 0, "replaced_count": 0}


def test_paraphrase_strips_think_blocks(conn, fake: FakeLlm) -> None:
    fake.responder = lambda p: "<think>reasoning...</think>\nA fresh sentence."
    _s, body = _post(conn, "/api/text/paraphrase", {"text": "An old sentence."})
    assert body["final"] == "A fresh sentence."
    assert body["noop"] is False


def test_paraphrase_noop_when_identical(conn, fake: FakeLlm) -> None:
    fake.responder = lambda p: "  An old sentence.\n"
    _s, body = _post(conn, "/api/text/paraphrase", {"text": "An old sentence."})
    assert body["noop"] is True
    assert body["final"].strip() == "An old sentence."


def test_paraphrase_noop_when_model_returns_nothing(conn, fake: FakeLlm) -> None:
    fake.responder = lambda p: "<think>all thinking, no answer</think>"
    status, body = _post(conn, "/api/text/paraphrase", {"text": "An old sentence."})
    assert status == 200
    assert body["noop"] is True
    assert body["final"] == "An old sentence."
    assert body["warnings"]


def test_paraphrase_chunks_long_text_sequentially(conn, fake: FakeLlm) -> None:
    paras = [f"Para {i}. " + "alpha beta gamma " * 60 for i in range(10)]
    text = "\n\n".join(paras)
    assert len(text) > 6000
    status, body = _post(conn, "/api/text/paraphrase", {"text": text})
    assert status == 200, body
    n = body["chunks"]
    assert n >= 2
    assert len(fake.prompts()) == n
    assert body["final"].count("REWRITTEN:") == n
    assert body["final"].count("\n\n") == text.count("\n\n")
    for prompt in fake.prompts():
        assert len(prompt.rsplit("\n---\n", 1)[-1]) <= 3000


def test_paraphrase_model_override_and_ollama_profile(conn, fake: FakeLlm) -> None:
    _s, body = _post(
        conn, "/api/text/paraphrase", {"text": "Hallo Welt, wie geht es dir?", "profile": "local"}
    )
    assert _s == 200, body
    assert body["profile"] == "local"
    assert body["model"] == "gemma4:12b"  # lazily resolved from /api/tags
    assert any(r["path"] == "/api/chat" for r in fake.requests)
    _s, body = _post(
        conn,
        "/api/text/paraphrase",
        {"text": "Hallo Welt, wie geht es dir?", "profile": "local", "model": "custom:1b"},
    )
    assert body["model"] == "custom:1b"


def test_paraphrase_llm_errors(conn, fake: FakeLlm, monkeypatch) -> None:
    fake.fail_status = 500
    status, body = _post(conn, "/api/text/paraphrase", {"text": "An old sentence."})
    assert status == 502
    assert body["error_code"] == "llm_error"
    dead = LlmProfile(
        id="dead", label="d", backend="openai-compatible", base_url="http://127.0.0.1:9", model="m"
    )
    monkeypatch.setattr(webui_llm, "_REGISTRY", Registry({"dead": dead}, "dead"))
    status, body = _post(conn, "/api/text/paraphrase", {"text": "An old sentence."})
    assert status == 502
    assert body["error_code"] == "llm_unreachable"


def test_paraphrase_503_without_profiles(conn, monkeypatch) -> None:
    monkeypatch.setattr(webui_llm, "_REGISTRY", Registry({}, None))
    status, body = _post(conn, "/api/text/paraphrase", {"text": "x"})
    assert status == 503
    assert body["ok"] is False


def test_api_post_invalid_body(conn) -> None:
    conn.request("POST", "/api/text/scan", body=b"not json", headers={"Content-Length": "8"})
    resp = conn.getresponse()
    resp.read()
    assert resp.status == 400


# ---------------------------------------------------------------- static + auth


@pytest.fixture
def webdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "web"
    d.mkdir()
    (d / "index.html").write_text("<!doctype html><title>UI</title>", encoding="utf-8")
    (d / "app.js").write_text("console.log(1)", encoding="utf-8")
    (d / "style.css").write_text("body{}", encoding="utf-8")
    (d / "logo.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
    (d / "secret.py").write_text("x=1", encoding="utf-8")
    (d / "sub").mkdir()
    (d / "sub" / "nested.js").write_text("1", encoding="utf-8")
    (tmp_path / "outside.js").write_text("leak", encoding="utf-8")
    monkeypatch.setattr(webui_api, "WEB_DIR", d)
    return d


def test_static_index_and_assets(conn, webdir) -> None:
    status, resp, data = _get(conn, "/")
    assert status == 200
    assert b"<title>UI</title>" in data
    assert resp.getheader("Content-Type").startswith("text/html")
    assert resp.getheader("Cache-Control") == "no-store"
    assert resp.getheader("X-Content-Type-Options") == "nosniff"
    csp = resp.getheader("Content-Security-Policy")
    assert "default-src 'self'" in csp
    assert "style-src 'self' 'unsafe-inline'" in csp
    assert "img-src 'self' data:" in csp
    for path, ctype in (
        ("/ui/app.js", "javascript"),
        ("/ui/style.css", "text/css"),
        ("/ui/logo.svg", "image/svg+xml"),
        ("/ui/index.html", "text/html"),
    ):
        status, resp, _data = _get(conn, path)
        assert status == 200, path
        assert ctype in resp.getheader("Content-Type")
        assert resp.getheader("X-Content-Type-Options") == "nosniff"


@pytest.mark.parametrize(
    "path",
    [
        "/ui/secret.py",
        "/ui/missing.js",
        "/ui/sub/nested.js",
        "/ui/sub",
        "/ui/../outside.js",
        "/ui/%2e%2e/outside.js",
        "/ui/..%2foutside.js",
        "/ui/.hidden",
        "/ui/",
        "/ui",
        "/ui/app.js%00.png",
    ],
)
def test_static_whitelist_and_traversal(conn, webdir, path: str) -> None:
    status, _resp, data = _get(conn, path)
    assert status == 404, path
    assert b"leak" not in data


def test_static_symlink_escape_is_blocked(conn, webdir, tmp_path) -> None:
    link = webdir / "evil.js"
    try:
        link.symlink_to(tmp_path / "outside.js")
    except OSError:
        pytest.skip("symlinks unavailable")
    status, _resp, data = _get(conn, "/ui/evil.js")
    assert status == 404
    assert b"leak" not in data


def test_placeholder_when_no_index(conn, tmp_path, monkeypatch) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(webui_api, "WEB_DIR", empty)
    status, resp, data = _get(conn, "/")
    assert status == 200
    assert resp.getheader("Content-Type").startswith("text/html")
    assert b"Web UI" in data


def test_static_exempt_from_bearer_but_api_protected(conn, webdir, monkeypatch) -> None:
    monkeypatch.setattr(server, "API_KEY", "sekret")
    status, _resp, _data = _get(conn, "/")
    assert status == 200
    status, _resp, _data = _get(conn, "/ui/app.js")
    assert status == 200
    for path in ("/api/llm/profiles", "/api/llm/models", "/health", "/capabilities"):
        status, _resp, _data = _get(conn, path)
        assert status == 401, path
    status, _body = _post(conn, "/api/text/scan", {"text": "x"})
    assert status == 401
    status, _resp, _data = _get(
        conn, "/api/llm/profiles", headers={"Authorization": "Bearer sekret"}
    )
    assert status == 200


def test_openapi_includes_web_api(conn) -> None:
    status, body = _get_json(conn, "/openapi.json")
    assert status == 200
    paths = body["paths"]
    assert "get" in paths["/api/llm/profiles"]
    assert "get" in paths["/api/llm/models"]
    for p in ("/api/text/scan", "/api/text/clean", "/api/text/paraphrase"):
        post = paths[p]["post"]
        assert post["requestBody"]["required"] is True
        assert "502" in post["responses"] or p == "/api/text/scan"
    assert "503" in paths["/api/text/paraphrase"]["post"]["responses"]


def test_openapi_spec_validates() -> None:
    validator = pytest.importorskip("openapi_spec_validator")
    validator.validate(server.openapi_spec())
