"""In-process fake LLM server for the web UI tests (no real network).

Mimics the two wire formats the service talks to:

* LM Studio / OpenAI-compatible: ``GET /v1/models`` and ``POST /v1/chat/completions``
* Ollama: ``GET /api/tags`` and ``POST /api/chat``
* SlopTotal classifier sidecar: ``POST /api/analyze`` (body ``{"text": ...}``)

Every request is recorded on ``FakeLlm.requests`` so tests can assert on the
path, headers and decoded JSON body. ``responder(prompt) -> str`` produces the
model reply; ``fail_status`` makes chat endpoints return that HTTP status.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

OPENAI_MODELS = ["text-embedding-nomic-embed-text-v1.5", "qwen/qwen3.8-27b", "gemma-fake"]
OLLAMA_MODELS = ["nomic-embed-text:latest", "gemma4:12b"]


def default_responder(prompt: str) -> str:
    """Judge prompts get strict JSON; rewrite prompts get a deterministic rewrite."""
    if "ai_probability" in prompt:
        return json.dumps(
            {
                "ai_probability": 0.9,
                "reasons": ["uniform cadence", "stock transitions"],
                "suspect_sentences": ["Moreover, it is important to note this."],
            }
        )
    body = prompt.rsplit("\n---\n", 1)[-1]
    return "REWRITTEN: " + body.strip()


class FakeLlm:
    """Threaded fake server; use as a context manager or via ``start()``/``stop()``."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.responder: Callable[[str], str] = default_responder
        self.fail_status: int | None = None
        self.openai_models: list[str] = list(OPENAI_MODELS)
        self.ollama_models: list[str] = list(OLLAMA_MODELS)
        # SlopTotal sidecar (POST /api/analyze) emulation.
        self.analyze_status: int = 200
        self.analyze_response: dict[str, Any] = {
            "overall_score": 82.5,
            "overall_verdict": "Slop detected",
            "engines_flagged": 17,
            "engines_total": 23,
            "word_count": 120,
            "language": "en",
            "language_support": "supported",
        }
        self.analyze_delay: float = 0.0
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        assert self._server is not None
        return self._server.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def prompts(self) -> list[str]:
        """The user prompt of every chat request, in arrival order."""
        out = []
        for req in self.requests:
            if req["method"] == "POST" and req["path"] != "/api/analyze":
                out.append(req["body"]["messages"][0]["content"])
        return out

    def start(self) -> FakeLlm:
        fake = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: object) -> None:
                pass

            def _send(self, status: int, payload: dict[str, Any]) -> None:
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                fake.requests.append(
                    {"method": "GET", "path": self.path, "headers": dict(self.headers)}
                )
                if self.path == "/v1/models":
                    self._send(
                        200,
                        {"object": "list", "data": [{"id": m} for m in fake.openai_models]},
                    )
                elif self.path == "/api/tags":
                    self._send(200, {"models": [{"name": m} for m in fake.ollama_models]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                fake.requests.append(
                    {
                        "method": "POST",
                        "path": self.path,
                        "headers": dict(self.headers),
                        "body": body,
                    }
                )
                if self.path == "/api/analyze":
                    if fake.analyze_delay:
                        time.sleep(fake.analyze_delay)
                    self._send(fake.analyze_status, fake.analyze_response)
                    return
                if fake.fail_status is not None:
                    self._send(fake.fail_status, {"error": {"message": "model exploded"}})
                    return
                prompt = body["messages"][0]["content"]
                reply = fake.responder(prompt)
                if self.path == "/v1/chat/completions":
                    self._send(
                        200, {"choices": [{"message": {"role": "assistant", "content": reply}}]}
                    )
                elif self.path == "/api/chat":
                    self._send(200, {"message": {"role": "assistant", "content": reply}})
                else:
                    self._send(404, {"error": "not found"})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self) -> FakeLlm:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
