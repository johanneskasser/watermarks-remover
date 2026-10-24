"""`/clean` option layer_a_only: Layer A only, no server-side Layer B rewrite."""

from __future__ import annotations

import base64

import pytest
import server


def test_layer_a_only_skips_layer_b_even_without_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "_DEFAULT_STRATEGY", None)

    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("Layer B must not run with layer_a_only")

    monkeypatch.setattr(server, "_apply_layer_b", boom)
    options = server._parse_clean_options({"layer_a_only": True})
    payload = server._clean_payload("zero​width".encode(), "note.txt", options)
    assert base64.b64decode(payload["cleaned"]) == b"zerowidth"
    assert payload["report"]["layer_b"]["applied"] is False


def test_layer_a_only_must_be_boolean() -> None:
    with pytest.raises(ValueError):
        server._parse_clean_options({"layer_a_only": "yes"})
