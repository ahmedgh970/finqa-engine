"""Tests for the Ollama client: a real call (needs a running Ollama) plus an
offline check that the request payload is shaped right."""

import pytest
import requests

from src.llm.client import generate
from src.llm.config import LLMConfig


def _ollama_up() -> bool:
    try:
        requests.get("http://localhost:11434/api/tags", timeout=2)
        return True
    except requests.exceptions.RequestException:
        return False


@pytest.mark.eval
@pytest.mark.skipif(not _ollama_up(), reason="no local Ollama running")
def test_generate_returns_string():
    answer = generate("Reply with exactly: OK", LLMConfig())
    assert isinstance(answer, str)


class _FakeResp:
    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": "ok"}}


def _capture_post(monkeypatch) -> dict:
    """Patch requests.post to record the JSON body and return a canned reply."""
    from src.llm import client

    captured: dict = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured.update(json or {})
        return _FakeResp()

    monkeypatch.setattr(client.requests, "post", fake_post)
    return captured


def test_generate_strips_prefix_and_disables_thinking(monkeypatch):
    # The ollama_chat/ prefix (repo naming convention) is stripped before the
    # call, and thinking is disabled so a model's reasoning can't exhaust
    # num_predict and return empty content.
    captured = _capture_post(monkeypatch)
    generate("hi", LLMConfig(model="ollama_chat/qwen3.5:4b"))
    assert captured["model"] == "qwen3.5:4b"
    assert captured["think"] is False
    assert captured["options"]["temperature"] == 0.0
    assert captured["url"].endswith("/api/chat")


def test_generate_sends_a_system_turn_only_when_asked(monkeypatch):
    captured = _capture_post(monkeypatch)
    generate("hi", LLMConfig())
    assert [m["role"] for m in captured["messages"]] == ["user"]

    captured = _capture_post(monkeypatch)
    generate("hi", LLMConfig(), system="You are a judge.")
    assert captured["messages"][0] == {"role": "system", "content": "You are a judge."}
    assert captured["messages"][1] == {"role": "user", "content": "hi"}


def test_a_completion_says_when_the_output_budget_cut_it(monkeypatch):
    """Measured: a 4B model reasoned aloud past max_tokens and stopped mid-step; the
    text alone read like an answer that went nowhere."""
    from src.llm import client

    def reply(done_reason):
        class Resp(_FakeResp):
            def json(self):
                return {"message": {"content": "Step 1: ..."}, "done_reason": done_reason}

        monkeypatch.setattr(client.requests, "post", lambda url, json=None, timeout=None: Resp())
        return generate("hi", LLMConfig())

    cut = reply("length")
    assert cut == "Step 1: ..." and isinstance(cut, str) and cut.truncated
    assert not reply("stop").truncated


def test_a_stoppable_call_streams_and_gives_up_when_told(monkeypatch):
    """The demo's stop button: the call in progress is abandoned, the next never made."""
    import json as _json
    import threading

    from src.llm import client

    stop = threading.Event()
    closed = []

    class Streamed:
        def __init__(self, lines):
            self.lines = lines

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            closed.append(True)

        def raise_for_status(self):
            pass

        def iter_lines(self):
            for line in self.lines:
                yield _json.dumps(line).encode()
                stop.set()  # the user stops the run while tokens arrive

    lines = [{"message": {"content": "Step"}, "done": False}, {"done": True, "done_reason": "stop"}]
    monkeypatch.setattr(client.requests, "post", lambda *a, **kw: Streamed(lines))
    with client.stoppable(stop):
        with pytest.raises(client.Cancelled):
            generate("hi", LLMConfig())
        assert closed  # the connection is closed, which makes Ollama stop
        with pytest.raises(client.Cancelled):
            generate("next call", LLMConfig())  # never sent


def test_a_stoppable_call_returns_the_streamed_text_and_reason(monkeypatch):
    import json as _json
    import threading

    from src.llm import client

    class Streamed:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def raise_for_status(self):
            pass

        def iter_lines(self):
            yield _json.dumps({"message": {"content": "1."}, "done": False}).encode()
            yield _json.dumps({"message": {"content": "33"}, "done": False}).encode()
            yield _json.dumps(
                {"message": {"content": ""}, "done": True, "done_reason": "length"}
            ).encode()

    monkeypatch.setattr(client.requests, "post", lambda *a, **kw: Streamed())
    with client.stoppable(threading.Event()):
        answer = generate("hi", LLMConfig())
    assert answer == "1.33" and answer.truncated
