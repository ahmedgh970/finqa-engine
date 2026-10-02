"""Thin Ollama client: one function to generate a completion locally.

The whole pipeline runs on local Ollama models (generation, judging, serving),
so this talks to Ollama's native ``/api/chat`` endpoint directly -- no provider
abstraction, no per-minute token guard (there is no quota locally). Transient
transport failures (connection dropped, timeout, 5xx) are retried with backoff;
a bad request fails immediately since retrying can't help.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
import threading
from collections.abc import Iterator
from typing import TypeVar

import requests
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.llm.config import LLMConfig

T = TypeVar("T", bound=BaseModel)

# Calls go to Ollama over plain HTTP, which the LangChain instrumentation never sees, so
# each one opens its own LLM span. Without a registered tracer provider (tracing off,
# the default) the OpenTelemetry API hands out no-op spans; without the library at all
# there is nothing to open.
try:
    from opentelemetry import trace as _otel_trace
except ImportError:  # tracing extra not installed
    _otel_trace = None
# The graph node making the call is a LangChain span that the instrumentation does not
# make the current OpenTelemetry context; asked for explicitly, it parents the LLM span,
# so a question's calls sit under their node instead of in traces of their own.
try:
    from openinference.instrumentation.langchain import get_current_span as _node_span
except ImportError:
    _node_span = None

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Upper bound on the auto-sized context window: a bigger KV cache than this
# would not fit an 8 GB GPU and would spill to CPU.
NUM_CTX_CAP = 32768

# The stop signal of the run the current call belongs to, when a caller made it stoppable.
# A context variable, so it follows the call into the threads a graph runs its nodes in.
_stop: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar(
    "ollama_stop", default=None
)


class Cancelled(Exception):
    """The run this call belongs to was stopped; nothing should be retried or reported."""


@contextlib.contextmanager
def stoppable(stop: threading.Event) -> Iterator[None]:
    """Make the calls made inside this block stop as soon as ``stop`` is set.

    The call then streams from Ollama and closes the connection when the signal comes,
    which makes Ollama abandon the generation; the next call is not made at all. Outside
    such a block nothing changes: one request, one complete answer.
    """
    token = _stop.set(stop)
    try:
        yield
    finally:
        _stop.reset(token)


# Transport-level failures worth retrying; a 4xx (bad request) is not among them.
_TRANSIENT_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def _model_name(model: str) -> str:
    """Strip a ``provider/`` prefix, keeping the Ollama model name.

    Configs still name models ``ollama_chat/llama3.1:8b`` (the prefix drives the
    output-file naming across the repo); Ollama itself wants just ``llama3.1:8b``.
    """
    return model.split("/", 1)[1] if "/" in model else model


def estimated_context(prompt: str, num_predict: int) -> int:
    """Context a prompt needs: ~len/3 tokens, plus the output budget and a margin.

    len/3 is deliberately conservative for number-dense financial text, which
    tokenizes denser than prose. Shared with the workflow, which uses it to decide
    how many passages fit a pinned context rather than letting Ollama truncate.
    """
    return len(prompt) // 3 + num_predict + 256


def _auto_num_ctx(prompt: str, num_predict: int) -> int:
    """Size the context window to the prompt so retrieved chunks aren't truncated."""
    rounded = ((estimated_context(prompt, num_predict) + 511) // 512) * 512
    return min(NUM_CTX_CAP, max(2048, rounded))


class Completion(str):
    """Generated text that also says why generation stopped.

    A ``str``, so every caller keeps treating a completion as text. ``truncated`` is
    for the caller that must know the output budget ran out before the answer did:
    the text then ends mid-sentence, which reads like an answer that went nowhere.
    """

    done_reason: str | None = None

    @property
    def truncated(self) -> bool:
        return self.done_reason == "length"


def _completion(content: str, done_reason: str | None) -> Completion:
    completion = Completion(content)
    completion.done_reason = done_reason
    return completion


@retry(
    retry=retry_if_exception_type(_TRANSIENT_ERRORS),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=20),
)
def generate(
    prompt: str, config: LLMConfig, schema: dict | None = None, system: str | None = None
) -> Completion:
    """Generate a completion for ``prompt`` on a local Ollama model.

    ``think: False`` disables the reasoning channel on thinking-capable models
    (qwen3.5, gemma4): left on, their chain-of-thought is emitted first and can
    exhaust ``num_predict`` before any answer token, returning empty content.
    Plain-instruct models ignore the flag, so it is always safe to send.

    ``schema`` is a JSON Schema passed to Ollama's ``format`` field: decoding is then
    constrained to emit a matching JSON object. Use :func:`generate_structured` for
    the typed version.

    ``system``, when given, is sent as a system turn before the prompt -- for models
    trained on a fixed system prompt, such as the Prometheus judge.

    The text comes back as a :class:`Completion`, carrying Ollama's ``done_reason``.
    """
    num_ctx = config.num_ctx or _auto_num_ctx(prompt, config.max_tokens)
    messages = [{"role": "user", "content": prompt}]
    if system is not None:
        messages.insert(0, {"role": "system", "content": system})
    payload: dict = {
        "model": _model_name(config.model),
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": config.temperature,
            "num_predict": config.max_tokens,
            "num_ctx": num_ctx,
        },
    }
    if schema is not None:
        payload["format"] = schema
    stop = _stop.get()
    with _llm_span(payload) as span:
        if stop is None:
            response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=1800)
            response.raise_for_status()
            body = response.json()
            content = body.get("message", {}).get("content", "")
        else:
            content, body = _chat_until_stopped(payload, stop)
        if span is not None:
            _record_output(span, body, content)
    return _completion(content, body.get("done_reason"))


def _chat_until_stopped(payload: dict, stop: threading.Event) -> tuple[str, dict]:
    """Stream one chat call, giving up the moment ``stop`` is set.

    Returns the text and Ollama's closing message, which carries ``done_reason`` and the
    token counts. Closing the connection is what makes Ollama stop generating.
    """
    if stop.is_set():
        raise Cancelled
    parts: list[str] = []
    with requests.post(
        f"{OLLAMA_URL}/api/chat", json={**payload, "stream": True}, stream=True, timeout=1800
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if stop.is_set():
                raise Cancelled
            if not line:
                continue
            message = json.loads(line)
            parts.append(message.get("message", {}).get("content", ""))
            if message.get("done"):
                return "".join(parts), message
    return "".join(parts), {}


def _llm_span(payload: dict):
    """An OpenInference LLM span around one Ollama call, or a no-op context."""
    if _otel_trace is None:
        return contextlib.nullcontext()
    return _otel_trace.get_tracer(__name__).start_as_current_span(
        "ollama.chat",
        context=_parent_context(),
        attributes={
            "openinference.span.kind": "LLM",
            "llm.model_name": payload["model"],
            "llm.invocation_parameters": json.dumps(payload["options"]),
            "input.value": payload["messages"][-1]["content"],
        },
    )


def _parent_context():
    """The OpenTelemetry context of the graph node calling, if it is traced."""
    if _node_span is None:
        return None
    try:
        parent = _node_span()
    except Exception:  # outside a traced LangChain run
        return None
    return _otel_trace.set_span_in_context(parent) if parent is not None else None


def _record_output(span, body: dict, content: str) -> None:
    """The answer and Ollama's own token counts, on the call's span."""
    prompt_tokens = body.get("prompt_eval_count", 0)
    completion_tokens = body.get("eval_count", 0)
    span.set_attribute("output.value", content)
    span.set_attribute("llm.token_count.prompt", prompt_tokens)
    span.set_attribute("llm.token_count.completion", completion_tokens)
    span.set_attribute("llm.token_count.total", prompt_tokens + completion_tokens)


class StructuredOutputError(ValueError):
    """Ollama returned something that does not validate against the requested schema.

    Constrained decoding makes this rare, but it stays possible (an empty response
    when the output budget runs out, for instance), so callers must handle it rather
    than assume a valid object.
    """


def generate_structured(prompt: str, config: LLMConfig, model: type[T]) -> T:
    """Generate a JSON object constrained to ``model``'s schema and validate it.

    Note the trade-off of constrained decoding: the model is forced to emit a valid
    object, so abstention has to be *representable in the schema* (a list of booleans
    that can be all-false, say) rather than expressed by refusing to answer.
    """
    raw = generate(prompt, config, schema=model.model_json_schema())
    try:
        return model.model_validate_json(raw)
    except ValidationError as e:
        raise StructuredOutputError(f"invalid structured output: {raw[:300]!r}") from e
