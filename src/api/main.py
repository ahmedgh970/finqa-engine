"""FastAPI serving app: one grounded answer per request, from the advanced RAG workflow.

The served pipeline is the workflow graph of ``WORKFLOW_CONFIG`` (default: grading,
neighbour expansion and the verified calculator over reranked(dense), ADR 0005).
``pipeline: "naive"`` runs the same graph with every node switched off -- retrieve then
generate, the baseline the workflow is measured against -- so the two can be compared
on one question through one code path.

The heavy models (BGE-M3 embedder, cross-encoder reranker) and the expansion corpus
are loaded once at startup; each graph is compiled once per (pipeline, model, k) and
reused. The collection is fixed by the config: expansion reads the chunk file the
collection was built from, so the two cannot be switched apart per request.

Tracing to a local Phoenix collector is switched on with ``TRACING=1`` (needs the
``agents`` extra): one trace per question, a span per graph node and per LLM call.
Everything is local (Ollama); no external provider.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.api.demo import (
    LiveRuns,
    Replay,
    RunSettings,
    available_chunk_sizes,
    report_dense_stage,
    stream_run,
)
from src.llm.client import _model_name
from src.retrieval.registry import build_retriever
from src.workflow.config import WorkflowConfig, load_workflow_config
from src.workflow.graph import build_graph, run_graph

WORKFLOW_CONFIG = os.getenv("WORKFLOW_CONFIG", "configs/workflow/serve_grade_exp1_calc_1024.yaml")
REPLAY_CONFIG = os.getenv("REPLAY_CONFIG", "configs/demo/replay.yaml")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
PHOENIX_URL = os.getenv("PHOENIX_URL", "http://localhost:6006")
TRACING = os.getenv("TRACING", "").lower() in ("1", "true", "yes")
STATIC_DIR = Path(__file__).parent / "static"

# Ollama models that aren't chat generators (embedders, judges, benchmark variants).
_NON_GENERATORS = ("bge", "prometheus", "ragas-critic", "genbench", "embed", "nomic")

Pipeline = Literal["workflow", "naive"]

log = logging.getLogger(__name__)
_state: dict = {}
_graphs: dict[tuple[str, str, int], object] = {}  # (pipeline, model, k) -> compiled graph


def _config_for(pipeline: Pipeline, model: str | None, k: int | None) -> WorkflowConfig:
    """The served config, with the request's overrides and, for naive, every node off."""
    cfg: WorkflowConfig = _state["cfg"]
    update: dict = {"k": k or cfg.k}
    if model is not None:
        update["llm"] = cfg.llm.model_copy(update={"model": model})
    if pipeline == "naive":
        update["grading"] = cfg.grading.model_copy(update={"enabled": False})
        update["expansion"] = cfg.expansion.model_copy(update={"enabled": False})
        update["calculator"] = cfg.calculator.model_copy(update={"enabled": False})
    return cfg.model_copy(update=update)


def _get_graph(cfg: WorkflowConfig, pipeline: Pipeline):
    """Get (or compile once and cache) the graph for this pipeline, model and depth."""
    key = (pipeline, cfg.llm.model, cfg.k)
    if key not in _graphs:
        _graphs[key] = build_graph(_state["retriever"], cfg)
    return _graphs[key]


def _setup_tracing() -> None:
    """Export traces to the local Phoenix collector, or say why it is not possible."""
    try:
        from src.tracing import setup_tracing

        setup_tracing(project_name="finqa-api", batch=True)
    except ImportError:
        log.warning("TRACING=1 but the Phoenix libraries are missing: uv sync --extra agents")


class _GpuReleasing:
    """A retriever that hands back the GPU memory its search cached.

    The embedder and the reranker share one GPU with the generator. PyTorch keeps the
    activations of a rerank pass cached -- gigabytes for fifty long passages -- and
    Ollama, which places the model by the memory free when it loads it, would then
    push layers to the CPU. Emptying the cache after each search leaves only the
    weights resident.
    """

    def __init__(self, inner):
        self.inner = inner

    def retrieve(self, query: str, k: int = 5, doc_id: str | None = None):
        try:
            return self.inner.retrieve(query, k=k, doc_id=doc_id)
        finally:
            _release_gpu_cache()


def _release_gpu_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _warm_up(retriever) -> None:
    """Load the embedder and the reranker now, not on the first question (~20 s)."""
    try:
        retriever.retrieve("warm-up", k=1)
    except Exception as exc:  # Qdrant down at startup must not stop the API
        log.warning("retrieval warm-up skipped: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if TRACING:
        _setup_tracing()
    cfg = load_workflow_config(WORKFLOW_CONFIG)
    _state["cfg"] = cfg
    # Loads the embedder and the reranker once; every graph shares this retriever.
    _state["retriever"] = _GpuReleasing(report_dense_stage(build_retriever(cfg.retriever, cfg)))
    _retrievers[(cfg.retriever, cfg.collection_name, cfg.rerank_prefetch)] = _state["retriever"]
    _get_graph(_config_for("workflow", None, None), "workflow")  # warm the default
    _warm_up(_state["retriever"])
    yield
    _state.clear()
    _graphs.clear()
    _retrievers.clear()
    _demo_graphs.clear()


app = FastAPI(title="FinQA Engine", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --- demo: the UI, runs streamed node by node, recorded runs replayed --------------------

_retrievers: dict[tuple[str, str, int], object] = {}  # (retriever, collection, prefetch)
_demo_graphs: dict[str, object] = {}  # settings JSON -> compiled graph
_live_runs = LiveRuns()


def _retriever_for(cfg: WorkflowConfig):
    key = (cfg.retriever, cfg.collection_name, cfg.rerank_prefetch)
    if key not in _retrievers:
        _retrievers[key] = _GpuReleasing(report_dense_stage(build_retriever(cfg.retriever, cfg)))
    return _retrievers[key]


def _graph_for(settings: RunSettings):
    key = settings.model_dump_json()
    if key not in _demo_graphs:
        cfg = settings.config(_state["cfg"])
        _demo_graphs[key] = build_graph(_retriever_for(cfg), cfg)
    return _demo_graphs[key]


def _replay() -> Replay | None:
    if "replay" not in _state:
        _state["replay"] = Replay.load(REPLAY_CONFIG, _state["cfg"].golden_set_path)
    return _state["replay"]


def _sse(events) -> StreamingResponse:
    """Server-sent events, one JSON object per event; a failure becomes an event too."""

    def encode():
        try:
            for event in events:
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:  # shown in the UI rather than a dropped connection
            log.exception("demo run failed")
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        encode(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


def _qdrant_collections() -> list[str]:
    try:
        resp = requests.get(f"{_state['cfg'].qdrant_location}/collections", timeout=10)
        resp.raise_for_status()
        return [c["name"] for c in resp.json()["result"]["collections"]]
    except (requests.exceptions.RequestException, KeyError):
        return []


def _ollama_loaded(model: str) -> dict | None:
    """Where Ollama holds ``model`` right now: its share on GPU, from ``/api/ps``.

    Ollama silently falls back to the CPU when it misses the GPU at startup, which
    turns a question of minutes into one of tens of minutes; the UI says so up front.
    """
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/ps", timeout=5)
        resp.raise_for_status()
    except requests.exceptions.RequestException:
        return None
    for m in resp.json().get("models", []):
        if m["name"].removesuffix(":latest") == model and m.get("size"):
            share = round(100 * m.get("size_vram", 0) / m["size"])
            return {"loaded": model, "gpu_share": share, "on_cpu": share == 0}
    return None


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/demo/options")
def demo_options() -> dict:
    """What the UI offers: models, indexed chunk sizes, defaults, replay and tracing."""
    replay = _replay()
    defaults = RunSettings.from_config(_state["cfg"])
    return {
        "models": _ollama_models(),
        "ollama": _ollama_loaded(defaults.model),
        "chunk_sizes": available_chunk_sizes(_qdrant_collections()),
        "defaults": defaults.model_dump(),
        "replay": {"label": replay.sources.label, "settings": replay.sources.settings.model_dump()}
        if replay
        else None,
        "tracing": TRACING,
        "phoenix_url": PHOENIX_URL,
    }


@app.get("/demo/examples")
def demo_examples() -> list[dict]:
    """The golden-set questions, with what the recorded run made of each."""
    replay = _replay()
    return replay.examples() if replay else []


class RunRequest(BaseModel):
    question: str
    doc_id: str | None = None
    settings: RunSettings = RunSettings()


@app.post("/demo/run")
async def demo_run(req: RunRequest) -> StreamingResponse:
    """Run the workflow live with the UI's settings, streaming each node as it finishes.

    A new run stops the one before it. A client that disconnects -- the UI's stop button
    -- stops its run: the LLM call in progress is abandoned, the next one never made.
    """

    def events():
        yield {"type": "plan", "nodes": req.settings.plan(), "settings": req.settings.model_dump()}
        yield from stream_run(_graph_for(req.settings), req.question, req.doc_id)

    run = await asyncio.to_thread(_live_runs.start, events)

    async def encode():
        try:
            while True:
                try:
                    event = await asyncio.to_thread(run.queue.get, True, 0.5)
                except queue.Empty:
                    continue
                if event is None:
                    break
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            # Reached when the run ends and when the client leaves: either way nobody
            # reads this run any more.
            run.stop.set()

    return StreamingResponse(
        encode(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


@app.get("/demo/replay/{qa_id}")
def demo_replay(qa_id: str, pace: float = 1.0) -> StreamingResponse:
    """A recorded run of one golden-set question, streamed as the events of a live one."""
    replay = _replay()
    if replay is None or qa_id not in replay.answers:
        raise HTTPException(status_code=404, detail=f"no recorded run for {qa_id}")
    return _sse(replay.events(qa_id, pace=pace))


class AskRequest(BaseModel):
    question: str
    doc_id: str | None = None  # scope retrieval to one filing; None = search all
    pipeline: Pipeline = "workflow"
    model: str | None = None  # Ollama model name; None = served default
    k: int | None = None  # passages retrieved (and graded); None = served default


class Source(BaseModel):
    doc_id: str
    page: int
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_s: float
    pipeline: Pipeline
    model: str
    k: int
    collection: str
    # What the workflow did, for the UI and for debugging a single answer.
    llm_calls: int
    n_retrieved: int
    n_kept_by_grade: int
    n_dropped_to_fit: int
    computed: str | None  # the figure the calculator verified, if it ran and accepted
    calc_error: str | None  # why it declined, if it ran and refused
    node_latencies: dict[str, float]


def _ollama_models() -> list[str]:
    """Locally available chat generators (non-generators filtered out)."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException:
        return []
    out = []
    for m in resp.json().get("models", []):
        name = m["name"].removesuffix(":latest")
        if "/" in name or any(bad in name.lower() for bad in _NON_GENERATORS):
            continue
        out.append(name)
    return sorted(out)


@app.get("/health")
def health() -> dict:
    """Liveness + which config is being served."""
    cfg = _state.get("cfg")
    return {"status": "ok" if cfg else "loading", "config": WORKFLOW_CONFIG}


@app.get("/options")
def options() -> dict:
    """Choices for the UI: available models, pipelines, and the defaults."""
    cfg: WorkflowConfig = _state["cfg"]
    return {
        "models": _ollama_models(),
        "pipelines": ["workflow", "naive"],
        "defaults": {
            "model": _model_name(cfg.llm.model),
            "k": cfg.k,
            "pipeline": "workflow",
            "collection": cfg.collection_name,
        },
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Run the served graph on one question: the answer, its sources and what it did."""
    cfg = _config_for(req.pipeline, req.model, req.k)
    result = run_graph(_get_graph(cfg, req.pipeline), req.question, req.doc_id)
    return AskResponse(
        answer=result.answer,
        sources=[Source(doc_id=c.doc_id, page=c.page, text=c.text[:500]) for c in result.sources],
        latency_s=result.latency_s,
        pipeline=req.pipeline,
        model=_model_name(cfg.llm.model),
        k=cfg.k,
        collection=cfg.collection_name,
        llm_calls=result.llm_calls,
        n_retrieved=result.n_retrieved,
        n_kept_by_grade=result.n_kept_by_grade,
        n_dropped_to_fit=result.n_dropped_to_fit,
        computed=result.computed,
        calc_error=result.calc_error,
        node_latencies=result.node_latencies,
    )
