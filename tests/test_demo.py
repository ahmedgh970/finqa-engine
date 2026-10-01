"""Tests for the demo surface: settings as configs, a run as events, the stream endpoint.

No GPU, no Qdrant, no Ollama: the graph runs on a fake retriever and a stubbed LLM.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from src.api import main
from src.api.demo import RunSettings, stream_run
from src.ingestion.schema import Chunk
from src.retrieval.base import ScoredChunk
from src.workflow.config import WorkflowConfig, load_workflow_config
from src.workflow.graph import build_graph
from src.workflow.schemas import ChunkGrade

SERVED = "configs/workflow/serve_grade_exp1_calc_1024.yaml"


class FakeRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

    def retrieve(self, query: str, k: int = 5, doc_id: str | None = None):
        return [ScoredChunk(chunk=c, score=1.0 - i / 10) for i, c in enumerate(self.chunks[:k])]


def _chunks(n: int) -> list[Chunk]:
    return [Chunk(chunk_id=f"D::{i}", doc_id="D", page=i, text=f"passage {i}") for i in range(n)]


def test_the_served_config_reads_back_as_its_settings():
    served = load_workflow_config(SERVED)
    settings = RunSettings.from_config(served)
    assert settings == RunSettings()  # the UI's defaults are the served workflow
    rebuilt = settings.config(served)
    assert rebuilt.model_dump() == served.model_dump()


def test_settings_switch_the_corpus_and_the_nodes_together():
    served = load_workflow_config(SERVED)
    cfg = RunSettings(chunk_size=256, reranker=False, grading=False, calculator=False).config(
        served
    )
    assert cfg.chunks_path.endswith("chunks_256.jsonl") and "256" in cfg.collection_name
    assert cfg.retriever == "dense"
    assert not cfg.grading.enabled and cfg.expansion.enabled and not cfg.calculator.enabled
    assert cfg.llm.model == "ollama_chat/granite4.1:8b"


def test_the_plan_lists_the_switched_on_nodes_in_order():
    assert RunSettings().plan() == ["retrieve", "grade", "expand", "route", "calculate", "generate"]
    naive = RunSettings(grading=False, expansion=False, calculator=False)
    assert naive.plan() == ["retrieve", "generate"]


def test_a_run_streams_each_grade_then_each_node(monkeypatch):
    grades = iter([3, 0, 2])
    monkeypatch.setattr(
        "src.workflow.nodes.generate_structured",
        lambda prompt, config, model: ChunkGrade(grade=next(grades)),
    )
    monkeypatch.setattr("src.workflow.nodes.generate", lambda prompt, config: "the answer")
    cfg = WorkflowConfig(
        chunks_path="unused.jsonl",
        collection_name="c",
        k=3,
        grading={"enabled": True, "keep_threshold": 2, "min_chunks": 0},
    )
    events = list(stream_run(build_graph(FakeRetriever(_chunks(3)), cfg), "q", "D"))

    assert [e["grade"] for e in events if e["type"] == "progress"] == [3, 0, 2]
    nodes = [e for e in events if e["type"] == "node"]
    assert [e["node"] for e in nodes] == ["retrieve", "grade", "generate"]
    assert [p["id"] for p in nodes[0]["data"]["passages"]] == ["D::0", "D::1", "D::2"]
    assert nodes[1]["data"]["kept"] == ["D::0", "D::2"]
    assert nodes[2]["data"]["answer"] == "the answer"
    assert nodes[2]["data"]["sources"] == ["D::0", "D::2"]
    assert events[-1]["type"] == "done" and events[-1]["llm_calls"] == 4


def test_the_run_endpoint_streams_server_sent_events(monkeypatch):
    main._graphs.clear()
    main._demo_graphs.clear()
    monkeypatch.setattr(main, "build_retriever", lambda name, cfg: object())
    monkeypatch.setattr(main, "build_graph", lambda retriever, cfg: cfg)
    monkeypatch.setattr(
        main,
        "stream_run",
        lambda graph, question, doc_id: iter([{"type": "done", "latency_s": 0.1, "llm_calls": 1}]),
    )
    with TestClient(main.app) as client:
        resp = client.post("/demo/run", json={"question": "q", "settings": {"grading": False}})
    events = [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert events[0]["type"] == "plan" and "grade" not in events[0]["nodes"]
    assert events[-1]["type"] == "done"


def test_a_failing_run_ends_with_an_error_event(monkeypatch):
    main._demo_graphs.clear()
    monkeypatch.setattr(main, "build_retriever", lambda name, cfg: object())
    monkeypatch.setattr(main, "build_graph", lambda retriever, cfg: cfg)

    def broken(graph, question, doc_id):
        raise ConnectionError("Ollama is not reachable")
        yield  # pragma: no cover

    monkeypatch.setattr(main, "stream_run", broken)
    with TestClient(main.app) as client:
        resp = client.post("/demo/run", json={"question": "q"})
    last = json.loads(resp.text.strip().splitlines()[-1][6:])
    assert last == {"type": "error", "message": "Ollama is not reachable"}
