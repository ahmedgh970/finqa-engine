"""Contract tests for the FastAPI serving app.

The retriever build and the graph are mocked, so these run fast and offline (no
Qdrant, no GPU, no Ollama) -- they check the endpoint wiring, the per-request config
and the response schema, not the model.
"""

from fastapi.testclient import TestClient

from src.api import main
from src.ingestion.schema import Chunk
from src.workflow.graph import WorkflowAnswer

CHUNK = Chunk(
    chunk_id="ACME_2022_10K::5",
    doc_id="ACME_2022_10K",
    text="Total revenue was $5,000 million in FY2022.",
    page=12,
)


def _client(monkeypatch, built: list | None = None) -> TestClient:
    """A client whose graphs record the config they were built from."""
    main._graphs.clear()
    monkeypatch.setattr(main, "build_retriever", lambda name, cfg: object())

    def fake_build(retriever, cfg):
        if built is not None:
            built.append(cfg)
        return cfg

    monkeypatch.setattr(main, "build_graph", fake_build)
    return TestClient(main.app)


def _answer(graph, question, doc_id=None) -> WorkflowAnswer:
    return WorkflowAnswer(
        answer="$5,000 million",
        sources=[CHUNK],
        latency_s=0.42,
        n_retrieved=20,
        n_kept_by_grade=4,
        computed="5000.00",
        llm_calls=22,
        node_latencies={"retrieve": 0.1, "generate": 0.3},
    )


def test_health_names_the_served_workflow(monkeypatch):
    with _client(monkeypatch) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["config"].startswith("configs/workflow/serve_")


def test_options_lists_models_pipelines_and_defaults(monkeypatch):
    monkeypatch.setattr(main, "_ollama_models", lambda: ["granite4.1:8b", "qwen3.5:4b"])
    with _client(monkeypatch) as client:
        body = client.get("/options").json()
    assert body["models"] == ["granite4.1:8b", "qwen3.5:4b"]
    assert body["pipelines"] == ["workflow", "naive"]
    assert body["defaults"] == {
        "model": "granite4.1:8b",
        "k": 20,
        "pipeline": "workflow",
        "collection": "docling_hybrid_1024_bge-m3",
    }


def test_ask_returns_the_answer_and_what_the_workflow_did(monkeypatch):
    monkeypatch.setattr(main, "run_graph", _answer)
    with _client(monkeypatch) as client:
        resp = client.post(
            "/ask", json={"question": "What was FY2022 revenue?", "k": 5, "model": "qwen3.5:4b"}
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "$5,000 million"
    assert body["pipeline"] == "workflow"
    assert body["model"] == "qwen3.5:4b"  # request override echoed back
    assert body["k"] == 5
    assert body["collection"] == "docling_hybrid_1024_bge-m3"
    assert body["sources"] == [{"doc_id": "ACME_2022_10K", "page": 12, "text": CHUNK.text}]
    assert body["computed"] == "5000.00" and body["llm_calls"] == 22


def test_the_served_workflow_switches_on_every_node(monkeypatch):
    built: list = []
    with _client(monkeypatch, built):
        pass
    served = built[0]  # the default graph, compiled at startup
    assert served.retriever == "reranked" and served.base_retriever == "dense"
    assert served.grading.enabled and served.expansion.enabled and served.calculator.enabled
    assert served.expansion.window == 1 and served.llm.num_ctx == 12288


def test_naive_is_the_same_graph_with_every_node_off(monkeypatch):
    built: list = []
    monkeypatch.setattr(main, "run_graph", _answer)
    with _client(monkeypatch, built) as client:
        client.post("/ask", json={"question": "q", "pipeline": "naive"})
        client.post("/ask", json={"question": "q", "pipeline": "naive"})
    naive = built[-1]
    assert len(built) == 2  # startup graph + naive, the second request reuses it
    assert not (naive.grading.enabled or naive.expansion.enabled or naive.calculator.enabled)
    assert naive.retriever == built[0].retriever and naive.k == built[0].k


def test_ask_requires_a_question(monkeypatch):
    with _client(monkeypatch) as client:
        resp = client.post("/ask", json={})
    assert resp.status_code == 422  # pydantic validation: question is required


def test_an_unknown_pipeline_is_refused(monkeypatch):
    with _client(monkeypatch) as client:
        resp = client.post("/ask", json={"question": "q", "pipeline": "agent"})
    assert resp.status_code == 422
