"""The demo surface of the API: run settings, a live event stream, and recorded replays.

A run is described by ``RunSettings`` -- the knobs a workflow YAML would set, exposed
so a UI can change them per question -- and streamed as events, one per graph node as
it finishes, plus one per passage while the grader works. The UI draws the graph from
them as it runs.

A replay streams the same events from a benchmark run already on disk: the retrieval,
the grades, the widened context and the answer of the reference row, and the verdict
it received. Nothing is recomputed, so it is instant and the GPU stays free.
"""

from __future__ import annotations

import contextlib
import json
import queue
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal

import yaml
from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field

from src.ingestion.schema import Chunk
from src.llm.client import Cancelled, stoppable
from src.llm.prompts import build_prompt
from src.workflow.calculator import routes_to_calculator
from src.workflow.config import WorkflowConfig

# Where each chunk budget's corpus lives, and the collection it was indexed into.
CHUNKS_PATH = "data/processed/docling/chunked/hybrid/chunks_{size}.jsonl"
COLLECTION = "docling_hybrid_{size}_bge-m3"
CHUNK_SIZES = (256, 512, 1024)


class RunSettings(BaseModel):
    """What a workflow YAML sets, as a UI changes it for one question."""

    model: str = "granite4.1:8b"
    chunk_size: Literal[256, 512, 1024] = 1024
    reranker: bool = True
    prefetch: int = Field(default=50, ge=10, le=200)
    k: int = Field(default=20, ge=1, le=50)
    grading: bool = True
    keep_threshold: int = Field(default=2, ge=0, le=3)
    min_chunks: int = Field(default=3, ge=0, le=20)
    expansion: bool = True
    window: int = Field(default=1, ge=0, le=6)
    calculator: bool = True
    num_ctx: int = Field(default=12288, ge=2048, le=32768)
    max_tokens: int = Field(default=1024, ge=128, le=4096)

    @classmethod
    def from_config(cls, cfg: WorkflowConfig) -> RunSettings:
        """The settings a served config amounts to."""
        size = next((s for s in CHUNK_SIZES if CHUNKS_PATH.format(size=s) == cfg.chunks_path), 1024)
        return cls(
            model=cfg.llm.model.split("/", 1)[-1],
            chunk_size=size,
            reranker=cfg.retriever == "reranked",
            prefetch=cfg.rerank_prefetch,
            k=cfg.k,
            grading=cfg.grading.enabled,
            keep_threshold=cfg.grading.keep_threshold,
            min_chunks=cfg.grading.min_chunks,
            expansion=cfg.expansion.enabled,
            window=cfg.expansion.window,
            calculator=cfg.calculator.enabled,
            num_ctx=cfg.llm.num_ctx or 12288,
            max_tokens=cfg.llm.max_tokens,
        )

    def config(self, base: WorkflowConfig) -> WorkflowConfig:
        """``base`` with these settings applied: the config one run is built from."""
        return base.model_copy(
            update={
                "chunks_path": CHUNKS_PATH.format(size=self.chunk_size),
                "collection_name": COLLECTION.format(size=self.chunk_size),
                "retriever": "reranked" if self.reranker else "dense",
                "base_retriever": "dense",
                "rerank_prefetch": self.prefetch,
                "k": self.k,
                "grading": base.grading.model_copy(
                    update={
                        "enabled": self.grading,
                        "keep_threshold": self.keep_threshold,
                        "min_chunks": self.min_chunks,
                    }
                ),
                "expansion": base.expansion.model_copy(
                    update={"enabled": self.expansion, "window": self.window}
                ),
                "calculator": base.calculator.model_copy(update={"enabled": self.calculator}),
                "llm": base.llm.model_copy(
                    update={
                        "model": f"ollama_chat/{self.model}",
                        "num_ctx": self.num_ctx,
                        "max_tokens": self.max_tokens,
                    }
                ),
            }
        )

    def plan(self) -> list[str]:
        """The graph nodes these settings switch on, in the order they run."""
        nodes = ["retrieve"]
        nodes += ["grade"] if self.grading else []
        nodes += ["expand"] if self.expansion else []
        nodes += ["route", "calculate"] if self.calculator else []
        return [*nodes, "generate"]


def available_chunk_sizes(collections: list[str]) -> list[int]:
    """Chunk budgets that are both indexed and on disk (expansion reads the file)."""
    return [
        size
        for size in CHUNK_SIZES
        if COLLECTION.format(size=size) in collections
        and Path(CHUNKS_PATH.format(size=size)).exists()
    ]


# --- live runs ------------------------------------------------------------------------


class DenseStageReporter:
    """Wraps the first stage of a reranked retriever and reports what it found.

    The graph's retrieve node returns the reranked top-k only: the shortlist the dense
    search gave the cross-encoder, and its cosine scores, never leave the retriever.
    Reported while the node runs, they let a UI show the two stages apart -- and which
    passages the reranker pulled up from deep in the dense ranking. Outside a streamed
    graph (``POST /ask``, a benchmark run) the report goes nowhere.
    """

    def __init__(self, inner):
        self.inner = inner

    def retrieve(self, query: str, k: int = 5, doc_id: str | None = None):
        results = self.inner.retrieve(query, k=k, doc_id=doc_id)
        with contextlib.suppress(RuntimeError):  # not inside a graph run
            get_stream_writer()(
                {
                    "node": "retrieve",
                    "stage": "dense",
                    "passages": [_passage(sc.chunk, sc.score) for sc in results],
                }
            )
        return results


def report_dense_stage(retriever):
    """``retriever`` with its dense stage reporting, when it has one to report."""
    if hasattr(retriever, "base") and not isinstance(retriever.base, DenseStageReporter):
        retriever.base = DenseStageReporter(retriever.base)
    return retriever


def _passage(chunk: Chunk, score: float | None = None) -> dict:
    return {
        "id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "page": chunk.page,
        "score": score,
        "text": chunk.text,
    }


def _prompt_tokens(question: str, sources: list[Chunk]) -> int:
    """The generation prompt's size, by the estimate the trim itself uses (len / 3)."""
    return len(build_prompt(question, sources)) // 3


def _node_seconds(update: dict, node: str) -> float | None:
    return (update.get("node_latencies") or {}).get(node)


def _summary(node: str, update: dict, question: str, known: set[str]) -> dict:
    """What a UI needs to show about one finished node."""
    if node == "retrieve":
        scores = update.get("scores") or [None] * len(update.get("chunks", []))
        passages = [_passage(c, s) for c, s in zip(update.get("chunks", []), scores, strict=True)]
        known.update(p["id"] for p in passages)
        return {"passages": passages}
    if node == "grade":
        return {
            "grades": update.get("grades", []),
            "kept": [c.chunk_id for c in update.get("graded", [])],
            "n_kept_by_grade": update.get("n_kept_by_grade", 0),
            "n_kept_by_floor": update.get("n_kept_by_floor", 0),
        }
    if node == "expand":
        widened = update.get("expanded", [])
        added = [_passage(c) for c in widened if c.chunk_id not in known]
        known.update(p["id"] for p in added)
        return {"ids": [c.chunk_id for c in widened], "added": added}
    if node == "route":
        return {"is_numeric": update.get("is_numeric", False)}
    if node == "calculate":
        return {
            "computed": update.get("computed"),
            "calc_error": update.get("calc_error"),
            "calculation": update.get("calculation"),
        }
    if node == "generate":
        sources = update.get("sources", [])
        added = [_passage(c) for c in sources if c.chunk_id not in known]
        known.update(p["id"] for p in added)
        return {
            "answer": update.get("answer", ""),
            "sources": [c.chunk_id for c in sources],
            "added": added,
            "n_dropped_to_fit": update.get("n_dropped_to_fit", 0),
            "prompt_tokens": _prompt_tokens(question, sources),
            "computed_used": update.get("computed_used", False),
            "truncated": update.get("truncated", False),
        }
    return {}


def stream_run(graph, question: str, doc_id: str | None) -> Iterator[dict]:
    """Run ``graph`` on one question, yielding an event as each node finishes."""
    started = time.perf_counter()
    known: set[str] = set()
    llm_calls = 0
    latencies: dict[str, float] = {}
    for mode, chunk in graph.stream(
        {"question": question, "doc_id": doc_id}, stream_mode=["updates", "custom"]
    ):
        if mode == "custom":
            yield {"type": "progress", **chunk}
            continue
        for node, update in chunk.items():
            llm_calls = update.get("llm_calls", llm_calls)
            latencies = update.get("node_latencies", latencies)
            yield {
                "type": "node",
                "node": node,
                "seconds": _node_seconds(update, node),
                "data": _summary(node, update, question, known),
            }
    yield {
        "type": "done",
        "latency_s": time.perf_counter() - started,
        "llm_calls": llm_calls,
        "node_latencies": latencies,
    }


class LiveRun:
    """One live question run in its own thread, its events queued for the request.

    The thread, not the HTTP response, owns the run: a client that leaves mid-run only
    stops reading, so the run is told to stop instead of being left to finish -- or to
    hang -- with nobody listening. ``stop`` reaches the LLM client, which abandons the
    call in progress; ``finished`` is set however the run ends.
    """

    def __init__(self, events: Callable[[], Iterator[dict]]):
        self._events = events
        self.queue: queue.Queue[dict | None] = queue.Queue()
        self.stop = threading.Event()
        self.finished = threading.Event()
        self._thread = threading.Thread(target=self._work, name="demo-run", daemon=True)

    def start(self) -> LiveRun:
        self._thread.start()
        return self

    def _work(self) -> None:
        try:
            with stoppable(self.stop):
                for event in self._events():
                    self.queue.put(event)
                    if self.stop.is_set():
                        raise Cancelled
        except Cancelled:
            self.queue.put({"type": "cancelled", "message": "Run arrêté."})
        except Exception as exc:  # shown in the UI rather than a dropped connection
            self.queue.put({"type": "error", "message": str(exc)})
        finally:
            self.finished.set()
            self.queue.put(None)


class LiveRuns:
    """At most one live run: starting one stops the run before it, then waits for it.

    One GPU serves every run, so two at once would both crawl; and the run being
    replaced is one its user abandoned (stopped, or changed a setting and asked again).
    """

    def __init__(self, wait_s: float = 30.0):
        self._lock = threading.Lock()
        self._active: LiveRun | None = None
        self._wait_s = wait_s

    def start(self, events: Callable[[], Iterator[dict]]) -> LiveRun:
        with self._lock:
            if self._active is not None:
                self._active.stop.set()
                self._active.finished.wait(self._wait_s)
            self._active = LiveRun(events).start()
            return self._active


# --- recorded runs --------------------------------------------------------------------


class ReplaySources(BaseModel):
    """The files one recorded run is rebuilt from (``configs/demo/replay.yaml``)."""

    label: str
    settings: RunSettings
    retrieval: str  # materialised top-k, in reranker order
    grading: str  # the graded run over that same top-k
    expansion: str  # the graded selection widened to its neighbours
    answers: str  # the row's answers, after the trim and the calculator
    verdicts: str | None = None  # the judged grid of ``answers``
    # The expression and rows behind each verified figure (scripts/demo_calc_details.py).
    calculations: str | None = None


def _by_id(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in (json.loads(line) for line in f if line.strip())}


class Replay:
    """A recorded benchmark run, served back as the events of a live one."""

    def __init__(self, sources: ReplaySources, golden_set_path: str):
        self.sources = sources
        self.retrieval = _by_id(sources.retrieval)
        self.grading = _by_id(sources.grading)
        self.expansion = _by_id(sources.expansion)
        self.answers = _by_id(sources.answers)
        self.verdicts = _by_id(sources.verdicts) if sources.verdicts else {}
        self.calculations = (
            _by_id(sources.calculations)
            if sources.calculations and Path(sources.calculations).exists()
            else {}
        )
        with open(golden_set_path, encoding="utf-8") as f:
            self.gold = {
                r["financebench_id"]: r for r in (json.loads(line) for line in f if line.strip())
            }

    @classmethod
    def load(cls, path: str, golden_set_path: str) -> Replay | None:
        """The replay a YAML describes, or None when one of its files is missing."""
        if not Path(path).exists():
            return None
        sources = ReplaySources(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))
        files = [sources.retrieval, sources.grading, sources.expansion, sources.answers]
        if not all(Path(p).exists() for p in files):
            return None
        return cls(sources, golden_set_path)

    def examples(self) -> list[dict]:
        """Every golden-set question with what the recorded run made of it."""
        out = []
        for qa_id, gold in self.gold.items():
            answer = self.answers.get(qa_id, {})
            verdict = self.verdicts.get(qa_id, {})
            out.append(
                {
                    "id": qa_id,
                    "company": gold["company"],
                    "doc_id": gold["doc_name"],
                    "question": gold["question"],
                    "gold_answer": gold["answer"],
                    "justification": gold.get("justification"),
                    "routed": routes_to_calculator(gold["question"]),
                    "computed": answer.get("computed"),
                    "outcome": verdict.get("outcome"),
                    "recorded": qa_id in self.answers,
                }
            )
        return out

    def events(self, qa_id: str, pace: float = 1.0) -> Iterator[dict]:
        """The recorded run of ``qa_id`` as events, paced for the eye, not the clock."""
        top = self.retrieval[qa_id]["sources"]
        graded = self.grading[qa_id]
        widened = self.expansion[qa_id]["sources"]
        final = self.answers[qa_id]
        question = final["question"]

        # Files from different stages name passages differently; the text is shared.
        ids: dict[str, str] = {}

        def pid(passage: dict, fallback: str) -> str:
            return ids.setdefault(passage["text"].strip(), passage.get("chunk_id") or fallback)

        def passage(p: dict, fallback: str) -> dict:
            return {
                "id": pid(p, fallback),
                "doc_id": p["doc_id"],
                "page": p["page"],
                "score": None,
                "text": p["text"],
            }

        def wait(seconds: float) -> None:
            if pace:
                time.sleep(seconds * pace)

        settings = self.sources.settings
        yield {"type": "plan", "nodes": settings.plan(), "settings": settings.model_dump()}
        wait(0.6)
        retrieved = [passage(p, f"r{i}") for i, p in enumerate(top)]
        yield {"type": "node", "node": "retrieve", "seconds": None, "data": {"passages": retrieved}}

        grades = graded["grades"]
        for index, grade in enumerate(grades):
            wait(0.18)
            yield {"type": "progress", "node": "grade", "index": index, "grade": grade}
        kept = [pid(p, f"k{i}") for i, p in enumerate(graded["sources"])]
        yield {
            "type": "node",
            "node": "grade",
            "seconds": graded["node_latencies"].get("grade"),
            "data": {
                "grades": grades,
                "kept": kept,
                "n_kept_by_grade": graded["n_kept_by_grade"],
                "n_kept_by_floor": graded["n_kept_by_floor"],
            },
        }

        wait(0.6)
        known = {p["id"] for p in retrieved}
        widened_passages = [passage(p, f"x{i}") for i, p in enumerate(widened)]
        yield {
            "type": "node",
            "node": "expand",
            "seconds": None,
            "data": {
                "ids": [p["id"] for p in widened_passages],
                "added": [p for p in widened_passages if p["id"] not in known],
            },
        }
        known.update(p["id"] for p in widened_passages)

        wait(0.4)
        is_numeric = final.get("is_numeric", False)
        yield {"type": "node", "node": "route", "seconds": 0.0, "data": {"is_numeric": is_numeric}}
        if is_numeric:
            wait(1.0)
            yield {
                "type": "node",
                "node": "calculate",
                "seconds": final["node_latencies"].get("calculate"),
                "data": {
                    "computed": final.get("computed"),
                    "calc_error": final.get("calc_error"),
                    "calculation": self.calculations.get(qa_id, {}).get("calculation"),
                },
            }

        wait(1.4)
        sources = [passage(p, f"s{i}") for i, p in enumerate(final["sources"])]
        yield {
            "type": "node",
            "node": "generate",
            "seconds": final["node_latencies"].get("generate"),
            "data": {
                "answer": final["generated_answer"],
                "sources": [p["id"] for p in sources],
                "added": [p for p in sources if p["id"] not in known],
                "n_dropped_to_fit": final.get("n_dropped_to_fit", 0),
                "prompt_tokens": _prompt_tokens(
                    question,
                    [
                        Chunk(chunk_id=p["id"], doc_id=p["doc_id"], page=p["page"], text=p["text"])
                        for p in sources
                    ],
                ),
                "computed_used": final.get("computed_used", False),
            },
        }
        recorded = {**graded["node_latencies"], **final["node_latencies"]}
        verdict = self.verdicts.get(qa_id, {})
        yield {
            "type": "done",
            "latency_s": sum(recorded.values()),
            "llm_calls": graded["llm_calls"] + (1 if is_numeric else 0),
            "node_latencies": recorded,
            "outcome": verdict.get("outcome"),
            "verdict": verdict.get("justification"),
        }
