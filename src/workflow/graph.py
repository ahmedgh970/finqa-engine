"""The advanced RAG graph: retrieve -> [grade] -> [expand] -> [route -> calculate] -> generate.

Only the nodes enabled by the config are wired in, so one graph serves every row of
the ablation matrix. With grading off it reduces to retrieve -> generate, which is the
advanced-RAG baseline running on this exact code path -- that is what makes a
difference between rows attributable to the node under test and not to the plumbing.

There is no correction loop. Query reformulation was implemented, run over the 150
questions and measured: it fired on 15% of them and, when it did find passages the
grader accepted, those answers scored *worse* (29%) than simply keeping the original
retrieval (44%). The floor inside the grading node replaces it -- it repairs the same
failure, an empty or over-thin selection, without a second retrieval or a second
grading pass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from langgraph.graph import END, START, StateGraph

from src.ingestion.schema import Chunk
from src.retrieval.base import Retriever
from src.workflow.config import WorkflowConfig
from src.workflow.nodes import (
    make_calculate,
    make_expand,
    make_generate,
    make_grade,
    make_retrieve,
    make_route,
)
from src.workflow.state import CragState


@dataclass
class WorkflowAnswer:
    """A workflow answer with its sources and the instrumentation the eval needs."""

    answer: str
    sources: list[Chunk]
    latency_s: float
    n_retrieved: int = 0
    n_kept_by_grade: int = 0
    n_kept_by_floor: int = 0
    n_dropped_to_fit: int = 0
    n_expanded: int = 0  # passages added around the selection by the expansion node
    is_numeric: bool = False  # the wording states a computation the tool can verify
    computed: str | None = None  # the verified figure, None when the tool declined
    calc_error: str | None = None  # why it declined
    computed_used: bool = False  # whether the figure survived into the answer
    truncated: bool = False  # the answer hit max_tokens before it was complete
    grades: list[int] = field(default_factory=list)
    max_grade: int | None = None
    low_confidence: bool = False
    llm_calls: int = 0
    node_latencies: dict[str, float] = field(default_factory=dict)


def build_graph(retriever: Retriever, config: WorkflowConfig):
    """Compile the graph for ``config`` -- only the enabled nodes are wired in."""
    builder = StateGraph(CragState)
    builder.add_node("retrieve", make_retrieve(retriever, config))
    builder.add_node("generate", make_generate(config))
    builder.add_edge(START, "retrieve")

    selected_by = "retrieve"
    if config.grading.enabled:
        builder.add_node("grade", make_grade(config))
        builder.add_edge(selected_by, "grade")
        selected_by = "grade"
    if config.expansion.enabled:
        builder.add_node("expand", make_expand(config))
        builder.add_edge(selected_by, "expand")
        selected_by = "expand"

    if config.calculator.enabled:
        # The routing decision is a node so it lands in the state, and the branch it
        # opens always rejoins generate: a declined computation is not a dead end, it is
        # the ungraded answer the row would have produced anyway.
        builder.add_node("route", make_route())
        builder.add_node("calculate", make_calculate(config))
        builder.add_edge(selected_by, "route")
        builder.add_conditional_edges(
            "route",
            lambda state: "calculate" if state.get("is_numeric") else "generate",
            {"calculate": "calculate", "generate": "generate"},
        )
        builder.add_edge("calculate", "generate")
    else:
        builder.add_edge(selected_by, "generate")

    builder.add_edge("generate", END)
    return builder.compile()


def answer_workflow(
    question: str,
    retriever: Retriever,
    config: WorkflowConfig,
    doc_id: str | None = None,
) -> WorkflowAnswer:
    """Answer ``question`` by running the configured graph once."""
    return run_graph(build_graph(retriever, config), question, doc_id)


def run_graph(graph, question: str, doc_id: str | None = None) -> WorkflowAnswer:
    """Run an already compiled graph on one question.

    Compiling is separate so a server builds each graph once and reuses it: the graph
    holds no per-question state, and building it loads the expansion corpus.
    """
    start = time.perf_counter()
    state: CragState = graph.invoke({"question": question, "doc_id": doc_id})
    return WorkflowAnswer(
        answer=state.get("answer", ""),
        sources=state.get("sources", []),
        latency_s=time.perf_counter() - start,
        n_retrieved=len(state.get("chunks", [])),
        n_kept_by_grade=state.get("n_kept_by_grade", 0),
        n_kept_by_floor=state.get("n_kept_by_floor", 0),
        n_dropped_to_fit=state.get("n_dropped_to_fit", 0),
        n_expanded=len(state.get("expanded", [])),
        is_numeric=state.get("is_numeric", False),
        computed=state.get("computed"),
        calc_error=state.get("calc_error"),
        computed_used=state.get("computed_used", False),
        truncated=state.get("truncated", False),
        grades=state.get("grades", []),
        max_grade=state.get("max_grade"),
        low_confidence=state.get("low_confidence", False),
        llm_calls=state.get("llm_calls", 0),
        node_latencies=state.get("node_latencies", {}),
    )
