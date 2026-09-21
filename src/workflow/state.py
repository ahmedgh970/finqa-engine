"""State carried through the CRAG graph.

One flat TypedDict rather than nested objects: LangGraph merges the partial dict a
node returns into the state, so a flat shape keeps every node's contract obvious.
Fields are grouped by the phase that fills them; a disabled phase simply leaves its
fields unset, which is what lets one graph serve the whole ablation matrix.
"""

from __future__ import annotations

from typing import TypedDict

from src.ingestion.schema import Chunk


class CragState(TypedDict, total=False):
    """Everything the graph reads or writes for one question."""

    # Inputs
    question: str
    doc_id: str | None

    # Retrieval -- ``chunks`` arrives in the retriever's own order (cross-encoder
    # order when the retriever reranks), which is also the order the floor draws from.
    chunks: list[Chunk]
    scores: list[float]

    # Grading
    graded: list[Chunk]  # the context; unset when grading is off
    grades: list[int]  # one 0-3 grade per retrieved chunk, kept for offline calibration
    n_kept_by_grade: int  # passed the threshold on their own merit
    n_kept_by_floor: int  # added by the floor to reach min_chunks
    max_grade: int
    low_confidence: bool  # recorded, never acted upon

    # Neighbourhood expansion
    expanded: list[Chunk]  # the selection widened to the chunks around each passage

    # Numeric path
    is_numeric: bool
    computed: str | None

    # Output
    answer: str
    sources: list[Chunk]
    n_dropped_to_fit: int  # passages trimmed so the prompt fits the pinned num_ctx

    # Instrumentation
    node_latencies: dict[str, float]
    llm_calls: int
