"""Configuration schema for a retrieval evaluation (1 YAML = 1 retriever setup)."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class RetrievalEvalConfig(BaseModel):
    """Parameters of a retrieval evaluation.

    Carries ``qdrant_location`` / ``collection_name`` / ``embedding_model`` so it
    plugs straight into ``build_retriever``. ``chunks_path`` is the same corpus
    that was indexed — it is used to resolve each evidence to its physical page.
    """

    golden_set_path: str = "data/jsons/financebench_open_source.jsonl"
    chunks_path: str
    collection_name: str
    embedding_model: str = "BAAI/bge-m3"
    retriever: str = "dense"
    doc_scoped: bool = False  # True: restrict retrieval to each QA's gold document
    qdrant_location: str = Field(
        default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333")
    )
    k_values: list[int] = [1, 3, 5, 10]

    # Only used when retriever == "reranked".
    base_retriever: str | None = None  # dense | bm25 | hybrid, reranked on top of
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_prefetch: int = 50  # candidates pulled from base_retriever before rerank

    # Only used when retriever == "replay": the materialised retrieval to score, which
    # caps k_values at the depth it was stored with and needs neither GPU nor Qdrant.
    replay_path: str | None = None


def load_retrieval_eval_config(path: str) -> RetrievalEvalConfig:
    """Load and validate a retrieval-evaluation config from a YAML file."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    return RetrievalEvalConfig(**data)
