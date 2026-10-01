"""Configuration schema for a naive RAG run (1 YAML = 1 experiment)."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from src.llm.config import LLMConfig


class RagConfig(BaseModel):
    """Parameters of a naive RAG run: build a retriever, answer every QA, save results.

    The retrieval fields mirror ``RetrievalEvalConfig`` so the same YAML shape plugs
    straight into ``build_retriever``.
    """

    golden_set_path: str = "data/jsons/financebench_open_source.jsonl"
    chunks_path: str
    collection_name: str
    embedding_model: str = "BAAI/bge-m3"
    retriever: str = "dense"
    doc_scoped: bool = False
    qdrant_location: str = Field(
        default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333")
    )
    k: int = 10

    # Only used when retriever == "replay": a materialised retrieval to serve instead
    # of recomputing one, so several configurations share byte-identical passages.
    replay_path: str | None = None

    # Only used when retriever == "reranked".
    base_retriever: str | None = None
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_prefetch: int = 50
    # None keeps the checkpoint's precision (every benchmark row); "float16" when the
    # reranker shares a GPU with the generator, as in serving.
    reranker_dtype: str | None = None

    llm: LLMConfig = Field(default_factory=LLMConfig)


def load_rag_config(path: str) -> RagConfig:
    """Load and validate a RAG config from a YAML file."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    return RagConfig(**data)
