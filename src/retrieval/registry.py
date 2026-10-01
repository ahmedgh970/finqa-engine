"""Registry of retrieval strategies, selectable by name.

Each builder constructs a retriever from a config object. New strategies (bm25,
hybrid, reranker) register a builder here; the eval harness builds retrievers via
``build_retriever`` and never has to change. Builders import their retriever
lazily so unused strategies don't pull heavy dependencies.
"""

from __future__ import annotations

from collections.abc import Callable

from src.retrieval.base import Retriever


def _build_dense(cfg) -> Retriever:
    from src.retrieval.dense import DenseRetriever
    from src.vectorstore.qdrant_store import get_client

    return DenseRetriever(
        client=get_client(cfg.qdrant_location),
        collection_name=cfg.collection_name,
        model_name=cfg.embedding_model,
    )


def _load_corpus(cfg) -> list:
    from src.ingestion.storage import read_chunks

    return list(read_chunks(cfg.chunks_path))


def _build_bm25(cfg) -> Retriever:
    from src.retrieval.bm25 import BM25Retriever

    return BM25Retriever(_load_corpus(cfg))


def _build_hybrid(cfg) -> Retriever:
    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.hybrid import HybridRetriever

    return HybridRetriever(dense=_build_dense(cfg), sparse=BM25Retriever(_load_corpus(cfg)))


def _build_reranked(cfg) -> Retriever:
    from src.retrieval.reranked import RerankedRetriever
    from src.retrieval.reranker import Reranker

    if not cfg.base_retriever:
        raise ValueError("retriever 'reranked' requires 'base_retriever' to be set")
    base = build_retriever(cfg.base_retriever, cfg)
    return RerankedRetriever(
        base=base,
        reranker=Reranker(cfg.reranker_model, getattr(cfg, "reranker_dtype", None)),
        prefetch=cfg.rerank_prefetch,
    )


def _build_replay(cfg) -> Retriever:
    from src.retrieval.replay import ReplayRetriever

    if not getattr(cfg, "replay_path", None):
        raise ValueError("retriever 'replay' requires 'replay_path' to be set")
    return ReplayRetriever(cfg.replay_path)


_BUILDERS: dict[str, Callable[[object], Retriever]] = {
    "dense": _build_dense,
    "replay": _build_replay,
    "bm25": _build_bm25,
    "hybrid": _build_hybrid,
    "reranked": _build_reranked,
}


def build_retriever(name: str, cfg) -> Retriever:
    """Build the retriever registered under ``name`` from ``cfg``."""
    if name not in _BUILDERS:
        raise ValueError(f"Unknown retriever '{name}'. Available: {list(_BUILDERS)}")
    return _BUILDERS[name](cfg)
