"""Score a retriever against the FinanceBench golden set.

For each QA the gold evidence is resolved to physical pages, the top-k chunks are
retrieved, and each hit is marked relevant when it lands on a gold page not
already reached by a better-ranked chunk. recall@k / precision@k / MRR / nDCG@k
are computed per QA and averaged over the QAs whose document is in the corpus.

Page-level relevance cannot tell whether a passage holds the figures the question
needs, so the top-k context is also read directly: its coverage of the evidence text
and its size (``coverage.py``). Scoring a replayed workflow output this way measures
the context a grader actually kept, whatever its length (every k at or above it
sees the whole context).
"""

from __future__ import annotations

from datetime import UTC, datetime

from tqdm import tqdm

from src.evaluation.common.golden_set import load_golden_set
from src.evaluation.common.matching import build_page_index, is_relevant, resolve_gold_pages
from src.evaluation.retrieval.config import RetrievalEvalConfig
from src.evaluation.retrieval.coverage import evidence_coverage
from src.evaluation.retrieval.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.retrieval.base import ScoredChunk
from src.retrieval.registry import build_retriever


def dedup_relevances(results: list[ScoredChunk], doc_name: str, gold_pages: set[int]) -> list[bool]:
    """Ranked relevance, one True per gold page (first chunk that reaches it)."""
    seen: set[int] = set()
    flags: list[bool] = []
    for sc in results:
        rel = is_relevant(sc.chunk, doc_name, gold_pages) and sc.chunk.page not in seen
        flags.append(rel)
        if rel:
            seen.add(sc.chunk.page)
    return flags


def score_qa(relevances: list[bool], num_gold: int, k_values: list[int]) -> dict[str, float]:
    """MRR plus recall / precision / nDCG at every k, for one QA."""
    scores: dict[str, float] = {"mrr": reciprocal_rank(relevances)}
    for k in k_values:
        scores[f"recall@{k}"] = recall_at_k(relevances, k, num_gold)
        scores[f"precision@{k}"] = precision_at_k(relevances, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(relevances, k, num_gold)
    return scores


def coverage_qa(
    results: list[ScoredChunk], evidence: list[str], k_values: list[int]
) -> dict[str, float | None]:
    """Evidence coverage and size of the top-k context at every k, for one QA."""
    scores: dict[str, float | None] = {}
    for k in k_values:
        top = results[:k]
        context = " ".join(sc.chunk.text for sc in top)
        for name, value in evidence_coverage(context, evidence).items():
            scores[f"{name}@{k}"] = value
        scores[f"passages@{k}"] = float(len(top))
    return scores


def aggregate(per_qa: list[dict[str, float | None]]) -> dict[str, float]:
    """Mean of every metric over the evaluated QAs, rounded to 4 decimals.

    A ``None`` means the metric does not apply to that QA (an evidence holding no
    number, say): it is left out of that metric's mean.
    """
    if not per_qa:
        return {}
    means = {}
    for k in per_qa[0]:
        values = [q[k] for q in per_qa if q[k] is not None]
        means[k] = round(sum(values) / len(values), 4) if values else None
    return means


def evaluate(config: RetrievalEvalConfig) -> dict:
    """Run the retriever on every QA and return the report (metrics + provenance)."""
    qas = load_golden_set(config.golden_set_path)
    page_index = build_page_index(config.chunks_path, {q.doc_name for q in qas})
    retriever = build_retriever(config.retriever, config)
    top_k = max(config.k_values)

    per_qa: list[dict[str, float | None]] = []
    skipped = 0
    for qa in tqdm(qas, desc=f"retrieval [{config.collection_name}]"):
        gold_pages = resolve_gold_pages(qa, page_index.get(qa.doc_name, {}))
        if not gold_pages:  # document not in the indexed corpus
            skipped += 1
            continue
        results = retriever.retrieve(
            qa.question, k=top_k, doc_id=qa.doc_name if config.doc_scoped else None
        )
        relevances = dedup_relevances(results, qa.doc_name, gold_pages)
        evidence = [ev.text for ev in qa.evidence]
        per_qa.append(
            score_qa(relevances, len(gold_pages), config.k_values)
            | coverage_qa(results, evidence, config.k_values)
        )

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "collection": config.collection_name,
        "embedding_model": config.embedding_model,
        "retriever": config.retriever,
        "base_retriever": config.base_retriever,
        "replay_path": config.replay_path,
        "doc_scoped": config.doc_scoped,
        "n_evaluated": len(per_qa),
        "n_skipped_doc_absent": skipped,
        "metrics": aggregate(per_qa),
    }
