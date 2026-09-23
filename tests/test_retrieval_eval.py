"""Unit tests for the retrieval evaluation (no Qdrant, no embedder)."""

from pathlib import Path

from src.evaluation.retrieval.config import load_retrieval_eval_config
from src.evaluation.retrieval.coverage import (
    evidence_coverage,
    key_figures,
    ngrams,
    numbers,
)
from src.evaluation.retrieval.evaluate import aggregate, coverage_qa, dedup_relevances, score_qa
from src.evaluation.run_retrieval import report_path
from src.ingestion.schema import Chunk
from src.retrieval.base import ScoredChunk


def _hit(page, doc="DOC", i=0):
    return ScoredChunk(Chunk(chunk_id=f"{doc}::{i}", doc_id=doc, text="...", page=page), 1.0)


def test_only_the_first_chunk_reaching_a_gold_page_counts():
    # Two chunks of gold page 46: a second hit on the same page is not a new relevant item.
    results = [_hit(3, i=0), _hit(46, i=1), _hit(46, i=2), _hit(58, i=3)]
    assert dedup_relevances(results, "DOC", {46, 58}) == [False, True, False, True]


def test_hits_from_another_document_are_never_relevant():
    assert dedup_relevances([_hit(46, doc="OTHER")], "DOC", {46}) == [False]


def test_score_qa_reports_every_metric_at_every_k():
    scores = score_qa([False, True], num_gold=1, k_values=[1, 2])
    assert scores["mrr"] == 0.5
    assert scores["recall@1"] == 0.0 and scores["recall@2"] == 1.0
    assert set(scores) == {
        "mrr",
        "recall@1",
        "precision@1",
        "ndcg@1",
        "recall@2",
        "precision@2",
        "ndcg@2",
    }


def test_aggregate_averages_over_the_evaluated_qas():
    assert aggregate([{"mrr": 1.0}, {"mrr": 0.0}]) == {"mrr": 0.5}
    assert aggregate([]) == {}


def test_aggregate_leaves_inapplicable_values_out_of_the_mean():
    per_qa = [{"evidence_numbers@5": 1.0}, {"evidence_numbers@5": None}]
    assert aggregate(per_qa) == {"evidence_numbers@5": 1.0}


def test_numbers_drop_thousands_separators_and_sentence_dots():
    assert numbers("Revenue was $1,234.5 million in 2019.") == {"1234.5", "2019"}


def test_evidence_coverage_reads_the_figures_not_the_page():
    evidence = ["Accounts payable 2019 = 292. Accounts payable 2018 = 253."]
    half = evidence_coverage("Accounts payable 2019 = 292.", evidence)
    assert half["evidence_numbers"] == 0.5  # 2019 and 292, not 2018 and 253
    assert half["all_evidence_numbers"] == 0.0
    full = evidence_coverage(evidence[0], evidence)
    assert full["evidence_numbers"] == full["all_evidence_numbers"] == 1.0


def test_key_figures_drop_years_and_small_counts():
    """A page is full of millésimes and small counts that match by coincidence."""
    assert key_figures("Accounts payable 2019 = 34,616. 7 segments, shares 2.2, total 93") == {
        "34616",
        "2.2",
    }


def test_span_overlap_needs_the_sequence_not_the_vocabulary():
    """The 5-gram measure is what separates a real quote from shared words."""
    evidence = ["the company repurchased 722,457 shares of its common stock"]
    quoted = evidence_coverage(f"note 7 {evidence[0]} during the quarter", evidence)
    shuffled = evidence_coverage(
        "shares 722,457 common stock company the of its repurchased", evidence
    )
    assert quoted["evidence_overlap"] == 1.0
    assert shuffled["evidence_overlap"] == 0.0
    # The looser word measure cannot tell them apart.
    assert shuffled["evidence_words"] == 1.0


def test_ngrams_are_ordered_windows_of_the_text():
    assert ngrams("a b c d e f", n=5) == {("a", "b", "c", "d", "e"), ("b", "c", "d", "e", "f")}
    assert ngrams("too short", n=5) == set()


def test_evidence_without_numbers_does_not_count_as_covered():
    cov = evidence_coverage("anything", ["Revenue grew on higher volumes"])
    assert cov["evidence_numbers"] is None and cov["all_evidence_numbers"] is None
    assert cov["evidence_figures"] is None and cov["all_evidence_figures"] is None


def test_coverage_at_k_reads_only_the_top_k_passages():
    results = [
        ScoredChunk(Chunk(chunk_id=f"D::{i}", doc_id="D", page=1, text=t), 1.0)
        for i, t in enumerate(["cash 100", "debt 200"])
    ]
    cov = coverage_qa(results, ["cash 100 debt 200"], [1, 5])
    assert cov["evidence_numbers@1"] == 0.5
    assert cov["evidence_numbers@5"] == 1.0
    assert cov["passages@1"] == 1.0 and cov["passages@5"] == 2.0


def test_report_is_named_after_the_config():
    path = report_path("configs/evaluation/retrieval/chunks512_dense.yaml", "20260911-120000")
    assert path == Path("docs/benchmarks/retrieval_chunks512_dense_20260911-120000.json")


def test_every_shipped_retrieval_config_loads():
    configs = sorted(Path("configs/evaluation/retrieval").glob("*.yaml"))
    assert len(configs) == 13
    for path in configs:
        cfg = load_retrieval_eval_config(str(path))
        assert cfg.doc_scoped
        assert (cfg.retriever == "reranked") == ("reranked" in path.stem)
