"""Tests for the replayed retrieval (no GPU, no Qdrant, no model)."""

from __future__ import annotations

import json

import pytest

from src.retrieval.replay import ReplayRetriever


@pytest.fixture
def stored(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "financebench_id_00001",
                "question": "What was capex?",
                "sources": [
                    {"doc_id": "D_2022_10K", "page": p, "text": f"passage {p}"} for p in (46, 12, 7)
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return str(path)


def test_replay_serves_the_stored_passages_in_their_stored_ranking(stored):
    results = ReplayRetriever(stored).retrieve("What was capex?", k=3)

    assert [sc.chunk.page for sc in results] == [46, 12, 7]
    assert [sc.chunk.text for sc in results] == ["passage 46", "passage 12", "passage 7"]
    # Scores are synthesised from the rank: only the ordering is meaningful.
    assert results[0].score > results[1].score > results[2].score


def test_replay_honours_k_by_taking_the_top_of_the_stored_ranking(stored):
    results = ReplayRetriever(stored).retrieve("What was capex?", k=2)
    assert [sc.chunk.page for sc in results] == [46, 12]


def test_replayed_chunks_carry_distinct_ids(stored):
    """The floor deduplicates by chunk_id, so replayed passages must not collide."""
    ids = [sc.chunk.chunk_id for sc in ReplayRetriever(stored).retrieve("What was capex?", k=3)]
    assert len(ids) == len(set(ids)) == 3


def test_an_unseen_question_fails_loudly_rather_than_returning_nothing(stored):
    """Silently returning an empty context would look like a retrieval failure."""
    with pytest.raises(KeyError, match="no stored retrieval"):
        ReplayRetriever(stored).retrieve("a question never materialised", k=3)


def test_replay_keeps_the_corpus_chunk_id_when_the_file_stored_it(tmp_path):
    """A passage must be traceable to its neighbours in the chunks file."""
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "financebench_id_00001",
                "question": "What was capex?",
                "sources": [
                    {"chunk_id": f"D_2022_10K::{i}", "doc_id": "D_2022_10K", "page": 3, "text": "t"}
                    for i in (41, 7)
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ids = [sc.chunk.chunk_id for sc in ReplayRetriever(str(path)).retrieve("What was capex?", k=2)]
    assert ids == ["D_2022_10K::41", "D_2022_10K::7"]
