"""Replay a retrieval that was computed once and stored.

Retrieval is deterministic and, for a fixed corpus and query, its result never
changes -- yet re-running it costs an embedding pass and, with a cross-encoder,
the dominant share of a run's compute. Materialised retrievals let every later
experiment reuse that work instead of recomputing it.

Two consequences beyond the saved time. Every configuration replaying the same
file sees *byte-identical* passages, so a difference between them is attributable
to what follows retrieval and not to retrieval drift. And no embedder or
cross-encoder is loaded, which leaves the whole GPU to the generator.

This is a benchmarking device, not a serving one: it can only answer questions it
has already seen, and it ignores the query text beyond looking it up.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.ingestion.schema import Chunk
from src.retrieval.base import ScoredChunk


class ReplayRetriever:
    """Serve stored passages, keyed by the exact question text they were fetched for."""

    def __init__(self, path: str):
        self._by_question: dict[str, list[Chunk]] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            self._by_question[record["question"]] = [
                Chunk(
                    # The corpus id when the file stored it, so a passage can be traced
                    # back to its neighbours; older files carry none, and position in
                    # the stored ranking is then a stable stand-in (the floor only needs
                    # it to avoid duplicates).
                    chunk_id=source.get("chunk_id") or f"{record['id']}::{i}",
                    doc_id=source["doc_id"],
                    page=source["page"],
                    text=source["text"],
                )
                for i, source in enumerate(record["sources"])
            ]
        self.path = path

    def retrieve(self, query: str, k: int = 5, doc_id: str | None = None) -> list[ScoredChunk]:
        """Return the first ``k`` stored passages, in the ranking they were stored in.

        ``doc_id`` is ignored: the stored retrieval was already document-scoped. Scores
        are synthesised from the rank, since only the ordering is meaningful here.
        """
        try:
            chunks = self._by_question[query]
        except KeyError:
            raise KeyError(
                f"no stored retrieval for this question in {self.path}. Replay only "
                f"serves questions it was materialised for: {query[:80]!r}"
            ) from None
        return [
            ScoredChunk(chunk=c, score=1.0 - i / max(len(chunks), 1))
            for i, c in enumerate(chunks[:k])
        ]
