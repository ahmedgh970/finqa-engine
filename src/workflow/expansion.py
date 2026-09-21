"""Widen a selection of passages to the chunks that surround them in their document.

A financial statement rarely fits one chunk: the chunker cuts a balance sheet into
consecutive pieces, and retrieval brings back whichever piece scored best -- often the
half without the figure the question needs. Retrieval works better on small chunks
(a focused passage embeds and re-scores more precisely), reading works better on whole
tables, so the two are decoupled: select small, then read the neighbourhood.

``window`` is a number of chunks on each side. Neighbours are taken from the same
document by chunk index, so a passage is only ever extended with text that physically
surrounds it. The selection keeps its order: each kept passage is followed by its
neighbourhood, and a passage already pulled in is never repeated, so two adjacent
selected chunks merge into one contiguous block instead of duplicating.
"""

from __future__ import annotations

from src.ingestion.schema import Chunk


def chunk_index(chunk_id: str) -> int | None:
    """Position of the chunk in its document, or ``None`` when the id has no index.

    Ids are ``{doc_id}::{index}``; a replayed file that stored no corpus id falls back
    to a synthesised one, which carries no neighbourhood and is left untouched.
    """
    doc, _, suffix = chunk_id.rpartition("::")
    return int(suffix) if doc and suffix.isdigit() else None


def expand(selection: list[Chunk], corpus: dict[str, Chunk], window: int) -> list[Chunk]:
    """``selection`` with each passage's neighbours within ``window`` chunks added."""
    if window <= 0:
        return list(selection)

    expanded: list[Chunk] = []
    seen: set[str] = set()
    for chunk in selection:
        index = chunk_index(chunk.chunk_id)
        if index is None:
            neighbourhood = [chunk]
        else:
            doc = chunk.chunk_id.rpartition("::")[0]
            ids = (f"{doc}::{i}" for i in range(index - window, index + window + 1))
            # A passage the corpus does not hold -- a different corpus, or an id the
            # replayed file synthesised -- keeps itself and simply gains no neighbour,
            # rather than dropping out of the context it was selected for.
            neighbourhood = [
                corpus.get(cid, chunk) for cid in ids if cid in corpus or cid == chunk.chunk_id
            ]
        for neighbour in neighbourhood:
            if neighbour.chunk_id not in seen:
                seen.add(neighbour.chunk_id)
                expanded.append(neighbour)
    return expanded
