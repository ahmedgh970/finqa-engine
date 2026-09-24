"""Widen the contexts of a stored run to their neighbouring chunks, offline.

Reads a file whose records carry a question and its passages -- a materialised
retrieval or a workflow answers file, whose sources are the context the grader kept --
and writes the same records with each passage extended to the chunks around it
(`src/workflow/expansion.py`). No retriever, no LLM: the neighbours come from the
chunks file by id.

The output has the materialised-prompts schema, so it is scored by the retrieval eval
(a config whose `replay_path` points at it) and replayed by a workflow run, which is
how a window is measured before being wired into the graph.

    uv run python scripts/expand_context.py --context <run.jsonl> \
        --chunks data/processed/docling/chunked/hybrid/chunks_256.jsonl --window 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ingestion.schema import Chunk
from src.ingestion.storage import read_chunks
from src.llm.prompts import build_prompt
from src.workflow.expansion import expand

OUT_DIR = Path("data/processed/prompts/expanded")


def _out_path(context_path: str, window: int) -> Path:
    return OUT_DIR / f"{Path(context_path).stem}_pm{window}.jsonl"


def run(context_path: str, chunks_path: str, window: int, out_path: Path | None = None) -> Path:
    corpus = {c.chunk_id: c for c in read_chunks(chunks_path)}
    # Files written before passages carried their corpus id are still usable: a stored
    # passage is the chunk's text verbatim, and chunk texts are unique in a corpus, so
    # the id is recovered by lookup rather than by re-running the retriever.
    by_text = {c.text: c.chunk_id for c in corpus.values()}
    out_path = out_path or _out_path(context_path, window)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    grew = 0
    with out_path.open("w", encoding="utf-8") as out:
        for line in Path(context_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            selection = [
                Chunk(
                    chunk_id=s.get("chunk_id") or by_text.get(s["text"], ""),
                    doc_id=s["doc_id"],
                    page=s["page"],
                    text=s["text"],
                )
                for s in record["sources"]
            ]
            widened = expand(selection, corpus, window)
            grew += len(widened) > len(selection)
            written += 1
            out.write(
                json.dumps(
                    {
                        "id": record["id"],
                        "question": record["question"],
                        "gold_answer": record.get("gold_answer"),
                        "doc_name": record.get("doc_name"),
                        "k": len(widened),
                        "n_chunks": len(widened),
                        "n_selected": len(selection),
                        "window": window,
                        "sources": [
                            {
                                "chunk_id": c.chunk_id,
                                "doc_id": c.doc_id,
                                "page": c.page,
                                "text": c.text,
                            }
                            for c in widened
                        ],
                        "prompt": build_prompt(record["question"], widened),
                    }
                )
                + "\n"
            )
    print(f"expanded {written} contexts (+-{window} chunks), {grew} grew -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extend stored contexts to neighbouring chunks.")
    parser.add_argument("--context", required=True, help="Answers or prompts JSONL to widen.")
    parser.add_argument("--chunks", required=True, help="Chunks JSONL of the same corpus.")
    parser.add_argument("--window", type=int, default=1, help="Chunks on each side (default 1).")
    parser.add_argument("--out", help="Output path (default: alongside, suffixed _pm{window}).")
    args = parser.parse_args()
    run(args.context, args.chunks, args.window, Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
