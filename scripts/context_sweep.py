"""What each (corpus, grade threshold, expansion window, context window) would read.

Everything that builds a context is deterministic -- the retrieval is replayed from a
stored file, grading ran at temperature 0 and its grades are recorded per passage, the
expansion is arithmetic and the trim is a function of the text -- so a configuration
can be evaluated without generating anything. Replaying two measured rows this way
reproduced their contexts exactly (150/150 passages identical), which is what makes
the sweep a prediction rather than an estimate.

What it cannot predict is the answer: a context holding more evidence does not always
produce more good answers, so the winner of a sweep is a candidate to run, not a
result.

A corpus is given as `<size>=<prompts file>[,<graded answers file>]`. Without the
graded file only threshold 0 is available, since there are no grades to filter on.

    uv run python scripts/context_sweep.py --corpus 1024=<prompts>,<graded> \
        --thresholds 0 1 2 3 --windows 0 1 2 3 4 --num-ctx 12288 24576 --out sweep.csv
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from src.evaluation.common.golden_set import load_golden_set
from src.evaluation.retrieval.coverage import evidence_coverage
from src.ingestion.schema import Chunk
from src.ingestion.storage import read_chunks
from src.llm.client import estimated_context
from src.llm.prompts import build_prompt
from src.workflow.expansion import expand
from src.workflow.nodes import _apply_floor

GOLDEN = "data/jsons/financebench_open_source.jsonl"
CHUNKS = "data/processed/docling/chunked/hybrid/chunks_{size}.jsonl"
FLOOR = 3  # min_chunks of the shipped grading rows
OUTPUT_BUDGET = 1024  # llm.max_tokens of the shipped rows, counted by the trim


def read_jsonl(path: str) -> dict[str, dict]:
    return {
        json.loads(line)["id"]: json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def fit(selection: list[Chunk], question: str, num_ctx: int) -> list[Chunk]:
    """The prefix of ``selection`` that fits ``num_ctx`` -- the trim of the graph."""
    fitting: list[Chunk] = []
    for chunk in selection:
        if estimated_context(build_prompt(question, [*fitting, chunk]), OUTPUT_BUDGET) > num_ctx:
            break
        fitting.append(chunk)
    return fitting or selection[:1]


def sweep(size: int, prompts_path: str, graded_path: str | None, thresholds, windows, contexts):
    """One record per (threshold, window, num_ctx) cell of this corpus."""
    corpus = {c.chunk_id: c for c in read_chunks(CHUNKS.format(size=size))}
    by_text = {c.text: c.chunk_id for c in corpus.values()}  # pre-chunk_id files
    qas = {qa.id: qa for qa in load_golden_set(GOLDEN)}
    prompts = read_jsonl(prompts_path)
    graded = read_jsonl(graded_path) if graded_path else {}

    def chunks_of(record) -> list[Chunk]:
        return [
            Chunk(
                chunk_id=s.get("chunk_id") or by_text[s["text"]],
                doc_id=s["doc_id"],
                page=s["page"],
                text=s["text"],
            )
            for s in record["sources"]
        ]

    rows = []
    for threshold in thresholds:
        if threshold > 0 and not graded:
            continue  # no grades for this corpus: only the ungraded row exists
        for window in windows:
            for num_ctx in contexts:
                anchors, passages, tokens, figures, complete, trimmed = [], [], [], [], [], 0
                for qa_id, record in prompts.items():
                    chunks = chunks_of(record)
                    if threshold > 0:
                        grades = graded[qa_id]["grades"]
                        kept = [c for c, g in zip(chunks, grades, strict=True) if g >= threshold]
                        kept = _apply_floor(kept, chunks, FLOOR)
                        rank = {c.chunk_id: i for i, c in enumerate(chunks)}
                        kept.sort(key=lambda c: rank[c.chunk_id])
                    else:
                        kept = chunks
                    widened = expand(kept, corpus, window)
                    read = fit(widened, record["question"], num_ctx)
                    trimmed += len(read) < len(widened)
                    coverage = evidence_coverage(
                        " ".join(c.text for c in read), [ev.text for ev in qas[qa_id].evidence]
                    )
                    anchors.append(len(kept))
                    passages.append(len(read))
                    tokens.append(coverage["tokens"])
                    if coverage["evidence_figures"] is not None:
                        figures.append(coverage["evidence_figures"])
                        complete.append(coverage["all_evidence_figures"])
                rows.append(
                    {
                        "chunks": size,
                        "threshold": threshold,
                        "window": window,
                        "num_ctx": num_ctx,
                        "anchors": round(statistics.mean(anchors), 1),
                        "passages": round(statistics.mean(passages), 1),
                        "tokens": round(statistics.mean(tokens)),
                        "trimmed": trimmed,
                        "figures": round(statistics.mean(figures), 3),
                        "complete": round(statistics.mean(complete), 3),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep context configurations offline.")
    parser.add_argument(
        "--corpus",
        action="append",
        required=True,
        metavar="SIZE=PROMPTS[,GRADED]",
        help="Corpus to sweep: its materialised retrieval and, optionally, a graded run.",
    )
    parser.add_argument("--thresholds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--windows", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--num-ctx", type=int, nargs="+", default=[12288])
    parser.add_argument("--out", help="Write every cell to this CSV.")
    args = parser.parse_args()

    rows = []
    for spec in args.corpus:
        size, _, paths = spec.partition("=")
        prompts, _, graded = paths.partition(",")
        rows += sweep(
            int(size), prompts, graded or None, args.thresholds, args.windows, args.num_ctx
        )

    header = f"{'chunks':>7}{'seuil':>6}{'fen.':>5}{'num_ctx':>8}"
    header += f"{'ancres':>7}{'lus':>6}{'tokens':>8}{'coupées':>8}{'montants':>10}{'tous':>7}"
    print(header)
    for r in sorted(rows, key=lambda r: (-r["complete"], r["tokens"])):
        print(
            f"{r['chunks']:>7}{r['threshold']:>6}{r['window']:>5}{r['num_ctx']:>8}"
            f"{r['anchors']:>7.1f}{r['passages']:>6.1f}{r['tokens']:>8.0f}"
            f"{r['trimmed']:>8}{r['figures']:>10.3f}{r['complete']:>7.3f}"
        )
    if args.out:
        columns = list(rows[0])
        lines = [",".join(columns)] + [",".join(str(r[c]) for c in columns) for r in rows]
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n{len(rows)} cellules -> {args.out}")


if __name__ == "__main__":
    main()
