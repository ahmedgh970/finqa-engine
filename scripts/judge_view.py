"""Render the judging view, one file per question, for one or more runs side by side.

For each question: the gold answer and its justification, the physical page each
gold evidence resolves to, then every run's *complete* answer with the passages
that were actually in its prompt and whether each gold page reached it. Passages
shared by several runs are printed once and referenced by id, and those sitting
on a gold page are flagged. Meant to be read question by question, never truncated.

    uv run python scripts/judge_view.py --start 1 --count 150 \
        --answers advanced=data/processed/answers/workflow/workflow_advanced_..._24kc.jsonl \
        --answers grade=data/processed/answers/workflow/workflow_grade_..._12kc.jsonl \
        --out-dir /tmp/views
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.common.golden_set import load_golden_set  # noqa: E402
from src.evaluation.common.io import read_jsonl  # noqa: E402
from src.evaluation.common.matching import build_page_index, resolve_evidence_page  # noqa: E402
from src.evaluation.common.schema import QAItem  # noqa: E402
from src.evaluation.judge.grid import evidence_retrieved  # noqa: E402


def render_question(
    n: int,
    qa: QAItem,
    justification: str,
    evidence_pages: list[int | None],
    answers: dict[str, dict],
) -> str:
    gold_pages = set(evidence_pages)
    out = [
        f"# Q{n} {qa.id} ({qa.doc_name})",
        f"QUESTION: {qa.question}",
        f"GOLD: {qa.answer}",
        f"GOLD JUSTIFICATION: {justification}",
        f"GOLD EVIDENCE PAGES: {evidence_pages}",
        "",
    ]
    passages: dict[str, str] = {}
    for r in answers.values():
        for s in r["sources"]:
            passages.setdefault(s["text"], f"P{len(passages) + 1}")
    for name, r in answers.items():
        in_prompt = {s["page"] for s in r["sources"] if s["doc_id"] == qa.doc_name}
        per_page = ", ".join(
            f"p.{p} {'IN' if p in in_prompt else 'MISSING'}" for p in evidence_pages
        )
        retrieved = evidence_retrieved(r["sources"], qa.doc_name, evidence_pages)
        mapping = ", ".join(
            f"Source {i + 1}={passages[s['text']]}" for i, s in enumerate(r["sources"])
        )
        out += [
            f"## {name} -- evidence retrieved: {retrieved} ({per_page})",
            f"SOURCES: {mapping}",
            "ANSWER:",
            r["generated_answer"].strip(),
            "",
        ]
    out.append("## PASSAGES")
    for text, pid in passages.items():
        s = next(s for r in answers.values() for s in r["sources"] if s["text"] == text)
        flag = " [GOLD PAGE]" if s["doc_id"] == qa.doc_name and s["page"] in gold_pages else ""
        out += [f"[{pid}] (p.{s['page']}){flag} {text}", ""]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Judging view, one file per question.")
    parser.add_argument(
        "--answers", action="append", required=True, help="NAME=answers.jsonl, repeat per run."
    )
    parser.add_argument("--start", type=int, required=True, help="1-based question index.")
    parser.add_argument("--count", type=int, required=True, help="Number of questions.")
    parser.add_argument("--golden-set", default="data/jsons/financebench_open_source.jsonl")
    parser.add_argument(
        "--chunks", default="data/processed/docling/chunked/hybrid/chunks_1024.jsonl"
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    justifications = {
        rec["financebench_id"]: rec.get("justification") or ""
        for rec in read_jsonl(args.golden_set)
    }
    qas = {qa.id: qa for qa in load_golden_set(args.golden_set)}
    runs = {name: read_jsonl(path) for name, path in (a.split("=", 1) for a in args.answers)}
    order = [r["id"] for r in next(iter(runs.values()))]
    by_id = {name: {r["id"]: r for r in rs} for name, rs in runs.items()}
    page_index = build_page_index(args.chunks, {qas[i].doc_name for i in order})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, qa_id in enumerate(order[args.start - 1 : args.start - 1 + args.count], args.start):
        qa = qas[qa_id]
        pages = [
            resolve_evidence_page(ev.text, page_index.get(qa.doc_name, {})) for ev in qa.evidence
        ]
        answers = {name: by_id[name][qa_id] for name in runs}
        text = render_question(n, qa, justifications.get(qa_id, ""), pages, answers)
        (out_dir / f"q{n:03d}.md").write_text(text)


if __name__ == "__main__":
    main()
