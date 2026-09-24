"""How much of each question's gold evidence physically reached the prompt.

Page-level recall answers a different question -- it says a passage landed on a page
the evidence sits on, not that the figures the answer needs are in front of the model.
This reads the context itself and compares it with the evidence text, per question:

- ``overlap``: share of the evidence's word 5-grams present in the context. A span of
  the evidence matches only when it is there in full, so this is the honest measure of
  "did the evidence reach the prompt";
- ``words`` / ``numbers``: the looser set-based shares, kept for continuity;
- ``all_numbers``: every figure of the evidence is in the context.

Any file whose records carry ``question`` and ``sources`` works: a materialised
retrieval, a widened context, or a workflow answers file -- whose sources are what the
generator actually read, after grading, expansion and the trim to the pinned window.

    uv run python scripts/evidence_overlap.py --context <label>=<file.jsonl> ... \
        [--out report.jsonl] [--ids 02987,04735]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from src.evaluation.common.golden_set import load_golden_set
from src.evaluation.retrieval.coverage import approx_tokens, evidence_coverage

GOLDEN = "data/jsons/financebench_open_source.jsonl"


def score_file(label: str, path: str, qas: dict, ids: set[str] | None) -> list[dict]:
    """One record per question: its context size and how much evidence it holds."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        qa = qas.get(record["id"])
        if qa is None or (ids and record["id"][-5:] not in ids):
            continue
        context = " ".join(s["text"] for s in record["sources"])
        coverage = evidence_coverage(context, [ev.text for ev in qa.evidence])
        rows.append(
            {
                "row": label,
                "id": record["id"][-5:],
                "passages": len(record["sources"]),
                "tokens": round(approx_tokens(context)),
                "evidences": len(qa.evidence),
                "overlap": coverage["evidence_overlap"],
                "words": coverage["evidence_words"],
                "numbers": coverage["evidence_numbers"],
                "figures": coverage["evidence_figures"],
                "all_numbers": coverage["all_evidence_numbers"],
                "all_figures": coverage["all_evidence_figures"],
            }
        )
    return rows


def _mean(values: list[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return statistics.mean(kept) if kept else None


def summarise(rows: list[dict]) -> None:
    """Per-row averages, plus how many questions hold the evidence almost in full."""
    print(
        f"{'ligne':<22}{'n':>4}{'passages':>9}{'tokens':>8}"
        f"{'prose':>8}{'mots':>7}{'chiffres':>10}{'montants':>10}{'tous':>7}"
    )
    for label in dict.fromkeys(r["row"] for r in rows):
        group = [r for r in rows if r["row"] == label]
        # The span overlap only means something on narrative evidence (see coverage.py),
        # so it is averaged over the questions whose evidence is prose to begin with.
        prose = [r["overlap"] for r in group if r["overlap"] and r["overlap"] > 0.3]
        print(
            f"{label:<22}{len(group):>4}{_mean([r['passages'] for r in group]):>9.1f}"
            f"{_mean([r['tokens'] for r in group]):>8.0f}{_mean(prose) or 0:>8.3f}"
            f"{_mean([r['words'] for r in group]):>7.3f}"
            f"{_mean([r['numbers'] for r in group]):>10.3f}"
            f"{_mean([r['figures'] for r in group]):>10.3f}"
            f"{_mean([r['all_figures'] for r in group]):>7.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidence actually present in each context.")
    parser.add_argument(
        "--context",
        action="append",
        required=True,
        metavar="LABEL=FILE",
        help="A context file to score, named for the table (repeatable).",
    )
    parser.add_argument("--out", help="Write the per-question records to this JSONL.")
    parser.add_argument("--ids", help="Comma-separated 5-digit QA ids to restrict to.")
    args = parser.parse_args()

    ids = set(args.ids.split(",")) if args.ids else None
    qas = {qa.id: qa for qa in load_golden_set(GOLDEN)}

    rows: list[dict] = []
    for spec in args.context:
        label, _, path = spec.partition("=")
        rows += score_file(label, path, qas, ids)

    summarise(rows)
    if args.out:
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        print(f"\nper-question -> {args.out}")


if __name__ == "__main__":
    main()
