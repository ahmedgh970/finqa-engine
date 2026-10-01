"""Run the extraction step alone and read what it produced, question by question.

Tuning a prompt against a score teaches you the score. This prints the object the model
returned, which check rejected it and what the arithmetic gave, so the prompt is fixed
against what it actually does: a quote that cannot be found, a period read off the wrong
column, a formula invented rather than taken from the wording.

It writes nothing to the answers files and never generates: one structured call per
question, the deterministic checks, and the result on screen. Tune on questions the
measured rows do not target, then freeze the prompt before running the row itself.

    uv run python scripts/calc_tuning.py --config configs/workflow/grade_exp1_calc_1024_replayed.yaml \
        --ids 02608,06272,02981
"""

from __future__ import annotations

import argparse
import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from src.evaluation.common.golden_set import load_golden_set
from src.ingestion.schema import Chunk
from src.llm.client import StructuredOutputError, generate_structured
from src.workflow.calculator import (
    CalculationError,
    answer_format,
    compute,
    routes_to_calculator,
    rows_of,
)
from src.workflow.config import load_workflow_config
from src.workflow.nodes import _fit_context
from src.workflow.prompts import build_calc_prompt
from src.workflow.schemas import CalcSpec


def contexts(replay_path: str) -> dict[str, dict]:
    """The stored context of each question, keyed by QA id."""
    return {
        json.loads(line)["id"]: json.loads(line)
        for line in Path(replay_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _same_number(rendered: str, gold: str) -> bool:
    """Whether the gold's first figure equals the result at the result's own precision."""
    found = re.search(r"-?\d[\d,]*\.?\d*", gold)
    if not found:
        return False
    ours = Decimal(rendered.strip("%$"))
    theirs = Decimal(found.group(0).replace(",", ""))
    return ours == theirs.quantize(ours, rounding=ROUND_HALF_UP)


def report(spec: CalcSpec, rows: list, question: str, gold: str) -> None:
    """Print the specification with the rows it points to, then what the checks made of it."""
    places, percent = answer_format(question)
    print(f"    expression : {spec.expression}   (question: round_to={places} percent={percent})")
    by_number = {r.number: r for r in rows}
    for v in spec.variables:
        row = by_number.get(v.line)
        shown = f"{row.shown()}  [{row.statement}, {row.scale}]" if row else "(no such line)"
        print(f"    {v.name:<16} [{v.term}] -> {shown[:140]}")
    if not spec.variables:
        print("    (abstention: the model reports the values are not all present)")

    try:
        result = compute(spec, rows, question)
    except CalculationError as exc:
        print(f"    REFUSÉ    {exc}")
        return
    for v in spec.variables:
        if result.lines.get(v.name) not in (None, v.line):
            moved = by_number[result.lines[v.name]]
            print(
                f"    rattaché  {v.name}: L{v.line} -> {moved.shown()[:110]}  [{moved.statement}]"
            )
    rendered = result.rendered(places, percent)
    verdict = "=" if _same_number(rendered, gold) else "≠"
    print(f"    CALCULÉ   {rendered}   {verdict} gold {gold}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read what the extraction step produces.")
    parser.add_argument("--config", required=True, help="Workflow config whose context to reuse.")
    parser.add_argument("--ids", required=True, help="Comma-separated 5-digit QA ids.")
    parser.add_argument(
        "--model",
        help="Extraction model, e.g. ollama_chat/qwen3.5:9b (default: the config's). The "
        "checks have to hold whatever model points at the lines.",
    )
    args = parser.parse_args()

    config = load_workflow_config(args.config)
    if args.model:
        config = config.model_copy(
            update={"llm": config.llm.model_copy(update={"model": args.model})}
        )
    print(f"extraction model: {config.llm.model}")
    stored = contexts(config.replay_path)
    qas = {qa.id[-5:]: qa for qa in load_golden_set(config.golden_set_path)}

    for short in args.ids.split(","):
        qa = qas[short.strip()]
        record = stored[qa.id]
        chunks = _fit_context(
            [
                Chunk(
                    chunk_id=s.get("chunk_id", ""),
                    doc_id=s["doc_id"],
                    page=s["page"],
                    text=s["text"],
                )
                for s in record["sources"]
            ],
            qa.question,
            config,
        )
        routed = routes_to_calculator(qa.question)
        print(f"\n===== {short}  |  gold {qa.answer}  |  routé: {routed}")
        print(f"    {qa.question[:160]}")
        if not routed:
            continue
        rows = rows_of(chunks)
        try:
            spec = generate_structured(build_calc_prompt(qa.question, rows), config.llm, CalcSpec)
        except StructuredOutputError as exc:
            print(f"    SORTIE INVALIDE  {exc}")
            continue
        report(spec, rows, qa.question, str(qa.answer))


if __name__ == "__main__":
    main()
