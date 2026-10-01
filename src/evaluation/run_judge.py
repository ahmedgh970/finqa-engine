"""CLI: judge generated answers against gold with one of the judging protocols.

The protocol comes from the config (``configs/evaluation/judge/{protocol}.yaml``):

- ``grid``: the judge says whether each answer is correct or a refusal, writing
  ``data/processed/judged/verdicts/{answers stem}.{judge}.jsonl``; once every answer
  has a verdict, the answers are placed in the evidence-grounded outcome grid,
  ``..._grid_by_{judge}.jsonl`` (ADR 0004).
- ``correct_grounded``: correct / grounded verdicts, ``..._judged_by_{judge}.jsonl``.
- ``prometheus``: Prometheus-2 1-5 scores, ``..._judged_by_{judge}.jsonl`` (ADR 0003).

Resumable: ids already judged are skipped, so an interrupted run continues where it
stopped, and a complete verdicts file (an external judge's, say) goes straight to
the grid.

Usage:
    python -m src.evaluation.run_judge --config configs/evaluation/judge/grid.yaml
    python -m src.evaluation.run_judge --config configs/evaluation/judge/grid.yaml \
        --answers data/processed/answers/<run>.jsonl --model ollama_chat/qwen3.5:9b
    python -m src.evaluation.run_judge --config configs/evaluation/judge/correct_grounded.yaml \
        --answers data/processed/answers/naive/*_k20.jsonl
    python -m src.evaluation.run_judge --config configs/evaluation/judge/prometheus.yaml --id financebench_id_03029
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from src.evaluation.common.io import append_jsonl, derived_path, done_ids, read_jsonl, select
from src.evaluation.judge.config import JudgeConfig, load_judge_config
from src.evaluation.judge.grid import (
    JUDGED_DIR,
    Outcome,
    build_grid,
    grid_path,
    verdicts_path,
    with_gold_justification,
)
from src.evaluation.judge.protocols import get_protocol


def judged_path(config: JudgeConfig) -> Path:
    """Where the verdicts go: the grid's verdicts file, else ``..._judged_by_{judge}.jsonl``."""
    if config.protocol == "grid":
        return verdicts_path(config)
    return derived_path(config.answers_path, JUDGED_DIR, f"judged_by_{config.judge_name}")


def with_model(config: JudgeConfig, model: str) -> JudgeConfig:
    """``config`` graded by ``model`` instead, re-validated, under its own output tag."""
    data = config.model_dump()
    data["llm"]["model"] = model
    data["name"] = None
    return JudgeConfig(**data)


def score(config: JudgeConfig, qa_id: str | None = None, limit: int | None = None) -> Path:
    """Judge the not-yet-judged answers of ``config.answers_path``; returns the verdicts file."""
    protocol = get_protocol(config.protocol)
    records = select(read_jsonl(config.answers_path), qa_id, limit)
    out_path = judged_path(config)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = done_ids(out_path)
    remaining = [r for r in records if r["id"] not in done]
    if config.protocol == "grid":
        remaining = with_gold_justification(remaining, config.golden_set_path)
    for r in tqdm(remaining, desc=f"{config.protocol} [{config.judge_name}]"):
        verdict = protocol.judge(r, config.llm)
        append_jsonl(
            out_path,
            {
                "id": r["id"],
                "question": r["question"],
                "gold_answer": r["gold_answer"],
                "generated_answer": r["generated_answer"],
                **verdict,
                "judge_model": config.llm.model,
            },
        )

    judged = [v for v in read_jsonl(out_path) if v["id"] in {r["id"] for r in records}]
    print(
        f"{Path(config.answers_path).name}: judged {len(remaining)} new, "
        f"skipped {len(records) - len(remaining)} -> {out_path}\n  {protocol.summarize(judged)}"
    )
    return out_path


def grid(config: JudgeConfig) -> Path:
    """Write the outcome grid of one judged answers file and print its counts."""
    records = build_grid(config)
    out_path = grid_path(config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    counts = Counter(r["outcome"] for r in records)
    n = len(records)
    print(f"{n} answers -> {out_path}")
    for o in Outcome:
        print(f"  {o.value:<14} {counts[o.value]:>4}  {100 * counts[o.value] / n:5.1f}%")
    missed = sum(r["outcome"] == Outcome.DONT_KNOW and r["evidence_retrieved"] for r in records)
    print(f"  dont_know with the evidence in context: {missed}")
    return out_path


def judge(config: JudgeConfig, qa_id: str | None = None, limit: int | None = None) -> Path:
    """Score one answers file; for the grid protocol, then place it in the grid.

    The grid covers the whole answers file, so a ``qa_id`` / ``limit`` spot check
    only writes verdicts.
    """
    out_path = score(config, qa_id, limit)
    if config.protocol != "grid":
        return out_path
    if qa_id is not None or limit is not None:
        print("  grid not built: it needs a verdict for every answer (drop --id / --limit)")
        return out_path
    return grid(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge answers against gold.")
    parser.add_argument("--config", required=True, help="Judge YAML config (picks the protocol).")
    parser.add_argument("--answers", nargs="+", help="Answers JSONL(s), overriding the config's.")
    parser.add_argument("--model", help="Judge model, overriding the config's.")
    parser.add_argument("--id", help="Judge only this QA id.")
    parser.add_argument("--limit", type=int, help="Judge only the first N answers per file.")
    args = parser.parse_args()

    config = load_judge_config(args.config)
    if args.model:
        config = with_model(config, args.model)
    for answers in args.answers or [config.answers_path]:
        judge(config.model_copy(update={"answers_path": answers}), args.id, args.limit)


if __name__ == "__main__":
    main()
