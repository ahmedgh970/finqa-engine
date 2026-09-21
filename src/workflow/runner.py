"""CLI runner: answer every FinanceBench QA with the deterministic CRAG workflow.

Builds the retriever once, runs the configured graph per QA, and writes answer +
sources + the workflow instrumentation (rewrite rounds, calculator trigger,
per-node latency) to a JSONL for later judging. Resumable: QA ids already in the
output file are skipped, so an interrupted run continues where it stopped.

The output filename encodes the ablation cell and the pinned context window, so each
row of the matrix, and each window a row is run at, is a separate experiment instead
of overwriting the previous one.

Usage:
    python -m src.workflow.runner --config configs/workflow/advanced.yaml
    python -m src.workflow.runner --config configs/workflow/advanced.yaml --limit 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.evaluation.common.golden_set import load_golden_set
from src.evaluation.common.schema import QAItem
from src.retrieval.registry import build_retriever
from src.tracing import setup_tracing
from src.workflow.config import WorkflowConfig, load_workflow_config, variant_name
from src.workflow.graph import answer_workflow


def _answered_ids(path: Path) -> set[str]:
    """QA ids already answered in ``path`` (empty set if it doesn't exist yet)."""
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def _select(qas: list[QAItem], qa_id: str | None) -> list[QAItem]:
    """All QA, or just the one matching ``qa_id`` (for quick single-question checks)."""
    if qa_id is None:
        return qas
    selected = [qa for qa in qas if qa.id == qa_id]
    if not selected:
        raise SystemExit(f"No QA with id {qa_id!r} in the golden set.")
    return selected


def _context_tag(num_ctx: int | None) -> str:
    """Short name of the pinned context window: ``12kc`` for 12288 tokens.

    Empty when the window is not pinned (the client then sizes it per prompt); a
    window that is not a multiple of 1024 keeps its exact size (``10000c``).
    """
    if num_ctx is None:
        return ""
    return f"{num_ctx // 1024}kc" if num_ctx % 1024 == 0 else f"{num_ctx}c"


def _output_path(config: WorkflowConfig) -> Path:
    """One file per (ablation cell, retriever, corpus, LLM, k, context window).

    The window is part of a row's definition: it decides how many passages survive
    the trim, and the same prompt answered at two windows gives different answers.
    """
    model = config.llm.model.replace("/", "_")
    ctx = _context_tag(config.llm.num_ctx)
    return (
        Path("data/processed/answers/workflow")
        / f"workflow_{variant_name(config)}_{config.retriever}_{config.collection_name}"
        f"_{model}_k{config.k}{f'_{ctx}' if ctx else ''}.jsonl"
    )


def run(
    config: WorkflowConfig,
    qa_id: str | None = None,
    limit: int | None = None,
    trace: bool = False,
) -> str:
    """Answer every not-yet-answered QA (or just ``qa_id``) and append to a JSONL."""
    if trace:
        setup_tracing(project_name="financerag-workflow")

    qas = _select(load_golden_set(config.golden_set_path), qa_id)
    if limit is not None:
        qas = qas[:limit]

    out_path = _output_path(config)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = _answered_ids(out_path)
    remaining = [qa for qa in qas if qa.id not in done]
    if not remaining:
        print(f"All {len(qas)} QA already answered -> {out_path}")
        return str(out_path)

    retriever = build_retriever(config.retriever, config)

    with out_path.open("a", encoding="utf-8") as f:
        for qa in tqdm(remaining, desc=f"workflow [{variant_name(config)}]"):
            result = answer_workflow(
                qa.question,
                retriever,
                config,
                doc_id=qa.doc_name if config.doc_scoped else None,
            )
            record = {
                "id": qa.id,
                "question": qa.question,
                "gold_answer": qa.answer,
                "generated_answer": result.answer,
                "sources": [
                    {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "page": c.page, "text": c.text}
                    for c in result.sources
                ],
                "latency_s": result.latency_s,
                "n_retrieved": result.n_retrieved,
                "n_kept_by_grade": result.n_kept_by_grade,
                "n_kept_by_floor": result.n_kept_by_floor,
                "n_dropped_to_fit": result.n_dropped_to_fit,
                "n_expanded": result.n_expanded,
                "grades": result.grades,
                "max_grade": result.max_grade,
                "low_confidence": result.low_confidence,
                "llm_calls": result.llm_calls,
                "node_latencies": result.node_latencies,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()

    print(f"Answered {len(remaining)} new QA (skipped {len(done)} already present) -> {out_path}")
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Answer FinanceBench QA with the deterministic CRAG workflow."
    )
    parser.add_argument("--config", required=True, help="Path to a workflow YAML config.")
    parser.add_argument("--id", help="Answer only this QA id, skipping the rest.")
    parser.add_argument("--limit", type=int, help="Only the first N QA (quick subset run).")
    parser.add_argument(
        "--trace", action="store_true", help="Export the graph run to local Phoenix."
    )
    args = parser.parse_args()
    run(load_workflow_config(args.config), qa_id=args.id, limit=args.limit, trace=args.trace)


if __name__ == "__main__":
    main()
