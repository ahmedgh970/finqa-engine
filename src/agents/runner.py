"""CLI runner: answer every FinanceBench QA with the agentic RAG pipeline.

Builds the (dense) retriever once, answers each QA with the depth-escalating
tool-using agent, and writes the results -- answer, sources actually retrieved,
latency, and per-tool call counts -- to a JSONL for later judging. Resumable: QA
ids already in the output file are skipped and new answers appended, so a run cut
short (or a GPU hiccup) doesn't lose progress -- rerun the same command to resume.

Usage:
    python -m src.agents.runner --config configs/agents/agent_dense_granite3b.yaml
    python -m src.agents.runner --config configs/agents/agent_dense_granite3b.yaml --id financebench_id_03029
    python -m src.agents.runner --config configs/agents/agent_dense_granite3b.yaml --limit 20 --trace
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.agents.agent_rag import answer_agentic
from src.agents.config import AgentConfig, load_agent_config
from src.evaluation.common.golden_set import load_golden_set
from src.evaluation.common.schema import QAItem
from src.retrieval.registry import build_retriever
from src.tracing import setup_tracing


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


def _output_path(config: AgentConfig) -> Path:
    """Where answers are written: one file per (retriever, corpus, LLM, depths, prompt).

    The ``agent_`` prefix and the escalating depths distinguish these from the naive
    runner's files, so the resume logic never mixes the two; the prompt version is
    encoded too, so prompt variants are separate experiments rather than overwrites.
    """
    model = config.llm.model.replace("/", "_")
    depths = "-".join(map(str, config.depths))
    return (
        Path("data/processed/answers")
        / f"agent_{config.retriever}_{config.collection_name}_{model}"
        f"_d{depths}_{config.prompt_version}.jsonl"
    )


def run(
    config: AgentConfig,
    qa_id: str | None = None,
    limit: int | None = None,
    trace: bool = False,
) -> str:
    """Answer every not-yet-answered QA (or just ``qa_id``) and append results to a JSONL."""
    if trace:
        setup_tracing(project_name="financerag-agent")

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
        for qa in tqdm(remaining, desc=f"agent [{config.llm.model}]"):
            result = answer_agentic(
                qa.question,
                retriever,
                config.llm,
                doc_id=qa.doc_name if config.doc_scoped else None,
                depths=config.depths,
                num_ctx=config.num_ctx,
                recursion_limit=config.recursion_limit,
                prompt_version=config.prompt_version,
            )
            record = {
                "id": qa.id,
                "question": qa.question,
                "gold_answer": qa.answer,
                "generated_answer": result.answer,
                "sources": [
                    {"doc_id": c.doc_id, "page": c.page, "text": c.text} for c in result.sources
                ],
                "latency_s": result.latency_s,
                "n_llm_calls": result.n_llm_calls,
                "n_retrieve": result.n_retrieve,
                "n_calculator": result.n_calculator,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()

    print(f"Answered {len(remaining)} new QA (skipped {len(done)} already present) -> {out_path}")
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Answer FinanceBench QA with the agentic RAG pipeline."
    )
    parser.add_argument("--config", required=True, help="Path to an agent YAML config.")
    parser.add_argument("--id", help="Answer only this QA id, skipping the rest.")
    parser.add_argument("--limit", type=int, help="Only the first N QA (quick subset run).")
    parser.add_argument(
        "--trace", action="store_true", help="Export the agent loop to local Phoenix."
    )
    parser.add_argument("--prompt", help="Override the config's prompt_version (e.g. v2).")
    args = parser.parse_args()
    config = load_agent_config(args.config)
    if args.prompt:
        config = config.model_copy(update={"prompt_version": args.prompt})
    run(config, qa_id=args.id, limit=args.limit, trace=args.trace)


if __name__ == "__main__":
    main()
