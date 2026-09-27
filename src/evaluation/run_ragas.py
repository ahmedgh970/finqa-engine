"""CLI: score generated answers with Ragas.

Scores every answer of one or more answers files on the config's metrics
(faithfulness, answer relevancy, context precision / recall) with a local critic,
writing ``data/processed/ragas/{answers stem}_ragas_by_{critic}.jsonl``. Resumable:
ids already scored are skipped, so an interrupted run continues where it stopped.

Usage:
    python -m src.evaluation.run_ragas --config configs/evaluation/ragas/ragas.yaml
    python -m src.evaluation.run_ragas --config configs/evaluation/ragas/ragas.yaml \
        --answers data/processed/answers/naive/*_k10.jsonl --limit 50
    python -m src.evaluation.run_ragas --config configs/evaluation/ragas/ragas.yaml \
        --model ollama_chat/llama3.1:8b --id financebench_id_03029
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from src.evaluation.common.io import append_jsonl, derived_path, done_ids, read_jsonl, select
from src.evaluation.ragas.config import RagasConfig, load_ragas_config
from src.evaluation.ragas.critic import critic_config, ensure_critic
from src.evaluation.ragas.metrics import build_metrics, score_record

RAGAS_DIR = Path("data/processed/ragas")


def ragas_path(config: RagasConfig) -> Path:
    """``data/processed/ragas/{answers stem}_ragas_by_{critic}.jsonl``."""
    critic = critic_config(config.llm).model
    return derived_path(config.answers_path, RAGAS_DIR, f"ragas_by_{critic}")


def score(config: RagasConfig, qa_id: str | None = None, limit: int | None = None) -> Path:
    """Score the not-yet-scored answers of ``config.answers_path``; returns the output file."""
    records = select(read_jsonl(config.answers_path), qa_id, limit)
    out_path = ragas_path(config)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = done_ids(out_path)
    remaining = [r for r in records if r["id"] not in done]
    if remaining:
        ensure_critic(config.llm)
        critic = critic_config(config.llm)
        metrics = build_metrics(critic, config.embedding_model, config.metrics)
        for r in tqdm(remaining, desc=f"ragas [{critic.model}]"):
            append_jsonl(
                out_path, {"id": r["id"], "question": r["question"], **score_record(r, metrics)}
            )

    ids = {r["id"] for r in records}
    scored = [s for s in read_jsonl(out_path) if s["id"] in ids] if out_path.exists() else []
    means = {
        k: sum(s[k] for s in scored) / len(scored)
        for k in (scored[0].keys() - {"id", "question"} if scored else ())
    }
    summary = " | ".join(f"{k}={v:.3f}" for k, v in sorted(means.items()))
    print(
        f"{Path(config.answers_path).name}: scored {len(remaining)} new, "
        f"skipped {len(records) - len(remaining)} -> {out_path}\n  {summary}"
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Score answers with Ragas.")
    parser.add_argument("--config", required=True, help="Ragas YAML config.")
    parser.add_argument("--answers", nargs="+", help="Answers JSONL(s), overriding the config's.")
    parser.add_argument("--model", help="Critic model, overriding the config's.")
    parser.add_argument("--id", help="Score only this QA id.")
    parser.add_argument("--limit", type=int, help="Score only the first N answers per file.")
    args = parser.parse_args()

    config = load_ragas_config(args.config)
    if args.model:
        config = config.model_copy(
            update={"llm": config.llm.model_copy(update={"model": args.model})}
        )
    for answers in args.answers or [config.answers_path]:
        score(config.model_copy(update={"answers_path": answers}), args.id, args.limit)


if __name__ == "__main__":
    main()
