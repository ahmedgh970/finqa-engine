"""Record, for the demo's replay, the calculation behind each verified figure of a run.

An answers file keeps the figure the calculator verified, not the expression and the
table rows it came from. This reruns the calculator node on the exact context the run
gave it (the stored sources) with the run's own settings, and keeps the detail only
when it reproduces the recorded figure -- a detail that computed something else would
explain a different answer. One structured LLM call per verified figure.

    uv run python scripts/demo_calc_details.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.api.demo import ReplaySources
from src.ingestion.schema import Chunk
from src.workflow.config import load_workflow_config
from src.workflow.nodes import make_calculate


def main() -> None:
    parser = argparse.ArgumentParser(description="Record the calculations of a replayed run.")
    parser.add_argument("--replay", default="configs/demo/replay.yaml")
    parser.add_argument("--base", default="configs/workflow/serve_grade_exp1_calc_1024.yaml")
    parser.add_argument("--out", default="data/processed/demo/calc_details.jsonl")
    args = parser.parse_args()

    sources = ReplaySources(**yaml.safe_load(Path(args.replay).read_text(encoding="utf-8")))
    config = sources.settings.config(load_workflow_config(args.base))
    calculate = make_calculate(config)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with open(sources.answers, encoding="utf-8") as f, out.open("w", encoding="utf-8") as sink:
        for record in (json.loads(line) for line in f if line.strip()):
            if not record.get("computed"):
                continue
            chunks = [
                Chunk(
                    chunk_id=s.get("chunk_id") or f"s{i}",
                    doc_id=s["doc_id"],
                    page=s["page"],
                    text=s["text"],
                )
                for i, s in enumerate(record["sources"])
            ]
            result = calculate({"question": record["question"], "chunks": chunks})
            same = result.get("computed") == record["computed"]
            print(
                f"{record['id'][-5:]}  recorded {record['computed']:<9} rerun {result.get('computed')}  {'kept' if same else 'DROPPED'}"
            )
            if same and result.get("calculation"):
                sink.write(
                    json.dumps({"id": record["id"], "calculation": result["calculation"]}) + "\n"
                )
                kept += 1
    print(f"{kept} calculations -> {out}")


if __name__ == "__main__":
    main()
