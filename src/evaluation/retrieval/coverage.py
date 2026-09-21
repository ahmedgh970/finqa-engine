"""How much of the gold evidence text a retrieved context actually contains.

Recall is defined at the page level: a passage counts as soon as it lands on a gold
page, whether or not it holds the figures the question needs. With small chunks a
page is split across several passages, so recall can stay flat while the evidence
that reaches the prompt shrinks. These measures read the context itself:

- ``evidence_words``: share of the evidence's words found in the context;
- ``evidence_numbers``: share of the evidence's numbers found in the context --
  the facts a financial answer is built from, and less diluted by common words;
- ``all_evidence_numbers``: 1 when every evidence number is in the context. Strict,
  since an evidence is often a whole statement page of which the answer uses two
  figures;
- ``tokens``: context size, approximated at 4 characters per token (within a few
  percent of the BGE-M3 tokenizer on this corpus).

Questions whose evidence holds no number get ``None`` for the two number measures
and are left out of their average rather than counted as fully covered.
"""

from __future__ import annotations

import re

from src.evaluation.common.matching import words

_NUMBER = re.compile(r"\d[\d,]*\.?\d*")


def numbers(text: str) -> set[str]:
    """Numbers in ``text``, thousands separators and trailing dots dropped."""
    found = (n.replace(",", "").rstrip(".") for n in _NUMBER.findall(text))
    return {n for n in found if n}


def approx_tokens(text: str) -> float:
    """Token count estimated at 4 characters per token."""
    return len(text) / 4


def evidence_coverage(context: str, evidence: list[str]) -> dict[str, float | None]:
    """Coverage of the ``evidence`` texts by ``context`` (see the module docstring)."""
    ev_text = " ".join(evidence)
    ev_words = words(ev_text)
    ev_numbers = numbers(ev_text)
    ctx_numbers = numbers(context)
    return {
        "evidence_words": len(ev_words & words(context)) / len(ev_words) if ev_words else 1.0,
        "evidence_numbers": (
            len(ev_numbers & ctx_numbers) / len(ev_numbers) if ev_numbers else None
        ),
        "all_evidence_numbers": float(ev_numbers <= ctx_numbers) if ev_numbers else None,
        "tokens": approx_tokens(context),
    }
