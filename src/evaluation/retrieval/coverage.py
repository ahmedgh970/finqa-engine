"""How much of the gold evidence text a retrieved context actually contains.

Recall is defined at the page level: a passage counts as soon as it lands on a gold
page, whether or not it holds the figures the question needs. With small chunks a
page is split across several passages, so recall can stay flat while the evidence
that reaches the prompt shrinks. These measures read the context itself:

- ``evidence_words``: share of the evidence's words found in the context, as sets.
  Generous: a financial page shares "total", "net" and its years with any other page;
- ``evidence_overlap``: share of the evidence's word 5-grams found in the context.
  Strict, and only meaningful for narrative evidence: the benchmark stores a table as
  the PDF's raw dump while the parser serialises it as ``label, column = value``, so a
  table's spans never match however completely its figures reached the prompt. Read it
  on prose, read the figures on tables;
- ``evidence_numbers``: share of the evidence's numbers found in the context --
  the facts a financial answer is built from, and less diluted by common words;
- ``evidence_figures``: the same restricted to the figures that carry the information,
  dropping years and one- or two-digit counts, which match by chance;
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

from src.evaluation.common.matching import _WORD, words

_NUMBER = re.compile(r"\d[\d,]*\.?\d*")
NGRAM = 5  # words per span; below this an evidence is compared word by word


def numbers(text: str) -> set[str]:
    """Numbers in ``text``, thousands separators and trailing dots dropped."""
    found = (n.replace(",", "").rstrip(".") for n in _NUMBER.findall(text))
    return {n for n in found if n}


def key_figures(text: str) -> set[str]:
    """Numbers in ``text`` that carry information: no years, no one- or two-digit counts.

    A financial page is full of years and small counts that match by coincidence; the
    amounts an answer is built from have three significant digits or a decimal part.
    """
    kept = set()
    for value in numbers(text):
        digits = value.replace(".", "").lstrip("0")
        is_year = value.isdigit() and 1900 <= int(value) <= 2100
        if not is_year and (len(digits) >= 3 or "." in value):
            kept.add(value)
    return kept


def ngrams(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    """Word n-grams of ``text``, in order (``words()`` is a set and loses the order)."""
    tokens = _WORD.findall(text.lower())
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def approx_tokens(text: str) -> float:
    """Token count estimated at 4 characters per token."""
    return len(text) / 4


def evidence_coverage(context: str, evidence: list[str]) -> dict[str, float | None]:
    """Coverage of the ``evidence`` texts by ``context`` (see the module docstring)."""
    ev_text = " ".join(evidence)
    ev_words = words(ev_text)
    ev_numbers = numbers(ev_text)
    ctx_numbers = numbers(context)
    ev_spans = ngrams(ev_text)
    ev_figures = key_figures(ev_text)
    ctx_figures = key_figures(context)
    return {
        "evidence_words": len(ev_words & words(context)) / len(ev_words) if ev_words else 1.0,
        "evidence_overlap": (len(ev_spans & ngrams(context)) / len(ev_spans) if ev_spans else None),
        "evidence_numbers": (
            len(ev_numbers & ctx_numbers) / len(ev_numbers) if ev_numbers else None
        ),
        "evidence_figures": (
            len(ev_figures & ctx_figures) / len(ev_figures) if ev_figures else None
        ),
        "all_evidence_numbers": float(ev_numbers <= ctx_numbers) if ev_numbers else None,
        "all_evidence_figures": float(ev_figures <= ctx_figures) if ev_figures else None,
        "tokens": approx_tokens(context),
    }
