"""Prompts for the workflow's judgement nodes.

These ask the model to *judge*, never to control the flow: the graph reads the
judgement and decides what happens. Kept apart from the generation prompt
(:mod:`src.llm.prompts`), which the baseline shares.
"""

from __future__ import annotations

from src.ingestion.schema import Chunk

_GRADE_SCALE = """3 = the passage is dedicated to the question and states the answer.
2 = the passage holds part of what the answer is built from, even if it is buried in a
    table or stated indirectly. A passage carrying ONE of the line items a computed
    metric needs -- a revenue, a cost, an asset balance -- is at least a 2.
1 = the passage is on a related topic but carries nothing the answer is built from.
0 = the passage is unrelated: wrong topic, wrong period, or wrong statement."""


def build_grading_prompt(question: str, chunk: Chunk) -> str:
    """Ask for a graded relevance judgement on ONE passage.

    The four points come from UMBRELA, the relevance assessor adopted by the TREC RAG
    track, which reports Kendall tau above 0.87 against human assessors. The one
    addition is the financial clause inside grade 2: a ratio is computed from two line
    items, and neither answers the question on its own -- judged by a yes/no criterion
    both get rejected, which is precisely how a grader loses the evidence it needs.
    """
    return (
        "You are assessing how relevant a passage from an SEC filing is to a financial "
        "question, on a four-point scale.\n\n"
        f"QUESTION: {question}\n\n"
        f"PASSAGE (page {chunk.page}):\n{chunk.text}\n\n"
        f"SCALE:\n{_GRADE_SCALE}\n\n"
        "Answer with the grade only."
    )


_CALC_RULES = """1. Take the formula from the QUESTION. If the question does not say how the metric is
   computed, return an empty variables list.
2. One variable per quantity, for every year the formula needs. For each: `term` names
   the quantity (revenue, operating income, D&A...); `line` is the number of the line
   holding it (17 for L17). Never copy a figure: the program reads it from the line.
3. Choose lines from the statement and the year the question names.
4. `expression` writes the formula as the question states it, over your variable names
   and plain numbers. Treat every amount as positive: the program handles the signs.
   An average of a ratio over N years is (a1 / b1 + ... + aN / bN) / N. Do not multiply
   by 100 for a percent.
5. If a quantity has no line, return an empty variables list. Never let a neighbouring
   line stand in for a missing one."""

_CALC_EXAMPLE = """QUESTION: What is the FY2024 current ratio for Acme? Current ratio is defined as:
total current assets / total current liabilities.

LINES:
L1 [balance sheet, page 12]: Total current assets, December 31, 2024 = 1,250.7
L2 [balance sheet, page 12]: Total current assets, December 31, 2023 = 1,101.2
L3 [balance sheet, page 12]: Total current liabilities, December 31, 2024 = 890.4

EXPECTED: expression "tca / tcl", variables:
  tca -> term "total current assets", line 1
  tcl -> term "total current liabilities", line 3"""

_STATEMENT_NAMES = {
    "income_statement": "income statement",
    "balance_sheet": "balance sheet",
    "cash_flow": "cash flow statement",
    "other": "other table",
}


def build_calc_prompt(question: str, rows: list) -> str:
    """Ask which lines of the context hold the inputs of the formula the question defines.

    The model is shown every ``label = amount`` row of its context, numbered, each tagged
    with the statement the code recognised it in, and it answers with line numbers and
    the words of the question each line stands for: it never copies a figure. It is asked
    only what cannot be read off the text -- which line is which quantity, and how they
    combine -- and every answer it gives is checked by the code
    (``src.workflow.calculator``). Kept short on purpose: measured, each rule added to fix
    one question moved the model's choices on others.
    """
    lines = "\n".join(
        f"L{r.number} [{_STATEMENT_NAMES.get(r.statement, r.statement)}, page {r.page}]: "
        f"{r.label} = {r.printed}"
        for r in rows
    )
    return (
        "You are reading table lines from SEC filings to set up a calculation. You do not "
        "perform it: you state the formula the question gives and point to the lines "
        "holding its inputs, so that a program can read and check them and compute the "
        "result.\n\n"
        f"RULES:\n{_CALC_RULES}\n\n"
        f"EXAMPLE\n{_CALC_EXAMPLE}\n\n"
        f"QUESTION: {question}\n\n"
        f"LINES:\n{lines}"
    )
