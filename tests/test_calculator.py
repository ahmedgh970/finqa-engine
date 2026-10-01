"""Tests for the verified calculator (no GPU, no LLM, no Qdrant).

The rejection tests are measured failure modes, not hypotheticals: each one is a way a
value was found to be wrong while looking right when the contexts were read.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.ingestion.schema import Chunk
from src.workflow.calculator import (
    CalculationError,
    answer_format,
    compute,
    evaluate,
    parse_amount,
    routes_to_calculator,
    rows_of,
    statement_of,
    states_figure,
)
from src.workflow.prompts import build_calc_prompt
from src.workflow.schemas import CalcSpec, CalcVariable

BALANCE_SHEET = (
    "ACME CORP CONSOLIDATED BALANCE SHEETS (In Millions) ASSETS, May 31, 2020 = . "
    "Total current assets, May 31, 2020 = 5,121.3. Total current assets, May 26, 2019 = "
    "4,186.5. Total current liabilities, May 31, 2020 = 7,491.5."
)
CASH_FLOWS = (
    "ACME CORP CONSOLIDATED STATEMENTS OF CASH FLOWS Net cash provided by operating "
    "activities, 2020 = 3,676.2. Purchases of land, buildings, and equipment, 2020 = (460.8). "
    "Dividends paid, 2020 = (1,244.5)."
)
INCOME = (
    "ACME CORP CONSOLIDATED STATEMENTS OF EARNINGS (In Millions) Net sales, 2020 = 17,626.6. "
    "Net earnings attributable to Acme, 2020 = 2,181.2."
)


def _chunk(text: str, page: int = 1, doc: str = "ACME_2020_10K") -> Chunk:
    return Chunk(chunk_id=f"{doc}::{page}", doc_id=doc, page=page, text=text)


def _rows(*texts: str):
    return rows_of([_chunk(t, page=i + 1) for i, t in enumerate(texts)])


def _line(rows, label_start: str) -> int:
    return next(r.number for r in rows if r.label.startswith(label_start))


def _spec(expression: str, **variables: tuple[str, int]) -> CalcSpec:
    """``name=(term, line)``: the words of the question it stands for, and its line."""
    return CalcSpec(
        expression=expression,
        variables=[
            CalcVariable(name=name, term=term, line=line)
            for name, (term, line) in variables.items()
        ],
    )


# --- reading the context -------------------------------------------------------------


def test_rows_are_numbered_across_the_whole_context_and_empty_cells_skipped():
    rows = _rows(BALANCE_SHEET, CASH_FLOWS)
    labels = [r.label for r in rows]
    assert "ASSETS, May 31, 2020" not in labels  # "= ." is a header cell, not a value
    assert labels[0] == "Total current assets, May 31, 2020"
    assert [r.number for r in rows] == list(range(1, len(rows) + 1))
    assert rows[-1].printed == "(1,244.5)"


def test_the_statement_is_recognised_by_its_title():
    assert statement_of(BALANCE_SHEET) == "balance_sheet"
    assert statement_of(CASH_FLOWS) == "cash_flow"
    assert statement_of(INCOME) == "income_statement"


def test_an_untitled_statement_is_recognised_by_the_lines_only_it_prints():
    """Measured: a balance sheet split across chunks started at "(In Millions) ASSETS"."""
    untitled = "(In Millions) Total current assets, 2020 = 5.0. Total current liabilities = 4.0."
    assert statement_of(untitled) == "balance_sheet"
    assert statement_of("Revenue grew on higher volumes.") == "other"


def test_a_chunk_without_a_scale_takes_its_documents():
    """A cash flow chunk cut from mid-table does not repeat "(In Millions)"."""
    rows = _rows(INCOME, CASH_FLOWS)
    assert {r.scale for r in rows} == {"millions"}


def test_amounts_are_read_the_way_filings_print_them():
    assert parse_amount("16,865.2") == Decimal("16865.2")
    assert parse_amount("$ 5,121.3") == Decimal("5121.3")
    # Accounting parentheses are a negative amount, not decoration.
    assert parse_amount("(460.8)") == Decimal("-460.8")
    with pytest.raises(CalculationError):
        parse_amount("n/a")


# --- arithmetic ----------------------------------------------------------------------


def test_arithmetic_is_exact_rather_than_binary_floating_point():
    values = {"a": Decimal("0.1"), "b": Decimal("0.2")}
    assert evaluate("a + b", values) == Decimal("0.3")
    assert evaluate("(a + b) * 10", values) == Decimal("3.0")


def test_the_expression_is_walked_not_executed():
    values = {"a": Decimal(2)}
    for hostile in ("__import__('os').system('ls')", "a.__class__", "open('f')", "b + 1"):
        with pytest.raises(CalculationError):
            evaluate(hostile, values)


def test_division_by_zero_is_a_failure_not_a_crash():
    with pytest.raises(CalculationError):
        evaluate("a / b", {"a": Decimal(1), "b": Decimal(0)})


# --- computing from the lines the model points to ------------------------------------

RATIO_Q = (
    "What is Acme's FY2020 working capital ratio? Define working capital ratio as total "
    "current assets divided by total current liabilities. Round your answer to two "
    "decimal places. Use the balance sheet."
)


def test_a_ratio_is_computed_from_the_lines_pointed_to_and_rounded():
    rows = _rows(BALANCE_SHEET)
    spec = _spec(
        "tca / tcl",
        tca=("total current assets", _line(rows, "Total current assets, May 31")),
        tcl=("total current liabilities", _line(rows, "Total current liabilities")),
    )
    assert compute(spec, rows, RATIO_Q).rendered(*answer_format(RATIO_Q)) == "0.68"


def test_the_figure_comes_from_the_line_so_a_transcription_error_cannot_happen():
    """Measured: asked to copy, the model wrote 7,617 for 7,616. It no longer copies."""
    rows = _rows(CASH_FLOWS)
    question = "What were dividends paid in FY2020? Use the cash flow statement."
    spec = _spec("d", d=("dividends paid", _line(rows, "Dividends paid")))
    assert compute(spec, rows, question).inputs["d"] == Decimal("1244.5")


def test_a_loss_keeps_its_sign_where_an_outflow_does_not():
    """Parentheses on a cash outflow are presentation; on a net loss they are the value."""
    rows = _rows("STATEMENTS OF OPERATIONS (In Millions) Net income (loss), 2022 = (546).")
    question = "What is the FY2022 net income? Use the income statement."
    spec = _spec("n", n=("net income", 1))
    assert compute(spec, rows, question).inputs["n"] == Decimal("-546")


def test_a_formula_written_as_the_question_states_it_gives_a_positive_payout():
    """Measured: the model negated dividends in one draft and not in the next, so the sign
    of an outflow is decided by the code and the model writes the formula as stated."""
    rows = _rows(CASH_FLOWS, INCOME)
    question = (
        "What is Acme's FY2020 dividend payout ratio (using total cash dividends paid and net "
        "income)? Round answer to two decimal places. Use the cash flow statement and the "
        "income statement."
    )
    spec = _spec(
        "dividends / net_income",
        dividends=("total cash dividends paid", _line(rows, "Dividends paid")),
        net_income=("net income", _line(rows, "Net earnings attributable")),
    )
    assert compute(spec, rows, question).rendered(*answer_format(question)) == "0.57"


def test_a_percentage_is_scaled_then_rounded_by_the_code():
    rows = _rows(CASH_FLOWS, INCOME)
    question = "What is FY2020 capex as a % of revenue for Acme? Round to one decimal place."
    spec = _spec(
        "capex / revenue",
        capex=("capex", _line(rows, "Purchases of land")),
        revenue=("revenue", _line(rows, "Net sales")),
    )
    assert compute(spec, rows, question).rendered(*answer_format(question)) == "2.6%"


def test_a_sum_is_answered_in_the_scale_the_question_asks_for():
    thousands = _chunk(
        "NETFLIX INC. STATEMENTS OF OPERATIONS (in thousands) Revenues, 2015 = 6,779,511."
    )
    rows = rows_of([thousands])
    question = "What were FY2015 revenues? Answer in USD millions."
    result = compute(_spec("r", r=("revenues", rows[0].number)), rows, question)
    assert result.rendered(2, False) == "6779.51"


# --- the answer's format is read off the question, never asked -----------------------


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Round your answer to two decimal places.", (2, False)),
        ("Answer in units of percents and round to one decimal place.", (1, True)),
        ("What is the 3 year average net profit margin (as a %)?", (None, True)),
        ("What is the dividend payout ratio? Round answer to two decimal places.", (2, False)),
        ("Round to the nearest whole number.", (0, False)),
    ],
)
def test_rounding_and_percent_are_read_off_the_wording(question, expected):
    """Measured: asked, the model flipped a payout ratio into a percentage."""
    assert answer_format(question) == expected


# --- refusals: the measured failure modes --------------------------------------------


def test_a_quarterly_column_is_refused_for_an_annual_question():
    """Measured: an interest coverage question answered from three-month figures."""
    rows = _rows(
        "STATEMENTS OF OPERATIONS Interest expense, Three Months Ended December 31, 2022 = (137,132)."
    )
    question = "What was the FY2022 interest coverage using interest expense?"
    with pytest.raises(CalculationError, match="quarterly"):
        compute(_spec("i", i=("interest expense", 1)), rows, question)


def test_a_segment_line_is_refused_where_the_formula_wants_the_consolidated_one():
    """Measured: reportable segment net revenues stood in for consolidated revenues."""
    rows = _rows("Reportable segment net revenues, Year Ended December 31, 2019 = 12,286,257.")
    question = "What is the FY2019 capex as a % of revenue?"
    with pytest.raises(CalculationError, match="segment"):
        compute(_spec("r", r=("revenue", 1)), rows, question)


def test_a_year_the_question_does_not_ask_for_is_refused():
    """Measured: a 2017 revenue was reported under a 2018 label."""
    rows = _rows(BALANCE_SHEET)
    question = "What is FY2020 total current assets? Use the balance sheet."
    spec = _spec("t", t=("total current assets", _line(rows, "Total current assets, May 26")))
    with pytest.raises(CalculationError, match="not a year the question asks for"):
        compute(spec, rows, question)


def test_every_year_inside_an_asked_range_is_accepted():
    """A 'FY2018 - FY2020' average needs 2019 too, although the wording never names it."""
    rows = _rows(BALANCE_SHEET)
    question = "What is the FY2018 - FY2020 total current assets? Use the balance sheet."
    spec = _spec("t", t=("total current assets", _line(rows, "Total current assets, May 26")))
    compute(spec, rows, question)


def test_a_statement_the_question_names_but_no_line_comes_from_is_refused():
    """Measured: D&A 'from the cash flow statement' replaced by an income-statement line."""
    rows = _rows(INCOME)
    question = "What is the FY2020 EBITDA margin? Use D&A from the cash flow statement."
    spec = _spec("r", r=("margin", _line(rows, "Net sales")))
    with pytest.raises(CalculationError, match="cash_flow"):
        compute(spec, rows, question)


def test_a_line_naming_another_quantity_than_its_term_is_refused():
    """Measured: 'operating income' answered from 'Income before income taxes'."""
    rows = _rows(
        "STATEMENTS OF OPERATIONS Operating income, 2021 = 2,112. Income before income taxes, "
        "2021 = 2,397."
    )
    question = "What is the FY2021 operating income? Use the income statement."
    with pytest.raises(CalculationError, match="operating_income, the line is pretax_income"):
        compute(_spec("o", o=("operating income", 2)), rows, question)
    compute(_spec("o", o=("operating income", 1)), rows, question)


def test_a_quantity_the_question_never_mentions_is_refused():
    """A genuine line of the right statement, but not one of the formula's inputs."""
    rows = _rows(INCOME)
    spec = _spec("n", n=("net income", _line(rows, "Net earnings attributable")))
    with pytest.raises(CalculationError, match="never mentions net_income"):
        compute(spec, rows, "What is the FY2020 net sales? Use the income statement.")


def test_a_margin_names_its_revenue_without_saying_the_word():
    """Measured: an operating margin's 'net sales' was refused as absent from the question."""
    rows = _rows(INCOME)
    question = "What is the FY2020 operating income % margin? Use the income statement."
    compute(_spec("r", r=("net sales", _line(rows, "Net sales"))), rows, question)


def test_an_n_year_average_has_to_be_divided_by_n():
    """Measured: a three year average written as a sum of three margins."""
    rows = _rows(INCOME)
    question = "What is the FY2020 3 year average of net sales? Use the income statement."
    rev = ("net sales", _line(rows, "Net sales"))
    with pytest.raises(CalculationError, match="not divided by 3"):
        compute(_spec("a + a + a", a=rev), rows, question)
    compute(_spec("(a + a + a) / 3", a=rev), rows, question)


def test_an_average_between_two_dates_has_to_be_halved():
    rows = _rows(BALANCE_SHEET)
    question = (
        "What is the ratio of total current assets to average total current assets between "
        "FY2019 and FY2020? Use the balance sheet."
    )
    now = ("total current assets", _line(rows, "Total current assets, May 31"))
    before = ("total current assets", _line(rows, "Total current assets, May 26"))
    with pytest.raises(CalculationError, match="not halved"):
        compute(_spec("a / (a + b)", a=now, b=before), rows, question)
    compute(_spec("a / ((a + b) / 2)", a=now, b=before), rows, question)


def test_a_line_number_that_does_not_exist_is_refused():
    rows = _rows(BALANCE_SHEET)
    with pytest.raises(CalculationError, match="does not exist"):
        compute(_spec("x", x=("total current assets", 999)), rows, RATIO_Q)


def test_an_empty_specification_is_how_the_model_abstains():
    """Constrained decoding removes refusal in prose, so it must exist in the schema."""
    with pytest.raises(CalculationError, match="not all in the context"):
        compute(CalcSpec(expression="a / b", variables=[]), _rows(BALANCE_SHEET), RATIO_Q)


# --- routing -------------------------------------------------------------------------

ROUTED = [
    "What is the FY2019 fixed asset turnover ratio for X? Fixed asset turnover ratio is "
    "defined as: FY2019 revenue / (average PP&E between FY2018 and FY2019).",
    "What is X's FY2021 net working capital? Define net working capital as total current "
    "assets less total current liabilities. Answer in USD millions.",
    "What was X's interest coverage ratio using FY2022 Adjusted EBIT as the numerator and "
    "annual Interest Expense as the denominator?",
    "What is the FY2018 - FY2020 3 year average of capex as a % of revenue for X?",
    "What is X's FY2021 unadjusted operating income + depreciation and amortization from "
    "the cash flow statement in USD millions?",
    "What is X's FY2022 dividend payout ratio (using total cash dividends paid and net "
    "income attributable to shareholders)?",
]

NOT_ROUTED = [
    "Did X's net earnings as a percent of sales increase in Q2 FY2023 compared to Q2 FY2022?",
    "Which of X's business segments had the lowest net revenue in 2021 Q1?",
    "What drove the reduction in SG&A expense as a percent of net sales in FY2023?",
    "What is the quantity of restructuring costs directly outlined in X's income statement?",
    "Has X reported any materially important ongoing legal battles from 2022 and 2023?",
]


@pytest.mark.parametrize("question", ROUTED)
def test_a_question_that_carries_its_formula_is_routed(question):
    assert routes_to_calculator(question)


@pytest.mark.parametrize("question", NOT_ROUTED)
def test_a_question_without_a_computation_to_verify_is_not_routed(question):
    """Routing reads the shape of the question, not its subject: no company, no id."""
    assert not routes_to_calculator(question)


def test_the_prompt_shows_numbered_lines_with_their_statement():
    """The model answers with line numbers, so each line must carry one it can cite."""
    prompt = build_calc_prompt(RATIO_Q, _rows(BALANCE_SHEET))
    assert "L1 [balance sheet, page 1]: Total current assets, May 31, 2020 = 5,121.3" in prompt
    # The abstention path has to be stated, since constrained decoding removes refusal.
    assert "empty variables list" in " ".join(prompt.split())


def test_a_qualifier_the_question_never_uses_marks_the_wrong_line():
    """Measured: 'from continuing operations' answered a plain net profit margin."""
    rows = _rows(
        "STATEMENTS OF EARNINGS (In Millions) Net earnings from continuing operations, 2017 "
        "= 1,207. Net earnings, 2017 = 1,228."
    )
    question = "What is the FY2017 net profit margin? Use the income statement."
    continuing = ("net profit", _line(rows, "Net earnings from continuing"))
    with pytest.raises(CalculationError, match="continuing operations"):
        compute(_spec("n", n=continuing), rows, question)
    compute(_spec("n", n=("net profit", _line(rows, "Net earnings, 2017"))), rows, question)


def test_a_qualifier_the_question_names_is_accepted():
    rows = _rows("STATEMENTS OF EARNINGS Net earnings from continuing operations, 2017 = 1,207.")
    question = "What were FY2017 net earnings from continuing operations? Use the income statement."
    compute(_spec("n", n=("net earnings from continuing operations", 1)), rows, question)


def test_a_redundant_percent_factor_is_removed_not_applied_twice():
    """Measured: told not to, the model still wrote '/ 3 * 100' for a margin."""
    rows = _rows(INCOME)
    question = "What is the FY2020 net profit margin (as a %)? Round to one decimal place."
    net = ("net profit", _line(rows, "Net earnings attributable"))
    rev = ("margin", _line(rows, "Net sales"))
    plain = compute(_spec("n / r", n=net, r=rev), rows, question)
    scaled = compute(_spec("n / r * 100", n=net, r=rev), rows, question)
    assert plain.rendered(*answer_format(question)) == scaled.rendered(*answer_format(question))
    assert plain.rendered(*answer_format(question)) == "12.4%"


def test_a_factor_of_100_stays_when_the_question_asks_no_percent():
    rows = _rows(INCOME)
    question = "What is 100 times FY2020 net earnings? Use the income statement."
    spec = _spec("n * 100", n=("net earnings", _line(rows, "Net earnings attributable")))
    assert compute(spec, rows, question).value == Decimal("218120.0")


NOTE_COPY = "NOTE 14 — EQUITY Net earnings attributable to Acme, 2020 = 2,181.2."


def test_a_row_read_from_a_note_moves_to_the_statement_that_prints_it_too():
    """Measured: the note's copy of net income was chosen, the statement's copy existed."""
    rows = _rows(INCOME, NOTE_COPY)
    note = next(r.number for r in rows if r.statement == "other")
    question = "What is FY2020 net income? Use the income statement."
    result = compute(_spec("n", n=("net income", note)), rows, question)
    assert result.inputs["n"] == Decimal("2181.2")


def test_a_row_is_not_moved_to_a_line_with_a_different_figure():
    """Moving a row may change where it is read from, never the number read."""
    rows = _rows(INCOME.replace("2,181.2", "2,000.0"), NOTE_COPY)
    note = next(r.number for r in rows if r.statement == "other")
    question = "What is FY2020 net income? Use the income statement."
    with pytest.raises(CalculationError, match="income_statement"):
        compute(_spec("n", n=("net income", note)), rows, question)


def test_a_total_including_noncontrolling_interests_is_the_net_profit_itself():
    """Measured: 'including noncontrolling interests' gave the gold; refusing it was wrong."""
    rows = _rows(
        "STATEMENTS OF EARNINGS (In Millions) Net earnings including noncontrolling "
        "interests, 2017 = 1,228. Net income attributable to noncontrolling interests, "
        "2017 = 29."
    )
    question = "What is the FY2017 net profit margin? Use the income statement."
    compute(_spec("n", n=("net profit", _line(rows, "Net earnings including"))), rows, question)
    with pytest.raises(CalculationError, match="attributable to noncontrolling"):
        compute(
            _spec("n", n=("net profit", _line(rows, "Net income attributable"))), rows, question
        )


def test_an_average_of_ratios_must_be_n_ratios_not_a_precedence_accident():
    """Measured: '(oi + da / rev + ...) / 3' divided only the D&A by revenue."""
    rows = _rows(INCOME)
    question = "What is the FY2020 3 year average net profit margin (as a %)?"
    n = ("net profit", _line(rows, "Net earnings attributable"))
    r = ("margin", _line(rows, "Net sales"))
    with pytest.raises(CalculationError, match="not the sum of 3 ratios"):
        compute(_spec("(n + n / r + n / r) / 3", n=n, r=r), rows, question)
    compute(_spec("(n / r + n / r + n / r) / 3", n=n, r=r), rows, question)


def test_an_average_of_amounts_needs_no_ratio_shape():
    rows = _rows(INCOME)
    question = "What is the FY2020 3 year average net sales in USD millions?"
    compute(_spec("(s + s + s) / 3", s=("net sales", _line(rows, "Net sales"))), rows, question)


def test_an_adjusted_line_does_not_answer_an_unadjusted_question():
    """Anticipated: a non-GAAP reconciliation prints 'Adjusted operating income' next to
    the GAAP line, and both are operating income to the lexicon."""
    rows = _rows(
        "STATEMENTS OF OPERATIONS (In Millions) Operating income, 2022 = 11,512. Adjusted "
        "operating income, 2022 = 12,030. Core, Non-GAAP operating profit, 2022 = 12,100."
    )
    unadjusted = "What is the FY2022 unadjusted operating income? Use the income statement."
    compute(_spec("o", o=("operating income", 1)), rows, unadjusted)
    for line in (2, 3):
        with pytest.raises(CalculationError, match="qualifies the line"):
            compute(_spec("o", o=("operating income", line)), rows, unadjusted)
    adjusted = "What is the FY2022 adjusted operating income? Use the income statement."
    compute(_spec("o", o=("adjusted operating income", 2)), rows, adjusted)


def test_a_scale_note_in_a_label_is_read_as_the_scale_not_as_a_qualifier():
    """Measured: '(in millions, except per share amounts)' folded into a net loss label
    read as a per share figure, and a correct return on assets was refused."""
    rows = _rows(
        "STATEMENTS OF OPERATIONS Net income (loss), 2022.(in millions, except per share "
        "amounts) = (546)."
    )
    assert rows[0].label == "Net income (loss), 2022"
    assert rows[0].scale == "millions"
    question = "What is the FY2022 net income? Use the income statement."
    assert compute(_spec("n", n=("net income", 1)), rows, question).inputs["n"] == Decimal("-546")


def test_a_line_without_a_year_is_refused_for_a_question_that_names_one():
    """Measured: a 'Full Year' column of a segment table answered a FY2017 question."""
    rows = _rows("STATEMENTS OF INCOME (In Millions) Consolidated net income, Full Year = 6,527.")
    dated = "What is the FY2017 net income? Use the income statement."
    with pytest.raises(CalculationError, match="no year on the line"):
        compute(_spec("n", n=("net income", 1)), rows, dated)
    undated = "What is the net income? Use the income statement."
    compute(_spec("n", n=("net income", 1)), rows, undated)


def test_ebit_is_not_the_ebitdar_printed_next_to_it():
    """Measured: 'Adjusted EBITDAR' answered a coverage ratio on adjusted EBIT."""
    rows = _rows("RECONCILIATION (In Thousands) Adjusted EBITDAR, 2022 = 3,497,254.")
    question = "What was the FY2022 interest coverage ratio using Adjusted EBIT?"
    with pytest.raises(CalculationError, match="ebit"):
        compute(_spec("e", e=("Adjusted EBIT", 1)), rows, question)


def test_a_term_outside_the_lexicon_needs_all_its_words_on_the_line():
    rows = _rows("STATEMENTS OF INCOME (In Millions) Restructuring charges, 2022 = 120.")
    question = "What were FY2022 restructuring and impairment charges?"
    compute(_spec("c", c=("restructuring charges", 1)), rows, question)
    with pytest.raises(CalculationError, match="impairment"):
        compute(_spec("c", c=("impairment charges", 1)), rows, question)
    with pytest.raises(CalculationError, match="names no quantity"):
        compute(_spec("c", c=("total", 1)), rows, question)


def test_an_interest_expense_in_parentheses_is_a_cost_not_a_negative_ratio():
    """Measured: an interest expense printed in parentheses flipped a coverage ratio."""
    rows = _rows(
        "STATEMENTS OF OPERATIONS (In Millions) Operating income, 2022 = 1,200. Interest "
        "expense, 2022 = (300)."
    )
    question = "What is the FY2022 interest coverage ratio (operating income / interest expense)?"
    spec = _spec("o / i", o=("operating income", 1), i=("interest expense", 2))
    assert compute(spec, rows, question).value == Decimal(4)


def test_a_named_statement_left_empty_takes_the_twin_of_another_row():
    """Measured: D&A read off the income statement for a question on the income and cash
    flow statements, both printing the same 636 for 2021."""
    rows = _rows(
        "STATEMENTS OF OPERATIONS (In Millions) Operating income, 2021 = 1,196. "
        "Depreciation and amortization, 2021 = 636.",
        "STATEMENTS OF CASH FLOWS (In Millions) Depreciation and amortization expense, 2021 = 636.",
    )
    question = (
        "Using the cash flow statement and the income statement, what is FY2021 operating "
        "income + depreciation and amortization?"
    )
    result = compute(
        _spec("o + d", o=("operating income", 1), d=("depreciation", 2)), rows, question
    )
    assert result.value == Decimal(1832)
    assert result.lines["d"] == 3


def test_a_twin_must_print_the_same_figure_for_an_overlapping_year():
    rows = _rows(
        "STATEMENTS OF OPERATIONS (In Millions) Operating income, 2021 = 1,196. "
        "Depreciation and amortization, 2021 = 636.",
        "STATEMENTS OF CASH FLOWS (In Millions) Depreciation and amortization, 2020 = 636.",
    )
    question = (
        "Using the cash flow statement and the income statement, what is FY2021 operating "
        "income + depreciation and amortization?"
    )
    with pytest.raises(CalculationError, match="cash_flow"):
        compute(_spec("o + d", o=("operating income", 1), d=("depreciation", 2)), rows, question)


def test_a_per_share_block_is_marked_on_its_rows_once_the_scale_note_is_gone():
    """Measured: selected financial data prints per-share lines under a heading, their
    own label naming net income; only the heading says the figure is per share."""
    note = ".(in millions, except per share amounts)"
    rows = _rows(
        f"SELECTED FINANCIAL DATA Net income attributable to AES, 2022{note} = (546). "
        f"Basic earnings (loss) per share:, 2022{note} = . "
        f"Net income attributable to AES common stockholders, 2022{note} = $ (0.82). "
        f"Income from discontinued operations, 2022{note} = -. "
        f"Net income attributable to AES common stockholders, 2021{note} = $ (0.61). "
        f"Balance Sheet Data at December 31:, 2022{note} = . Total assets, 2022{note} = 38,363."
    )
    assert [r.label for r in rows] == [
        "Net income attributable to AES, 2022",
        "Net income attributable to AES common stockholders, 2022 (per share)",
        "Net income attributable to AES common stockholders, 2021 (per share)",
        "Total assets, 2022",
    ]
    assert {r.scale for r in rows} == {"millions"}
    question = "What is the FY2022 return on assets (net income / total assets)?"
    compute(_spec("n / a", n=("net income", 1), a=("total assets", 4)), rows, question)
    with pytest.raises(CalculationError, match="per share"):
        compute(_spec("n / a", n=("net income", 2), a=("total assets", 4)), rows, question)


def test_a_per_share_block_cut_by_the_chunker_goes_on_in_the_next_chunk():
    """Measured: the heading closed one chunk and its rows opened the next."""
    note = ".(in millions, except per share amounts)"
    head = _chunk(
        f"SELECTED FINANCIAL DATA Net income, 2022{note} = (546). "
        f"Basic earnings (loss) per share:, 2022{note} = . ",
        page=7,
    )
    rows_text = f"Net income attributable to common stockholders, 2022{note} = (0.82)."
    following = Chunk(chunk_id="ACME_2020_10K::8", doc_id="ACME_2020_10K", page=8, text=rows_text)
    elsewhere = Chunk(chunk_id="ACME_2020_10K::20", doc_id="ACME_2020_10K", page=20, text=rows_text)
    assert rows_of([head, following])[-1].label.endswith("(per share)")
    assert not rows_of([head, elsewhere])[-1].label.endswith("(per share)")


@pytest.mark.parametrize(
    ("answer", "figure", "kept"),
    [
        ("FCF = $3,215.4 million", "3215.40", True),  # measured: formatted, same value
        ("net working capital is **$5,818 million**", "5818.00", True),
        ("the average is **55.5%**", "55.1%", False),  # measured: a different figure
        ("ROA = (0.02)", "-0.02", True),
        ("ROA = 0.02", "-0.02", False),
    ],
)
def test_a_verified_figure_is_found_in_the_answer_by_value(answer, figure, kept):
    assert states_figure(answer, figure) is kept
