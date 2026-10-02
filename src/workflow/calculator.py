"""Verify the values a model read, then compute the formula the question stated.

A language model predicts tokens; it does not compute. So the arithmetic leaves the
model entirely: it states the expression the question defines and the values it found,
and this module checks those values against the passages before evaluating the
expression in exact decimal arithmetic. Nothing the model produces is executed -- the
expression is parsed into an AST and walked against a whitelist, never ``eval``.

The checks are not defensive programming, they are the measured failure modes. Reading
the contexts of the questions whose wording carries its own formula turned up three
ways a value can be wrong while looking right:

- a figure taken from the cash flow statement, where the line "Accounts payable" is the
  *change* over the year, when the formula wants the balance sheet's closing balance;
- a segment revenue used as the consolidated one, a 5% difference invisible from the
  number alone;
- a quarterly amount used as an annual one.

None is detectable from the number alone, which is why the model does not report
numbers at all. The context is cut into numbered table rows (``label = amount``); the
model says which row holds each quantity, and the figure, its statement, its scale and
its period are read off that row by code. Transcription errors -- measured: a digit
off, a year's column swapped, an addition typed where a number belonged -- become
impossible by construction, and every check below runs on the filing's text, never on
a label the model supplied. A failed check falls back to the generator instead of
guessing.

Known limitation: a row whose label names a segment is refused outright, because every
question whose wording carries its own formula asks for consolidated figures. A
question that legitimately asked for a segment would be refused rather than answered
wrongly -- the generator then answers as it would without the tool.
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, DivisionByZero, InvalidOperation

from src.ingestion.schema import Chunk
from src.workflow.expansion import chunk_index
from src.workflow.schemas import CalcSpec

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SCALE = {
    "thousands": Decimal(1_000),
    "millions": Decimal(1_000_000),
    "billions": Decimal(1_000_000_000),
}

_QUARTER = re.compile(r"\b(q[1-4]|first|second|third|fourth)\s+quarter|\bthree months\b", re.I)
_YEAR = re.compile(r"(?<!\d)(19[89]\d|20[0-4]\d)(?!\d)")
# "(in thousands)" in a statement header, "Answer in USD millions" in a question.
_SCALE_IN = re.compile(r"\bin (?:usd |us\$ ?|\$ ?)?(thousands|millions|billions)\b", re.I)
# What precedes the first row of a chunk: the company and statement title in capitals,
# then the scale in parentheses. Neither belongs to the row's label. The title is matched
# case-sensitively on purpose: its whole signal is being in capitals.
_SCALE_NOTE = re.compile(r"^.*?\(in (?:thousands|millions|billions)[^)]*\)\s*", re.I | re.S)
_CAPS_TITLE = re.compile(r"^[A-Z0-9 ,.&'|\-\u2013\u2014:]{12,}\s+(?=[A-Z][a-z])")
# A scale note folded into a row's label from its column header, just before the value.
_ROW_SCALE_NOTE = re.compile(r"\.?\s*\(\s*in (thousands|millions|billions)\b[^)]*\)\s*$", re.I)
_ANY_SCALE_NOTE = re.compile(r"\(\s*in (?:thousands|millions|billions)\b[^)]*\)", re.I)
# A sub-heading's empty cell ("= ."), not a nil value ("= -."), which is a row.
_EMPTY_CELL = re.compile(r"=\s*\.(?=\s|$)")
_PER_SHARE = re.compile(r"\bper (common |diluted |basic )?share\b", re.I)
_SEGMENT = re.compile(r"\bsegments?\b", re.I)
# Words that make a line a different quantity from the one its head noun names. A line
# carrying one the question never uses is the wrong line, however close its label.
# Measured: "Net earnings from continuing operations" answered a net profit margin whose
# gold uses total net earnings -- right company, right year, wrong quantity, and every
# other check passed.
#
# Precision matters: "including noncontrolling interests" is the consolidated total and
# answers a net profit margin (measured, it gave the gold); "attributable to
# noncontrolling interests" is the minority's slice, a different quantity.
_QUALIFIERS = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"\bcontinuing operations\b",
        r"\bdiscontinued operations\b",
        r"\battributable to non-?controlling\b",
        r"\bper (diluted |basic )?share\b",
        # Anticipated rather than observed: a non-GAAP reconciliation prints "Adjusted
        # operating income" beside the GAAP line, and the lexicon alone would take it for
        # operating income. Matched as whole words, so "unadjusted" never triggers it.
        r"\badjusted\b",
        r"\bnon-?gaap\b",
        r"\bpro forma\b",
        r"\bexcluding\b",
    )
)
# One serialised table row: "Receivables, May 31, 2020 = 1,615.1". The label is whatever
# precedes the sign back to the previous sentence break; an empty cell ("= .") is no row.
_ROW_VALUE = re.compile(r"=\s*\$?\s?(\(?-?\d[\d,]*(?:\.\d+)?\)?)")

# A chunk does not always carry its statement's title -- a balance sheet split across
# chunks can start at "(In Millions) ASSETS" -- so the statement is recognised by its
# title when there is one, and otherwise by the line items only it prints.
_TITLE = {
    "cash_flow": re.compile(r"statements? of cash flows?", re.I),
    "balance_sheet": re.compile(
        r"balance sheets?|statements? of financial (position|condition)", re.I
    ),
    "income_statement": re.compile(
        r"statements? of (operations|income|earnings)|income statements?", re.I
    ),
}
_MARKERS = {
    "cash_flow": re.compile(
        r"operating activities|investing activities|financing activities|net cash provided", re.I
    ),
    "balance_sheet": re.compile(
        r"total current assets|total current liabilities|total assets|total liabilities", re.I
    ),
    "income_statement": re.compile(
        r"cost of (sales|revenue|goods sold)|gross (profit|margin)|operating (income|profit)"
        r"|income before income taxes|earnings per share",
        re.I,
    ),
}

# Standard line items, as statements name them. A chosen line must name the same item as
# the words of the question it stands for: "operating income" answered from "Income
# before income taxes" (measured) is a genuine line, the right year, the right statement,
# and the wrong quantity. Ordered from specific to general, since "cost of sales" contains
# "sales" and "total current assets" contains "assets". Accounting vocabulary shared by
# every filer -- it checks a choice, it never supplies a formula.
_LINE_ITEMS = (
    # Three different aggregates, each a prefix of the next: "Adjusted EBITDAR" answered
    # a question on adjusted EBIT (measured), so the longest name is matched first.
    ("ebitdar", r"\bebitdar\b"),
    ("ebitda", r"\bebitda\b"),
    ("ebit", r"\bebit\b"),
    ("pretax_income", r"(income|earnings) before (provision for )?(income )?tax|pre-?tax income"),
    ("cogs", r"cost of (sales|revenues?|goods sold|products sold)|\bcogs\b"),
    ("gross_profit", r"gross (profit|margin)"),
    ("sga", r"selling, general|\bsg&a\b"),
    ("deferred_revenue", r"(deferred|unearned) revenue"),
    ("operating_income", r"operating (income|profit|earnings|loss)|income from operations"),
    ("net_income", r"net (income|earnings|profit|loss)"),
    ("revenue", r"\brevenues?\b|net sales|\bsales\b|\bmargin\b"),
    ("da", r"\bd&a\b|depreciation"),
    (
        "capex",
        r"capital expenditures?|\bcapex\b|purchases? of (property|land|plant|buildings|equipment)"
        r"|additions to property",
    ),
    (
        "cfo",
        r"cash (provided by|from|generated by|used in|flows? from) operati|operating cash flow",
    ),
    ("dividends", r"dividends?"),
    ("interest_expense", r"interest expense"),
    ("current_assets", r"current assets"),
    ("current_liabilities", r"current liabilities"),
    ("total_assets", r"total assets|\bassets\b"),
    ("inventory", r"inventor(y|ies)"),
    ("receivables", r"receivables?"),
    ("payables", r"accounts payable|\bpayables?\b"),
    ("ppe", r"property,? plant|\bpp&e\b|property and equipment|fixed assets"),
)
_LINE_ITEM_PATTERNS = [(name, re.compile(pattern, re.I)) for name, pattern in _LINE_ITEMS]
# Costs and outflows, which filings print in parentheses as a matter of presentation --
# an expense on the income statement, a payment on the cash flow statement -- rather than
# as a negative quantity. A formula names them as positive amounts ("cash from
# operations - capex", "dividends / net income", "EBIT / interest expense"), so they are
# read as magnitudes. A result is different: a net or operating loss is the quantity and
# keeps its sign. Measured twice: asked to handle signs, the model negated dividends in one
# draft and not the next, and an interest expense in parentheses flipped a coverage ratio.
_COSTS = {"capex", "dividends", "interest_expense", "cogs", "sga", "da"}

# How the question wants the figure. Written in the wording, so read there, not asked:
# asking made the model flip a payout ratio into a percentage between two prompt drafts.
_ROUND = re.compile(
    r"round(?:ed)?\b[^.?]{0,30}?\b(zero|one|two|three|four|[0-4])\s+decimals?", re.I
)
_WHOLE = re.compile(r"nearest (whole number|integer)", re.I)
_PERCENT = re.compile(r"%|\bpercent(age|s)?\b", re.I)
# What a question averages when it averages a ratio rather than an amount.
_RATIO = re.compile(r"%|\bpercent|\bmargin\b|\bratio\b|\bturnover\b", re.I)
_NUMBER_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4}
# Words of a term that say nothing about which line it is.
_STOPWORDS = {"total", "unadjusted", "annual", "fiscal", "year", "average", "between", "from"}

# Which statement a question sends the reader to. A question that names two of them and
# gets answered from one is answering from the wrong line, whatever the line says.
_STATEMENTS = {
    "income_statement": re.compile(
        r"income statement|statement of income|p&l|profit and loss", re.I
    ),
    "balance_sheet": re.compile(r"balance sheet|statement of financial position", re.I),
    "cash_flow": re.compile(r"cash flow|statement of cash flows", re.I),
}

# Routing reads the SHAPE of the question, never its subject: a wording that carries its
# own formula, that asks for a value, and that is not a yes/no or an explanation. On the
# 150 questions it fires on 30 -- the 22 whose wording carries a formula, plus 8 other
# metrics-generated ones, which is the intended reach rather than a false positive. A
# question it misroutes costs one structured call and falls back to the generator.
_FORMULA = re.compile(
    r"\bdefine[sd]? [^.?]{0,60}\bas\b"  # "X is defined as", "Define X as"
    r"|\bas the numerator\b|\bas the denominator\b"
    r"|\b(three|3)[ -]year average\b"
    r"|\bas a (%|percent(age)?) of\b"
    r"|[a-z]{4}\s[+\-]\s[a-z]{4}"  # an expression written inline: "income + depreciation"
    r"|\bdivided by\b"
    r"|\busing [^.?]{0,90}\band\b",  # "the ratio (using cash dividends and net income)"
    re.I,
)
_ASKS_VALUE = re.compile(
    r"\bwhat (is|was|are|were)\b|\bhow much\b|\bcalculate\b|\banswer in\b", re.I
)
_NOT_A_VALUE = re.compile(
    r"^\s*(did|is|are|was|were|does|do|which|who|why|has|have)\b"
    r"|\bwhat drove\b|\bincrease or decrease\b|\bcompared to\b",
    re.I,
)


def line_item(text: str) -> str | None:
    """The standard line item a label or a term names, if the lexicon knows it."""
    for name, pattern in _LINE_ITEM_PATTERNS:
        if pattern.search(text):
            return name
    return None


def line_items_in(question: str) -> set[str]:
    """Every standard line item the question mentions, including one a margin implies."""
    return {name for name, pattern in _LINE_ITEM_PATTERNS if pattern.search(question)}


# Where a question defines its formula: "ROA is defined as: ...", "Define net working
# capital as ...". Only there does a year written before a quantity fix that quantity's
# year -- in the question's own title, "FY2019 inventory turnover ratio" names a metric.
_DEFINITION = re.compile(r"\bdefined? (?:[^.:?]{0,80}? )?as\b:?|\bformula\b", re.I)
# A year, then the words that follow it up to the next symbol or figure.
_YEAR_THEN_WORDS = re.compile(
    r"(?<!\d)(?:fy\s?)?(19[89]\d|20[0-4]\d)(?!\d)\s+([a-z][a-z&'-]*(?:\s+[a-z][a-z&'-]*){0,3})",
    re.I,
)


def _item_at_start(text: str) -> str | None:
    for name, pattern in _LINE_ITEM_PATTERNS:
        found = pattern.search(text)
        if found and found.start() == 0:
            return name
    return None


def bound_years(question: str) -> dict[str, set[int]]:
    """Line items the question's formula pins to a year: "FY2020 revenue / (...)".

    A year binds the line item written right after it, or after one qualifier
    ("FY2021 unadjusted operating income"); anything else between them -- a figure, a
    second year, "3 year average of" -- binds nothing. An average "between FY2019 and
    FY2020" pins no year either: both are wanted.
    """
    definition = _DEFINITION.search(question)
    if definition is None:
        return {}
    bound: dict[str, set[int]] = {}
    for found in _YEAR_THEN_WORDS.finditer(question[definition.end() :]):
        words = found.group(2).split()
        for start in (0, 1):
            item = _item_at_start(" ".join(words[start:]))
            if item:
                bound.setdefault(item, set()).add(int(found.group(1)))
                break
    return bound


def answer_format(question: str) -> tuple[int | None, bool]:
    """The rounding and the percent the question asks for, read off its wording."""
    places = None
    if found := _ROUND.search(question):
        word = found.group(1).lower()
        places = int(word) if word.isdigit() else _NUMBER_WORDS[word]
    elif _WHOLE.search(question):
        places = 0
    return places, bool(_PERCENT.search(question))


def routes_to_calculator(question: str) -> bool:
    """Whether the question states a computation this tool can verify and evaluate."""
    return bool(
        _FORMULA.search(question)
        and _ASKS_VALUE.search(question)
        and not _NOT_A_VALUE.search(question)
    )


@dataclass(frozen=True)
class Row:
    """A table row of the context, with what the code -- not the model -- read off it."""

    number: int
    label: str
    printed: str
    statement: str
    scale: str | None
    page: int

    def shown(self) -> str:
        return f"L{self.number}: {self.label} = {self.printed}"


def statement_of(text: str) -> str:
    """Which financial statement a passage is, from its title or failing that its rows."""
    head = text[:300]
    for name, title in _TITLE.items():
        if title.search(head):
            return name
    counts = {name: len(marker.findall(text)) for name, marker in _MARKERS.items()}
    best = max(counts, key=counts.get)
    ties = sum(count == counts[best] for count in counts.values())
    return best if counts[best] and ties == 1 else "other"


def rows_of(chunks: list[Chunk]) -> list[Row]:
    """Every ``label = amount`` row of the passages, numbered across the whole context.

    The scale ("in millions") is printed once in a statement's header, so a chunk cut
    from the middle of a table does not carry it. A filing prints its statements in one
    scale, so a chunk that does not state it takes the one its document states most
    often -- the alternative, refusing, would reject a cash flow line for being unlabelled.
    """
    stated: dict[str, list[str]] = {}
    for chunk in chunks:
        found = _SCALE_IN.search(chunk.text[:300])
        if found:
            stated.setdefault(chunk.doc_id, []).append(found.group(1).lower())
    by_document = {doc: max(set(s), key=s.count) for doc, s in stated.items()}

    rows: list[Row] = []
    per_share = False
    before: Chunk | None = None
    for chunk in chunks:
        statement = statement_of(chunk.text)
        found = _SCALE_IN.search(chunk.text[:300])
        scale = found.group(1).lower() if found else by_document.get(chunk.doc_id)
        previous = 0
        # A block cut by the chunker goes on in the next chunk of the document, which
        # expansion places right after it; any other chunk starts outside every block.
        index = chunk_index(chunk.chunk_id)
        follows = (
            before is not None
            and index is not None
            and before.doc_id == chunk.doc_id
            and chunk_index(before.chunk_id) == index - 1
        )
        per_share = per_share and follows
        before = chunk
        for match in _ROW_VALUE.finditer(chunk.text):
            gap = chunk.text[previous : match.start()]
            label = re.split(r"\.\s+|\n", gap)[-1]
            # The parser folds a column's header into every row, scale note included:
            # "Net income, 2022.(in millions, except per share amounts)". The note says how
            # the row is scaled, not what it measures -- left in, its "per share" made a
            # correct net income look like a per-share figure (measured).
            note = _ROW_SCALE_NOTE.search(label)
            row_scale = note.group(1).lower() if note else scale
            label = _ROW_SCALE_NOTE.sub("", label)
            if not previous:
                label = _CAPS_TITLE.sub("", _SCALE_NOTE.sub("", label))
            # What the note said about per-share amounts must not go with it. Rows without
            # a value are sub-headings, each opening a block, and under "Basic earnings per
            # share:" the lines drop the words: "Net income attributable to common
            # stockholders = (0.61)" is a per-share amount in a table otherwise in millions
            # (measured). So in a table whose note excepts per-share amounts, the block's
            # heading is carried onto its rows. Only there: elsewhere "per share" in the
            # text before a row is a par value or an EPS note's title, and the millions
            # of its numerator are not per share (measured).
            per_share = _block_after(gap, per_share)
            label = re.sub(r"\s+", " ", label).strip(" .|")
            excepts = note is not None and _PER_SHARE.search(note.group(0))
            if label and excepts and per_share and not _PER_SHARE.search(label):
                label = f"{label} (per share)"
            previous = match.end()
            if label:
                rows.append(
                    Row(len(rows) + 1, label, match.group(1), statement, row_scale, chunk.page)
                )
        per_share = _block_after(chunk.text[previous:], per_share)
    return rows


def _block_after(text: str, per_share: bool) -> bool:
    """Whether rows following ``text`` sit in a per-share block.

    A sub-heading in ``text`` opens a new block, per share when it says so; without one
    the current block goes on.
    """
    empty = list(_EMPTY_CELL.finditer(text))
    if not empty:
        return per_share
    headings = _ANY_SCALE_NOTE.sub("", text[: empty[-1].start()])
    return bool(_PER_SHARE.search(headings))


class CalculationError(Exception):
    """A computation that cannot be trusted. Always caught: the generator takes over."""


@dataclass(frozen=True)
class Calculation:
    """A verified result, and the trace that makes it auditable."""

    value: Decimal
    expression: str
    inputs: dict[str, Decimal]
    lines: dict[str, int] = field(default_factory=dict)  # the line each input was read from

    def rendered(self, round_to: int | None, as_percent: bool) -> str:
        """The number as the question asked for it -- rounding is the code's job.

        A question that states no rounding still expects a figure, not twenty decimals:
        the convention of the corpus is one decimal for a percentage, two otherwise.
        """
        value = self.value * 100 if as_percent else self.value
        places = round_to if round_to is not None else (1 if as_percent else 2)
        value = value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
        return f"{value}%" if as_percent else str(value)


def parse_amount(printed: str) -> Decimal:
    """The number as filings print it: thousands separators, currency, parentheses.

    Accounting parentheses mean a negative amount -- a capex of ``(460.8)`` is a
    subtraction the formula already accounts for, so dropping the sign would double it.
    """
    text = printed.strip().replace(",", "").replace("$", "").replace("%", "").strip()
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").strip()
    if text.startswith("-"):
        negative, text = True, text[1:]
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise CalculationError(f"not a number: {printed!r}") from exc
    return -value if negative else value


_FIGURE = re.compile(r"\(\s*\d[\d,]*(?:\.\d+)?\s*\)|-?\d[\d,]*(?:\.\d+)?")


def states_figure(text: str, figure: str) -> bool:
    """Whether ``text`` states ``figure`` as a number, however it is formatted.

    Compared by value, not by string: the generator writes a verified 3215.40 as
    "$3,215.4 million" and 5818.00 as "$5,818" (measured), the same figure a substring
    test reads as dropped. A different figure is not the same one rounded -- 55.5% for a
    verified 55.1% is a figure the generator did not keep.
    """
    wanted = parse_amount(figure)
    for match in _FIGURE.finditer(text):
        try:
            if parse_amount(match.group(0)) == wanted:
                return True
        except CalculationError:
            continue
    return False


def check_row(row: Row, question: str | None) -> None:
    """Refuse a row whose own text says it is not what an annual, consolidated formula wants.

    Every check reads the filing's wording: a quarterly column, a segment line, a year the
    question never mentions. The model's opinion of the row is not consulted.
    """
    if _QUARTER.search(row.label) and not (question and _QUARTER.search(question)):
        raise CalculationError(f"L{row.number}: quarterly figure for an annual question")
    if _SEGMENT.search(row.label):
        raise CalculationError(f"L{row.number}: segment figure, not consolidated")
    for qualifier in _QUALIFIERS:
        found = qualifier.search(row.label)
        if found and not qualifier.search(question or ""):
            raise CalculationError(
                f"L{row.number}: '{found.group(0)}' qualifies the line, not the question"
            )
    asked = [int(y) for y in _YEAR.findall(question or "")]
    shown = {int(y) for y in _YEAR.findall(row.label)}
    if asked and not shown:
        # A dated question needs a dated figure: a line that shows no year cannot be
        # placed in the period asked for, whatever column it was read from.
        raise CalculationError(f"L{row.number}: no year on the line for a question that names one")
    if asked and not shown & set(range(min(asked), max(asked) + 1)):
        raise CalculationError(
            f"L{row.number}: {sorted(shown)} is not a year the question asks for"
        )
    # The range is not enough when the formula dates a quantity: "FY2020 revenue /
    # (average total assets between FY2019 and FY2020)" covers 2019, yet its revenue is
    # 2020's. Measured: the 2019 revenue line was pointed at and a 1.22 accepted for 1.33.
    pinned = bound_years(question or "").get(line_item(row.label) or "")
    if pinned and shown and not shown & pinned:
        raise CalculationError(
            f"L{row.number}: {sorted(shown)}, but the formula asks for {line_item(row.label)} "
            f"of {sorted(pinned)}"
        )


def check_term(term: str, row: Row, question: str | None) -> None:
    """The line must be a quantity the question talks about, and the one the term names.

    Two checks, both on text. The line's label must name the same standard line item as
    ``term`` -- that is what stops a neighbouring line, or two variables swapped, from
    passing. And that line item must be one the question mentions: the model describes a
    line in its own words ("net sales" for a margin's revenue, measured), so what is
    checked is the quantity, not the wording. A term the lexicon does not know falls back
    to sharing words with both the label and the question; a synonym outside the lexicon
    is refused rather than trusted.
    """
    wanted = line_item(term)
    if wanted is not None:
        found = line_item(row.label)
        if found != wanted:
            raise CalculationError(f"L{row.number}: {term!r} is {wanted}, the line is {found}")
        if question is not None and wanted not in line_items_in(question):
            raise CalculationError(f"L{row.number}: the question never mentions {wanted}")
        return
    words = _content_words(term)
    if not words:
        raise CalculationError(f"L{row.number}: {term!r} names no quantity")
    if missing := words - _content_words(row.label):
        raise CalculationError(
            f"L{row.number}: {sorted(missing)} of {term!r} not in the line's label"
        )
    if question is not None and not words & _content_words(question):
        raise CalculationError(f"L{row.number}: nothing of {term!r} in the question")


def check_average(expression: str, question: str) -> None:
    """An average the question asks for has to be in the expression, in the right shape.

    Measured, twice: a three year average written as a sum of three margins, three times
    too large; and one whose operator precedence divided only the D&A by revenue, which
    passed every other check but the statement one. The shape is checkable. An N-year
    average is divided by N at the top; when what is averaged is a ratio -- a margin, a
    percentage, a turnover -- the numerator is a sum of exactly N quotients. An average
    between two balance dates divides a sum by 2 somewhere.
    """
    tree = ast.parse(expression, mode="eval").body
    if found := re.search(r"\b(two|three|four|five|[2-5])[ -]year average\b", question, re.I):
        years = found.group(1).lower()
        n = int(years) if years.isdigit() else {"two": 2, "three": 3, "four": 4, "five": 5}[years]
        top = isinstance(tree, ast.BinOp) and isinstance(tree.op, ast.Div)
        if not (top and isinstance(tree.right, ast.Constant) and tree.right.value == n):
            raise CalculationError(f"a {n} year average is not divided by {n}: {expression!r}")
        if _RATIO.search(question):
            terms = _addends(tree.left)
            quotients = [t for t in terms if isinstance(t, ast.BinOp) and isinstance(t.op, ast.Div)]
            if len(terms) != n or len(quotients) != n:
                raise CalculationError(
                    f"a {n} year average of a ratio is not the sum of {n} ratios: {expression!r}"
                )
    elif re.search(r"\baverage\b[^.?]{0,60}\bbetween\b", question, re.I):
        halves = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Div)
            and isinstance(node.right, ast.Constant)
            and node.right.value == 2
        ]
        if not halves:
            raise CalculationError(f"an average between two dates is not halved: {expression!r}")


def _addends(node: ast.AST) -> list[ast.AST]:
    """The terms of a sum, flattened: (a + b) + c gives [a, b, c]."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _addends(node.left) + _addends(node.right)
    return [node]


def check_statements(chosen: list[Row], question: str) -> None:
    """Every statement the question sends you to must be among the rows used.

    Measured: a question asking for operating income *and D&A from the cash flow
    statement* was answered from three income-statement lines, one of them a pension
    credit standing in for depreciation. Each row was genuine, so only the shape of the
    answer betrays it. The statement of a row is read from the filing, not declared.
    """
    used = {row.statement for row in chosen}
    for name, pattern in _STATEMENTS.items():
        if pattern.search(question) and name not in used:
            raise CalculationError(f"the question asks for the {name}, no row comes from it")


def _without_percent_scaling(expression: str, question: str | None) -> str:
    """Drop a top-level "* 100" from a percentage the code will scale itself.

    Told not to, the model still multiplied a margin by 100 (measured), which the code's
    own scaling would have turned into 1030%. Only the outermost factor is removed, and
    only when the question asks for a percent: it changes the form, never the quantity.
    """
    if not question or not answer_format(question)[1]:
        return expression
    tree = ast.parse(expression, mode="eval").body
    if isinstance(tree, ast.BinOp) and isinstance(tree.op, ast.Mult):
        for kept, factor in ((tree.left, tree.right), (tree.right, tree.left)):
            if isinstance(factor, ast.Constant) and factor.value == 100:
                return ast.unparse(kept)
    return expression


def _rehome(chosen: list[Row], rows: list[Row], question: str) -> list[Row]:
    """Move a row to the statement the question names when that statement prints it too.

    The same line often appears twice -- on the income statement and in a note or the
    equity statement. Measured: the model pointed at the note's copy of "Net income
    attributable to shareowners = 9,542", and the statement check refused a correct
    ratio. A row is moved only to one printing the same figure and carrying either the
    same label, or the same standard line item for an overlapping year (a note and a
    statement word the same line differently), so the number cannot change: only where
    it is read from.

    A named statement no row comes from is filled the same way, from a row of another
    named statement that is not its statement's only one. Measured: D&A read off the
    income statement for a question on the income and cash flow statements, which print
    the same 636 for the same year.
    """
    named = {name for name, pattern in _STATEMENTS.items() if pattern.search(question)}
    rehomed = []
    for row in chosen:
        if row.statement not in named:
            twin = next(
                (other for other in rows if other.statement in named and _twins(row, other)),
                None,
            )
            row = twin or row
        rehomed.append(row)
    for statement in sorted(named - {row.statement for row in rehomed}):
        for i, row in enumerate(rehomed):
            rest = {other.statement for j, other in enumerate(rehomed) if j != i}
            if row.statement in named and row.statement not in rest:
                continue
            twin = next(
                (other for other in rows if other.statement == statement and _twins(row, other)),
                None,
            )
            if twin is not None:
                rehomed[i] = twin
                break
    return rehomed


def _twins(row: Row, other: Row) -> bool:
    """Whether two rows print the same figure for the same line, whatever their wording."""
    if _squash(other.label) == _squash(row.label):
        return other.printed == row.printed
    item = line_item(row.label)
    if item is None or line_item(other.label) != item:
        return False
    try:
        same = parse_amount(other.printed) == parse_amount(row.printed)
        if item in _COSTS:
            same = same or abs(parse_amount(other.printed)) == abs(parse_amount(row.printed))
    except CalculationError:
        return False
    years = set(_YEAR.findall(row.label))
    return same and bool(years) and bool(years & set(_YEAR.findall(other.label)))


def _common_scale(chosen: list[Row]) -> tuple[dict[int, Decimal], str | None]:
    """A multiplier per row so that every value is in one scale, and that scale.

    Rows printed in the same scale are used as printed: the result is then in that
    scale, which is what an "answer in USD millions" question expects of a statement in
    millions. Mixed scales are converted to units; a mix involving a row whose scale is
    unknown cannot be reconciled and is refused.
    """
    scales = {row.scale for row in chosen}
    if len(scales) == 1:
        return {row.number: Decimal(1) for row in chosen}, scales.pop()
    if None in scales:
        raise CalculationError("rows in different scales, one of them unknown")
    return {row.number: _SCALE[row.scale] for row in chosen}, "units"


def compute(spec: CalcSpec, rows: list[Row], question: str | None = None) -> Calculation:
    """Read every chosen row, check it, then evaluate the expression. Raises rather than guessing."""
    if not spec.variables:
        raise CalculationError("no variables: the values are not all in the context")
    by_number = {row.number: row for row in rows}
    chosen = []
    for variable in spec.variables:
        if variable.line not in by_number:
            raise CalculationError(f"{variable.name}: line {variable.line} does not exist")
        chosen.append(by_number[variable.line])
    try:
        expression = _without_percent_scaling(spec.expression, question)
    except SyntaxError as exc:
        raise CalculationError(f"unparsable expression: {spec.expression!r}") from exc
    if question:
        chosen = _rehome(chosen, rows, question)
    for variable, row in zip(spec.variables, chosen, strict=True):
        check_row(row, question)
        check_term(variable.term, row, question)
    if question:
        check_statements(chosen, question)
        check_average(expression, question)

    multiplier, scale = _common_scale(chosen)
    values = {}
    for v, row in zip(spec.variables, chosen, strict=True):
        amount = parse_amount(row.printed)
        if line_item(row.label) in _COSTS:
            amount = abs(amount)
        values[v.name] = amount * multiplier[row.number]
    result = evaluate(expression, values)

    wanted = _SCALE_IN.search(question or "")
    if wanted and scale not in (None, wanted.group(1).lower()):
        result = result * (_SCALE.get(scale, Decimal(1)) / _SCALE[wanted.group(1).lower()])
    lines = {v.name: row.number for v, row in zip(spec.variables, chosen, strict=True)}
    return Calculation(result, expression, values, lines)


def evaluate(expression: str, values: dict[str, Decimal]) -> Decimal:
    """Evaluate ``expression`` over ``values`` in exact decimal arithmetic.

    The expression is parsed, then walked: only numbers, the named variables and the
    five arithmetic operators survive. A call, an attribute or an unknown name raises
    rather than resolving, so nothing the model writes can reach the interpreter.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculationError(f"unparsable expression: {expression!r}") from exc

    def walk(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return Decimal(str(node.value))
        if isinstance(node, ast.Name):
            if node.id not in values:
                raise CalculationError(f"unknown variable: {node.id}")
            return values[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        raise CalculationError(f"unsupported expression element: {type(node).__name__}")

    try:
        return walk(tree.body)
    except (DivisionByZero, InvalidOperation) as exc:
        raise CalculationError(f"arithmetic error in {expression!r}") from exc


def _squash(text: str) -> str:
    """Case, punctuation and spacing aside, so a term copied from the question matches it."""
    return re.sub(r"[^a-z0-9&%]+", " ", text.lower()).strip()


def _content_words(text: str) -> set[str]:
    """The words of four letters or more that say which quantity a text is about."""
    return {w.rstrip("s") for w in re.findall(r"[a-z]{4,}", text.lower())} - _STOPWORDS
