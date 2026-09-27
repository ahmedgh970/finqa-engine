"""Pydantic schemas for the workflow's structured LLM outputs.

Passed to Ollama as JSON Schema, so decoding is constrained to a valid object. The
consequence worth keeping in mind: the model can no longer abstain by refusing, so
abstention must be *expressible in the schema* -- here, the grade 0.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChunkGrade(BaseModel):
    """Graded relevance of a single passage, on the four-point UMBRELA scale.

    A graded scale rather than a boolean, for two reasons. It matches how relevance
    actually distributes -- a passage carrying one of the two line items a ratio needs
    is neither irrelevant nor an answer -- and it turns the keep/drop decision into a
    *threshold applied afterwards*, so one run yields the whole precision/recall curve
    instead of one irreversible verdict.

    ``Literal`` rather than a bounded int: it compiles to a JSON Schema enum, which
    constrained decoding actually enforces, where a numeric range may not be.
    """

    grade: Literal[0, 1, 2, 3] = Field(description="0 irrelevant, 1 related, 2 partial, 3 exact")


class CalcVariable(BaseModel):
    """One quantity the formula needs: the words of the question it stands for, and the
    numbered line of the context holding it.

    The model points, it does not copy. Asked to transcribe figures, an 8B model read the
    right line and still wrote 7,617 for 7,616, a 2017 revenue under a 2018 label, and an
    addition where a number belonged. So it names a line, and the code reads the figure,
    the statement, the scale and the period off that line itself. ``term`` is what makes
    the choice of line checkable: the line's label must name the same accounting quantity,
    and that quantity must be one the question talks about.
    """

    name: str = Field(description="Identifier used in the expression, e.g. revenue_2019")
    term: str = Field(description="What this quantity is, e.g. operating income or revenue")
    line: int = Field(description="Number of the context line holding the value, e.g. 17 for L17")


class CalcSpec(BaseModel):
    """A computation the code can verify, then evaluate itself.

    The model states the expression the question defines and points to the lines holding
    its inputs -- nothing else. How to round and whether to answer in percent are written
    in the question, so the code reads them there rather than asking. ``variables`` empty
    is how the model abstains: constrained decoding removes its ability to refuse in
    prose, so abstention has to be expressible here.
    """

    expression: str = Field(description="Formula over the variable names, e.g. rev / ppe_avg")
    variables: list[CalcVariable] = Field(description="Empty when the values are not all present")
