"""Configuration schema for a CRAG workflow run (1 YAML = 1 reproducible experiment).

Extends the RAG config with the switches that select a row of the ablation matrix.
Grading and the calculator act on different failure modes -- noisy retrieval versus
numeric questions -- so their effects should be attributable independently.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from src.rag.config import RagConfig


class GradingConfig(BaseModel):
    """Graded relevance filtering of the retrieved passages.

    Every passage is graded 0-3; ``keep_threshold`` turns those grades into a context.
    Because the grades are recorded per passage, the threshold can be re-examined
    offline from a single run rather than requiring one run per value.

    ``min_chunks`` is the floor: measured, the grader keeps 1-2 passages on 45% of the
    questions and those collapse to 45% accuracy against 73% when 3-5 survive, so a
    selection thinner than the floor is topped up with the best-ranked passages it did
    not already keep. The value doubles as the choice of policy -- 0 disables the floor,
    1 only rescues an empty selection, 3 always guarantees three passages.

    Whether that ranking comes from a cross-encoder is not a switch here: it follows
    from ``retriever`` (dense or reranked) on the parent config.
    """

    enabled: bool = False
    keep_threshold: int = Field(default=2, ge=0, le=3)
    min_chunks: int = Field(default=3, ge=0)
    # Records low_confidence when the best grade across all passages stays below this:
    # the grader saw nothing usable anywhere, which is CRAG's second threshold. No
    # action follows -- it is kept as the hook a later node could branch on, so the
    # signal exists in the records rather than requiring a re-run to obtain. Note it
    # carries information of its own only when it differs from keep_threshold; at equal
    # values it restates n_kept_by_grade == 0. Deliberately unused, not dead.
    low_confidence_below: int = Field(default=2, ge=0, le=3)


class ExpansionConfig(BaseModel):
    """Widen the selected passages to the chunks around them in their document.

    Retrieval is precise on small chunks and reading needs whole tables, so the two
    are decoupled: the grader selects on 256-token passages, then each survivor is read
    with ``window`` chunks of its neighbourhood on each side (``expansion.py``).
    Measured on the 256-token corpus, it is what a selection costs in evidence when a
    statement is cut across consecutive chunks and only one of them scored well.

    The window is a number of chunks, not a token budget: the trim at generation time
    already enforces the window, and keeping it in chunks makes a row reproducible
    whatever the corpus it replays.
    """

    enabled: bool = False
    window: int = Field(default=1, ge=0)


class CalculatorConfig(BaseModel):
    """Deterministic arithmetic path for numeric questions."""

    enabled: bool = False
    routing: str = "heuristic"


class WorkflowConfig(RagConfig):
    """Parameters of a CRAG workflow run."""

    # Names the ablation cell when the switches no longer describe it: a row whose
    # selection was computed by an earlier run and is replayed from it has grading off
    # here, yet it is not the ungraded baseline and must not be filed as one.
    name: str | None = None
    grading: GradingConfig = Field(default_factory=GradingConfig)
    expansion: ExpansionConfig = Field(default_factory=ExpansionConfig)
    calculator: CalculatorConfig = Field(default_factory=CalculatorConfig)


def load_workflow_config(path: str) -> WorkflowConfig:
    """Load and validate a workflow config from a YAML file."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    return WorkflowConfig(**data)


def variant_name(config: WorkflowConfig) -> str:
    """Short name of the ablation cell this config selects.

    Used in the output filename so each cell of the matrix is a separate, resumable
    experiment rather than an overwrite of the previous one. An explicit ``name`` on the
    config wins: it is how a replayed selection declares which cell it belongs to.
    """
    if config.name:
        return config.name
    name = {
        (False, False): "advanced",
        (True, False): "grading",
        (False, True): "calc",
        (True, True): "crag_full",
    }[(config.grading.enabled, config.calculator.enabled)]
    # The expansion window changes what the generator reads, so it belongs to the cell.
    return f"{name}_pm{config.expansion.window}" if config.expansion.enabled else name
