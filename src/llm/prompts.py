"""Prompt template assembling a question and retrieved chunks into an LLM prompt.

Financial questions often need a precise number, so the prompt asks
explicitly for the exact figure and its unit, grounds the answer in the
given context only, and asks which source it came from.
"""

from __future__ import annotations

from src.ingestion.schema import Chunk

_TEMPLATE = """You are a financial analyst assistant. Answer the question using ONLY the context below — do not use outside knowledge.

If the answer is a number, state the exact figure with its unit (e.g. "$1,577 million", "12.4%"). Cite which source (e.g. "Source 2") the answer comes from. If the context does not contain the answer, say so explicitly instead of guessing.

Context:
{context}

Question: {question}

Answer:"""

_VERIFIED = """A calculation tool has computed this answer from the context and checked every value it used against the passages: {verified}

Use this exact figure as your final answer. If it contradicts your own reading, still report it: the values behind it were verified line by line.

"""


def _format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[Source {i + 1}] {c.text}" for i, c in enumerate(chunks))


def build_prompt(question: str, chunks: list[Chunk], verified: str | None = None) -> str:
    """Assemble a grounded QA prompt from the question and its retrieved chunks.

    ``verified`` carries a figure a calculation tool produced and checked against those
    same passages. It is presented as established rather than suggested, because the
    failure it exists to prevent is the model recomputing it and drifting. With no
    figure the prompt is byte-identical to the baseline's, which is what keeps the
    ablation rows comparable.
    """
    prompt = _TEMPLATE.format(context=_format_context(chunks), question=question)
    if verified is None:
        return prompt
    return prompt.replace("Question: ", _VERIFIED.format(verified=verified) + "Question: ")
