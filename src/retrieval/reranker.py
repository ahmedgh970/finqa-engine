"""Cross-encoder reranking: score a query against a small candidate set.

Unlike a bi-encoder (query and chunk embedded independently, then compared by
cosine), a cross-encoder scores the pair jointly with full cross-attention
between the two texts. That is far more precise but too slow to run over an
entire collection, so it is only used to re-score the shortlist a first-stage
retriever already narrowed down.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

_models: dict[tuple[str, str | None], CrossEncoder] = {}


def _get_model(model_name: str, dtype: str | None = None) -> CrossEncoder:
    if (model_name, dtype) not in _models:
        kwargs = {"model_kwargs": {"torch_dtype": dtype}} if dtype else {}
        _models[(model_name, dtype)] = CrossEncoder(model_name, **kwargs)
    return _models[(model_name, dtype)]


class Reranker:
    """Score (query, text) pairs with a cross-encoder.

    ``dtype`` (e.g. ``"float16"``) halves the weights and the activations when the
    reranker shares a GPU with a generator; ``None`` keeps the checkpoint's precision,
    the setting every benchmark row was measured with.
    """

    def __init__(self, model_name: str, dtype: str | None = None):
        self.model_name = model_name
        self.dtype = dtype

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return one relevance score per text, in the same order as ``texts``."""
        model = _get_model(self.model_name, self.dtype)
        pairs = [(query, text) for text in texts]
        return model.predict(pairs).tolist()
