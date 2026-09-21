"""Deterministic Advanced RAG workflow: the graph controls the flow, the LLM only judges.

Contrast with :mod:`src.agents`, where the LLM decides which tool to call and when.
Here every branch is a conditional edge evaluated by the graph; the LLM produces
judgements (is this chunk relevant, what are the operands) but never controls
execution. Nodes are switched on and off from the YAML config so that every row of
the ablation matrix runs on this same code path.
"""
