"""Retrievers. Every retriever subclasses ``Retriever`` and outputs TREC runs (CONTRACT.md 4.6-4.7).

Owners: A -> tfidf, bm25, boolean, positional; B -> dense, rocchio, hyde, fusion.
"""

from phonesense.retrieval.base import Retriever, run_retriever

__all__ = ["Retriever", "run_retriever"]
