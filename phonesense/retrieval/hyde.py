"""HyDE: Hypothetical Document Embeddings, adapted to sparse retrieval (owner: B).

Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels* (2022): instead of
matching a short query against documents, ask an LLM to write the document that would answer
it, then search with that text. A user who types "phone whose battery lasts two days" uses
different words from a listing that says "6000mAh battery, up to 48 hours of use"; the
hypothetical listing bridges that vocabulary gap.

This module covers the generation step. Every call goes through
:class:`~phonesense.llm.client.LLMClient`, so each prompt and answer is logged under
``runs/llm_logs/hyde.jsonl`` and repeated calls are replayed rather than re-sent. How the
hypothetical text is folded into the query is in :mod:`phonesense.retrieval.rocchio` (the
``hypothetical`` weight) and :class:`HydeRetriever` below.

The model only *imagines* a listing, so its text can contain invented specs or brands. That is
why the text is never shown to users and never used as a document: it only reweights query
terms, and the original query keeps its own weight (expansion drift is the known risk).
"""

from __future__ import annotations

import re

from phonesense.io import Ranking
from phonesense.llm.client import LLMClient
from phonesense.retrieval.rocchio import RocchioRetriever
from phonesense.retrieval.tfidf import TfidfRetriever

PURPOSE = "hyde"

PROMPT = """\
You write product listings for an online smartphone store.

A shopper typed this search: "{query}"

Write the listing of ONE phone that fits this search perfectly, in the style of an Amazon \
listing: a title line, 3 to 5 feature lines, then a 2-sentence description. Where the search \
implies a spec, use concrete numbers and units (RAM in GB, storage in GB, battery in mAh, \
screen in inches, main camera in MP, price in USD). Name a brand or model only if the search \
names one. Output only the listing, with no introduction and no markdown formatting.
"""


def build_prompt(query_text: str) -> str:
    """The exact prompt sent to the LLM for ``query_text`` (also what gets logged)."""
    text = " ".join(query_text.split())
    if not text:
        raise ValueError("query text is empty")
    return PROMPT.format(query=text.replace('"', "'"))


def clean_listing(raw: str, max_chars: int = 1200) -> str:
    """Strip code fences and markdown markers, collapse blank lines, cap the length."""
    text = re.sub(r"^```[a-zA-Z]*\s*$", "", raw, flags=re.MULTILINE)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]\s+|#+\s*)", "", line.replace("**", "")).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)[:max_chars].strip()


class HydeGenerator:
    """Turns a query into a hypothetical phone listing with an :class:`LLMClient`."""

    def __init__(self, client: LLMClient, max_chars: int = 1200) -> None:
        self.client = client
        self.max_chars = max_chars

    def generate(self, query_text: str) -> str:
        """Return the cleaned hypothetical listing (empty string if the model said nothing)."""
        raw = self.client.complete(build_prompt(query_text), purpose=PURPOSE)
        return clean_listing(raw, self.max_chars)


class HydeRetriever(RocchioRetriever):
    """TF-IDF + Rocchio + HyDE: ``q' = alpha*q + beta*centroid(top m) + delta*h``.

    ``h`` is the lnc vector of the hypothetical listing for the query. The first pass that picks
    the pseudo-relevant documents still uses the original query only, so a bad hypothetical
    listing can shift the weights but cannot choose the feedback set. If the generator returns
    nothing the result equals plain Rocchio.
    """

    def __init__(
        self,
        generator: HydeGenerator,
        base: TfidfRetriever | None = None,
        m: int = 10,
        alpha: float = 1.0,
        beta: float = 0.75,
        gamma: float = 0.0,
        n_terms: int = 30,
        delta: float = 0.75,
    ) -> None:
        super().__init__(
            base, m=m, alpha=alpha, beta=beta, gamma=gamma, n_terms=n_terms, delta=delta
        )
        self.generator = generator
        self.name = "hyde" + self.name.removeprefix("rocchio")

    def score_text(self, text: str, k: int = 100) -> Ranking:
        hypothetical = self.generator.generate(text)
        return self.score_vector(self.expanded_query(text, hypothetical), k)
