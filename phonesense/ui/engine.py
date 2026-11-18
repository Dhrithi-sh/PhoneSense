"""Search engine behind the demo UI (owner: B, T24). No Streamlit imports: testable on its own.

The UI (``app.py``) only draws; everything that loads data, builds retrievers and turns a
ranking into displayable results lives here.

Data: the phones file is looked up in this order: the ``PHONESENSE_PHONES`` environment
variable, ``paths.phones`` (full data), ``paths.sample_dir/phones.jsonl`` (the early sample).
When no override is given and a saved index exists (``scripts/build_index.py``) it is loaded;
otherwise an index is built in memory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from phonesense.config import get_path
from phonesense.index import InvertedIndex, index_path
from phonesense.io import read_phones
from phonesense.retrieval.tfidf import TfidfRetriever
from phonesense.schemas import Phone, Query


class NoDataError(FileNotFoundError):
    """No phones file was found; the message says how to create one."""


@dataclass(frozen=True)
class Result:
    """One search result, ready to draw."""

    rank: int
    doc_id: str
    title: str
    brand: str | None
    price: float | None
    currency: str
    avg_rating: float | None
    rating_count: int
    score: float

    @property
    def price_text(self) -> str:
        return "price unknown" if self.price is None else f"{self.price:,.2f} {self.currency}"

    @property
    def rating_text(self) -> str:
        if self.avg_rating is None:
            return "no ratings"
        return f"{self.avg_rating:.1f}/5 ({self.rating_count:,} ratings)"


def find_phones_file() -> tuple[Path, bool]:
    """Return ``(path, is_sample)`` of the phones file to use, or raise :class:`NoDataError`."""
    env = os.environ.get("PHONESENSE_PHONES")
    if env:
        path = Path(env)
        if not path.is_file():
            raise NoDataError(f"PHONESENSE_PHONES points to {path}, which does not exist")
        return path, False
    full = get_path("phones")
    if full.is_file():
        return full, False
    sample = get_path("sample_dir") / "phones.jsonl"
    if sample.is_file():
        return sample, True
    raise NoDataError(
        "no phones file found. Build the data first: python scripts/build_data.py "
        "(or point PHONESENSE_PHONES at a phones.jsonl)"
    )


class SearchEngine:
    """TF-IDF search over a phone collection, returning displayable :class:`Result` rows."""

    def __init__(self, phones: list[Phone], index: InvertedIndex | None = None) -> None:
        self.phones = {p.doc_id: p for p in phones}
        self.tfidf = TfidfRetriever()
        if index is not None:
            self.tfidf.fit_index(index)
        else:
            self.tfidf.fit(phones)

    @classmethod
    def from_default_data(cls) -> SearchEngine:
        """Load the phones (see module docstring) and a saved index when one is usable."""
        path, is_sample = find_phones_file()
        phones = read_phones(path)
        index = None
        if not os.environ.get("PHONESENSE_PHONES"):
            saved = index_path(is_sample)
            if (saved / "meta.json").exists():
                try:
                    index = InvertedIndex.load(saved)
                except Exception:  # stale or unreadable index: fall back to building one
                    index = None
        return cls(phones, index)

    def __len__(self) -> int:
        return len(self.phones)

    def _results(self, ranking: list[tuple[str, float]]) -> list[Result]:
        out = []
        for rank, (doc_id, score) in enumerate(ranking, start=1):
            p = self.phones[doc_id]
            out.append(Result(rank, doc_id, p.title, p.brand, p.price, p.currency,
                              p.avg_rating, p.rating_count, score))
        return out

    def search(self, text: str, k: int = 10) -> list[Result]:
        """Top ``k`` phones for ``text`` (empty text gives no results)."""
        if not text.strip():
            return []
        query = Query(qid="q00", text=text, split="dev")
        return self._results(self.tfidf.search(query, k))
