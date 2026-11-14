"""Inverted / positional index (owner: A). T03.

    from phonesense.index import Analyzer, InvertedIndex, Posting, load_index
"""

from phonesense.index.analyzer import Analyzer, AnalyzerConfig
from phonesense.index.inverted import FIELDS, InvertedIndex, StaleIndexError, field_text
from phonesense.index.postings import Posting, intersect, skip_pointers


def index_path(sample: bool = False):
    """Default on-disk location: ``paths.index_dir`` (``data/index``), or its ``sample/``."""
    from phonesense.config import get_path

    base = get_path("index_dir")
    return base / "sample" if sample else base


def load_index(sample: bool = False, analyzer: Analyzer | None = None) -> InvertedIndex:
    """Load the saved full (or sample) index built by ``scripts/build_index.py``."""
    return InvertedIndex.load(index_path(sample), analyzer)


__all__ = [
    "FIELDS",
    "Analyzer",
    "AnalyzerConfig",
    "InvertedIndex",
    "Posting",
    "StaleIndexError",
    "field_text",
    "index_path",
    "intersect",
    "load_index",
    "skip_pointers",
]
