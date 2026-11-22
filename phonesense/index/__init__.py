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


def open_index(sample: bool = False, fallback: bool = True) -> tuple[InvertedIndex, bool]:
    """Load the saved index for scripts; returns ``(index, is_sample)``.

    Without ``sample`` and with ``fallback``, uses the sample index (printing a notice) if the
    full one hasn't been built. Raises ``FileNotFoundError`` with the build command if neither
    exists, and :class:`StaleIndexError` if the index needs rebuilding.
    """
    path = index_path(sample)
    if (
        not sample
        and fallback
        and not (path / "meta.json").exists()
        and (index_path(True) / "meta.json").exists()
    ):
        print(
            "note: full index not built; using the sample index (run `make index` for the full one)"
        )
        path, sample = index_path(True), True
    if not (path / "meta.json").exists():
        flag = " --sample" if sample else ""
        raise FileNotFoundError(
            f"no index at {path}. Build it first: python scripts/build_index.py{flag}"
        )
    return InvertedIndex.load(path), sample


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
    "open_index",
    "skip_pointers",
]
