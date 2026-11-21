"""Text analysis shared by documents and queries (T03).

One :class:`Analyzer` turns raw text into index terms. The *same* analyzer instance (same
config) must be used for indexing and for queries; the index stores a hash of the config and
refuses to load with a different one (see :mod:`phonesense.index.inverted`).

Pipeline, in order:

1. Unicode NFKC normalization, lowercasing, typographic quotes mapped to ASCII.
2. Number clean-up: thousands separators are dropped (``5,000`` -> ``5000``).
3. Unit joining: a number followed by a unit becomes one token, with or without a space, and
   unit spellings are canonicalized: ``128 GB`` / ``128GB`` -> ``128gb``,
   ``5,000 mAh`` -> ``5000mah``, ``6.1"`` / ``6.1 inches`` -> ``6.1in``,
   ``48 megapixel`` -> ``48mp``, ``120 Hz`` -> ``120hz``, ``5 G`` -> ``5g``.
4. Tokenization that keeps phone-search tokens whole: model codes (``a54``, ``s23``,
   ``13c``, ``x5``), ``5g`` / ``4g``, decimals (``6.1``) and unit tokens from step 3.
5. Stopword removal with the built-in list in :mod:`phonesense.index.stopwords`. The
   protected words (``5g``, ``4g``, ``pro``, ``max``, ``plus``, ``mini``, ``ultra``,
   ``lite``) are never removed.
6. Optional stemming (``analyzer.stem`` in config.yaml, default true). ``stemmer: porter``
   uses NLTK's Porter stemmer (only the algorithm, no corpus download); ``stemmer: light``
   is a cheap plural-only "S-stemmer" (Harman 1991). This keeps the midterm project's idea
   of comparing stemmer variants (none / light / aggressive = no stem / light / porter).
   Tokens containing a digit and protected words are never stemmed.

Positions: :meth:`Analyzer.analyze_with_positions` returns each kept term with its position
in the token stream *before* stopword removal, so a removed stopword leaves a gap
(``"phone with camera"`` -> ``[("phone", 0), ("camera", 2)]``). Phrase and proximity search
(T07) then treat documents and queries the same way. :meth:`Analyzer.analyze` returns just
the terms in order.

The overall structure (normalize -> tokenize -> optional stemmer variant, one analyzer for
documents and queries) is adapted from our midterm project Dhvani-lingual
(``text/analyzer.py``); the Hindi/Hinglish normalization and stemmers were replaced.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

from phonesense.index.stopwords import PROTECTED, STOPWORDS

ANALYZER_VERSION = 1  # bump when the analysis logic changes, so old indexes become stale
STEMMERS = ("porter", "light")

# Canonical unit spelling for every accepted variant (longest variants first in the regex).
_UNIT_CANON: dict[str, str] = {
    "tb": "tb", "terabyte": "tb", "terabytes": "tb",
    "gb": "gb", "gigabyte": "gb", "gigabytes": "gb",
    "mb": "mb", "megabyte": "mb", "megabytes": "mb",
    "mah": "mah",
    "mp": "mp", "megapixel": "mp", "megapixels": "mp",
    "hz": "hz", "khz": "khz", "mhz": "mhz", "ghz": "ghz",
    "w": "w", "watt": "w", "watts": "w", "wh": "wh",
    "mm": "mm", "cm": "cm", "nm": "nm",
    "in": "in", "inch": "in", "inches": "in", '"': "in", "''": "in",
    "nits": "nits",
    "g": "g",  # "5 G" -> 5g (network); also grams, which is harmless
}
_UNIT_ALT = "|".join(sorted((re.escape(u) for u in _UNIT_CANON), key=len, reverse=True))
# number, optional spaces, unit, then not followed by a letter/digit. Bare "in" after an
# integer with a space ("2 in 1") is ambiguous, so it is handled separately below.
_UNIT_RE = re.compile(rf"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s*({_UNIT_ALT})(?![a-z0-9])")
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                         "′": "'", "″": '"'})
# Tokens: decimals with an optional unit suffix (6.1in, 2.4ghz), or alphanumeric runs with
# optional inner apostrophes (don't, phone's).
_TOKEN_RE = re.compile(r"\d+\.\d+[a-z]*|[a-z0-9]+(?:'[a-z]+)*")
_DIGIT_RE = re.compile(r"\d")


def _join_unit(m: re.Match[str]) -> str:
    number, unit = m.group(1), m.group(2)
    spaced = m.group(0)[len(number)].isspace()
    if unit == "in" and spaced and "." not in number:
        return m.group(0)  # "2 in 1", "5 in stock": not inches
    if unit in ('"', "''") and spaced:
        return m.group(0)  # 'rated 5 "great"': a quotation, not inches
    return f"{number}{_UNIT_CANON[unit]}"


def _light_stem(token: str) -> str:
    """Harman's S-stemmer: only plural endings (ies -> y, es -> e, s -> '')."""
    if len(token) > 3 and token.endswith("ies") and not token.endswith(("eies", "aies")):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("es") and not token.endswith(("aes", "ees", "oes")):
        return token[:-1]
    if len(token) > 2 and token.endswith("s") and not token.endswith(("us", "ss")):
        return token[:-1]
    return token


@dataclass(frozen=True)
class AnalyzerConfig:
    """Settings that change which terms are produced (and therefore the index)."""

    stopwords: bool = True
    stem: bool = True
    stemmer: str = "porter"

    def __post_init__(self) -> None:
        if self.stemmer not in STEMMERS:
            raise ValueError(f"analyzer.stemmer must be one of {STEMMERS}, got {self.stemmer!r}")

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> AnalyzerConfig:
        d = dict(d or {})
        unknown = set(d) - {"stopwords", "stem", "stemmer"}
        if unknown:
            raise ValueError(f"unknown analyzer config keys: {sorted(unknown)}")
        return cls(**d)


class Analyzer:
    """Turns text into index terms. Use one instance for both documents and queries.

    >>> Analyzer().analyze("Samsung Galaxy A54 5G, 128 GB, 5,000 mAh")
    ['samsung', 'galaxi', 'a54', '5g', '128gb', '5000mah']
    """

    def __init__(self, config: AnalyzerConfig | None = None, **kwargs: Any) -> None:
        if config is not None and kwargs:
            raise TypeError("pass either config or keyword settings, not both")
        self.config = config if config is not None else AnalyzerConfig(**kwargs)
        self._stem_fn = self._make_stemmer()

    @classmethod
    def from_config(cls) -> Analyzer:
        """Analyzer built from the ``analyzer:`` section of config.yaml."""
        from phonesense.config import get_config

        return cls(AnalyzerConfig.from_dict(get_config().get("analyzer")))

    def _make_stemmer(self):
        if not self.config.stem:
            return None
        if self.config.stemmer == "light":
            return lru_cache(maxsize=200_000)(_light_stem)
        from nltk.stem import PorterStemmer

        return lru_cache(maxsize=200_000)(PorterStemmer().stem)

    # ------------------------------------------------------------------ identity

    def config_dict(self) -> dict[str, Any]:
        """Config plus the analyzer version, as stored in the index metadata."""
        return {"version": ANALYZER_VERSION, **asdict(self.config)}

    def config_hash(self) -> str:
        """Stable hash of everything that affects the produced terms.

        Covers the analyzer version, the config, the stopword and protected lists, so any
        change to them makes previously built indexes stale.
        """
        payload = {
            **self.config_dict(),
            "stopwords_list": sorted(STOPWORDS),
            "protected": sorted(PROTECTED),
        }
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()[:16]

    # ------------------------------------------------------------------ analysis

    @staticmethod
    def normalize(text: str) -> str:
        """NFKC, lowercase, ASCII quotes, thousands separators removed, units joined."""
        text = unicodedata.normalize("NFKC", text).translate(_QUOTES).lower()
        text = _THOUSANDS_RE.sub("", text)
        return _UNIT_RE.sub(_join_unit, text)

    @staticmethod
    def tokenize(normalized: str) -> list[str]:
        """Split normalized text into raw tokens (before stopwords and stemming)."""
        return _TOKEN_RE.findall(normalized)

    def analyze_with_positions(self, text: str) -> list[tuple[str, int]]:
        """Return ``[(term, position), ...]``; positions count tokens before stopword removal."""
        out: list[tuple[str, int]] = []
        for pos, tok in enumerate(self.tokenize(self.normalize(text))):
            if tok in PROTECTED:
                out.append((tok, pos))
                continue
            if self.config.stopwords and tok in STOPWORDS:
                continue
            if "'" in tok:
                tok = tok[:-2] if tok.endswith("'s") else tok
                tok = tok.replace("'", "")
                if not tok or (self.config.stopwords and tok in STOPWORDS):
                    continue
            if self._stem_fn is not None and not _DIGIT_RE.search(tok):
                tok = self._stem_fn(tok)
            out.append((tok, pos))
        return out

    def analyze(self, text: str) -> list[str]:
        """Return the index terms of ``text`` in order (see :meth:`analyze_with_positions`)."""
        return [term for term, _ in self.analyze_with_positions(text)]

    def __repr__(self) -> str:
        return f"Analyzer({self.config})"
