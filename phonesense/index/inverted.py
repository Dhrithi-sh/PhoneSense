"""Positional inverted index over phone fields (T03).

Adapted from our midterm project Dhvani-lingual (``index/positional.py``, ``index/build.py``):
the positional postings ``(doc, tf, positions)`` per zone, duplicate-id checks and per-doc
lengths. What changed: zones ``headline``/``body`` became our fields ``title`` / ``features``
/ ``description`` and the combined ``text``; internal int doc ids; a forward index; per-field
average lengths for BM25; a versioned, VB-compressed on-disk format instead of pickle; and a
stale-index check against the analyzer config.

API used by later tasks (BM25 in T05, positional search in T07, Rocchio, the ablation)::

    from phonesense.index import Analyzer, InvertedIndex

    idx = InvertedIndex.build(phones, Analyzer.from_config())    # or InvertedIndex.load(path)
    idx.N                                  # number of documents
    idx.postings("camera", "text")         # [Posting(doc, tf, positions), ...] sorted by doc
    idx.df("camera"), idx.cf("camera")     # document / collection frequency
    idx.doc_len(d, "title"), idx.avg_doc_len("title")
    idx.doc_id(d) -> "B0..."; idx.doc_index("B0...") -> d
    idx.term_vector(d, "text")             # {term: tf}, forward index
    idx.analyzer.analyze(query_text)       # same analysis as the documents

Document length = number of indexed terms in the field (after stopword removal). Positions
are token offsets that keep gaps for removed stopwords (see :mod:`.analyzer`).

On-disk format (directory, default ``data/index/`` or ``data/index/sample/``):

* ``meta.json``: format version, analyzer config + hash, compression, fields, doc ids, doc
  lengths, source-file fingerprint.
* ``<field>.lex.json``: ``[[term, df, cf, byte_offset, n_bytes], ...]`` sorted by term.
* ``<field>.post``: all postings lists of the field, in lexicon order, encoded by
  :func:`phonesense.index.postings.encode_postings` (VB gaps, or plain uint32).
"""

from __future__ import annotations

import gc
import json
import os
import time
from array import array
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

from phonesense.index.analyzer import Analyzer
from phonesense.index.postings import (
    Posting,
    encode_postings,
    skip_pointers,
    unflatten_from,
    vb_decode,
)
from phonesense.schemas import Phone

FORMAT_VERSION = 1
FIELDS: tuple[str, ...] = ("title", "features", "description", "text")
COMPRESSIONS = ("vbyte", "none")


class StaleIndexError(RuntimeError):
    """The saved index was built with another format version or analyzer config."""


def field_text(phone: Phone, field: str) -> str:
    """The raw text of ``field`` for one phone (features are joined with newlines)."""
    if field == "text":
        return phone.text
    if field == "title":
        return phone.fields.title
    if field == "features":
        return "\n".join(phone.fields.features)
    if field == "description":
        return phone.fields.description
    raise ValueError(f"unknown field {field!r}; expected one of {FIELDS}")


def _rebuild_hint(path: Path) -> str:
    sample = " --sample" if path.name == "sample" else ""
    return f"Rebuild it with: python scripts/build_index.py{sample} --force"


class InvertedIndex:
    """Positional inverted index with per-field postings, lengths and a forward index.

    Build with :meth:`build` or :meth:`load`; the constructor is internal.
    """

    def __init__(self, analyzer: Analyzer, fields: Sequence[str], doc_ids: Sequence[str]):
        for f in fields:
            if f not in FIELDS:
                raise ValueError(f"unknown field {f!r}; expected a subset of {FIELDS}")
        if not fields:
            raise ValueError("need at least one field")
        self.analyzer = analyzer
        self.fields: tuple[str, ...] = tuple(fields)
        self._doc_ids: list[str] = list(doc_ids)
        self._doc_index: dict[str, int] = {}
        for i, d in enumerate(self._doc_ids):
            if d in self._doc_index:
                raise ValueError(f"duplicate doc_id {d!r}")
            self._doc_index[d] = i
        self._postings: dict[str, dict[str, list[Posting]]] = {f: {} for f in self.fields}
        self._cf: dict[str, dict[str, int]] = {f: {} for f in self.fields}
        self._doc_len: dict[str, list[int]] = {f: [0] * len(self._doc_ids) for f in self.fields}
        self._avg_len: dict[str, float] = {}
        self._forward: dict[str, list[dict[str, int]]] = {}
        # load() defers decoding each field's postings until it is first used
        self._pending: dict[str, Callable[[], None]] = {}
        self.build_seconds: float | None = None
        self.source: dict[str, Any] | None = None

    # ------------------------------------------------------------------ building

    @classmethod
    def build(
        cls,
        phones: Iterable[Phone],
        analyzer: Analyzer,
        fields: Sequence[str] = FIELDS,
    ) -> InvertedIndex:
        """Index ``phones`` (in the given order: the i-th phone gets internal id ``i``)."""
        t0 = time.perf_counter()
        phones = list(phones)
        idx = cls(analyzer, fields, [p.doc_id for p in phones])
        for f in idx.fields:
            acc: dict[str, list[Posting]] = defaultdict(list)
            forward: list[dict[str, int]] = []
            for doc, phone in enumerate(phones):
                positions: dict[str, list[int]] = defaultdict(list)
                for term, pos in analyzer.analyze_with_positions(field_text(phone, f)):
                    positions[term].append(pos)
                tv: dict[str, int] = {}
                for term, pos_list in positions.items():
                    acc[term].append(Posting(doc, len(pos_list), pos_list))
                    tv[term] = len(pos_list)
                forward.append(tv)
                idx._doc_len[f][doc] = sum(tv.values())
            # docs were visited in id order, so every postings list is already sorted
            idx._postings[f] = dict(sorted(acc.items()))
            idx._forward[f] = forward
            idx._cf[f] = {t: sum(p.tf for p in pl) for t, pl in idx._postings[f].items()}
        idx._compute_averages()
        idx.build_seconds = time.perf_counter() - t0
        return idx

    def _compute_averages(self) -> None:
        n = max(self.N, 1)
        self._avg_len = {f: sum(self._doc_len[f]) / n for f in self.fields}

    def _build_forward(self, field: str) -> list[dict[str, int]]:
        forward: list[dict[str, int]] = [{} for _ in range(self.N)]
        for term, plist in self._postings[field].items():
            for p in plist:
                forward[p.doc][term] = p.tf
        return forward

    # ------------------------------------------------------------------ queries on the index

    def _field(self, field: str) -> str:
        if field not in self._postings:
            raise KeyError(f"field {field!r} is not indexed; indexed fields: {self.fields}")
        if field in self._pending:
            self._pending.pop(field)()
        return field

    def materialize(self) -> None:
        """Decode every lazily loaded field now (see :meth:`load`)."""
        for f in self.fields:
            self._field(f)

    @property
    def N(self) -> int:  # noqa: N802 (IR notation)
        """Number of documents."""
        return len(self._doc_ids)

    def vocab_size(self, field: str = "text") -> int:
        """Number of distinct terms in ``field``."""
        return len(self._postings[self._field(field)])

    def vocabulary(self, field: str = "text") -> list[str]:
        """All terms of ``field``, sorted (useful for k-gram / wildcard indexes)."""
        return list(self._postings[self._field(field)])

    def __contains__(self, term: str) -> bool:
        """True if ``term`` occurs in the ``text`` field (or the first indexed field)."""
        f = self._field("text" if "text" in self._postings else self.fields[0])
        return term in self._postings[f]

    def df(self, term: str, field: str = "text") -> int:
        """Document frequency: number of documents whose ``field`` contains ``term``."""
        return len(self._postings[self._field(field)].get(term, ()))

    def cf(self, term: str, field: str = "text") -> int:
        """Collection frequency: total occurrences of ``term`` in ``field``."""
        return self._cf[self._field(field)].get(term, 0)

    def postings(self, term: str, field: str = "text") -> list[Posting]:
        """Postings of ``term`` in ``field``, sorted by doc; ``[]`` if absent. Do not mutate."""
        return self._postings[self._field(field)].get(term, [])

    def skip_pointers(self, term: str, field: str = "text") -> dict[int, int]:
        """sqrt(n)-spaced skip pointers for ``postings(term, field)`` (see :mod:`.postings`)."""
        return skip_pointers(self.postings(term, field))

    def total_postings(self, field: str = "text") -> int:
        """Number of (term, doc) pairs in ``field``."""
        return sum(len(pl) for pl in self._postings[self._field(field)].values())

    def doc_len(self, doc: int, field: str = "text") -> int:
        """Length of document ``doc`` in ``field``, in indexed terms."""
        return self._doc_len[self._field(field)][doc]

    def doc_lens(self, field: str = "text") -> list[int]:
        """All document lengths of ``field``, indexed by internal doc id. Do not mutate."""
        return self._doc_len[self._field(field)]

    def avg_doc_len(self, field: str = "text") -> float:
        """Average document length of ``field`` over all N documents."""
        return self._avg_len[self._field(field)]

    def doc_id(self, doc: int) -> str:
        """Contract ``doc_id`` string of internal id ``doc``."""
        return self._doc_ids[doc]

    def doc_index(self, doc_id: str) -> int:
        """Internal int id of a contract ``doc_id``; ``KeyError`` if unknown."""
        return self._doc_index[doc_id]

    @property
    def doc_ids(self) -> list[str]:
        """All contract doc ids, indexed by internal id. Do not mutate."""
        return self._doc_ids

    def term_vector(self, doc: int, field: str = "text") -> dict[str, int]:
        """Forward index: ``{term: tf}`` of document ``doc`` in ``field`` (a copy)."""
        f = self._field(field)
        if f not in self._forward:
            self._forward[f] = self._build_forward(f)
        return dict(self._forward[f][doc])

    def stats(self) -> dict[str, Any]:
        """Summary numbers (N, vocab size, postings, average lengths per field)."""
        self.materialize()
        return {
            "N": self.N,
            "vocab_size": {f: self.vocab_size(f) for f in self.fields},
            "total_postings": {f: self.total_postings(f) for f in self.fields},
            "total_positions": {f: sum(self._cf[f].values()) for f in self.fields},
            "avg_doc_len": {f: round(self.avg_doc_len(f), 4) for f in self.fields},
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InvertedIndex):
            return NotImplemented
        self.materialize()
        other.materialize()
        return (
            self.fields == other.fields
            and self._doc_ids == other._doc_ids
            and self._doc_len == other._doc_len
            and self._postings == other._postings
            and self._cf == other._cf
            and self.analyzer.config_hash() == other.analyzer.config_hash()
        )

    __hash__ = None  # type: ignore[assignment]

    # ------------------------------------------------------------------ persistence

    def save(
        self,
        path: str | Path,
        compression: str = "vbyte",
        source: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        """Write the index to directory ``path``; return ``{file name: bytes}``.

        ``compression``: ``"vbyte"`` (gap + variable-byte, default) or ``"none"`` (plain
        uint32, used to report the uncompressed size). ``source`` is an optional fingerprint
        of the input file, used by ``scripts/build_index.py`` to skip up-to-date rebuilds.
        """
        if compression not in COMPRESSIONS:
            raise ValueError(f"compression must be one of {COMPRESSIONS}")
        path = Path(path)
        self.materialize()
        path.mkdir(parents=True, exist_ok=True)
        sizes: dict[str, int] = {}
        for f in self.fields:
            lexicon, blob, offset = [], bytearray(), 0
            for term, plist in self._postings[f].items():
                data = encode_postings(plist, compression)
                lexicon.append([term, len(plist), self._cf[f][term], offset, len(data)])
                blob += data
                offset += len(data)
            (path / f"{f}.post").write_bytes(bytes(blob))
            lex = json.dumps(lexicon, ensure_ascii=False, separators=(",", ":"))
            (path / f"{f}.lex.json").write_text(lex, encoding="utf-8")
            sizes[f"{f}.post"] = len(blob)
            sizes[f"{f}.lex.json"] = (path / f"{f}.lex.json").stat().st_size
        meta = {
            "format_version": FORMAT_VERSION,
            "analyzer": self.analyzer.config_dict(),
            "analyzer_hash": self.analyzer.config_hash(),
            "compression": compression,
            "fields": list(self.fields),
            "doc_ids": self._doc_ids,
            "doc_len": self._doc_len,
            "source": source if source is not None else self.source,
        }
        (path / "meta.json").write_text(json.dumps(meta, separators=(",", ":")), encoding="utf-8")
        sizes["meta.json"] = (path / "meta.json").stat().st_size
        return sizes

    @staticmethod
    def read_meta(path: str | Path) -> dict[str, Any]:
        """Read ``meta.json`` of a saved index (no postings are loaded)."""
        meta_path = Path(path) / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"no index at {Path(path)}. {_rebuild_hint(Path(path)).replace(' --force', '')}"
            )
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def _decode_field(self, path: Path, f: str, compression: str) -> None:
        """Read and decode ``<f>.lex.json`` + ``<f>.post`` (called lazily by :meth:`load`)."""
        lexicon = json.loads((path / f"{f}.lex.json").read_text(encoding="utf-8"))
        blob = (path / f"{f}.post").read_bytes()
        if compression == "vbyte":
            ints, use_gaps = vb_decode(blob), True
        else:
            arr = array("I")
            arr.frombytes(blob)
            ints, use_gaps = arr.tolist(), False
        pos = 0
        postings: dict[str, list[Posting]] = {}
        cf: dict[str, int] = {}
        # millions of small, acyclic objects: pausing the cyclic GC makes this ~2-3x faster
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            for term, df, term_cf, _offset, _nbytes in lexicon:
                postings[term], pos = unflatten_from(ints, pos, df, use_gaps)
                cf[term] = term_cf
        finally:
            if gc_was_enabled:
                gc.enable()
        if pos != len(ints):
            raise StaleIndexError(f"corrupt index at {path}: {f}.post has trailing data")
        self._postings[f] = postings
        self._cf[f] = cf

    @classmethod
    def load(cls, path: str | Path, analyzer: Analyzer | None = None) -> InvertedIndex:
        """Load a saved index, checking the format version and the analyzer config.

        ``analyzer`` defaults to :meth:`Analyzer.from_config`. Raises
        :class:`StaleIndexError` (with the rebuild command) if the index was built with a
        different format version or analyzer config. Each field's postings are decoded on
        first use, so a retriever that only needs ``text`` doesn't pay for the other fields;
        call :meth:`materialize` to decode everything up front.
        """
        path = Path(path)
        analyzer = analyzer if analyzer is not None else Analyzer.from_config()
        meta = cls.read_meta(path)
        if meta.get("format_version") != FORMAT_VERSION:
            raise StaleIndexError(
                f"index at {path} has format version {meta.get('format_version')}, "
                f"code expects {FORMAT_VERSION}. {_rebuild_hint(path)}"
            )
        if meta.get("analyzer_hash") != analyzer.config_hash():
            raise StaleIndexError(
                f"index at {path} was built with analyzer {meta.get('analyzer')} "
                f"(hash {meta.get('analyzer_hash')}), but the current analyzer is "
                f"{analyzer.config_dict()} (hash {analyzer.config_hash()}). {_rebuild_hint(path)}"
            )
        compression = meta["compression"]
        idx = cls(analyzer, meta["fields"], meta["doc_ids"])
        idx.source = meta.get("source")
        for f in idx.fields:
            lens = meta["doc_len"][f]
            if len(lens) != idx.N:
                raise StaleIndexError(f"corrupt index at {path}: doc_len[{f}] size mismatch")
            idx._doc_len[f] = lens
            idx._pending[f] = partial(idx._decode_field, path, f, compression)
        idx._compute_averages()
        return idx


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    """``{path, size, mtime_ns}`` of a source file (to detect a changed phones.jsonl)."""
    st = os.stat(path)
    return {"path": str(Path(path).resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def dir_size(path: str | Path) -> int:
    """Total bytes of the files directly inside ``path``."""
    return sum(p.stat().st_size for p in Path(path).iterdir() if p.is_file())
