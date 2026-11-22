"""Dense retrieval: sentence embeddings + exact cosine search (owner: B).

Every phone's canonical ``text`` is embedded once with a sentence encoder. The unit-length
vectors are kept in one matrix, so cosine similarity is a single matrix-vector product and the
top ``k`` are taken exactly (no approximate index: the collection has only a few thousand
phones). Ties are broken by ``doc_id`` ascending, like every other retriever.

The embedding matrix is cached on disk under ``dense.cache_dir`` (config.yaml). The cache key
is a hash of the encoder name and of every (doc_id, text) pair, so it is rebuilt whenever the
collection or the model changes and reused otherwise.

Two encoders are provided:

* :class:`SentenceTransformerEncoder`: the real model (``dense.model`` in config.yaml). It needs
  ``pip install -r requirements-dense.txt`` and is imported lazily.
* :class:`HashingEncoder`: a deterministic bag-of-words hashing encoder with no dependencies.
  It is *not* a quality baseline; it exists for tests and offline smoke runs.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Protocol

import numpy as np

from phonesense.config import get_config, get_path
from phonesense.io import Ranking
from phonesense.retrieval.base import Retriever
from phonesense.schemas import Phone, Query


class Encoder(Protocol):
    """Anything that turns texts into unit-length float32 vectors."""

    name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape ``(len(texts), dim)`` with L2-normalised rows."""
        ...


def _normalise(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


class SentenceTransformerEncoder:
    """Wraps a sentence-transformers model; loaded on first use."""

    def __init__(self, model: str, batch_size: int = 64) -> None:
        self.name = model
        self.batch_size = batch_size
        self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise ImportError(
                    "sentence-transformers is not installed: "
                    "pip install -r requirements-dense.txt"
                ) from exc
            self._model = SentenceTransformer(self.name)
        vecs = self._model.encode(
            texts, batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False
        )
        return _normalise(vecs)


class HashingEncoder:
    """Deterministic hashed bag-of-words vectors (tests and offline runs only)."""

    _TOKEN = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.name = f"hashing{dim}"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in self._TOKEN.findall(text.lower()):
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "big")
                out[row, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _normalise(out)


def encoder_from_config() -> Encoder:
    cfg = get_config().get("dense", {})
    return SentenceTransformerEncoder(
        cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
        batch_size=int(cfg.get("batch_size", 64)),
    )


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", name).strip("-").lower()


def collection_key(encoder_name: str, doc_ids: list[str], texts: list[str]) -> str:
    """Cache key: changes when the model, the doc set or any document text changes."""
    h = hashlib.sha1(encoder_name.encode())
    for doc_id, text in zip(doc_ids, texts, strict=True):
        h.update(b"\x00" + doc_id.encode() + b"\x01" + text.encode())
    return h.hexdigest()[:16]


_DEFAULT_CACHE = object()


class DenseRetriever(Retriever):
    """Cosine retrieval over cached sentence embeddings of ``Phone.text``.

    ``cache_dir`` defaults to ``dense.cache_dir`` from config.yaml; pass ``None`` to disable
    the cache (nothing is read or written).
    """

    def __init__(self, encoder: Encoder | None = None, cache_dir=_DEFAULT_CACHE):
        self.encoder = encoder if encoder is not None else encoder_from_config()
        self.name = f"dense_{_slug(self.encoder.name)}"
        if cache_dir is _DEFAULT_CACHE:
            has_cfg = get_config().get("dense", {}).get("cache_dir")
            self.cache_dir: Path | None = get_path("dense_cache") if has_cfg else None
        else:
            self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.doc_ids: list[str] = []
        self.matrix: np.ndarray | None = None
        self.cache_hit = False

    # ------------------------------------------------------------------ fitting

    def fit(self, phones: list[Phone]) -> None:
        """Embed every phone (or load the cached matrix). Documents are kept sorted by doc_id."""
        ordered = sorted(phones, key=lambda p: p.doc_id)
        self.doc_ids = [p.doc_id for p in ordered]
        texts = [p.text for p in ordered]
        key = collection_key(self.encoder.name, self.doc_ids, texts)
        cached = self._load(key)
        self.cache_hit = cached is not None
        if cached is None:
            cached = self.encoder.encode(texts) if texts else np.zeros((0, 1), np.float32)
            self._save(key, cached)
        self.matrix = cached

    def _paths(self, key: str) -> tuple[Path, Path] | None:
        if self.cache_dir is None:
            return None
        stem = f"{_slug(self.encoder.name)}_{key}"
        return self.cache_dir / f"{stem}.npy", self.cache_dir / f"{stem}.json"

    def _load(self, key: str) -> np.ndarray | None:
        paths = self._paths(key)
        if paths is None or not (paths[0].exists() and paths[1].exists()):
            return None
        meta = json.loads(paths[1].read_text(encoding="utf-8"))
        mat = np.load(paths[0])
        if meta.get("key") != key or meta.get("doc_ids") != self.doc_ids:
            return None
        return mat

    def _save(self, key: str, mat: np.ndarray) -> None:
        paths = self._paths(key)
        if paths is None:
            return
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        np.save(paths[0], mat)
        paths[1].write_text(
            json.dumps({"key": key, "encoder": self.encoder.name, "doc_ids": self.doc_ids}),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ searching

    def score_text(self, text: str, k: int = 100) -> Ranking:
        """Top ``k`` ``(doc_id, cosine)`` for free text; ties by doc_id ascending."""
        if self.matrix is None:
            raise RuntimeError("call fit() before searching")
        if k <= 0 or not self.doc_ids:
            return []
        scores = self.matrix @ self.encoder.encode([text])[0]
        # Rows are already in doc_id order, so a stable sort on -score breaks ties by doc_id.
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.doc_ids[i], float(scores[i])) for i in order]

    def search(self, query: Query, k: int = 100) -> Ranking:
        """Rank phones for ``query.text`` (filters are ignored, as in plain TF-IDF)."""
        return self.score_text(query.text, k)
