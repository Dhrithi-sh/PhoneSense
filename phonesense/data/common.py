"""Shared helpers for the T02 data pipeline: settings, streaming JSONL, stage caching.

Every stage takes a :class:`Settings` object, so tests (and partial runs) can point the
whole pipeline at another directory without touching ``config.yaml``.
"""

from __future__ import annotations

import gzip
import json
import os
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from phonesense.config import get_config, get_path, get_seed

# Bump when a stage's logic changes so cached outputs are rebuilt.
PIPELINE_VERSION = "t02-v2"


@dataclass
class Settings:
    """All inputs, outputs and knobs of the pipeline."""

    raw_meta: Path
    raw_reviews: Path
    out_dir: Path  # where phones.jsonl, reviews.jsonl, logs and interim/ go
    phones: Path
    reviews: Path
    phone_meta: Path
    filter_log: Path
    dedupe_log: Path
    interim_dir: Path
    stats_path: Path
    threshold: float = 4.0
    min_reviews: int = 5
    currency: str = "USD"
    seed: int = 42
    limit: int | None = None
    handcheck_margin: float = 3.0
    urls: dict[str, str] = field(default_factory=dict)
    fallback_urls: dict[str, str] = field(default_factory=dict)
    progress: bool = True

    # Interim (stage-to-stage) files: plain JSON(L), not part of the contract.
    @property
    def candidates(self) -> Path:
        return self.interim_dir / "candidates.jsonl"

    @property
    def filter_summary(self) -> Path:
        return self.interim_dir / "filter_summary.json"

    @property
    def specs(self) -> Path:
        return self.interim_dir / "specs.jsonl"

    @property
    def dedupe_map(self) -> Path:
        return self.interim_dir / "dedupe_map.json"

    @property
    def review_candidates(self) -> Path:
        return self.interim_dir / "reviews_candidates.jsonl"

    @property
    def review_summary(self) -> Path:
        return self.interim_dir / "reviews_summary.json"

    @property
    def manifest(self) -> Path:
        return self.interim_dir / "manifest.json"

    def params(self) -> dict[str, Any]:
        """Settings that change outputs (used to invalidate the stage cache)."""
        return {
            "version": PIPELINE_VERSION, "limit": self.limit, "threshold": self.threshold,
            "min_reviews": self.min_reviews, "currency": self.currency,
            "raw_meta": str(self.raw_meta), "raw_reviews": str(self.raw_reviews),
        }

    @classmethod
    def from_config(cls, *, limit: int | None = None, out_dir: Path | None = None,
                    progress: bool = True) -> Settings:
        """Default settings from config.yaml. ``out_dir`` redirects every output there
        (used for partial runs, e.g. ``build_sample.py --limit``)."""
        cfg = get_config()
        ds, pl = cfg.get("dataset", {}), cfg.get("pipeline", {})
        if out_dir is None:
            paths = {k: get_path(k) for k in
                     ("phones", "reviews", "phone_meta", "filter_log", "dedupe_log",
                      "interim_dir", "data_stats")}
            base = get_path("processed_dir")
        else:
            base = Path(out_dir)
            paths = {
                "phones": base / "phones.jsonl", "reviews": base / "reviews.jsonl",
                "phone_meta": base / "phone_meta.jsonl",
                "filter_log": base / "filter_log.jsonl",
                "dedupe_log": base / "dedupe_log.jsonl", "interim_dir": base / "interim",
                "data_stats": base / "data_stats.json",
            }
        return cls(
            raw_meta=get_path("raw_meta"), raw_reviews=get_path("raw_reviews"),
            out_dir=base, phones=paths["phones"], reviews=paths["reviews"],
            phone_meta=paths["phone_meta"], filter_log=paths["filter_log"],
            dedupe_log=paths["dedupe_log"], interim_dir=paths["interim_dir"],
            stats_path=paths["data_stats"],
            threshold=float(pl.get("filter_threshold", 4.0)),
            min_reviews=int(pl.get("min_reviews", 5)),
            currency=str(ds.get("default_currency", "USD")),
            seed=get_seed(), limit=limit,
            handcheck_margin=float(pl.get("handcheck_margin", 3.0)),
            urls=dict(ds.get("urls", {})), fallback_urls=dict(ds.get("fallback_urls", {})),
            progress=progress,
        )


# --------------------------------------------------------------------------------------
# Raw file access
# --------------------------------------------------------------------------------------


def resolve_raw(path: Path) -> Path:
    """Return the raw file to read: ``x.jsonl.gz`` or, if only that exists, ``x.jsonl``."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return path
    if path.suffix == ".gz":
        plain = path.with_suffix("")
        if plain.exists() and plain.stat().st_size > 0:
            return plain
    raise FileNotFoundError(
        f"raw file not found: {path}\n"
        "  Run `python scripts/build_data.py` (downloads it) or download it manually into "
        "data/raw/ (see README, 'Data pipeline')."
    )


def open_text(path: Path):
    """Open a .gz or plain text file for streaming UTF-8 reading."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def iter_raw_lines(path: Path, *, limit: int | None = None, desc: str = "",
                   progress: bool = True) -> Iterator[str]:
    """Stream non-blank lines of a (possibly gzipped) JSONL file, at most ``limit`` lines."""
    with open_text(resolve_raw(path)) as fh, tqdm(
        desc=desc, unit=" lines", total=limit, disable=not progress, mininterval=1.0
    ) as bar:
        n = 0
        for line in fh:
            if limit is not None and n >= limit:
                break
            n += 1
            bar.update(1)
            if line.strip():
                yield line


def iter_raw_records(path: Path, *, limit: int | None = None, desc: str = "",
                     progress: bool = True) -> Iterator[dict]:
    for line in iter_raw_lines(path, limit=limit, desc=desc, progress=progress):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # a corrupt raw line is skipped, never fatal
        if isinstance(rec, dict):
            yield rec


# --------------------------------------------------------------------------------------
# Interim JSONL (streaming, atomic)
# --------------------------------------------------------------------------------------


def iter_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


class AtomicJsonlWriter:
    """Write JSONL to ``path.tmp`` and rename on success, so a crash never leaves a
    half-written file that a later run would mistake for a cached output."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.tmp = self.path.with_name(self.path.name + ".tmp")
        self.count = 0

    def __enter__(self) -> AtomicJsonlWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.tmp.open("w", encoding="utf-8", newline="\n")
        return self

    def write(self, rec: dict) -> None:
        self.fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.count += 1

    def __exit__(self, exc_type, exc, tb) -> None:
        self.fh.close()
        if exc_type is None:
            os.replace(self.tmp, self.path)
        else:
            self.tmp.unlink(missing_ok=True)


def write_jsonl_atomic(path: Path, records: Iterable[dict]) -> int:
    with AtomicJsonlWriter(path) as w:
        for r in records:
            w.write(r)
    return w.count


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------------------
# Stage cache
# --------------------------------------------------------------------------------------


def _load_manifest(s: Settings) -> dict:
    try:
        return read_json(s.manifest)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def stage_is_cached(s: Settings, stage: str, outputs: Iterable[Path],
                    inputs: Iterable[Path] = ()) -> bool:
    """True if every output exists (size > 0), was built with the current params, and is
    not older than any input (so re-running an upstream stage invalidates this one)."""
    outputs = [Path(p) for p in outputs]
    if not all(p.exists() and p.stat().st_size > 0 for p in outputs):
        return False
    if _load_manifest(s).get(stage) != s.params():
        return False
    oldest_out = min(p.stat().st_mtime for p in outputs)
    return all(not Path(p).exists() or Path(p).stat().st_mtime <= oldest_out for p in inputs)


def mark_stage_done(s: Settings, stage: str) -> None:
    m = _load_manifest(s)
    m[stage] = s.params()
    write_json(s.manifest, m)


# --------------------------------------------------------------------------------------
# Small text utilities
# --------------------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def clean_text(s: Any) -> str:
    """Collapse whitespace; non-strings become ''."""
    if not isinstance(s, str):
        return ""
    return _WS_RE.sub(" ", s).strip()


def as_str_list(v: Any) -> list[str]:
    """Raw list fields (features, description, categories) -> cleaned non-empty strings."""
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list):
        return []
    out = [clean_text(x) for x in v]
    return [x for x in out if x]


def as_dict(v: Any) -> dict[str, Any]:
    if isinstance(v, dict):
        return v
    if isinstance(v, str):  # some dumps store details as a JSON string
        try:
            d = json.loads(v)
            return d if isinstance(d, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def settings_dict(s: Settings) -> dict[str, Any]:
    return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(s).items()}
