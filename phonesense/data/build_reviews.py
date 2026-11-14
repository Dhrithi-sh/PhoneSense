"""Stage 5: second streaming pass over the raw reviews, keeping reviews of candidate phones.

* Only reviews whose ``parent_asin`` is a phone candidate are parsed (a regex pre-check
  skips JSON parsing for the ~95% of lines about accessories).
* Reviews of merged duplicate listings are remapped to the surviving ``doc_id``.
* Mapped to the contract's review schema, with ``review_id = make_review_id(...)`` and
  ``is_synthetic = false``.
* Dropped: empty text, missing/invalid user_id, rating or timestamp, exact duplicates
  (same review_id, or same user + phone + text).

``collect`` writes ``interim/reviews_candidates.jsonl`` and per-phone counts. The final
``reviews.jsonl`` is written by ``finalize`` (called from build_phones) once the final phone
set is known, so no review can point to a missing phone.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from typing import Any

from phonesense import io
from phonesense.data.common import (
    AtomicJsonlWriter,
    Settings,
    iter_jsonl,
    iter_raw_lines,
    mark_stage_done,
    read_json,
    stage_is_cached,
    write_json,
)
from phonesense.schemas import MIN_TIMESTAMP_MS, Review, make_review_id

_ASIN_RX = re.compile(r'"parent_asin"\s*:\s*"([^"]+)"')
_BR_RX = re.compile(r"<br\s*/?>", re.I)


def clean_review_text(s: Any) -> str:
    """Raw review strings contain HTML line breaks and entities; normalize them."""
    if not isinstance(s, str):
        return ""
    return html.unescape(_BR_RX.sub("\n", s)).strip()


def map_review(raw: dict, doc_id: str) -> tuple[dict | None, str | None]:
    """Raw review -> contract dict, or (None, drop_reason)."""
    user = raw.get("user_id")
    if not isinstance(user, str) or not user or any(c.isspace() for c in user):
        return None, "bad_user_id"
    rating = raw.get("rating")
    if isinstance(rating, bool) or not isinstance(rating, int | float) or rating != int(rating) \
            or not 1 <= rating <= 5:
        return None, "bad_rating"
    ts = raw.get("timestamp")
    if isinstance(ts, bool) or not isinstance(ts, int) or ts < MIN_TIMESTAMP_MS:
        return None, "bad_timestamp"
    text = clean_review_text(raw.get("text"))
    if not text:
        return None, "empty_text"
    helpful = raw.get("helpful_vote")
    helpful = helpful if isinstance(helpful, int) and not isinstance(helpful, bool) \
        and helpful >= 0 else 0
    verified = raw.get("verified_purchase")
    return {
        "review_id": make_review_id(user, doc_id, ts),
        "doc_id": doc_id,
        "user_id": user,
        "rating": int(rating),
        "title": clean_review_text(raw.get("title")),
        "text": text,
        "timestamp": ts,
        "helpful_votes": helpful,
        "verified": verified if isinstance(verified, bool) else False,
        "is_synthetic": False,
    }, None


def _content_key(r: dict) -> int:
    h = hashlib.blake2b(f"{r['user_id']}|{r['doc_id']}|{r['text']}".encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big")


def map_reviews(lines: Iterable[str], target: dict[str, str], drops: Counter
                ) -> Iterator[dict]:
    """Core of the pass. ``target`` maps every accepted parent_asin to its final doc_id."""
    seen_ids: set[int] = set()  # review_ids as ints: ~4x less memory than str
    seen_content: set[int] = set()
    for line in lines:
        m = _ASIN_RX.search(line)
        if not m or m.group(1) not in target:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            drops["bad_json"] += 1
            continue
        asin = raw.get("parent_asin")
        if asin not in target:
            continue
        rec, why = map_review(raw, target[asin])
        if rec is None:
            drops[why] += 1
            continue
        rid = int(rec["review_id"], 16)
        if rid in seen_ids:
            drops["duplicate_review_id"] += 1
            continue
        ck = _content_key(rec)
        if ck in seen_content:
            drops["duplicate_content"] += 1
            continue
        seen_ids.add(rid)
        seen_content.add(ck)
        yield rec


def collect(s: Settings, force: bool = False) -> None:
    outputs = [s.review_candidates, s.review_summary]
    if not force and stage_is_cached(s, "build_reviews", outputs,
                                     [s.candidates, s.dedupe_map]):
        print("[build_reviews] cached, skipping (use --force to rebuild)")
        return
    mapping: dict[str, str] = read_json(s.dedupe_map)
    target = {c["doc_id"]: mapping.get(c["doc_id"], c["doc_id"]) for c in iter_jsonl(s.candidates)}
    drops: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    rating_sums: Counter[str] = Counter()
    lines_seen = 0

    def counted(lines: Iterable[str]) -> Iterator[str]:
        nonlocal lines_seen
        for line in lines:
            lines_seen += 1
            yield line

    with AtomicJsonlWriter(s.review_candidates) as w:
        raw_lines = iter_raw_lines(s.raw_reviews, limit=s.limit, desc="reviews pass",
                                   progress=s.progress)
        for rec in map_reviews(counted(raw_lines), target, drops):
            w.write(rec)
            counts[rec["doc_id"]] += 1
            rating_sums[rec["doc_id"]] += rec["rating"]
    write_json(s.review_summary, {
        "raw_review_lines": lines_seen, "kept": w.count, "dropped": dict(drops),
        "counts": dict(counts), "rating_sums": dict(rating_sums),
    })
    mark_stage_done(s, "build_reviews")
    print(f"[build_reviews] scanned {lines_seen} raw reviews -> {w.count} phone reviews "
          f"(dropped {dict(drops)})")


def finalize(s: Settings, keep_doc_ids: set[str]) -> int:
    """Write the contract ``reviews.jsonl``: candidate reviews of final phones only."""
    def gen() -> Iterator[Review]:
        for d in iter_jsonl(s.review_candidates):
            if d["doc_id"] in keep_doc_ids:
                yield Review.from_dict(d)

    n = io.write_reviews_stream(s.reviews, gen())
    print(f"[build_reviews] wrote {n} reviews -> {s.reviews}")
    return n


def run(s: Settings, force: bool = False) -> None:
    collect(s, force=force)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(Settings.from_config(limit=a.limit), force=a.force)


if __name__ == "__main__":
    main()
