"""Early sample for teammates: N phones stratified by brand, plus all their reviews.

Writes ``<sample_dir>/phones.jsonl`` and ``<sample_dir>/reviews.jsonl`` (contract format).
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path

from phonesense import io
from phonesense.data.common import iter_jsonl
from phonesense.schemas import Phone, Review


def stratified_sample(phones: list[Phone], n: int, seed: int) -> list[Phone]:
    """Proportional allocation by brand (largest remainder), at least one phone per brand
    while there is room, random within each brand. Deterministic for a given seed."""
    if n >= len(phones):
        return sorted(phones, key=lambda p: p.doc_id)
    rng = random.Random(seed)
    groups: dict[str, list[Phone]] = defaultdict(list)
    for p in sorted(phones, key=lambda p: p.doc_id):
        groups[p.brand or "(unknown)"].append(p)
    brands = sorted(groups, key=lambda b: (-len(groups[b]), b))
    total = len(phones)
    exact = {b: n * len(groups[b]) / total for b in brands}
    quota = {b: min(len(groups[b]), max(1, math.floor(exact[b]))) for b in brands}
    # Too many brands for "one each": keep the largest brands' minimum.
    while sum(quota.values()) > n:
        smallest = min((b for b in brands if quota[b] > 0),
                       key=lambda b: (exact[b], b))
        quota[smallest] -= 1
    # Distribute what is left by largest remainder.
    order = sorted(brands, key=lambda b: (-(exact[b] - math.floor(exact[b])), b))
    while sum(quota.values()) < n:
        progressed = False
        for b in order:
            if sum(quota.values()) >= n:
                break
            if quota[b] < len(groups[b]):
                quota[b] += 1
                progressed = True
        if not progressed:
            break
    out: list[Phone] = []
    for b in brands:
        out.extend(rng.sample(groups[b], quota[b]))
    return sorted(out, key=lambda p: p.doc_id)


def build_sample(phones_path: Path, reviews_path: Path, out_dir: Path, n: int, seed: int
                 ) -> tuple[int, int]:
    phones = io.read_phones(phones_path)
    chosen = stratified_sample(phones, n, seed)
    ids = {p.doc_id for p in chosen}
    reviews = [Review.from_dict(d) for d in iter_jsonl(reviews_path) if d["doc_id"] in ids]
    out_dir = Path(out_dir)
    io.write_phones(out_dir / "phones.jsonl", chosen)
    io.write_reviews(out_dir / "reviews.jsonl", reviews)
    print(f"[sample] {len(chosen)} phones ({len({p.brand for p in chosen})} brands), "
          f"{len(reviews)} reviews -> {out_dir}")
    return len(chosen), len(reviews)
