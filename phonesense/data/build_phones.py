"""Stage 6: assemble ``phones.jsonl`` (CONTRACT.md 4.1), the ``phone_meta.jsonl`` sidecar and
the final ``reviews.jsonl``.

Final phone set = candidates that survived dedupe and have at least ``min_reviews`` kept
reviews. ``rating_count`` and ``avg_rating`` are computed from those kept reviews, so they
agree with reviews.jsonl. ``text = build_text(fields)``.

The sidecar ``data/processed/phone_meta.jsonl`` holds what the contract has no field for:
``refurbished`` flag, filter score, the source of every parsed spec, merged listings and the
raw ``details`` (one line per phone, same order as phones.jsonl). It is not a contract file.

Run: ``python -m phonesense.data.build_phones [--force]``
"""

from __future__ import annotations

from collections import defaultdict

from phonesense import io
from phonesense.data import build_reviews
from phonesense.data.common import (
    AtomicJsonlWriter,
    Settings,
    iter_jsonl,
    mark_stage_done,
    read_json,
    stage_is_cached,
)
from phonesense.data.normalize import canonical_brand, parse_price
from phonesense.schemas import Phone, PhoneFields, Specs, build_text


def make_phone(cand: dict, specs: dict, n_reviews: int, rating_sum: int, currency: str
               ) -> Phone:
    fields = PhoneFields(title=cand["title"], features=list(cand["features"]),
                         description=cand["description"])
    return Phone(
        doc_id=cand["doc_id"],
        title=cand["title"],
        brand=canonical_brand((cand.get("details") or {}).get("Brand"), cand.get("store"),
                              cand["title"]),
        price=parse_price(cand.get("price")),
        currency=currency,
        avg_rating=round(rating_sum / n_reviews, 3) if n_reviews else None,
        rating_count=n_reviews,
        specs=Specs.from_dict(specs),
        text=build_text(fields),
        fields=fields,
    )


def run(s: Settings, force: bool = False) -> None:
    outputs = [s.phones, s.reviews, s.phone_meta]
    inputs = [s.candidates, s.specs, s.dedupe_map, s.review_summary, s.review_candidates]
    if not force and stage_is_cached(s, "build_phones", outputs, inputs):
        print("[build_phones] cached, skipping (use --force to rebuild)")
        return
    mapping: dict[str, str] = read_json(s.dedupe_map)
    rsum = read_json(s.review_summary)
    counts, rating_sums = rsum["counts"], rsum["rating_sums"]
    specs = {r["doc_id"]: r for r in iter_jsonl(s.specs)}
    merged_from: dict[str, list[str]] = defaultdict(list)
    for merged, survivor in mapping.items():
        merged_from[survivor].append(merged)

    phones: list[Phone] = []
    meta: list[dict] = []
    too_few = 0
    for cand in iter_jsonl(s.candidates):
        doc_id = cand["doc_id"]
        if doc_id in mapping:
            continue
        n = counts.get(doc_id, 0)
        if n < s.min_reviews:
            too_few += 1
            continue
        sp = specs[doc_id]
        phones.append(make_phone(cand, sp["specs"], n, rating_sums.get(doc_id, 0), s.currency))
        meta.append({
            "doc_id": doc_id, "refurbished": cand["refurbished"],
            "filter_score": cand["score"], "spec_sources": sp["sources"],
            "merged_from": sorted(merged_from.get(doc_id, [])),
            "categories": cand["categories"], "store": cand["store"],
            "raw_price": cand["price"], "raw_details": cand["details"],
        })
    phones.sort(key=lambda p: p.doc_id)
    meta.sort(key=lambda m: m["doc_id"])
    io.write_phones(s.phones, phones)
    with AtomicJsonlWriter(s.phone_meta) as w:
        for m in meta:
            w.write(m)
    print(f"[build_phones] wrote {len(phones)} phones -> {s.phones} "
          f"({too_few} candidates dropped with < {s.min_reviews} kept reviews)")
    build_reviews.finalize(s, {p.doc_id for p in phones})
    mark_stage_done(s, "build_phones")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    run(Settings.from_config(), force=ap.parse_args().force)


if __name__ == "__main__":
    main()
