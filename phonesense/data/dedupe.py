"""Stage 4: merge near-duplicate listings of the same phone model.

``parent_asin`` already groups colour/size variants of one listing. Different listings of
the same model are merged when their key ``(brand, model name, storage_gb, refurbished)``
matches and the variant words in their titles (lite/plus/pro/max/mini/ultra/edge/xl/5g/
year...) agree. The listing with the most ratings (dataset ``rating_number``) survives; ties
go to the smaller ``parent_asin`` (deterministic). Listings without a brand, a specific
``details["Model Name"]`` (containing a digit) or a parsed storage size are never merged
(not enough evidence). New and renewed listings are kept apart (different products, prices).

Outputs ``interim/dedupe_map.json`` ({merged_asin: surviving_asin}) and
``data/processed/dedupe_log.jsonl`` (one line per merge).

Run: ``python -m phonesense.data.dedupe [--force]``
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from phonesense.data.common import (
    AtomicJsonlWriter,
    Settings,
    iter_jsonl,
    mark_stage_done,
    stage_is_cached,
    write_json,
)
from phonesense.data.normalize import canonical_brand, normalize_model, title_variants


def dedupe_key(candidate: dict[str, Any], specs: dict[str, Any]) -> tuple | None:
    details = candidate.get("details") or {}
    brand = canonical_brand(details.get("Brand"), candidate.get("store"),
                            candidate.get("title"))
    if brand is None:
        return None
    model = normalize_model(details.get("Model Name"), brand)
    storage = specs.get("storage_gb")
    if model is None or storage is None:
        return None
    variants = ",".join(sorted(title_variants(candidate.get("title"))))
    return (brand.lower(), model, float(storage), bool(candidate.get("refurbished")), variants)


def find_duplicates(items: list[tuple[dict, dict]]) -> tuple[dict[str, str], list[dict]]:
    """``items`` = [(candidate, specs), ...]. Returns ({merged: survivor}, merge log)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for cand, specs in items:
        key = dedupe_key(cand, specs)
        if key is not None:
            groups[key].append(cand)
    mapping: dict[str, str] = {}
    log: list[dict] = []
    for key in sorted(groups, key=repr):
        group = groups[key]
        if len(group) < 2:
            continue
        group.sort(key=lambda c: (-(c.get("rating_number") or 0), c["doc_id"]))
        survivor = group[0]
        for other in group[1:]:
            mapping[other["doc_id"]] = survivor["doc_id"]
            log.append({
                "merged": other["doc_id"], "into": survivor["doc_id"],
                "key": {"brand": key[0], "model": key[1], "storage_gb": key[2],
                        "refurbished": key[3], "title_variants": key[4]},
                "merged_title": other["title"], "into_title": survivor["title"],
                "merged_rating_number": other.get("rating_number"),
                "into_rating_number": survivor.get("rating_number"),
            })
    return mapping, log


def run(s: Settings, force: bool = False) -> None:
    outputs = [s.dedupe_map, s.dedupe_log]
    if not force and stage_is_cached(s, "dedupe", outputs, [s.candidates, s.specs]):
        print("[dedupe] cached, skipping (use --force to rebuild)")
        return
    specs = {r["doc_id"]: r["specs"] for r in iter_jsonl(s.specs)}
    items = [(c, specs[c["doc_id"]]) for c in iter_jsonl(s.candidates)]
    mapping, log = find_duplicates(items)
    with AtomicJsonlWriter(s.dedupe_log) as w:
        # A placeholder line keeps the file non-empty (size > 0) when nothing merged.
        w.write({"merges": len(log), "candidates": len(items)})
        for entry in log:
            w.write(entry)
    write_json(s.dedupe_map, mapping)
    mark_stage_done(s, "dedupe")
    print(f"[dedupe] {len(items)} candidates, merged {len(mapping)} duplicate listings "
          f"-> {len(items) - len(mapping)} phones")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    run(Settings.from_config(), force=ap.parse_args().force)


if __name__ == "__main__":
    main()
