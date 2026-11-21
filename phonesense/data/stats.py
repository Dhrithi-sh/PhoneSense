"""Stage 7: dataset statistics -> ``results/data_stats.json`` plus a short printed summary.

Run: ``python -m phonesense.data.stats``
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from phonesense import io
from phonesense.data.common import Settings, iter_jsonl, read_json, write_json
from phonesense.schemas import SPEC_FIELDS


def _quantiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    v = sorted(values)

    def q(p: float) -> float:
        return float(v[min(len(v) - 1, int(round(p * (len(v) - 1))))])

    return {"min": float(v[0]), "p25": q(0.25), "median": q(0.5), "p75": q(0.75),
            "p90": q(0.9), "p99": q(0.99), "max": float(v[-1]),
            "mean": round(statistics.fmean(v), 2)}


def _histogram(values: list[int]) -> dict[str, int]:
    edges = [5, 10, 20, 50, 100, 200, 500, 1000, 5000]
    out: dict[str, int] = {}
    lo = 0
    for hi in edges:
        out[f"{lo}-{hi - 1}"] = sum(lo <= x < hi for x in values)
        lo = hi
    out[f">={lo}"] = sum(x >= lo for x in values)
    return out


def compute(s: Settings) -> dict[str, Any]:
    fsum = read_json(s.filter_summary)
    rsum = read_json(s.review_summary)
    mapping = read_json(s.dedupe_map)
    phones = io.read_phones(s.phones)
    phone_ids = {p.doc_id for p in phones}

    # Stream reviews: per-user distinct phones without holding all reviews in memory.
    first_doc: dict[str, str] = {}
    multi: set[str] = set()
    n_reviews = 0
    per_phone: Counter[str] = Counter()
    missing_doc = 0
    for r in iter_jsonl(s.reviews):
        n_reviews += 1
        d, u = r["doc_id"], r["user_id"]
        per_phone[d] += 1
        if d not in phone_ids:
            missing_doc += 1
        prev = first_doc.setdefault(u, d)
        if prev != d:
            multi.add(u)
    rpp = [per_phone.get(p.doc_id, 0) for p in phones]

    n = len(phones) or 1
    spec_cov = {f: round(100 * sum(getattr(p.specs, f) is not None for p in phones) / n, 1)
                for f in SPEC_FIELDS}
    brands = Counter(p.brand or "(unknown)" for p in phones)
    prices = [p.price for p in phones if p.price is not None]
    return {
        "settings": {"limit": s.limit, "threshold": s.threshold, "min_reviews": s.min_reviews},
        "raw_items": fsum["raw_items"],
        "filter_decisions": fsum["decisions"],
        "rejection_signal_counts": fsum["negative_signal_counts_among_score_rejects"],
        "phone_candidates": fsum["kept"],
        "duplicate_listings_merged": len(mapping),
        "kept_phones": len(phones),
        "raw_review_lines_scanned": rsum["raw_review_lines"],
        "review_drops": rsum["dropped"],
        "reviews_kept": n_reviews,
        "reviews_pointing_to_missing_doc": missing_doc,
        "unique_reviewers": len(first_doc),
        "reviewers_with_2plus_phones": len(multi),
        "reviews_per_phone": _quantiles(rpp),
        "reviews_per_phone_histogram": _histogram(rpp),
        "spec_coverage_pct": spec_cov,
        "brand_count": len(brands),
        "top_brands": dict(brands.most_common(15)),
        "price_coverage_pct": round(100 * len(prices) / n, 1),
        "price_usd": _quantiles([round(x) for x in prices]),
        "refurbished_phones": sum(m["refurbished"] for m in iter_jsonl(s.phone_meta)),
    }


def summary_lines(st: dict[str, Any]) -> list[str]:
    rpp = st["reviews_per_phone"]
    return [
        f"raw items            {st['raw_items']:,}",
        f"phone candidates     {st['phone_candidates']:,}  "
        f"(merged duplicates: {st['duplicate_listings_merged']:,})",
        f"kept phones          {st['kept_phones']:,}  "
        f"(refurbished: {st['refurbished_phones']:,})",
        f"filter decisions     {st['filter_decisions']}",
        f"reviews kept         {st['reviews_kept']:,}  "
        f"(of {st['raw_review_lines_scanned']:,} raw lines scanned)",
        f"unique reviewers     {st['unique_reviewers']:,}  "
        f"(>=2 phones: {st['reviewers_with_2plus_phones']:,})",
        f"reviews/phone        median {rpp.get('median')}  mean {rpp.get('mean')}  "
        f"max {rpp.get('max')}",
        f"spec coverage %      {st['spec_coverage_pct']}",
        f"price coverage %     {st['price_coverage_pct']}",
        f"top brands           {dict(list(st['top_brands'].items())[:8])}",
    ]


def run(s: Settings, force: bool = False) -> dict[str, Any]:
    st = compute(s)
    if st["reviews_pointing_to_missing_doc"]:
        raise RuntimeError(f"{st['reviews_pointing_to_missing_doc']} reviews point to a "
                           "doc_id missing from phones.jsonl")
    write_json(s.stats_path, st)
    print(f"[stats] wrote {s.stats_path}")
    for line in summary_lines(st):
        print("  " + line)
    return st


def main() -> None:
    run(Settings.from_config())


if __name__ == "__main__":
    main()
