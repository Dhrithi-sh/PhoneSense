"""T02 end-to-end: tiny raw .jsonl.gz fixtures -> contract-valid phones/reviews."""

import csv
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from phonesense import io
from phonesense.data import (
    build_phones,
    build_reviews,
    dedupe,
    filter_phones,
    parse_specs,
    stats,
)
from phonesense.data.common import Settings
from phonesense.data.handcheck import make as make_handcheck
from phonesense.data.handcheck import score as score_handcheck
from phonesense.data.sample import build_sample, stratified_sample
from phonesense.schemas import build_text, make_review_id

RAW = Path(__file__).parent / "fixtures" / "raw"


def settings(out: Path) -> Settings:
    return Settings(
        raw_meta=RAW / "meta.jsonl.gz", raw_reviews=RAW / "reviews.jsonl.gz", out_dir=out,
        phones=out / "phones.jsonl", reviews=out / "reviews.jsonl",
        phone_meta=out / "phone_meta.jsonl", filter_log=out / "filter_log.jsonl",
        dedupe_log=out / "dedupe_log.jsonl", interim_dir=out / "interim",
        stats_path=out / "data_stats.json", threshold=4.0, min_reviews=5, seed=42,
        progress=False,
    )


def run_all(s: Settings, force: bool = False) -> dict:
    for mod in (filter_phones, parse_specs, dedupe, build_reviews, build_phones):
        mod.run(s, force=force)
    return stats.run(s)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    s = settings(tmp_path_factory.mktemp("run1"))
    st = run_all(s)
    return s, st


def test_outputs_validate(built):
    s, _ = built
    phones = io.read_phones(s.phones)
    reviews = io.read_reviews(s.reviews)
    # 0002 merged into 0001, 0004 has < 5 kept reviews, 0006 too few ratings, BACC* rejected.
    assert [p.doc_id for p in phones] == ["BPHONE0001", "BPHONE0003", "BPHONE0005"]
    ids = {p.doc_id for p in phones}
    assert all(r.doc_id in ids for r in reviews)
    for p in phones:
        assert p.text == build_text(p.fields)
        assert p.rating_count == sum(r.doc_id == p.doc_id for r in reviews)
        assert p.currency == "USD"
    for r in reviews:
        assert r.review_id == make_review_id(r.user_id, r.doc_id, r.timestamp)
        assert r.text.strip() and not r.is_synthetic


def test_phone_contents(built):
    s, _ = built
    phones = {p.doc_id: p for p in io.read_phones(s.phones)}
    a54 = phones["BPHONE0001"]
    assert a54.brand == "Samsung" and a54.price == 349.99
    assert a54.specs.ram_gb == 6.0 and a54.specs.storage_gb == 128.0
    assert a54.specs.battery_mah == 5000 and a54.specs.screen_in == 6.4
    assert a54.specs.main_camera_mp == 50.0 and a54.specs.os == "android"
    assert a54.specs.has_5g is True and a54.specs.release_year == 2023
    iphone = phones["BPHONE0003"]
    assert iphone.price == 289.0 and iphone.specs.os == "ios"
    # Store "Amazon Renewed" is a reseller; the brand comes from the title.
    assert iphone.specs.storage_gb == 64.0 and iphone.brand == "Apple"
    pixel = phones["BPHONE0005"]
    assert pixel.price is None and pixel.brand == "Google"
    assert (pixel.specs.ram_gb, pixel.specs.storage_gb) == (8.0, 128.0)
    assert pixel.specs.screen_in == 6.3 and pixel.specs.battery_mah == 4355
    assert pixel.specs.main_camera_mp == 50.0 and pixel.avg_rating == 4.0


def test_merge_remap_and_review_cleaning(built):
    s, st = built
    reviews = io.read_reviews(s.reviews)
    a54 = [r for r in reviews if r.doc_id == "BPHONE0001"]
    # 6 own reviews + USER7's review of the merged listing; USER0's duplicate text dropped.
    assert len(a54) == 7 and any(r.user_id == "USER7" for r in a54)
    assert not any(r.doc_id == "BPHONE0002" for r in reviews)
    iphone = [r for r in reviews if r.doc_id == "BPHONE0003"]
    assert len(iphone) == 5 and all("<br" not in r.text for r in iphone)
    meta = {m["doc_id"]: m for m in (json.loads(x) for x in s.phone_meta.open())}
    assert meta["BPHONE0001"]["merged_from"] == ["BPHONE0002"]
    assert meta["BPHONE0003"]["refurbished"] is True
    assert meta["BPHONE0001"]["spec_sources"]["ram_gb"] == "details"
    assert st["review_drops"]["empty_text"] == 1
    assert st["review_drops"]["duplicate_content"] == 1
    assert st["review_drops"]["duplicate_review_id"] == 1


def test_stats(built):
    _, st = built
    assert st["raw_items"] == 16 and st["kept_phones"] == 3
    assert st["filter_decisions"]["rejected:min_reviews"] == 1
    assert st["duplicate_listings_merged"] == 1
    assert st["reviews_kept"] == 17 and st["reviews_pointing_to_missing_doc"] == 0
    # USER1 and USER2 reviewed both phones.
    assert st["reviewers_with_2plus_phones"] == 2
    assert st["spec_coverage_pct"]["storage_gb"] == 100.0


def test_filter_log(built):
    s, _ = built
    log = [json.loads(x) for x in s.filter_log.open()]
    assert len(log) == 16
    assert all({"parent_asin", "kept", "score", "reasons"} <= e.keys() for e in log)


def test_deterministic_and_cached(built, tmp_path):
    s1, _ = built
    s2 = settings(tmp_path / "run2")
    run_all(s2)
    assert s1.reviews.read_text(encoding="utf-8") == s2.reviews.read_text(encoding="utf-8")
    assert s1.phones.read_text(encoding="utf-8") == s2.phones.read_text(encoding="utf-8")
    mtime = s2.phones.stat().st_mtime_ns
    run_all(s2)  # everything cached: phones.jsonl not rewritten
    assert s2.phones.stat().st_mtime_ns == mtime
    run_all(s2, force=True)
    assert s1.phones.read_text(encoding="utf-8") == s2.phones.read_text(encoding="utf-8")


def test_limit(tmp_path):
    s = settings(tmp_path / "lim")
    s.limit, s.min_reviews = 3, 0
    for mod in (filter_phones, parse_specs, dedupe, build_reviews, build_phones):
        mod.run(s)
    assert len([json.loads(x) for x in s.filter_log.open()]) == 3
    io.read_phones(s.phones)
    io.read_reviews(s.reviews)


def test_sample(built, tmp_path):
    s, _ = built
    n_phones, n_reviews = build_sample(s.phones, s.reviews, tmp_path / "sample", 1, 42)
    assert n_phones == 1
    phones = io.read_phones(tmp_path / "sample" / "phones.jsonl")
    reviews = io.read_reviews(tmp_path / "sample" / "reviews.jsonl")
    assert len(reviews) == n_reviews and {r.doc_id for r in reviews} == {phones[0].doc_id}


def test_stratified_sample_covers_brands(built):
    s, _ = built
    base = io.read_phones(s.phones)[0]
    brands = ["A"] * 20 + ["B"] * 8 + ["C"] * 2
    phones = [replace(base, doc_id=f"D{i:03d}", brand=b) for i, b in enumerate(brands)]
    sample = stratified_sample(phones, 10, 42)
    counts = Counter(p.brand for p in sample)
    assert len(sample) == 10 and set(counts) == {"A", "B", "C"}
    assert counts["A"] > counts["B"] >= counts["C"] >= 1
    assert sample == stratified_sample(phones, 10, 42)


def test_handcheck_roundtrip(built, tmp_path):
    s, _ = built
    specs_csv, filter_csv = tmp_path / "hs.csv", tmp_path / "hf.csv"
    n_rows, _ = make_handcheck(
        phones_path=s.phones, phone_meta_path=s.phone_meta, filter_log_path=s.filter_log,
        out_specs=specs_csv, out_filter=filter_csv, n_phones=50, n_rejects=20,
        threshold=4.0, margin=30.0, seed=42)
    assert n_rows == 3 * 8
    with pytest.raises(FileExistsError):
        make_handcheck(phones_path=s.phones, phone_meta_path=s.phone_meta,
                       filter_log_path=s.filter_log, out_specs=specs_csv,
                       out_filter=filter_csv, n_phones=50, n_rejects=20, threshold=4.0,
                       margin=30.0, seed=42)
    # Fill: everything correct except ram_gb; every phone is a phone; rejects are not.
    rows = list(csv.DictReader(specs_csv.open(encoding="utf-8")))
    for r in rows:
        r["correct"] = "n" if r["field"] == "ram_gb" else "y"
        r["is_phone"] = "y"
    with specs_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    frows = list(csv.DictReader(filter_csv.open(encoding="utf-8")))
    for r in frows:
        r["is_phone"] = "n"
    with filter_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(frows[0]))
        w.writeheader()
        w.writerows(frows)
    out = score_handcheck(specs_csv=specs_csv, filter_csv=filter_csv,
                          out_path=tmp_path / "summary.json")
    assert out["filter_precision_kept_sample"] == 1.0
    assert out["false_rejection_rate_near_threshold"] == 0.0
    per = out["spec_accuracy_per_field"]
    assert per["storage_gb"]["accuracy"] == 1.0
    assert per["ram_gb"]["accuracy"] == 0.0  # A54 and Pixel have a parsed RAM value
    assert per["ram_gb"]["labelled_parsed_rows"] == 2
