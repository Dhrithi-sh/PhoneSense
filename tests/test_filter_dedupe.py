"""T02 phone filter (rule-based score) and near-duplicate merging."""

import gzip
import json
from pathlib import Path

import pytest

from phonesense.data.dedupe import find_duplicates
from phonesense.data.filter_phones import is_refurbished, score_item
from phonesense.data.normalize import canonical_brand, normalize_model, parse_price

RAW = Path(__file__).parent / "fixtures" / "raw"
THRESHOLD = 4.0  # config.yaml pipeline.filter_threshold


def _meta():
    with gzip.open(RAW / "meta.jsonl.gz", "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


@pytest.mark.parametrize("item", _meta(), ids=lambda m: m["parent_asin"])
def test_filter_decision(item):
    score, reasons = score_item(item)
    assert (score >= THRESHOLD) is item["_expect_phone"], (item["title"], score, reasons)


@pytest.mark.parametrize("title,is_phone", [
    ("Google Pixel 6A 5G 128GB Factory Unlocked (GSM Only | not Compatible with Verizon)", True),
    ("Samsung Galaxy Note 3 16GB Unlocked Smartphone w/S Pen Stylus", True),
    ("Case Compatible with iPhone 15", False),
    ("Kids GPS Tracker Wearable Cell Phone", False),
])
def test_filter_titles(title, is_phone):
    score, _ = score_item({"title": title, "categories": [], "details": {}})
    assert (score >= THRESHOLD) is is_phone, score


def test_reasons_are_logged():
    score, reasons = score_item({"title": "Case for iPhone 15", "categories": [],
                                 "details": {}})
    assert score < THRESHOLD
    assert any("title_for_brand" in r for r in reasons)


def test_refurbished_flag():
    assert is_refurbished("Apple iPhone 12 (Renewed)")
    assert is_refurbished("Galaxy S21 Certified Refurbished")
    assert not is_refurbished("Galaxy S21 Unlocked")


def test_normalizers():
    assert canonical_brand("SAMSUNG", None) == "Samsung"
    assert canonical_brand(None, "Samsung Electronics") == "Samsung"
    assert canonical_brand("Generic", "Visit the Nokia Store") == "Nokia"
    assert canonical_brand("", None) is None
    assert parse_price("$1,099.00") == 1099.0
    assert parse_price(199.99) == 199.99
    assert parse_price("199.99 - 249.99") is None
    assert parse_price(0) is None and parse_price("—") is None
    assert normalize_model("Galaxy A54 5G", "Samsung") == "galaxya545g"
    assert normalize_model("Smartphone", "Samsung") is None
    assert normalize_model("iPhone", "Apple") is None


def _cand(asin, title, n, model="Galaxy A54 5G", brand="Samsung", refurbished=False):
    return {"doc_id": asin, "title": title, "store": None, "rating_number": n,
            "refurbished": refurbished, "details": {"Brand": brand, "Model Name": model}}


def test_dedupe_keeps_listing_with_more_reviews():
    specs = {"storage_gb": 128.0}
    items = [
        (_cand("B2", "Samsung Galaxy A54 5G 128GB", 10), specs),
        (_cand("B1", "SAMSUNG Galaxy A54 5G, 128GB, Black", 300, brand="SAMSUNG"), specs),
        (_cand("B3", "Galaxy A54 256GB", 999), {"storage_gb": 256.0}),     # other storage
        (_cand("B4", "Galaxy A54 (Renewed)", 999, refurbished=True), specs),  # renewed
        (_cand("B5", "Some phone", 999, model="Smartphone"), specs),         # generic model
        (_cand("B6", "Some phone 2", 1, model="Smartphone"), specs),
    ]
    mapping, log = find_duplicates(items)
    assert mapping == {"B2": "B1"}
    assert log[0]["merged"] == "B2" and log[0]["into"] == "B1"


def test_dedupe_keeps_model_variants_apart():
    specs = {"storage_gb": 128.0}
    items = [
        (_cand("B1", "Samsung Galaxy Note 10 Lite 128GB", 5, model="Galaxy Note 10"), specs),
        (_cand("B2", "Samsung Galaxy Note 10+ 128GB", 9, model="Galaxy Note 10"), specs),
        (_cand("B3", "Galaxy A3 (2016) 128GB", 5, model="Galaxy A3"), specs),
        (_cand("B4", "Galaxy A3 (2017) 128GB", 9, model="Galaxy A3"), specs),
        (_cand("B5", "Galaxy Note II 128GB", 5, model="Galaxy Note"), specs),  # no digit
        (_cand("B6", "Galaxy Note I717 128GB", 9, model="Galaxy Note"), specs),
    ]
    assert find_duplicates(items)[0] == {}


def test_dedupe_tie_breaks_on_asin():
    specs = {"storage_gb": 128.0}
    mapping, _ = find_duplicates([(_cand("BZ", "a", 5), specs), (_cand("BA", "b", 5), specs)])
    assert mapping == {"BZ": "BA"}
