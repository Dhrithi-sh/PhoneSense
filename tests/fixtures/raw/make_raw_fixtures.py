"""Regenerate the tiny raw-format fixtures used by the T02 pipeline tests:

    python tests/fixtures/raw/make_raw_fixtures.py

Writes ``meta.jsonl.gz`` and ``reviews.jsonl.gz`` next to this file, in the same format as
the Amazon Reviews 2023 files. ``_expect_phone`` is a test-only key (the pipeline ignores
unknown raw keys): whether the rule-based filter score should accept the item as a phone.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

HERE = Path(__file__).parent
CELL = ["Cell Phones & Accessories", "Cell Phones"]


def meta(asin, title, *, cats=(), details=None, features=(), description=(), price=None,
         rating_number=50, store=None, expect=True):
    return {
        "main_category": "Cell Phones & Accessories", "title": title,
        "average_rating": 4.2, "rating_number": rating_number, "features": list(features),
        "description": list(description), "price": price, "store": store,
        "categories": list(cats), "details": details or {}, "parent_asin": asin,
        "_expect_phone": expect,
    }


META = [
    meta("BPHONE0001", "Samsung Galaxy A54 5G 128GB Unlocked Smartphone, Awesome Black",
         cats=CELL, price=349.99, rating_number=120, store="Samsung",
         details={"Operating System": "Android 13", "RAM": "6 GB",
                  "Memory Storage Capacity": "128 GB", "Screen Size": "6.4 Inches",
                  "Battery Capacity": "5000 Milliamp Hours", "Brand": "SAMSUNG",
                  "Model Name": "Galaxy A54 5G", "Cellular Technology": "5G",
                  "Date First Available": "March 16, 2023"},
         features=["50MP main camera + 12MP ultrawide, 32MP front camera",
                   "Long-lasting 5000mAh battery"],
         description=["Awesome screen, awesome camera."]),
    # Same model as BPHONE0001, fewer ratings -> merged into BPHONE0001 by dedupe.
    meta("BPHONE0002", "SAMSUNG Galaxy A54 5G, 128GB, Black - Unlocked", cats=CELL,
         price="$329.00", rating_number=40, store="Samsung Electronics",
         details={"Brand": "Samsung", "Model Name": "Galaxy A54 5G",
                  "Memory Storage Capacity": "128 GB", "Operating System": "Android"}),
    meta("BPHONE0003", "Apple iPhone 12, 64GB, Black - Unlocked (Renewed)", price="$289.00",
         rating_number=900, store="Amazon Renewed"),
    # Few kept reviews in the reviews file -> dropped by min_reviews in build_phones.
    meta("BPHONE0004", "Motorola Moto G Stylus 5G | 2023 | Unlocked | 6/256GB", cats=CELL,
         price=199.99, details={"Operating System": "Android 12", "Brand": "Motorola"}),
    meta("BPHONE0005", "Google Pixel 7 5G Android Phone 8GB RAM 128GB Unlocked", cats=CELL,
         price=None, store="Google",
         features=["6.3-inch display", "4355mAh battery", "50MP + 12MP rear camera"]),
    # A phone, but rating_number < min_reviews -> rejected early by filter_phones.
    meta("BPHONE0006", "BLU View 3 B140DL 32GB Prepaid Smartphone 4G LTE", cats=CELL,
         rating_number=2),
    meta("BACC000001", "Case for iPhone 15 Pro Max, Shockproof Clear Cover",
         cats=["Cell Phones & Accessories", "Cases, Holsters & Sleeves", "Basic Cases"],
         details={"Compatible Phone Models": "iPhone 15 Pro Max", "Screen Size": "6.7 Inches"},
         expect=False),
    meta("BACC000002", "Screen Protector Compatible with iPhone 15, Tempered Glass, 3 Pack",
         expect=False),
    meta("BACC000003", "USB C Charger 20W Fast Charging Block for Samsung Galaxy S23",
         cats=["Cell Phones & Accessories", "Accessories", "Chargers & Power Adapters"],
         expect=False),
    meta("BACC000004", "Samsung Galaxy Tab A8 10.5 32GB Android Tablet",
         details={"Operating System": "Android 11", "RAM": "3 GB"}, expect=False),
    meta("BACC000005", "Samsung Galaxy Watch 5 44mm Bluetooth Smartwatch", expect=False),
    meta("BACC000006", "Car Phone Mount Holder for Dashboard, Compatible with All Phones",
         expect=False),
    meta("BACC000007", "Replacement Battery for iPhone 8, 1821mAh", expect=False),
    meta("BACC000008", "Stylus Pen for Touch Screens, 2 Pack", expect=False),
    meta("BACC000009", "Nano SIM Card Adapter Kit", expect=False),
    meta("BACC000010", "Silicone Skin Case Compatible with Samsung Galaxy S23 Ultra",
         expect=False),
]

T0 = 1_672_531_200_000  # 2023-01-01 in ms


def review(asin, user, i, text="Great phone, battery lasts all day.", rating=5.0):
    return {"rating": rating, "title": f"Review {i}", "text": text, "images": [],
            "asin": asin, "parent_asin": asin, "user_id": user, "timestamp": T0 + i * 86_400_000,
            "helpful_vote": i % 3, "verified_purchase": i % 2 == 0}


REVIEWS = []
for i in range(6):
    REVIEWS.append(review("BPHONE0001", f"USER{i}", i, rating=float(5 - i % 3)))
# Reviews of the merged listing are remapped to BPHONE0001.
REVIEWS.append(review("BPHONE0002", "USER7", 10, text="Same phone, other listing."))
# USER0 posted the same text on the duplicate listing -> duplicate content, dropped.
REVIEWS.append(review("BPHONE0002", "USER0", 11, text=REVIEWS[0]["text"]))
for i, u in enumerate(["USER1", "USER2", "USER8", "USER9", "USER10"]):
    REVIEWS.append(review("BPHONE0003", u, 20 + i, text=f"Renewed iPhone works fine #{i}<br />"))
REVIEWS.append(review("BPHONE0003", "USER11", 30, text="   "))  # empty text, dropped
REVIEWS.append(dict(REVIEWS[-2]))  # exact duplicate line, dropped
for i in range(2):
    REVIEWS.append(review("BPHONE0004", f"USER{20 + i}", 40 + i))
for i in range(5):
    REVIEWS.append(review("BPHONE0005", f"USER{30 + i}", 50 + i, rating=4.0))
for i in range(3):
    REVIEWS.append(review("BACC000001", f"USER{40 + i}", 60 + i, text="Nice case"))


def write_gz(path: Path, records: list[dict]) -> None:
    data = "".join(json.dumps(r) + "\n" for r in records).encode("utf-8")
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(data)


if __name__ == "__main__":
    write_gz(HERE / "meta.jsonl.gz", META)
    write_gz(HERE / "reviews.jsonl.gz", REVIEWS)
    print(f"wrote {len(META)} meta and {len(REVIEWS)} review records to {HERE}")
