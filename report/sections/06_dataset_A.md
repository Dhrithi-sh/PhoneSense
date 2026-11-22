# 6. Dataset

> Report section (owner A). Every number comes from `results/data_stats.json`, written by the
> T02 pipeline (`scripts/build_data.py`, `phonesense/data/`), unless another file is named.

## 6.1 Source

We use the *Cell Phones and Accessories* category of Amazon Reviews 2023 (Hou et al., 2024):
item metadata (title, store, feature bullets, description, a free-form `details` table and a
price where one was listed at crawl time) and the reviews written for those items (rating,
title, text, timestamp, helpful votes, verified-purchase flag). The category has 1,288,490 items,
and the review file has 20,812,945 lines. Most of the items are not phones but cases, chargers,
cables, screen protectors and other accessories, so the first job is to find the phones.

## 6.2 From raw items to phones

The pipeline streams both files once and writes the collection in four stages.

1. **Phone filter.** Each item gets a rule score: phone categories, phone-only `details` keys
   (e.g. a stated battery or screen size), a brand plus model number, the word "smartphone" or
   "unlocked", or a storage size in the title count for a phone; phrases such as "case for",
   "compatible with", accessory words, tablets, watches, hotspots and lots count against it.
   Items scoring at least 4.0 with at least 5 ratings are kept. This keeps 13,424 candidates,
   rejects 1,272,433 items on score (most often for an accessory word in the title, 1,192,253
   items, or an accessory category, 1,021,834) and 2,633 phone-like items with fewer than 5
   ratings.
2. **Specification parsing.** One rule per field reads, in priority order, the `details` table,
   the title and the feature bullets, normalises units (GB, mAh, inches, megapixels) and stores
   `null` for anything missing, unparseable or outside a plausible range. No value is guessed.
3. **Deduplication.** Listings of the same phone (same brand, model, storage and refurbished
   status) are merged into the most-reviewed listing: 2,824 listings are merged.
4. **Reviews.** Reviews of the remaining phones are kept after dropping empty texts (368),
   repeated review ids (7,452) and exact duplicate contents (585); reviews of merged listings
   are moved to the listing that was kept. Phones left with fewer than 5 kept reviews are
   dropped.

The result is the collection used everywhere in this report: **7,166 phones and 575,816 reviews
from 489,887 reviewers**. Each phone record has a stable id (the dataset's `parent_asin`), a
title, brand, price, the eight typed specifications below and a searchable text made of the
title, the feature bullets and the description (CONTRACT.md 4.1). A 500-phone sample,
stratified by brand, is used for development and ships with the code.

## 6.3 What the collection looks like

| Quantity | Value |
|---|---|
| Phones / reviews / reviewers | 7,166 / 575,816 / 489,887 |
| Reviewers with reviews of 2 or more phones | 56,227 |
| Reviews per phone: median (quartiles), maximum | 20 (9, 59), 8,215 |
| Brands | 332 (top: Samsung 1,181, BLU 660, Motorola 615, LG 615, Nokia 344) |
| Refurbished listings | 744 |
| Price known | 22.8% of phones; median $150 (quartiles $80, $279) |

*Table 6.1. The processed collection (`results/data_stats.json`).*

| Specification | Known for | Specification | Known for |
|---|---:|---|---:|
| release year | 96.7% | operating system | 76.1% |
| 5G yes/no | 93.0% | main camera (MP) | 63.9% |
| storage (GB) | 87.5% | RAM (GB) | 47.6% |
| screen size (in) | 78.3% | battery (mAh) | 35.7% |

*Table 6.2. Share of phones with a known value for each specification.*

Three properties of the collection matter for the experiments.

* **It is old and long-tailed.** Release years run from 2000 to 2023; 41% of the phones with a
  known year are from 2014 or earlier, so feature phones and refurbished models are common. The
  median phone has 20 reviews; a few flagship listings have thousands.
* **Prices are mostly missing.** Only 22.8% of phones have a price, because the metadata stores
  a price only for listings on sale when the data was collected (Section 9 analyses this). Under
  our filter semantics a phone with an unknown value never satisfies a constraint on it
  (Section 7.2), so any query with a price constraint can only reach about a quarter of the
  collection.
* **Specifications are incomplete and sometimes wrong.** RAM is known for fewer than half the
  phones and battery size for about a third. Parsing free text also produces occasional wrong
  values (for example a storage size read from an unrelated number in a title), which a spec
  filter then trusts.

## 6.4 Index

The searchable text is indexed with the analyzer of Section 7.1 into a positional inverted
index over four fields: title, features, description and the combined text
(`results/index_stats.json`). The combined field has a vocabulary of 20,020 terms and 891,013
postings; an average phone text has 224 indexed terms (title 15, features 72, description
138). Doc-id and position gaps are stored with variable-byte compression, which makes the
postings 3.4 times smaller than plain 32-bit integers (8.4 MB against 28.3 MB).

## 6.5 Quality check

The pipeline writes two hand-check sheets, a sample of 50 kept phones with their parsed
specifications and 20 near-threshold rejections (`results/handcheck_*.csv`), meant to measure
spec accuracy and the precision of the phone filter by hand. **These sheets were not completed
before the deadline**, so we report no measured accuracy for the filter or the parser. The
spec coverage in Table 6.2 is measured; its correctness is not. The limitations in Section 9
list the consequences: some accessories may remain, some phones may be missing, and a wrongly
parsed value can make a spec filter keep or drop the wrong phone.
