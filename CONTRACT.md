# PhoneSense data & interface contract

This file is the **single source of truth** for every shared file format and interface.
Its executable twin is [`phonesense/schemas.py`](phonesense/schemas.py) (validators) plus
[`phonesense/io.py`](phonesense/io.py) (readers/writers). One hand-written example of each
format lives in [`tests/fixtures/`](tests/fixtures/), and `tests/` checks that all three agree.

> **Changing the contract:** open a PR that edits `CONTRACT.md` + `schemas.py` (+ `io.py` if
> needed) + `tests/fixtures/` **together**, keep `make test` green, and tell the team.

General rules for every file:

* UTF-8, `\n` line endings, one record per line (JSONL / TREC / TSV). Blank lines are ignored.
* **Always read and write shared files via `phonesense.io`.** Writers validate before writing;
  readers validate and report `file:line` on errors.
* **Unknown = `null`.** Never use `""`, `"N/A"`, `"unknown"`, `0` or `-1` as placeholders.
* **Strict types.** `"8"` is not `8`; `true` is not `1`. An int is accepted where a float is
  expected (and stored as float); a float is never accepted where an int is expected.
* No unknown keys. Every listed key must be present (with `null` where allowed).
* IDs that appear in whitespace-separated files (`doc_id`, `user_id`, `review_id`, `qid`,
  run names) contain no whitespace.
* All paths come from [`config.yaml`](config.yaml) via `phonesense.config.get_path(key)`;
  the key is given in brackets below.

| § | What | Path [config key] | Format | Owner | Schema / IO |
|---|------|-------------------|--------|-------|-------------|
| 4.1 | Phones | `data/processed/phones.jsonl` [`phones`] | JSONL | A | `Phone` / `read_phones`, `write_phones` |
| 4.2 | Reviews | `data/processed/reviews.jsonl` [`reviews`] | JSONL | A | `Review` / `read_reviews`, `write_reviews` |
| 4.3 | Queries | `queries/queries.jsonl` [`queries`] | JSONL | B | `Query` / `read_queries`, `write_queries` |
| 4.4 | Boolean filters | inside queries | JSON | A (T06) | `validate_filter` |
| 4.5 | Qrels | `qrels/{qrels,llm,human}.txt` [`qrels`, `qrels_llm`, `qrels_human`] | TREC | B | `Qrel` / `read_qrels`, `write_qrels` |
| 4.6 | Runs | `runs/<run_name>.run` [`runs_dir`] | TREC | everyone | `RunEntry` / `read_run`, `write_run`, `write_rankings` |
| 4.7 | Retriever API | `phonesense/retrieval/base.py` | Python | everyone | `Retriever`, `run_retriever` |
| 4.8 | Graph edges, reviewer trust | `data/processed/graph_edges.tsv`, `reviewer_trust.tsv` | TSV | C | `GraphEdge`, `ReviewerTrust` / `read_graph_edges`, `read_reviewer_trust` |
| 4.9 | Results | `results/<run_name>.json` [`results_dir`] | JSON | A | `Results` / `read_results`, `write_results` |

Fixtures: `phones.jsonl`, `reviews.jsonl`, `queries.jsonl`, `sample_qrels.txt`, `sample.run`,
`graph_edges.tsv`, `reviewer_trust.tsv`, `sample_dev.json`.

---

## 4.1 Phones (documents)

`data/processed/phones.jsonl`: one phone per line, produced by T02 from the dataset's item
metadata after filtering out cases, chargers and other accessories.

| Key | Type | Rules |
|-----|------|-------|
| `doc_id` | str | Stable ID = the dataset's `parent_asin`. Unique. No whitespace. |
| `title` | str | Non-empty. |
| `brand` | str \| null | Non-empty if present (dataset `store` / `details.Brand`); null if unknown. |
| `price` | float \| null | > 0, in `currency`. null if missing or unparseable. |
| `currency` | str | ISO 4217 code, e.g. `"USD"` (default from `dataset.default_currency`). |
| `avg_rating` | float \| null | In [1, 5]. null if the item has no ratings. |
| `rating_count` | int | ≥ 0. |
| `specs` | object | See below. **All 8 keys always present.** |
| `text` | str | Non-empty searchable text. Should equal `build_text(fields)` (see below). |
| `fields` | object | `{"title": str, "features": [str, ...], "description": str}`. `title` non-empty; `features` may be `[]`; `description` may be `""`. |

**`specs`**: normalized, typed, `null` when unknown:

| Key | Type | Unit / allowed values | Valid range |
|-----|------|-----------------------|-------------|
| `ram_gb` | float \| null | gigabytes | (0, 64] |
| `storage_gb` | float \| null | gigabytes (internal storage) | (0, 4096] |
| `battery_mah` | int \| null | milliamp-hours | (0, 20000] |
| `screen_in` | float \| null | inches, diagonal | (0, 12] |
| `main_camera_mp` | float \| null | megapixels, main rear camera | (0, 400] |
| `os` | str \| null | `"android"`, `"ios"`, `"other"` (lower-case) | |
| `has_5g` | bool \| null | true only if 5G is stated; false only if a non-5G network is explicitly stated (e.g. "4G LTE only") | |
| `release_year` | int \| null | four-digit year | [2000, 2030] |

**Unit rules** (apply during parsing in T02):

* Memory/storage in **GB**, never MB: `512 MB` → `0.5`; `1 TB` → `1024` (1 TB = 1024 GB,
  1 GB = 1024 MB). Values out of range mean a parse error, so write null.
* Screen size in **inches**: `16.5 cm` → `16.5 / 2.54`. Round to 2 decimals.
* Battery in **mAh** (int). `Wh` values are **not** converted (needs voltage) → null.
* Camera in **MP**: the largest rear sensor (`"50MP + 8MP + 2MP"` → `50`).
* Anything ambiguous → `null`. Never guess and never use 0.

**Canonical text:** `schemas.build_text(fields)` = title, each feature, description, joined
with `\n`, skipping empty parts. T02 must write `text = build_text(fields)` (the fixtures do).
`fields` stays available for field-weighted retrieval.

## 4.2 Reviews

`data/processed/reviews.jsonl`: one review per line, only for `doc_id`s present in phones.

| Key | Type | Rules |
|-----|------|-------|
| `review_id` | str | **Must equal** `make_review_id(user_id, doc_id, timestamp)` = first 16 hex chars of `sha1("{user_id}\|{doc_id}\|{timestamp}")`. Unique per file. |
| `doc_id` | str | The phone's `doc_id` (dataset `parent_asin`). |
| `user_id` | str | Dataset `user_id`. Attack-simulator reviewers use the prefix `SYN_`. |
| `rating` | int | 1–5. |
| `title` | str | May be `""`. |
| `text` | str | May be `""`. |
| `timestamp` | int | **Milliseconds** since Unix epoch (the dataset is already in ms). Values < 1e11 are rejected as "probably seconds". |
| `helpful_votes` | int | ≥ 0 (dataset `helpful_vote`). |
| `verified` | bool | Dataset `verified_purchase`. |
| `is_synthetic` | bool | Optional, default `false`. **Only the attack simulator sets `true`.** |

`read_reviews(path)` **rejects** any synthetic review unless called with
`allow_synthetic=True`, so injected reviews can never leak into the real data file. Attack
experiments write their own files (e.g. `data/processed/reviews_attack10.jsonl`).

## 4.3 Queries

`queries/queries.jsonl`: 45 queries.

| Key | Type | Rules |
|-----|------|-------|
| `qid` | str | `"q01"` … `"q45"` (regex `^q\d{2}$`). Unique. |
| `text` | str | Non-empty natural-language need. |
| `split` | str | `"dev"` or `"test"`. |
| `filters` | object \| null | Optional. Boolean filter (§4.4) that the query implies. |
| `intent_notes` | str \| null | Optional. Guidance for labellers (what counts as relevant). |

**Dev** is for development and tuning only (BM25 k1/b, fusion weights, damping, …). **Test**
is evaluated **once** with the settings chosen on dev. Use `read_queries(path, split="dev")`.

## 4.4 Boolean filter syntax

A filter is a JSON **node**. A node is either a **leaf** or a **combinator**.

```json
{"and": [
  {"field": "ram_gb",  "op": ">=", "value": 8},
  {"field": "has_5g",  "op": "==", "value": true},
  {"or":  [{"field": "price", "op": "<", "value": 300},
           {"not": {"field": "brand", "op": "in", "value": ["Apple", "Samsung"]}}]}
]}
```

* **Leaf:** exactly the keys `field`, `op`, `value`.
* **Combinators:** an object with exactly **one** key: `"and": [node, ...]` (non-empty list),
  `"or": [node, ...]` (non-empty list), `"not": node` (a single node, not a list).
* **Fields and allowed ops:**

| Field | Kind | Allowed ops | Value type |
|-------|------|-------------|------------|
| `price`, `avg_rating`, `rating_count`, `ram_gb`, `storage_gb`, `battery_mah`, `screen_in`, `main_camera_mp`, `release_year` | number | `== != < <= > >= in` | number (int or float) |
| `brand` | string | `== != in` | non-empty string |
| `os` | string | `== != in` | `"android"`, `"ios"`, `"other"` |
| `has_5g` | bool | `== !=` | `true` / `false` |

* `in`: `value` is a non-empty list of values of the field's type; matches if the phone's value
  equals any of them. `value` is never `null`.
* String comparison is exact and case-sensitive (`brand` is compared as stored).
* Unknown fields/ops, wrong value types, and empty lists are rejected by `validate_filter`.

**Null semantics: unknown values never match a constraint.** Evaluation uses three-valued
(Kleene) logic. A leaf whose field is `null` on the phone evaluates to **UNKNOWN** (this
includes `!=`). Then:

| | `and` | `or` | `not` |
|---|---|---|---|
| any child FALSE | FALSE | (see others) | — |
| any child TRUE | (see others) | TRUE | — |
| otherwise, if any child UNKNOWN | UNKNOWN | UNKNOWN | — |
| `not` TRUE / FALSE / UNKNOWN | | | FALSE / TRUE / **UNKNOWN** |

A phone passes the filter **only if the root is TRUE**. So a phone with unknown RAM fails both
`ram_gb >= 8` and `not(ram_gb >= 8)`, but passes `or(ram_gb >= 8, has_5g == true)` if it
has 5G. The evaluator itself is implemented in T06 (`phonesense/retrieval/boolean.py`).

## 4.5 Qrels

TREC qrels format, whitespace-separated, one judgment per line:

```
qid 0 doc_id grade
q01 0 B0TEST0001 3
```

* Second field is always the literal `0`.
* `grade` ∈ {0, 1, 2, 3}: **0** not relevant, **1** marginal, **2** relevant, **3** highly relevant.
* At most one judgment per (qid, doc_id).
* **Binary metrics (P@k, R@k, F1@k, MAP) treat grade ≥ 2 as relevant**
  (`schemas.RELEVANCE_THRESHOLD = 2`). nDCG uses the graded values (gain = grade).
* Unjudged documents count as not relevant.
* Files: `qrels/llm.txt` (LLM judge over pooled results), `qrels/human.txt` (hand-labelled
  subset, used for Cohen's kappa against `llm.txt`), `qrels/qrels.txt` (final labels used by
  every evaluation). How `qrels.txt` is merged from the other two is decided in T09 and
  documented there.

## 4.6 Runs

TREC run format, whitespace-separated, `runs/<run_name>.run`:

```
qid Q0 doc_id rank score run_name
q01 Q0 B0TEST0001 1 12.5 bm25_k1.2_b0.75_dev
```

* Second field is always the literal `Q0`.
* Lines for one query are **contiguous**, `rank` starts at **1** with **no gaps**, `score`
  is a finite float and **non-increasing** with rank (ties allowed). Higher = better.
* At most **100** docs per query (`schemas.MAX_RUN_DEPTH`). No duplicate `doc_id` within a
  query. A query with no results simply has no lines.
* All lines share **one** `run_name`, and the file is named `<run_name>.run`.
* Scores are written with Python `repr(float)`, so write → read round-trips exactly.
* Ties: `rankings_to_entries` / `run_retriever` break score ties by `doc_id` ascending, so
  runs are deterministic.
* **Run naming:** `<method>[_<variant>]_<split>` where split ∈ `dev | test | all`
  (`all` = both splits, e.g. for building the judging pool). Allowed characters:
  letters, digits, `. + - _`. Examples: `bm25_k1.2_b0.75_dev`, `tfidf_lnc.ltc_test`,
  `rrf_bm25+dense_test`, `pr_trust_attack10_test`.

Every retriever, reranker and fusion method from every teammate outputs this format. That
keeps evaluation and the ablation plug-and-play.

## 4.7 Retriever interface

```python
from phonesense.retrieval.base import Retriever, run_retriever

class Retriever(ABC):
    name: str                                   # run-name prefix, e.g. "bm25_k1.2_b0.75"
    def fit(self, phones: list[Phone]) -> None: ...
    def search(self, query: Query, k: int = 100) -> list[tuple[str, float]]: ...
        # (doc_id, score), score descending, <= k items, no duplicate doc_ids

run_retriever(retriever, queries, k=100, run_name=None, runs_dir=None) -> Path
```

* `run_retriever` calls `search` for each query and writes `<runs_dir>/<run_name>.run`.
  `run_name` defaults to `f"{retriever.name}_{split}"`. `runs_dir` defaults to `runs/`.
  The run-name split must match the queries (`_dev` run ⇒ only dev queries).
* `search` receives the whole `Query`, so it can use `query.filters`.
* **Rerankers / fusion** (Rocchio, PageRank reranking, RRF, weighted fusion) read run files
  with `io.read_run(path) -> {qid: [(doc_id, score), ...]}` and write their output with
  `io.write_rankings(path, rankings, run_name)`. They don't need to subclass `Retriever`.

## 4.8 Graph files (owner: C, may evolve; paths are stable)

Tab-separated, **with a header line**:

`data/processed/graph_edges.tsv` [`graph_edges`]
```
src_doc_id	dst_doc_id	weight
B0TEST0001	B0TEST0002	1.0
```
* Directed rows. An undirected co-review graph writes both directions.
* `weight` is a finite float > 0. No self-loops. No duplicate (src, dst) pairs.

`data/processed/reviewer_trust.tsv` [`reviewer_trust`]
```
user_id	trust
AE7K2QLXDZ4P	0.92
```
* `trust` is a float in [0, 1] (1 = fully trusted). One row per `user_id`.

C may add columns or change semantics, but must update this section, the schema and the
fixtures in the same PR. The paths stay the same.

## 4.9 Results

`results/<run_name>.json`, written by the evaluation harness (T11):

```json
{
  "run": "bm25_k1.2_b0.75_dev",
  "qrels": "qrels/qrels.txt",
  "split": "dev",
  "metrics":   {"P@10": 0.41, "R@10": 0.33, "F1@10": 0.36, "MAP": 0.38, "nDCG@10": 0.52},
  "per_query": {"q01": {"P@10": 0.5, "R@10": 0.4, "F1@10": 0.44, "MAP": 0.47, "nDCG@10": 0.61}},
  "meta": {"k1": 1.2, "b": 0.75}
}
```

* `run` is a valid run name and `split` must equal its suffix.
* `metrics`: aggregate = **mean over queries** of the per-query values.
* `per_query` is **required** (paired significance tests need it): one entry per evaluated
  qid, each with **exactly the same metric names** as `metrics`. Under the `"MAP"` key a
  per-query entry holds that query's AP.
* Metric names: `P@k`, `R@k`, `F1@k`, `MAP`, `nDCG@k` (e.g. `P@10`, `nDCG@5`).
* `meta` is optional and free-form (parameters, timestamps, …).

## 4.10 Conventions

* **Branch per task** (`t04-tfidf`, `t16-graph`, …), PRs into `main`, at least one teammate
  reviews. `make test` (or `pytest`) and `ruff check .` must pass before merging.
* **No hand-typed numbers in the report.** Every table and figure is generated by a script
  into `results/` or `report/figures/`.
* **Randomness:** the seed comes only from `config.yaml` (`from phonesense.config import SEED`).
  Never hard-code a seed.
* **Paths and dataset settings** come only from `config.yaml` via `phonesense.config`.
* **Secrets** (LLM API keys) go in `.env` (see `.env.example`). Never commit them.
* **Own implementations:** no BM25 or PageRank libraries. `networkx` is allowed only in tests
  to cross-check our PageRank.
* **If you need to change this contract, open a PR that edits CONTRACT.md + schemas +
  fixtures together and tell the team.**
