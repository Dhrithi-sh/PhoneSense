# PhoneSense

A search engine for smartphones. Type a need like *"phone under $300 with good battery and
5G"* and get a ranked list of phones. It combines classic IR (TF-IDF, BM25, Boolean spec
filters, phrase search), neural IR (dense retrieval, Rocchio, HyDE, rank fusion), and
graph-based trust ranking (PageRank over a co-review graph, reviewer trust).

**Research question:** does trust-weighted ranking resist review manipulation (injected fake
reviews and reviewers) better than plain ranking, while keeping retrieval quality the same?

Information Retrieval final project. Deadline: **Sat 21 Nov 2026**. Submission: a ZIP of
this repo plus the report.

All file formats and interfaces are defined in **[CONTRACT.md](CONTRACT.md)**. Read it first.

## Team

| Member | Area | Owns |
|--------|------|------|
| **A** | Data, core IR, evaluation | `phonesense/data`, `phonesense/index`, `phonesense/retrieval/{tfidf,bm25,boolean,positional}.py`, `phonesense/eval`, ablation, README/tests/final assembly |
| **B** | Queries, labels, dense, fusion, LLM, UI | `queries/`, `qrels/`, `phonesense/retrieval/{dense,rocchio,hyde,fusion}.py`, `phonesense/llm`, `phonesense/ui` |
| **C** | Graph, trust, attacks | `phonesense/graph`, `phonesense/trust`, `phonesense/attacks`, attack tab in the UI |

## Setup (Python 3.11)

With `make`:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
make setup      # pip install -e . && pip install -r requirements.txt
make test       # pytest
make lint       # ruff check .
```

Plain commands (Windows PowerShell, no `make`):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
pip install -r requirements.txt
pytest
ruff check .
python scripts/build_data.py                   # = make data (T02, see "Data pipeline")
python scripts/build_index.py                  # = make index (T03, see "Index & TF-IDF")
python scripts/run_tfidf.py --split dev        # TF-IDF run (T04)
python scripts/run_eval.py                     # = make eval (TODO, T11)
```

Optional extras: `pip install -r requirements-dense.txt` (B, dense retrieval),
`pip install -r requirements-ui.txt` (demo UI), `pip install -r requirements-dev.txt`
(networkx, only to cross-check PageRank in tests). LLM keys: copy `.env.example` to `.env`.

Validate a run against qrels (metrics arrive in T11):

```bash
python -m phonesense.eval.cli --run tests/fixtures/sample.run --qrels tests/fixtures/sample_qrels.txt
```

Reproduce everything: `make all` (test → data → eval). This becomes the single reproducible
command as tasks land.

## Data pipeline (T02)

Turns the raw Amazon Reviews 2023 `Cell_Phones_and_Accessories` files into the contract files
`data/processed/phones.jsonl` and `reviews.jsonl`. Code: `phonesense/data/`, settings and
URLs: `config.yaml` (`dataset.urls`, `dataset.fallback_urls`, `pipeline.*`).

```bash
python scripts/build_data.py                  # full run (= make data)
python scripts/build_data.py --limit 200000   # quick test on the first N raw lines (= make data-quick)
python scripts/build_data.py --force          # rebuild every stage instead of using the cache
python scripts/build_sample.py                # (re)write the 500-phone sample (= make sample)
python scripts/build_sample.py --limit 2000000  # sample from a partial run if the full run is slow
```

Stages (each `python -m phonesense.data.<stage>` also works on its own):

| Stage | What it does | Output |
|-------|--------------|--------|
| `download` | Fetches both raw files into `data/raw/`; skips files already there. UCSD mirror first, then Hugging Face (`McAuley-Lab/Amazon-Reviews-2023`, served as plain `.jsonl`, gzipped on the fly). If both fail it prints where to put a manually downloaded file (`.jsonl.gz` or `.jsonl`). | `data/raw/*.jsonl.gz` |
| `filter_phones` | Rule-based phone score: phone categories, phone-only `details` keys, brand+model / "smartphone" / "unlocked" / GB in the title (+); "case for", "compatible with <brand>", accessory words, tablet, watch, hotspot, lots (−). Keep if score ≥ `pipeline.filter_threshold`, then drop items with `rating_number` < `min_reviews`. | `filter_log.jsonl` (`parent_asin, kept, score, reasons`) |
| `parse_specs` | One regex function per spec field, priority `details` → title → features; units normalized, impossible values → null; source recorded. | `interim/specs.jsonl` |
| `dedupe` | Merges listings with the same (brand, model name, storage, refurbished); keeps the most-rated one. | `dedupe_log.jsonl` |
| `build_reviews` | Streams the reviews once, keeps reviews of candidate phones, remaps merged listings, drops empty texts and duplicates. | `interim/reviews_candidates.jsonl` |
| `build_phones` | Final phones (≥ `min_reviews` kept reviews), `rating_count`/`avg_rating` from kept reviews, `text = build_text(fields)`; writes the final reviews of those phones only. | `phones.jsonl`, `reviews.jsonl`, `phone_meta.jsonl` |
| `stats` | Counts, coverage, reviewer overlap (for the graph task), brands, prices. | `results/data_stats.json` |

* **Caching:** a stage is skipped if its outputs exist, are newer than its inputs, and were
  built with the same settings (`interim/manifest.json`). `--force` rebuilds everything.
* **`--limit N`** writes to `data/processed/partial_<N>/` (own stats and sample), so a test run
  never overwrites the full build.
* **Sidecar `phone_meta.jsonl`** (not a contract file): `refurbished` flag, filter score, the
  source of each parsed spec, merged listings, raw `details`. Same `doc_id`s as `phones.jsonl`.
* **Sample for teammates:** `data/processed/sample/{phones,reviews}.jsonl` = 500 phones
  stratified by brand (seed from config) + all their reviews. Data files are gitignored; share
  the sample folder directly.
* **Memory/runtime:** everything streams from the `.gz` files; the largest in-memory sets are
  the review-dedupe hashes and the per-reviewer map (a few hundred MB), so 8 GB RAM is
  enough. Measured on a laptop: download ~5 min (meta, 0.9 GB gz) + 15–60 min (reviews,
  depends on bandwidth); filter ~3 min; reviews pass ~10–15 min; `--limit 200000` ~1 min.

**Hand check (quality numbers for the report):**

```bash
python scripts/make_handcheck.py      # = make handcheck
#   fill results/handcheck_specs.csv:  correct = y/n on every row (for a null: y if the listing
#                                      really doesn't state it, n if we missed it);
#                                      is_phone = y/n once per phone
#   fill results/handcheck_filter.csv: is_phone = y/n (y = false rejection)
python scripts/score_handcheck.py     # = make handcheck-score -> results/handcheck_summary.json
```

`make_handcheck.py` refuses to overwrite existing CSVs (they may be filled in) unless `--force`.
The filled CSVs and the summary are exempt from `.gitignore`: commit them.

## Index & TF-IDF (T03, T04)

Our own positional inverted index (`phonesense/index/`) and an lnc.ltc TF-IDF retriever
(`phonesense/retrieval/tfidf.py`). No IR libraries: NLTK supplies only the Porter stemmer
algorithm (no corpus download). Index and weighting logic are adapted from our midterm project
(see [`legacy/PORTING.md`](legacy/PORTING.md)).

```bash
python scripts/build_index.py --sample                 # 500 phones -> data/index/sample/  (= make index-sample)
python scripts/build_index.py                          # full phones.jsonl -> data/index/  (= make index)
python scripts/run_tfidf.py --query "samsung 5g 128 gb"    # top 10 (doc_id, score, title)
python scripts/run_tfidf.py --split dev                # runs/tfidf_lnc.ltc_dev.run + latency (needs queries/queries.jsonl)
```

`build_index.py` skips the build when the saved index is up to date (`--force` rebuilds) and
writes `results/index_stats.json` (`index_stats_sample.json` for `--sample`): N, vocabulary,
postings and average length per field, build/load time, size with VB compression vs plain
uint32. `run_tfidf.py` prints mean/p95 query latency and writes
`results/latency_<run_name>.json`. Indexes live in `data/index/` (gitignored).

**Analyzer** (`analyzer:` in config.yaml, same for documents and queries): NFKC + lowercase;
units joined (`128 GB`/`128GB` -> `128gb`, `5,000 mAh` -> `5000mah`, `6.1"` -> `6.1in`,
`48 megapixel` -> `48mp`); model codes kept (`a54`, `13c`, `5g`); built-in stopword list
(never removes `5g 4g pro max plus mini ultra lite`); Porter stemming except for tokens with
digits. Positions keep gaps where stopwords were removed. Changing the analyzer config makes a
saved index stale: `load()` refuses it and prints the rebuild command.

**API for teammates** (`from phonesense.index import Analyzer, InvertedIndex, load_index`):

| Call | Returns |
|------|---------|
| `InvertedIndex.build(phones, analyzer, fields=FIELDS)` / `load_index(sample=False)` / `idx.save(path)` | index over fields `title`, `features`, `description`, `text` |
| `idx.N`, `idx.vocab_size(field)`, `idx.vocabulary(field)` | doc count, vocabulary |
| `idx.df(term, field="text")`, `idx.cf(term, field)` | document / collection frequency |
| `idx.postings(term, field)` | `[Posting(doc, tf, positions), ...]` sorted by internal doc id |
| `idx.doc_len(doc, field)`, `idx.avg_doc_len(field)` | lengths in indexed terms (for BM25) |
| `idx.doc_id(doc)`, `idx.doc_index(doc_id)` | internal int id <-> contract `doc_id` |
| `idx.term_vector(doc, field)` | `{term: tf}` forward index (Rocchio) |
| `idx.skip_pointers(term, field)`, `phonesense.index.intersect(p1, p2)` | sqrt(n) skips, AND of two postings lists (T07) |
| `idx.analyzer.analyze(text)` | query terms analysed exactly like the documents |

**lnc.ltc** (Manning et al., IIR ch. 6), term-at-a-time over postings, top k by heap, ties
by `doc_id`:

* document: `w(t,d) = 1 + log10 tf(t,d)`, cosine-normalized (no idf)
* query: `w(t,q) = (1 + log10 tf(t,q)) * log10(N / df_t)`, cosine-normalized; OOV terms skipped
* `score(q,d) = sum_t w(t,q) * w(t,d)`

`TfidfRetriever(field="text", index=idx)` reuses a loaded index; `TfidfRetriever().fit(phones)`
builds its own. The run name is `tfidf_lnc.ltc` (`tfidf_lnc.ltc.<field>` for other fields).

## Repo map

```
config.yaml              seed, dataset name/category, every path
CONTRACT.md              formats + interfaces (source of truth)
phonesense/
  config.py              SEED, get_config(), get_path(key)
  schemas.py             dataclasses + validators for every shared record
  io.py                  read/write helpers for every shared file
  retrieval/base.py      Retriever ABC + run_retriever()
  data/ index/ eval/     A
  retrieval/             A (tfidf, bm25, boolean, positional), B (dense, rocchio, hyde, fusion)
  llm/ ui/               B
  graph/ trust/ attacks/ C
data/raw, data/processed dataset files (gitignored)
queries/ qrels/          B: queries.jsonl, llm.txt, human.txt, qrels.txt
runs/ results/           generated run files and metrics (gitignored)
report/figures/          generated figures
scripts/                 entry-point scripts (build_data.py, build_index.py, run_tfidf.py, ...)
legacy/midterm/          our midterm project (reference only; not linted or tested), see legacy/PORTING.md
tests/ (+ fixtures/)     contract tests and one tiny example per file format
```

## Tasks

| Task | Owner | | Task | Owner |
|------|-------|-|------|-------|
| T01 repo, env, contract | A | | T14 Rocchio | B |
| T02 data pipeline | A | | T15 HyDE | B |
| T03 port index code | A | | T16 phone graph | C |
| T04 TF-IDF (lnc.ltc) | A | | T17 PageRank | C |
| T05 BM25 + tuning | A | | T18 damping sweep | C |
| T06 Boolean filters | A | | T19 reviewer trust | C |
| T07 positional search | A | | T20 attack simulator | C |
| T08 queries | B | | T21 attack experiments | C |
| T09 LLM judge, pooling, kappa | B | | T22 ablation table | A |
| T11 evaluation harness | A | | T23 buying guide | B |
| T12 dense retrieval | B | | T24 demo UI | B |
| T13 fusion (RRF, weighted) | B | | T25 attack tab | C |

## Add a new retriever in 5 steps

1. Create a branch (`git checkout -b t04-tfidf`) and a module, e.g. `phonesense/retrieval/tfidf.py`.
2. Subclass `Retriever`: set `name` (e.g. `"tfidf_lnc.ltc"`) and implement
   `fit(phones)` and `search(query, k) -> [(doc_id, score), ...]` (score descending).
3. Load inputs through the contract: `io.read_phones(get_path("phones"))`,
   `io.read_queries(get_path("queries"), split="dev")`.
4. Write the run with `run_retriever(retriever, queries)`, which creates `runs/tfidf_lnc.ltc_dev.run`.
   Check it with `python -m phonesense.eval.cli --run runs/tfidf_lnc.ltc_dev.run --qrels qrels/qrels.txt`.
5. Add a test on the fixtures (`tests/fixtures/phones.jsonl`), run `make test && make lint`,
   and open a PR to `main`.
