# Relevance judgments

| File | What | Source |
|---|---|---|
| `llm.txt` | 800 graded judgments (0-3), 495 dev and 305 test | LLM judge |
| `human.txt` | hand labels (not there yet) | the three of us, see below |
| `qrels.txt` | final qrels: a human grade replaces the LLM grade where both exist | `python scripts/kappa.py --merge` |

Until `human.txt` exists, `qrels.txt` equals `llm.txt` and there is **no kappa**.
Format (CONTRACT 4.5): `qid 0 doc_id grade`. Grades: 0 not relevant, 1 marginal, 2 relevant,
3 perfect; for binary metrics grade >= 2 is relevant.

## How the LLM labels were made

* **Pool:** the top 5 of TF-IDF lnc.ltc, BM25 (default k1, b), Rocchio, HyDE + Rocchio and dense
  (all-MiniLM-L6-v2) for each of the 45 queries. A query with no relevant phone after that was
  re-pooled to depth 10 (dev: q04 q14 q25 q27 q29 q31; test: q11 q23 q34).
* **Judge:** a 0-3 rubric prompt (`phonesense/llm/judge.py`). It sees the query, its labelling
  notes and filters, and the phone's title, brand, price, known specs, a few features and the start
  of the description. It never sees ratings, reviews or which system found the phone. A spec the
  listing does not state counts as unconfirmed, never as a 0 on its own.
* **Models:** two free Groq models, one per query (round robin over the sorted query ids):
  `openai/gpt-oss-120b` for 23 queries and `qwen/qwen3.8-27b` for 22.
* **Unjudged phones count as not relevant** (pooling bias). A depth-5 pool is shallow, so systems
  that returned many phones outside the pool are scored lower than they might deserve.

## Queries with no relevant phone

These drop out of every per-query metric (the harness reports `excluded`). **Decision: keep them
and report them as "no relevant phone in the collection"**, not as failures of any system.

| Query | Split | Why |
|---|---|---|
| q04 iPhone with 128GB storage under $500 | dev | price is almost never stated, so nothing is confirmed |
| q27 Google Pixel 7a | dev | no phone with this name in the collection |
| q29 OnePlus Nord N30 | dev | no phone with this name in the collection |
| q31 iPhone SE 2022 | dev | only the 2016 iPhone SE is in the collection |
| q11 phone rated 4.5 stars or higher under $250 | test | no pooled phone was judged relevant |
| q23 phone that gets long software update support | test | no pooled phone was judged relevant |

So dev has 23 of 27 and test 16 of 18 queries that count.

## Human labels (still to do)

Everyone labels a disjoint third of the same 90-pair sample drawn from the test pool, without
seeing the LLM grade, then `python scripts/label_sheet.py import --sheet <file>` merges the
grades into `human.txt` and `python scripts/kappa.py --merge` reports agreement and writes the
final `qrels.txt`.

## LLM call logs

The raw logs (`runs/llm_logs/*.jsonl`) contain Amazon listing text, so they are not in this
public repo; they live in the team's private data repo.
