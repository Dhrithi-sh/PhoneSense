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
python scripts/build_data.py                   # = make data (TODO, T02)
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
scripts/                 entry-point scripts (build_data.py, run_eval.py, ...)
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
