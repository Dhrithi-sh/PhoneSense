# Commands

Everything needed to set up, run and evaluate Dhvani. Run all commands from the repo root.

## Setup (once)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r requirements-dense.txt      # optional: dense re-ranking (about 1 GB)
```

Put the frozen corpus at `data/news.jsonl` (5,000 articles; not in git), then build the four indexes (about 30 s):

```bash
.venv/bin/python -m index.build --input data/news.jsonl
```

Without the indexes, the app and CLI fall back to a 20-article sample index.

## The app (Streamlit)

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

Opens at http://localhost:8501. Type in Hindi, Hinglish or English; results show in three columns (no stemming, light stemming, auto). The ranking model (Net score, lnc.ltc, BM25, Fusion) is under the search box; everything else is in **Filters**: newspaper, section, state, dates, results per column, and switches for smart query parsing, authority, duplicate collapsing, pushing listing pages down, Rocchio feedback, MMR, date-aware kal, translation, dense re-ranking, plus a speed-up menu.

The **Judging** link in the top bar opens the judging page (or go to http://localhost:8501/judge).

## Search from the terminal

```bash
.venv/bin/python app/cli.py "भूकंप के झटके"
```

| Option | What it does |
|---|---|
| `--k 5` | how many results (default 5) |
| `--ranker net\|lnc\|bm25\|rrf` | ranking model (default net; rrf is fusion of all) |
| `--stem none\|light\|aggr\|auto` | which index (default none) |
| `--explain` | print every step: query, vector, postings, parser stages, score breakdown |
| `--speedup elim\|champions\|tiers\|clusters\|impact` | score fewer articles with a Lecture 7 speed-up |
| `--dense` | re-rank with the e5 model (needs requirements-dense.txt) |
| `--prf` | Rocchio pseudo-relevance feedback |
| `--diversify` | MMR, so the top results cover more stories |
| `--no-parser` | skip the query parser |
| `--no-xling` | no English to Hindi translation |
| `--no-kal` | no date-aware kal |
| `--no-authority` | plain recency instead of recency + PageRank + first to publish |
| `--no-collapse` | don't merge copies of the same story |
| `--keep-listings` | don't push listing pages and horoscopes down |

Examples:

```bash
.venv/bin/python app/cli.py "bhukamp ke jhatke" --stem light
.venv/bin/python app/cli.py "earthquake tremors delhi" --ranker bm25 --k 10
.venv/bin/python app/cli.py "kal mausam kaisa rahega" --explain
.venv/bin/python app/cli.py "chardham yatra record" --ranker rrf --dense
.venv/bin/python app/cli.py "shreyas iyer shatak" --speedup champions
```

## Everything at once

```bash
.venv/bin/python scripts/rishit_results.py            # tests, every evaluation, demo queries (about 3 min)
.venv/bin/python scripts/rishit_results.py --quick    # tests and demo queries only
.venv/bin/python scripts/rishit_results.py --dense    # include the e5 feature in learning to rank
```

## Evaluation, one step at a time

```bash
.venv/bin/python -m dhvani.eval.experiments --out data/eval/full   # runs, P/R/MAP/nDCG, significance, speed-ups
.venv/bin/python -m dhvani.eval.sanity                             # agreement between query forms and systems
.venv/bin/python -m dhvani.eval.ltr --dense                        # learning to rank (leave-one-need-out)
.venv/bin/python -m dhvani.eval.corpus_stats --out data/eval/full  # stop words, idf, Zipf
.venv/bin/python scripts/rishit_figures.py                         # graphs into documentation/figures/
```

## Judging

```bash
.venv/bin/python -m dhvani.eval.judge     # add newly ranked articles to judgments/pool.tsv
```

Then judge on the app's Judging page; marks are saved to `judgments/qrels_<name>.txt`.

## Rebuilding generated files

```bash
.venv/bin/python -m dhvani.rank.learn_dict   # learned translations from Jagran's bilingual headlines
.venv/bin/python -m dhvani.rank.quality      # page types (listing pages, horoscopes)
```

## Tests

```bash
.venv/bin/python -m pytest -q partwise-tests
.venv/bin/python -m pytest -q tests          # Dhrithi's tests; run separately (two file names clash)
```
