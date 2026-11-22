# Handoff

What I've built, where it lives, and how the rest of the team can use it. The module notes below only cover work that's committed on the `rishit` branch; work in progress and teammates' status are tracked in `documentation/to-dos/rishit-todo.md`, and the reasons behind choices are in `documentation/decisions.md`.

Last updated: 8 Oct, final: graphs and the sanity check on the final runs

## Start here (for anyone, or any AI tool, picking this up)

**The project.** Dhvani is a Hindi + Hinglish news search engine for CSD358 Track 5 (multilingual and Indic-language search). Queries can be Hindi, Hinglish (Roman-script Hindi, any spelling), English or a mix, and every search runs on three indexes side by side: no stemming, stemming and auto. The full plan is `documentation/dhvani-plan.md` and the shared data formats are `documentation/formats.md`.

**Who owns what.**
| Person | Branch | Part | Code |
|---|---|---|---|
| Riya | `riya` | Crawler and corpus | `dhvani/crawl/`, `dhvani/eval/pool.py` |
| Dhrithi | `Dhrithi` | Text processing and indexes | top-level `text/`, `index/`, `scripts/`, tests in top-level `tests/` |
| Viraja | `Viraja` | Hinglish phonetic layer | `dhvani/query/` |
| Rishit | `rishit` | Ranking, cross-lingual layer, evaluation, app | `dhvani/rank/`, `dhvani/eval/` (except `pool.py`), `app/` |

Each person commits to their own branch. All four branches are merged into `main`; `rishit` is on `main` up to rank fusion.

Every command, with all the options, is in `documentation/commands.md`.

**Regenerate the graphs** (PR curves, rankers, stemming, translation, learning to rank) into `documentation/figures/`: `.venv/bin/python scripts/rishit_figures.py`.

**Run everything in my part at once** (tests, every evaluation, then a demo of the best queries; about 3 minutes):
```bash
.venv/bin/python scripts/rishit_results.py            # add --quick for just the tests and the demo
```

**Setup.**
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest partwise-tests/rishit -q
# put Riya's crawl in data/ first (it isn't in git); data/news.jsonl is the full crawl without its one repeated article
.venv/bin/python -m index.build --input data/news.jsonl
.venv/bin/streamlit run app/streamlit_app.py
```
Without the built indexes the app and CLI fall back to the 20-article sample index. Dense re-ranking is optional: `.venv/bin/pip install -r requirements-dense.txt` (sentence-transformers and torch, about 500 MB); the model, about 470 MB, downloads on first use.

**Rules for working in this repo.**
- Commits go under the person who did the work. No AI co-author lines or tool names in commit messages.
- Small commits with short, plain messages. Ask Rishit before pushing anything from this branch.
- With every commit on this branch, update this handoff, the to-do list and the decisions log.
- No em dashes in the docs; keep the writing plain.
- Article text never goes in git. Only the top-level `data/` folder is ignored, so crawled articles, indexes and downloads go there.
- Don't change a shared format in `formats.md` without telling the group.

**Where things stand right now.** The app and CLI run on the real pieces: Dhrithi's index built on Riya's full crawl (5,000 articles), Viraja's `build_query` with phonetic variants, and the cross-lingual layer, all through `dhvani/rank/real_index.py`. Hindi, Hinglish and English versions of a need find the same articles (for example भूकंप के झटके, bhukamp ke jhatke delhi and earthquake delhi). The tests stay on the 20-article sample.
- Riya's 300-article sample is `data/news_sample_300.jsonl` on her branch, with every field in `formats.md`. It's kept out of git on `main`; copy it into your local `data/` folder.
- **The frozen corpus** is `data/news.jsonl`: Riya's full crawl (`data/news_dedup.jsonl`, 5,001 lines) with its one repeated article (`jagran_40397148`) removed, 5,000 articles. Neither file is in git. The four indexes build from it in about 30 s, and Dhrithi's builder now also skips repeats itself. It wasn't cleaned further, so HTML in 249 bodies, 106 astrology pages and about 50 section pages are known limitations.

## My part
Ranking, the cross-lingual layer, evaluation, and the app. Code lives in `dhvani/rank/`, `dhvani/eval/` and `app/`, tests in `partwise-tests/rishit/`.

**My novelty**, in short: cross-lingual ranking inside the vector space model, the three stemming columns with match types, date-aware कल, authority g(d) from PageRank and first to publish, duplicate collapsing, BM25, dense re-ranking with e5, and rank fusion. Learning-to-rank comes after judging. The to-do has the full table with what's done and what's left.

## Committed so far

| Commit | What it does |
|---|---|
| `574e7c8` set up the dhvani package and a sample index to build ranking on | `dhvani` package, `.gitignore`, and `dhvani/rank/sample_index.py` |
| `9a8a1ba` lnc.ltc scoring with heap top-k | `dhvani/rank/vsm.py` and `dhvani/rank/query_stub.py` |
| `23cbb1a` net score with zone weights, proximity and recency | `dhvani/rank/scoring.py` |
| `0009026` added bm25 | `dhvani/rank/bm25.py` |
| `e1958f1` filters for source, section, state and date | `dhvani/rank/filters.py`, plus a `doc_filter` option on all three rankers |
| `728c3e4` sample index keeps article text for display | `SampleIndex.articles` holds headline and body |
| `43f3c50` first version of the streamlit app with the three stemming columns | `app/streamlit_app.py` |
| `896eab0` highlighted query words and snippets in the app | best-matching sentence for each result, matched words coloured |
| `6d88458` match chips, filters and score breakdown in the app | rest of `app/streamlit_app.py` |
| `4c95b31` redesigned the app in an apple style | new look for `app/streamlit_app.py` |
| `202fac6` search from the terminal with --explain | `app/cli.py` |
| `bdd6b29` query parser that tries the exact phrase first | `dhvani/rank/parser.py` |
| `6063301` parser stages shown in the app and the cli | `app/streamlit_app.py`, `app/cli.py` |
| `8cc6b07` cross-lingual layer for english queries | `dhvani/rank/xling.py` and the English to Hindi news dictionary |
| `69faf76` translation switch in the app and cli | `app/streamlit_app.py`, `app/cli.py` |
| `561d129` metrics for P@k, R@k, MAP, nDCG and PR curves | `dhvani/eval/metrics.py` |
| `a39e4fa` experiment runner with the stemming comparison | `dhvani/eval/experiments.py`, sample queries and judgments |
| `ae172cd` stop words, idf and zipf analysis | `dhvani/eval/corpus_stats.py`, a no-idf option in `vsm.py` |
| `fea28e2` feedback match type in the app and parser | `app/streamlit_app.py`, `dhvani/rank/parser.py` |
| `60ac8d1` index elimination: skip low-idf words and score only real contenders | `dhvani/rank/speedups.py` |
| `bedb0e8` champion lists: precompute each term's top articles and score only those | `dhvani/rank/speedups.py` |
| `1f6f519` recency tiers: fresh news first, older tiers as fallback | `dhvani/rank/speedups.py` |
| `a5b7682` added requirements.txt for the whole team | `requirements.txt` |
| `4354852` date-aware kal: tell yesterday from tomorrow and boost the right day | `dhvani/rank/kal.py` |
| `e48f17f` date-aware kal in the app and cli, with the boost shown in explain | `app/streamlit_app.py`, `app/cli.py`, `dhvani/rank/kal.py` |
| `62400c3` authority ranking: pagerank and first-to-publish credit in g(d) | `dhvani/rank/authority.py`, a `static=` option in `scoring.py` |
| `fb53231` authority g(d) in the app and cli, with its parts shown in score details | `app/streamlit_app.py`, `app/cli.py`, a `static=` option in `parser.py` |
| `26f980d` duplicate collapsing: one result per wire story, with the other papers listed | `dhvani/rank/collapse.py`, `app/streamlit_app.py`, `app/cli.py`, a fix in `kal.py` |
| `e545370` renamed the to-do to rishit-todo and crossed out what's done | `documentation/to-dos/rishit-todo.md` |
| `24b3804` speed-ups in the app and cli, with how many articles were scored | `app/streamlit_app.py`, `app/cli.py` |
| `5feba40` speed-ups results table: how much of the top k each speed-up keeps | `dhvani/eval/experiments.py` |
| `612c363` my 8 information needs from stories in riya's crawl | `documentation/needs/rishit-needs.md` |
| `3714640` real index and viraja's query layer plugged into the app and cli | `dhvani/rank/real_index.py`, `app/streamlit_app.py`, `app/cli.py`, more words in the news dictionary |
| `8e75d5a` cluster pruning: leaders and followers, compared with the other speed-ups | `dhvani/rank/speedups.py`, `app/streamlit_app.py`, `app/cli.py`, `dhvani/eval/experiments.py` |
| `21dd828` results file with early speed-ups numbers on riya's 300 articles | `documentation/results/rishit-results.md` |
| `c302928` impact-ordered postings: read each word's best articles first and stop early | `dhvani/rank/speedups.py`, `app/streamlit_app.py`, `app/cli.py`, `dhvani/eval/experiments.py` |
| `1fd899d` dense re-ranking with multilingual e5, optional | `dhvani/rank/dense.py`, `app/streamlit_app.py`, `app/cli.py`, `requirements-dense.txt` |
| `2c5f883` speed-ups, stop words and zipf on riya's full crawl | `documentation/results/rishit-results.md`, two plots in `documentation/figures/` |
| `a18d81e` rank fusion (rrf), and the to-do and handoff brought up to date | `dhvani/rank/fusion.py`, `dhvani/rank/parser.py`, `app/streamlit_app.py`, `app/cli.py` |
| `1d0a6d3` experiments and corpus stats run on the real index, run files for pooling | `dhvani/eval/experiments.py`, `dhvani/eval/corpus_stats.py` |
| `ca8c138` mmr diversification so the top results cover more stories | `dhvani/rank/diversify.py`, `app/streamlit_app.py`, `app/cli.py` |
| `4c254e4` query difficulty hint: low confidence when every word is common or nothing has all the words | `dhvani/rank/difficulty.py`, `app/streamlit_app.py`, `app/cli.py` |
| `b241c0d` k-gram index built with document frequencies (from_index) | `dhvani/rank/real_index.py` |
| `349e81e` frozen corpus noted in the to-do, handoff and results | `documentation/` |
| `0e4192e` sanity check without judgments: agreement between query forms and between systems | `dhvani/eval/sanity.py` |
| `ca32b9d` judging page and per-need pool, judgments kept in the repo | `dhvani/eval/judge.py`, `app/pages/judge.py`, `judgments/` |
| `ec45400` judging page highlights whole words only | `app/pages/judge.py` |
| `537cb56` judging link in the search page's top bar, and a way back | `app/streamlit_app.py`, `app/pages/judge.py` |
| `5117981`, `f52476a` my relevance judgments so far (23 of 345) | `judgments/qrels_rishit.txt` |
| `47a924f` judgment-free results rerun after viraja's spelling fix | `documentation/results/rishit-results.md` |
| `b63bacc` docs brought up to date | `documentation/` |
| `e3f1762` translations learned from jagran's bilingual headlines | `dhvani/rank/learn_dict.py`, `dhvani/rank/data/en_hi_learned.tsv`, `dhvani/rank/xling.py` |
| `8ce491a` listing pages and horoscopes pushed below real articles | `dhvani/rank/quality.py`, `dhvani/rank/data/page_types.tsv`, `app/streamlit_app.py`, `app/cli.py` |
| `81f5271` judging pool rebuilt from the current runs (641 articles) | `judgments/pool.tsv`, `documentation/` |
| `726bce6` decode html entities in headlines before showing them | `app/streamlit_app.py` |
| `06800e3` evaluation runs push listing pages down, pool extended (802 articles) | `dhvani/eval/experiments.py`, `dhvani/eval/judge.py`, `judgments/pool.tsv` |
| `b798dc3` significance tests and learning-to-rank | `dhvani/eval/significance.py`, `dhvani/eval/ltr.py`, `dhvani/eval/experiments.py` |
| `0f9211c` one script for all my tests, evaluations and a demo | `scripts/rishit_results.py`, `app/cli.py` |
| `380926e` rocchio pseudo-relevance feedback switch | `dhvani/rank/feedback.py`, `app/streamlit_app.py`, `app/cli.py` |
| `9e54c6a` all my relevance judgments (297) | `judgments/qrels_rishit.txt` |
| `247c889` final evaluation results | `documentation/results/rishit-results.md` |
| `15f2652` judging pool extended with riya's needs and the new auto-stemming runs | `judgments/pool.tsv` |
| `c5bd88d` evaluation on all 32 needs (provisional), dhrithi's needs and judgments added | `documentation/needs/dhriti-needs.md`, `judgments/`, `documentation/` |
| `2091dee` final results on all 32 needs, stop word experiment and difficulty check | `documentation/` |
| graphs for the results, sanity check on the final runs | `scripts/rishit_figures.py`, `documentation/figures/rishit-*.png`, `documentation/results/rishit-results.md` |

## How to use it

**Sample index** (`dhvani/rank/sample_index.py`)
20 hand-written Hindi articles across weather, sports, politics, business, crime, education, entertainment and national news. It has the same methods as the real index in `formats.md`: `postings(term, zone)`, `df(term)`, `N`, `vocab`, `doc_norm`, `meta`. Two articles are the same wire story and a few link to each other, for duplicate and PageRank testing.
```python
from dhvani.rank.sample_index import SampleIndex
idx = SampleIndex.load("none")
```

**Query stub** (`dhvani/rank/query_stub.py`)
Builds the query object from `formats.md` with exact matches only. The real pipeline now uses Viraja's `build_query` (see "Real index and query layer" below); the stub is only used with the sample index and in the tests.
```python
from dhvani.rank.query_stub import exact_query
q = exact_query("दिल्ली बारिश")
```

**lnc.ltc** (`dhvani/rank/vsm.py`)
```python
from dhvani.rank.vsm import search
search(q, idx, k=10)   # [(doc_id, score, {term: contribution})]
```

**Net score** (`dhvani/rank/scoring.py`)
cosine + 0.2 x zone + 0.1 x proximity + 0.1 x recency. The explain dict has every part.
```python
from dhvani.rank.scoring import rank
rank(q, idx, k=10)     # [(doc_id, net, {"cosine", "terms", "zone", "proximity", "recency", "net"})]
```

**BM25** (`dhvani/rank/bm25.py`)
k1 = 1.2, b = 0.75, idf that stays positive, optional headline boost.
```python
from dhvani.rank.bm25 import search_bm25
search_bm25(q, idx, k=10)
```

**Filters** (`dhvani/rank/filters.py`)
Source, section, state and date range. Every filter that's set has to match, and they're applied before the top K is picked, so a filtered search still gets its best results.
```python
from dhvani.rank.filters import make_filter
f = make_filter(idx, sources=["jagran"], states=["delhi"], date_from="2026-10-05")
rank(q, idx, k=10, doc_filter=f)     # same for search() and search_bm25()
```

**Article text** (`SampleIndex.articles`)
`idx.articles[doc_id]` gives `{"headline", "body"}` for showing results. With the real index, `load_index()` points `articles` at Dhrithi's stored text, so the same code works for both.

**The app** (`app/streamlit_app.py`)
Styled after Apple's design guidelines. A translucent bar sits at the top, and a big centred search field has suggestion pills under it. A segmented control picks the ranking model (net score, lnc.ltc, BM25 or Fusion), and a Filters popover next to it holds newspaper, section, state, results per column and date range. Results show in three grouped lists side by side: no stemming, stemming and auto. Each row shows the paper, section, place, date and score, the headline and best-matching sentence with matched words tinted by match type, chips for how each word matched (exact, phonetic, translated, or feedback for Rocchio terms), an "Only here" tag if the result isn't in the other columns, a small label for the parser stage that matched it, and a "Score details" disclosure with the full breakdown. The Filters popover has switches for "Smart query parsing", "Authority (PageRank + first to publish)", "Collapse duplicate stories", "Date-aware kal" and "Translate English words" (all on by default) and "Dense re-ranking (e5)" (off by default, only shown when it's installed), plus a "Speed-up" menu. The suggestion pills are real stories from the crawl. Works in light and dark mode, and respects reduced motion, transparency and contrast settings. Each column uses its own index from Dhrithi's builder, or the sample index if none is built.
```bash
.venv/bin/streamlit run app/streamlit_app.py
```

**Query parser** (`dhvani/rank/parser.py`)
Turns one query into stricter-to-looser searches: exact phrase, part of the phrase (two neighbouring words), all words, all words with variants, then any word. It stops once it has k results. Articles found at a stricter stage rank above looser ones, and within a stage the chosen ranker decides. Each result's explain dict gets `stage` and `stages_run`, and its word scores are always under `terms`. Feedback tokens from Rocchio (all expansions tagged `prf`) don't take part in the phrase and AND stages; they only count in "any word" and in the score.
```python
from dhvani.rank.parser import parse_and_rank
parse_and_rank(q, idx, k=10, ranker="net", doc_filter=None)   # ranker: net, lnc, bm25 or rrf
```

**Terminal search and --explain** (`app/cli.py`)
Searches from the terminal. With `--explain` it prints every step: the query object, the query vector (tf, df, idf and weights, or the BM25 idf), the postings each term touched with positions, how many articles became candidates and how the heap picks the top K, and a full score breakdown for each result.
```bash
.venv/bin/python app/cli.py "दिल्ली बारिश" --explain
.venv/bin/python app/cli.py "कोहली शतक" --ranker bm25 --k 3 --explain
```
With `--explain` there's also a parser step showing how many articles each stage found and where it stopped. Other flags: `--ranker net|lnc|bm25|rrf`, `--stem none|light|aggr|auto`, `--speedup elim|champions|tiers|clusters|impact`, `--dense`, `--diversify`, `--prf`, `--keep-listings`, `--no-parser`, `--no-xling`, `--no-kal`, `--no-authority` and `--no-collapse`.

**Cross-lingual layer** (`dhvani/rank/xling.py`)
English query words become weighted Hindi terms inside the query vector, so "weather tomorrow" is scored against the same Hindi terms as "कल का मौसम". A word's weight is split across its translations, multi-word entries like "prime minister" are matched as phrases, and English stop words are dropped. The dictionary is `dhvani/rank/data/en_hi_news.tsv`; if the MUSE English-Hindi dictionary is saved at `data/muse/en-hi.txt` it's merged in too.
```python
from dhvani.rank.xling import translate
q = translate(exact_query("delhi rain"))   # adds ("दिल्ली", w, "xling"), ("बारिश", w, "xling") ...
```

**Metrics** (`dhvani/eval/metrics.py`)
P@k, R@k, F1, AP, MAP, DCG, nDCG and the 11-point interpolated PR curve, checked against the lecture's own examples (MAP comes out as 0.53). Also reads and writes TREC run files and judgment files.
```python
from dhvani.eval.metrics import evaluate, read_qrels, read_run
per_query, means = evaluate(rankings, qrels_by_query, k=10)   # means: P@10, R@10, MAP, nDCG@10
```

**Experiment runner** (`dhvani/eval/experiments.py`)
Runs every stemming mode x every ranker (net, lnc.ltc, BM25, fusion) x every query, pushing listing pages and horoscopes below real articles as the app does (top 30 ranked, demoted, cut to 10), writes TREC run files to `<out>/runs/`, and prints the full results table (overall and per query form), the stemming comparison, per-query wins and losses against no stemming, and English queries with translation off vs on. With the real index built, it uses Dhrithi's index for each mode and the real query pipeline (`real_index.make_query`), reads the queries from the tsv blocks in `documentation/needs/*.md` (`read_needs()`), and skips modes that aren't built (yass). Without judgments it writes the run files for pooling and the speed-ups table and stops there. Without the real index it falls back to the sample index and the made-up queries and judgments in `dhvani/eval/sample/`. On the full crawl, 64 queries x 4 modes x 4 rankers take about 6 minutes.
```bash
.venv/bin/python -m dhvani.eval.experiments --out data/eval/full                        # run files for pooling
.venv/bin/python -m dhvani.eval.experiments --out data/eval/full                        # with judgments/ filled in: every metric
.venv/bin/python -m dhvani.eval.pool --runs data/eval/full/runs/*.txt --top-k 10         # Riya's pooling script
```

**Stop words, idf and Zipf** (`dhvani/eval/corpus_stats.py`)
df, collection frequency and idf for every term, the most frequent terms (Hindi function words like में, का, की come out at the top with idf near 0), a stop word list taken from the data, Zipf's law with a fitted slope, and a stop word experiment: no idf (lnc.lnc) vs idf (lnc.ltc) vs stop words removed. It uses Dhrithi's no-stemming index when it's built, and the experiment runs only when judgments are passed with `--queries` and `--qrels` (on the sample index it uses the made-up ones). Plots go to `data/eval/` when matplotlib is installed; the ones for the report are copied to `documentation/figures/`, and the numbers on the full crawl are in `documentation/results/rishit-results.md`.
```bash
.venv/bin/python -m dhvani.eval.corpus_stats
```

**Index elimination** (`dhvani/rank/speedups.py`)
Scores fewer articles. Query words with low idf (below 0.3 by default) are skipped, always keeping at least the rarest word. With 3 or more query words, an article has to contain at least 75% of them (3 of 4, as in Lecture 7). If that leaves fewer than k articles it relaxes one word at a time. Returns `(results, stats)`, where stats says how many articles were scored compared with full search. `overlap_at_k()` measures how much of the exact top k was kept.
```python
from dhvani.rank.speedups import search_index_elimination, overlap_at_k
results, stats = search_index_elimination(q, idx, k=10)
```

**Champion lists** (`dhvani/rank/speedups.py`)
`ChampionLists(index, r=50, static_scores=None)` keeps each term's r highest-weight articles; with `static_scores` (g(d)) they're ordered by weight + g(d), as in Lecture 7. `search_champions(q, idx, champions, k)` scores only those articles and falls back to the full postings if there are fewer than k. Returns `(results, stats)` like index elimination.

**Recent-news tiers** (`dhvani/rank/speedups.py`)
`RecencyTiers(index, tier_days=(2, 7, None))` puts each article in a tier by age (last 2 days, last week, older), measured from the newest article. `search_tiered(q, idx, tiers, k)` searches tier 0 first and only adds older tiers if there are fewer than k results. Returns `(results, stats)` with the tiers used.

**Cluster pruning** (`dhvani/rank/speedups.py`)
`ClusterPruning(index, n_leaders=None, seed=0)` picks sqrt(N) random leaders (71 for 5,000 articles) and puts every other article in the cluster of its most similar leader, by cosine of their lnc vectors. `search_clusters(q, idx, clusters, k, b=1)` compares the query with the leaders only and scores the clusters of the b closest; if that gives fewer than k results it adds the next-closest leader. Returns `(results, stats)` with the leaders used. A fixed seed keeps the clusters the same between runs.

Numbers comparing all the speed-ups on the frozen corpus (5,000 articles) are in `documentation/results/rishit-results.md`.

**Impact-ordered postings** (`dhvani/rank/speedups.py`)
`ImpactOrdered(index)` sorts each word's postings by the word's lnc weight in the article, highest first. `search_impact(q, idx, impact, k, max_docs=20, min_share=0.0)` goes through the query words in decreasing idf and reads each list from the top, stopping after `max_docs` articles or once the weight drops below `min_share` x the list's best weight. Articles past the cut-off just miss that word's small contribution. Returns `(results, stats)` with postings read against the total. The app and CLI use `max_docs` = N / 15 (at least 20).

All five speed-ups are in the app and the CLI. In the app, the "Speed-up" menu in Filters picks one (off by default) and each column shows "scored X of Y". In the CLI it's `--speedup elim`, `--speedup champions`, `--speedup tiers`, `--speedup clusters` or `--speedup impact`, and the output says how many of the candidate articles were scored. `speedup_table()` in `dhvani/eval/experiments.py` compares each one with full lnc.ltc over all queries: average overlap of the top k and average share of articles scored, with champion lists at r = 2, 5, 10 and 50 cluster pruning at b = 1 and 3, and impact-ordered postings stopping after 20 or 50 articles or below half the best weight. The experiment runner prints it and writes `data/eval/speedups.csv`. On the 20-article sample every row is 1.0 because there's too little to skip; numbers on Riya's full crawl are in `documentation/results/rishit-results.md`. They all use lnc.ltc, since that's what the speed-ups approximate, and champion lists use r = N / 20 (at least 5) ordered by weight + g(d).

**Date-aware kal** (`dhvani/rank/kal.py`)
कल means both yesterday and tomorrow. `kal_intent(q)` works it out from the query: English "tomorrow"/"yesterday" decide directly; otherwise future cues (होगा, रहेगा, alert, forecast, weather words) mean tomorrow and past cues (हुआ, था, result) mean yesterday, defaulting to yesterday. `apply_kal(results, q, idx)` re-ranks any ranker's results: yesterday favours articles from the day before, tomorrow favours the newest articles written in the future tense. The boost multiplies the score, score x (1 + 0.5 x boost), and is recorded in `explain["kal"]`. It only reorders within a query-parser stage, so a stricter match stays above a looser one. In the app it's the "Date-aware kal" switch (on by default), boosted results get a "कल · tomorrow" or "कल · yesterday" tag, and Score details shows the boost. In the CLI, results show `[kal: tomorrow]`, `--explain` adds a step 4c, and `--no-kal` turns it off.
```python
from dhvani.rank.kal import apply_kal
results = apply_kal(search(q, idx, k=10), q, idx)
```

**Authority: PageRank and first to publish** (`dhvani/rank/authority.py`)
`pagerank(link_graph(idx))` runs power iteration (damping 0.85) over the links between crawled articles from `meta["links"]`; links to articles that weren't crawled and self-links are ignored, and articles with no outgoing links spread their vote evenly, so the scores sum to 1. `originals(idx)` finds the first-to-publish article of each duplicate cluster (the one later copies point to with `dup_of`). `static_scores(idx)` combines them into g(d) = 0.5 x recency + 0.3 x PageRank (scaled to [0, 1]) + 0.2 x original flag, and returns the parts too. Pass it to the net score with `rank(q, idx, static=g)`; without `static` the net score uses plain recency as before. Champion lists take it as `static_scores=g`, and the query parser as `parse_and_rank(q, idx, static=g)`. In the app it's the "Authority (PageRank + first to publish)" switch (on by default), and Score details shows "Authority g(d)" with its three parts. In the CLI it's on by default, `--explain` breaks g(d) into recency, PageRank and first to publish, and `--no-authority` goes back to plain recency.
```python
from dhvani.rank.authority import static_scores
g, parts = static_scores(idx)
rank(q, idx, k=10, static=g)
```

**Duplicate collapsing** (`dhvani/rank/collapse.py`)
When the same wire story runs in several papers, Riya's dedup gives every later copy a `dup_of` pointing at the earliest one. `collapse_duplicates(results, idx, k)` keeps only the best-ranked article of each story and records the other copies in `explain["duplicates"]` and their papers in `explain["also_in"]`, so the top k shows k different stories. Ask the ranker for `collapse_pool(k)` results (3k) first, then collapse to k. It works on the output of any ranker. In the app it's the "Collapse duplicate stories" switch (on by default) and results show "Also in amarujala". In the CLI results show `[also in: ...]`, and `--no-collapse` turns it off.
```python
from dhvani.rank.collapse import collapse_duplicates, collapse_pool
results = collapse_duplicates(rank(q, idx, k=collapse_pool(10)), idx, k=10)
```

**Real index and query layer** (`dhvani/rank/real_index.py`)
`load_index(mode)` loads Dhrithi's index from `indexes/<mode>.pkl` and gives it an `articles` view of its stored text, so everything that used the sample index works unchanged. If the index isn't built it falls back to the sample index, and `DHVANI_INDEX=sample` forces the sample (the tests set this). `make_query(raw, mode)` builds the query the way the articles were indexed: Viraja's `build_query` with phonetic variants from a k-gram index over the unstemmed vocabulary (built with `KGramIndex.from_index`, so it knows each word's document frequency and prefers common words among sound-alikes, e.g. मोदी over मोड़), then the cross-lingual layer, then every term through Dhrithi's analyzer for that mode. Phonetic variants with weight under 0.05 are dropped, and Roman words match every letter case in the index ("iyer" finds "Iyer"). The app builds one query per column, and its suggestions are real stories from the crawl.
```python
from dhvani.rank.real_index import load_index, make_query
idx = load_index("light")
rank(make_query("bhukamp ke jhatke delhi", "light"), idx, k=10)
```

**Dense re-ranking** (`dhvani/rank/dense.py`, optional)
Re-scores the top 50 of any ranker with `intfloat/multilingual-e5-small`: score = 0.5 x first-stage score + 0.5 x e5 cosine, both min-max scaled over the 50 first (e5's cosines sit in a narrow 0.75 to 0.85 band, so unscaled they'd never change anything). The sparse ranker still picks the candidates. e5 handles Hindi and English but not Roman Hindi, so `dense_query_text()` sends the raw query plus the best Devanagari spelling of each Roman word: translations always, phonetic spellings only with weight 0.4 or more. "kal ka mausam" becomes "kal ka mausam कल का मौसम". Article vectors are computed once and cached in `data/dense/` (12.8 s for 300 articles). Like kal, it only reorders within a parser stage. In the app it's the "Dense re-ranking (e5)" switch, off by default and only shown if sentence-transformers is installed; Score details shows the e5 cosine. In the CLI it's `--dense`, and `--explain` adds a step 4b. Tests use a fake encoder, so they don't need the model.
```python
from dhvani.rank.dense import DenseIndex, SentenceEncoder, dense_rerank
dense = DenseIndex(idx, SentenceEncoder())
results = dense_rerank(rank(q, idx, k=50), q, dense)
```

**Rank fusion** (`dhvani/rank/fusion.py`)
`search_rrf(q, idx, k, depth=50, static=None, dense=None)` runs lnc.ltc, BM25 and the net score (top 50 each) and fuses them with reciprocal rank fusion: each article scores the sum of 1 / (60 + its rank) over the lists. With a `DenseIndex` it also adds a dense list: the articles the sparse lists found, ordered by e5 cosine. `rrf(lists, k)` does the fusing for any {name: results}. Only ranks are used, so cosine, BM25 and e5 scores never need to be on the same scale. explain gets `"rrf"` (the article's rank in each list) and `"rrf_score"`. It works with the query parser (`parse_and_rank(..., ranker="rrf", dense=...)`). In the app it's the "Fusion" ranking model, and Score details shows the rank in each list; with Dense on, dense becomes one of the fused lists instead of a re-ranker. In the CLI it's `--ranker rrf`, and `--explain` prints the ranks.
```python
from dhvani.rank.fusion import search_rrf
results = search_rrf(q, idx, k=10)
```

**Diversification (MMR)** (`dhvani/rank/diversify.py`)
`diversify(results, idx, k=None, lam=0.7)` re-orders any ranker's results with maximal marginal relevance: each next pick maximises 0.7 x relevance (the ranker's score scaled to [0, 1]) - 0.3 x its highest cosine with the articles already picked. Similarity uses log-tf, unit-length word vectors from each article's headline and body. It catches the same story told by several papers in different words, which duplicate collapsing (based on `dup_of`) misses. Like kal and dense it works within a parser stage. 0.05 s for 30 results on the full crawl. In the app it's the "Diversify results (MMR)" switch (off by default) and Score details shows the similarity to the closest earlier pick; in the CLI it's `--diversify`, with an explain step 4d.
```python
from dhvani.rank.diversify import diversify
results = diversify(parse_and_rank(q, idx, k=30), idx, k=10)
```

**Query difficulty** (`dhvani/rank/difficulty.py`)
`predict(q, results, idx)` estimates whether the results are probably poor, without judgments (query performance prediction). Signals: specificity, the highest idf among the query's words, where each word's idf comes from its most common strong spelling (weight 0.3 or more, so Hinglish "kya" counts as common because क्या is); scope, the share of articles containing any query word; clarity (Cronen-Townsend et al. 2002), the KL divergence in bits between the top 10's language model and the collection's; and the parser stage. It flags "low confidence" when specificity is below 1.0 (the rarest word is in more than 10% of articles) or only the "any word" stage matched. Clarity is reported but not used for the flag, because on the full crawl vague queries often hit near-identical listing pages, which look focused. In the app a "Low confidence: ..." line appears above the results; the CLI prints the same line, and `--explain` shows all the signals. About 0.18 s per query on the full crawl, mostly clarity.
```python
from dhvani.rank.difficulty import predict
hint = predict(q, parse_and_rank(q, idx, k=10), idx)   # hint["low_confidence"], hint["reasons"]
```

**Learned translations** (`dhvani/rank/learn_dict.py`)
Jagran's headlines end with an English version ("भूकंप के झटकों से कांपा ... - earthquake tremors felt ..."), so the frozen corpus has 973 Hindi/English headline pairs. `learn()` aligns words across them with the Dice coefficient (2 x pairs with both / (pairs with the English word + pairs with the Hindi word)), after dropping words in more than 5% of pairs, and keeps each English word's best Hindi word when it's seen in at least 3 pairs with Dice 0.5 or more. That gives 272 pairs (airport → एयरपोर्ट, arrested → गिरफ्तार, compensation → मुआवजा, border → सीमा, and many names and places), written to `dhvani/rank/data/en_hi_learned.tsv`. The translator loads them under the hand-made dictionary, which wins wherever both have a word. About 85% are right on a spot check; the wrong ones come from a single recurring story (students → वृंदावन). Numbers are in the results file.
```bash
.venv/bin/python -m dhvani.rank.learn_dict      # relearn from data/news.jsonl
```

**Page-type quality** (`dhvani/rank/quality.py`)
18% of the frozen corpus (900 pages) are section and city listing pages the crawler saved as articles ("देवरिया की सबसे ताज़ा खबर", "Guna News, Guna Samachar", amarujala.com/technology), and 2% (104) are daily horoscopes. A listing page stitches dozens of headlines together, so it contains almost any combination of words and was 21 to 33% of the top 10 for our needs queries. `classify(headline, url)` labels each page article, listing or horoscope, from its URL (site root, /live/, or a short section-style path with no article slug) and its headline (patterns like "सबसे ताज़ा खबर", "Samachar", "News in Hindi", with ताज़ा and ताजा treated the same; Jagran's bilingual article headlines are never listings). The index has no URLs, so `python -m dhvani.rank.quality` classifies the corpus file once and saves the non-article ids to `dhvani/rank/data/page_types.tsv` (ids only). `demote(results, idx)` multiplies non-article scores by 0.3 and puts them after every real article: they often reach a stricter parser stage than real articles, so demoting them within their stage wasn't enough. In the app it's the "Push listing pages down" switch (on by default) with a "Listing page" or "Horoscope" tag; in the CLI it's on by default, results show `[listing]`, and `--keep-listings` turns it off.
```bash
.venv/bin/python -m dhvani.rank.quality      # reclassify data/news.jsonl
```

**Significance tests** (`dhvani/eval/significance.py`)
`compare(a, b)` takes two {qid: score} dicts (per-query AP) and returns both means, the difference, a paired randomization test p-value (10,000 random swaps, Smucker et al. 2007) and a paired t-test p-value (t distribution computed in the module, no SciPy). The experiment runner prints them for every ranker against lnc.ltc and every stemming mode against no stemming, on judged queries only.

**Learning to rank** (`dhvani/eval/ltr.py`)
Every judged query-article pair becomes a row of 10 features (cosine, BM25, zone, proximity, recency, PageRank, first to publish, real article or not, parser stage, dense cosine), and a logistic regression learns how much each says about relevance (grade 1 or 2 = relevant). It's tested leave-one-need-out: for each need, a model trained on the other needs re-ranks that need's net-score top 30, and MAP, P@10 and a significance test compare it with the net score. It prints the average learned weights too.
```bash
.venv/bin/python -m dhvani.eval.ltr            # add --dense to include the e5 cosine
```

**Pseudo-relevance feedback** (`dhvani/rank/feedback.py`)
`feedback_query(q, idx, mode)` searches once, takes the top 5 real articles (after listing pages are pushed down), and folds their strongest words back into the query with Viraja's Rocchio update (alpha 1.0, beta 0.75), adding the top 5 new terms with source "prf"; the search then runs again. Article vectors are tf x idf from each article's own text: with plain tf the best new terms were के, में, की, and building from the text avoids scanning the whole vocabulary (0.03 to 0.05 s per query). In the app it's the "Pseudo-relevance feedback (Rocchio)" switch (off by default) and added words show as "Feedback" chips; in the CLI it's `--prf`, which prints the added terms. It helps when the first results are right ("iyer century" adds श्रेयस) and drifts when they're mixed, so it's off by default; numbers are in the results file.

**Sanity check** (`dhvani/eval/sanity.py`)
Evaluation that needs no judgments, from the run files. Cross-form agreement: for each need, how much of the Hindi form's top 10 its Hinglish, messy and English forms also return, plus the English form with translation off. System agreement: top-10 overlap and Kendall's tau between every pair of rankers and of stemming modes. Numbers are in the results file.
```bash
.venv/bin/python -m dhvani.eval.sanity --runs data/eval/full/runs
```

**Judging** (`dhvani/eval/judge.py`, `app/pages/judge.py`, `judgments/`)
`python -m dhvani.eval.judge` pools the top 10 of every run for every form of a need into one list per need (`judgments/pool.tsv`, 802 articles for the 16 needs, about 50 each). New runs only ever add to it, so nothing already judged is dropped; `--fresh` starts over. Riya's `dhvani/eval/pool.py` pools per need the same way, so each article is judged once per need, as `formats.md` says. The "judge" page in the app's sidebar shows the need, then each pooled article (paper, date, headline and the start of the body, with the Hindi query words highlighted), with buttons for 0 not relevant, 1 partly, 2 fully, and Skip. It's reached from the "Judging" link in the search page's top bar, and has a link back. It picks your needs by name (R for Rishit, V for Viraja), shows progress, and saves after every click to `judgments/qrels_<person>.txt` in the `formats.md` judgment format. One file per person, all in git (ids and grades only, no article text), so four people can judge at once without conflicts. `all_judgments()` merges the files, keeping the higher grade if two people judged the same pair, and the experiment runner uses them by default.
```bash
.venv/bin/python -m dhvani.eval.judge --runs data/eval/full/runs   # rebuild the pool
.venv/bin/streamlit run app/streamlit_app.py                         # then open "judge" in the sidebar
.venv/bin/python -m dhvani.eval.experiments --out data/eval/full     # metrics once there are judgments
```

## Tests
219 tests in `partwise-tests/rishit/`, all passing.
```bash
.venv/bin/python -m pytest partwise-tests/rishit -q
```

## What I need from others
Nothing for my code. The team stopped judging at 57% of the pool, and the results are reported with that limitation.

## Next
My code and results are done. Left: my report section, my video segment (`scripts/rishit_results.py --quick` runs the demo queries), and putting the final report and video together.
