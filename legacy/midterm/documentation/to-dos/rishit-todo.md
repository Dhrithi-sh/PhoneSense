# Rishit's to-do

The code side of my part only: ranking, the cross-lingual layer, evaluation code and the app. Done items are crossed out. How each piece works is in `documentation/handoffs/rishit.md`, why it was built that way is in `documentation/decisions.md`, and the numbers are in `documentation/results/rishit-results.md`.

Last updated: 8 Oct, around 01:20 (code and results complete)

## My novelty

What my part adds beyond the lecture basics, and where each one stands.

| Idea | What it does | Status |
|---|---|---|
| Cross-lingual ranking | English query words become weighted Hindi terms inside the lnc.ltc query vector, so "weather tomorrow" and "कल का मौसम" are scored against the same words | ~~Done~~ |
| Stemming columns with match types | Every search runs on no stemming, light stemming and auto side by side, and each result shows how each word matched (exact, phonetic, translated, feedback) | ~~Done~~ |
| Date-aware कल | Works out from the query whether कल means yesterday or tomorrow and boosts articles about that day | ~~Done~~ |
| Authority g(d): PageRank and first to publish | g(d) = recency + PageRank over the links between articles + credit for the paper that ran a wire story first | ~~Done~~ |
| Duplicate collapsing | A wire story carried by several papers shows once, with "also in" the others (22 story clusters in the full crawl) | ~~Done~~ |
| BM25 | Second ranking model next to lnc.ltc and the net score | ~~Done~~ |
| Dense re-ranking | Multilingual e5 re-scores the top 50, with Viraja's Devanagari spellings added to Hinglish queries so e5 understands them | ~~Done~~ |
| Rank fusion (RRF) | Fuses lnc.ltc, BM25, the net score and dense by rank | ~~Done~~ |
| Speed-ups study | All five Lecture 7 speed-ups compared on the same Hindi news queries (articles scored vs top 10 kept) | ~~Done on the frozen corpus~~ |
| MMR diversification | Re-orders results so the top 10 covers more different stories | ~~Done~~ |
| Query difficulty hint | Flags "low confidence" queries from idf, scope, clarity and the parser stage, without judgments | ~~Done~~ |
| Sanity check without judgments | Agreement between the four forms of a need and between systems, from the run files | ~~Done~~ |
| Judging page | Per-need pooling and a page in the app to mark each article 0, 1 or 2, saved to git per person | ~~Done~~; judging in progress |
| Learned translations | English to Hindi pairs learned from Jagran's bilingual headlines with Dice alignment, no outside data | ~~Done~~ |
| Page-type quality | Listing pages and horoscopes (20% of the corpus) recognised from URL and headline and pushed below real articles | ~~Done~~ |
| Rocchio feedback switch | Viraja's Rocchio wired into search with idf-weighted article vectors from real articles only | ~~Done~~; off by default (it drifts as often as it helps) |
| Learning-to-rank | Learn the weights of the score's parts from our judgments (logistic regression, leave-one-need-out) | ~~Done~~: MAP 0.807 vs 0.690 for the net score over 32 needs, p < 0.001 |

## Done

### Ranking
- [x] ~~lnc.ltc with heap top-K~~
- [x] ~~Net score: cosine + zone weights + proximity + g(d)~~
- [x] ~~BM25 (k1 1.2, b 0.75)~~
- [x] ~~Filters for newspaper, section, state and date, on every ranker~~
- [x] ~~Query parser: exact phrase, part of the phrase, all words, all words with variants, then any word~~
- [x] ~~Date-aware कल, as a re-ranking step that keeps parser stages in order~~
- [x] ~~PageRank (damping 0.85, power iteration) and first-to-publish credit in g(d)~~
- [x] ~~Duplicate collapsing with "also in"~~
- [x] ~~Listing pages and horoscopes pushed below real articles (page-type quality)~~
- [x] ~~Pseudo-relevance feedback switch (Rocchio, top 5 articles, 5 terms)~~
- [x] ~~Dense re-ranking with multilingual e5 (optional install)~~
- [x] ~~Rank fusion (RRF) of lnc.ltc, BM25, the net score and dense~~
- [x] ~~MMR diversification (lambda 0.7), within parser stages~~
- [x] ~~Query difficulty hint (specificity, scope, clarity, parser stage)~~
- [x] ~~Full crawl embedded for dense re-ranking (about 1 minute, cached locally)~~

### Query side
- [x] ~~Cross-lingual layer with an English to Hindi news dictionary (about 180 entries)~~
- [x] ~~272 more pairs learned from Jagran's bilingual headlines (Dice alignment), under the hand-made dictionary~~
- [x] ~~Real query pipeline: Viraja's `build_query` with phonetic variants, then translation, then Dhrithi's analyzer for each index mode~~
- [x] ~~Weak phonetic variants dropped, letter case matched for Roman words~~

### Speed-ups (Lecture 7)
- [x] ~~Index elimination~~
- [x] ~~Champion lists, with the high/low fallback and optional g(d) ordering~~
- [x] ~~Recent-news tiers~~
- [x] ~~Cluster pruning (leaders and followers)~~
- [x] ~~Impact-ordered postings with early stopping~~

### Evaluation code
- [x] ~~Metrics: P@k, R@k, F1, MAP, nDCG and 11-point PR, checked against the lecture examples~~
- [x] ~~Experiment runner: every stemming mode x every ranker x every query, TREC run files, tables, wins and losses, translation off vs on~~
- [x] ~~Speed-ups table (articles scored vs top 10 kept)~~
- [x] ~~Stop words, idf and Zipf analysis with plots~~
- [x] ~~My 8 information needs (R01 to R08), in four forms each~~
- [x] ~~Experiment runner and corpus stats on the real index: queries read from the needs files, real query pipeline per mode, run files for pooling (64 queries x 4 modes x 4 rankers so far)~~

### App and CLI
- [x] ~~Streamlit app: three stemming columns, highlighted snippets, match chips, filters, score breakdown, parser stage, "only here" tags~~
- [x] ~~Switches for parsing, authority, collapsing, kal, translation and dense, plus a speed-up menu~~
- [x] ~~`app/cli.py` with `--explain` showing every step, and flags for every feature~~

### Integration
- [x] ~~Dhrithi's index and Viraja's query layer plugged into the app and CLI, with a fallback to the sample index~~
- [x] ~~End-to-end: Hindi, Hinglish and English versions of a need find the same articles~~
- [x] ~~Running on Riya's full crawl: 5,000 articles, four indexes built in about 30 s, searches in under 0.1 s~~
- [x] ~~Merged into `main` up to rank fusion~~

## Remaining

### Can do now
- [ ] Facet counts next to each filter option, e.g. "Jagran (12)"
- [ ] Evaluation tab in the app: P@10, MAP, nDCG and PR curves per ranker and stemming mode
- [ ] Autocomplete with prefix search over the term dictionary

### On the frozen corpus
- [x] ~~Corpus frozen: Riya's full crawl minus its one repeated article, `data/news.jsonl`, 5,000 articles (not in git)~~
- [x] ~~Four indexes built from it, dense vectors embedded~~
- [x] ~~Speed-ups table, stop words, idf and Zipf on it~~
- [x] ~~Run all queries (128 from all four needs files) through every system and write the run files for pooling~~

### Judging
- [x] ~~Sanity check without judgments: cross-form agreement and system agreement on the frozen corpus~~
- [x] ~~Pool per need from the run files (641 articles for 16 needs, rebuilt after Viraja's spelling fix; Riya's pooling script gives the same set)~~
- [x] ~~Judging page in the app, one judgments file per person in the repo, linked from the search page~~
- [x] ~~Evaluation runs end to end on the judgments (checked with the first ones)~~
- [x] ~~Judgment-free results rerun after Viraja's rare-spelling fix (cross-form agreement up about 0.1, 6 queries flagged instead of 11)~~
- [x] ~~Judge my 8 needs (all 297 done)~~
- [x] ~~Pool extended with Riya's and Dhrithi's needs and the fixed auto stemming (1,659 articles)~~
- [x] ~~Evaluation and learning-to-rank on all 32 needs (57% of the pool judged)~~
- [x] ~~Judging stopped by the team at 57% of the pool; the 32-need evaluation is reported as final with that limitation~~
- [x] ~~Graphs for every result (PR curves, rankers, stemming, translation, learning to rank) and the sanity check on the final runs~~

### Once there are judgments
- [x] ~~Stemming: none vs light vs auto (P@10, MAP, nDCG, PR curves)~~
- [x] ~~Translation off vs on for the English queries~~
- [x] ~~lnc.ltc vs BM25 vs net score vs fusion, and dense on vs off~~
- [x] ~~Stop word experiment: no idf vs idf vs stop words removed~~
- [x] ~~Wins and losses per query~~
- [x] ~~Check the difficulty hint against the judgments: do flagged queries really have lower P@10?~~
- [x] ~~Learning-to-rank final run on the full judgments, with and without the dense feature~~

## What my code is waiting on
| From | What | Why it matters for my part |
|---|---|---|
| Viraja | "delhi" still goes to देल्ही instead of दिल्ली, "iyer" to एयर instead of अय्यर (her fix sorted "bhukamp" and "modi") | Hinglish queries with these words miss their articles; English ones still work through translation |

## Known issues in my part
- The evaluation covers all 32 needs but only 57% of the pool is judged (the team stopped there); unjudged articles count as not relevant, so absolute scores are lower than they'd be with full judging. This goes in the report's limitations.
- The frozen corpus wasn't cleaned beyond the one repeated article: 249 bodies have HTML tags, and 900 pages (18%) are section and listing pages and 104 (2%) are horoscopes. Ranking now pushes the listing pages and horoscopes below real articles; the HTML leftovers remain a limitation.
- Two test files share a name with Dhrithi's (`test_build.py`, `test_scoring.py`), so `tests/` and `partwise-tests/` have to be run as separate pytest commands.
- The dense mix (0.5 first stage, 0.5 e5) and the net score weights are hand-picked for now; learning-to-rank should set them.

## Future prospects

What we couldn't finish in the hackathon, and where the project could go next.

### Evaluation
- Judge the rest of the pool (43% is unjudged), so absolute scores aren't pulled down and smaller differences, like light stemming's gain on the first 16 needs, can be confirmed or ruled out.
- More needs (the plan aimed at 30 or more), written by people outside the team so they aren't shaped by what the system already does well.
- Measure agreement between judges by having two people judge the same needs (Cohen's kappa).

### Ranking
- Replace the hand-tuned net score with the learned weights in the app by default, since learning to rank beat it significantly.
- Pairwise or listwise learning to rank (ranking SVM, LambdaMART) once there are more judgments, instead of pointwise logistic regression.
- A fine-tuned dense model for Hindi and Hinglish news, used as a retriever rather than only a re-ranker, with a vector index.
- Tune the dense mix, the MMR lambda and the Rocchio settings on the judgments instead of choosing them by hand.

### Query side
- Fix the remaining spelling misses ("delhi" → दिल्ली, "iyer" → अय्यर) by weighting common corpus spellings more than spelling distance.
- Grow the translation dictionary: clean the learned pairs from Jagran's headlines (drop single-story mistakes like students → वृंदावन) and learn from other bilingual sources.
- Spelling suggestions ("did you mean") from the k-gram index, and autocomplete.

### Corpus
- Clean the crawl at the source: strip HTML from bodies and skip listing and horoscope pages in the crawler, instead of pushing them down at ranking time.
- A larger, continuously updated corpus with adaptive recrawling, so recency tiers and date-aware kal have older news to work with.
- More languages next to Hindi (Punjabi, Bengali, Marathi), reusing the same pipeline.

### App
- An evaluation tab in the app showing the metrics and graphs live.
- Facet counts next to the filters, e.g. "Jagran (12)".
- Explanations in plain language for why a result ranked where it did, built from the score breakdown.

