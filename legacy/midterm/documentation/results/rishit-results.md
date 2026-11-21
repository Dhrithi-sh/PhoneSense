# Rishit's results

Results for ranking, the cross-lingual layer and evaluation. Each section says which corpus, queries and settings it used, so early numbers and final numbers don't get mixed up. Plots go in `documentation/figures/` as `rishit-*.png`.

| Section | Status |
|---|---|
| Speed-ups vs exact lnc.ltc | Frozen corpus (5,000 articles) |
| Stemming: none vs light vs aggressive vs auto | Final (32 needs, 57% of the pool judged) |
| Translation off vs on | Final (32 needs, 57% of the pool judged) |
| lnc.ltc vs BM25 vs net score vs fusion | Final (32 needs, 57% of the pool judged) |
| Stop words, idf and Zipf | Frozen corpus; stop word experiment final (32 needs) |
| Dense re-ranking vs sparse only | Waiting for judgments |
| Rank fusion (RRF) vs single rankers | Waiting for judgments |
| Sanity check without judgments | Frozen corpus (5,000 articles) |
| Query difficulty hint | Frozen corpus; checked against the judgments |
| Learned translations | Frozen corpus (5,000 articles) |
| Page-type quality | Frozen corpus (5,000 articles) |
| Pseudo-relevance feedback (Rocchio) | Frozen corpus, early judgments (40 queries) |
| Learning-to-rank | Final (32 needs, 57% of the pool judged) |
| Wins and losses | Final (32 needs, 57% of the pool judged) |

## Speed-ups vs exact lnc.ltc

> **Final corpus.** These are on the frozen corpus: Riya's full crawl with its one repeated article removed, 5,000 articles. It wasn't cleaned further, so it still has HTML in 249 bodies, 106 astrology pages and about 50 section pages saved as articles; those are listed as limitations.

- **Corpus:** the frozen corpus, `data/news.jsonl` (Riya's `news_dedup.jsonl`, 5,001 lines, minus the one article saved twice: 5,000 articles), mostly from 5 to 7 Oct; no-stemming index, 83,560 distinct words
- **Queries:** 64, all four forms of Rishit's R01 to R08 and Viraja's V01 to V08, through the full query pipeline (Viraja's `build_query`, phonetic variants, translation)
- **k:** 10
- **Date run:** 7 Oct, rerun after Viraja's rare-spelling fix
- **How to rerun:** `speedup_table()` in `dhvani/eval/experiments.py`

"Scored" is the average share of the articles sharing a query word that each method actually scored (lower is faster). "Kept" is the average share of the exact lnc.ltc top 10 that the method also returned (higher is closer to exact).

| Method | Scored | Kept |
|---|---|---|
| Index elimination | 0.01 | 0.73 |
| Champion lists, r = 2 | 0.06 | 0.38 |
| Champion lists, r = 5 | 0.03 | 0.67 |
| Champion lists, r = 10 | 0.05 | 0.87 |
| Champion lists, r = 50 | 0.18 | 1.00 |
| Recent-news tiers | 0.95 | 0.97 |
| Cluster pruning, b = 1 | 0.03 | 0.27 |
| Cluster pruning, b = 3 | 0.07 | 0.41 |
| Impact-ordered, first 20 per word | 0.09 | 0.65 |
| Impact-ordered, first 50 per word | 0.18 | 0.80 |
| Impact-ordered, weight at least 0.5 x best | 0.36 | 0.70 |

**What it shows**
- **Champion lists give the best trade-off.** With r = 50 they score under a fifth of the candidates and still return exactly the full top 10. r = 10 keeps 87% while scoring only 5%. With r = 2 the lists are often too short for 10 results, so it falls back to the full postings more, which is why it scores more than r = 5.
- **Impact-ordered postings come second.** Reading the first 50 articles of each word's list scores the same 18% as champion lists with r = 50 but keeps 80%, because the cut-off isn't tuned in advance per word.
- **Index elimination is the cheapest that's still usable.** Dropping words like के and में and asking for most of the remaining words scores 1% and keeps 73%.
- **Cluster pruning is fast but loses the most.** One cluster scores 3% and keeps 27%. Articles about one story end up spread across several clusters, so the closest leader only covers part of them.
- **Recent-news tiers barely cut anything.** Almost all articles are from the last two days (3,150 from 7 Oct alone), so tier 0 is nearly the whole corpus.

### Earlier run on the 300-article sample

Same queries and k on Riya's 300-article sample, kept for comparison. With so few articles every method looked better, which is why the speed-ups had to be checked on the frozen corpus.

| Method | Scored | Kept |
|---|---|---|
| Index elimination | 0.21 | 0.72 |
| Champion lists, r = 5 | 0.27 | 0.93 |
| Champion lists, r = 10 | 0.37 | 0.99 |
| Cluster pruning, b = 1 | 0.15 | 0.28 |
| Impact-ordered, first 20 per word | 0.51 | 0.94 |

## Stop words, idf and Zipf

> **Final corpus.** Same frozen 5,000-article corpus as above.

- **Corpus:** 5,000 articles, no-stemming index, 83,560 distinct words, 40,255 of them seen only once
- **How to rerun:** `term_stats()`, `stop_words()` and `zipf_fit()` in `dhvani/eval/corpus_stats.py`
- **Plots:** `documentation/figures/rishit-zipf.png`, `documentation/figures/rishit-idf-histogram.png`

The 15 most widespread words, by document frequency (df), with collection frequency (cf) and idf = log10(N / df):

| Word | df | cf | idf |
|---|---|---|---|
| के | 4,926 | 92,206 | 0.006 |
| की | 4,903 | 66,394 | 0.009 |
| में | 4,886 | 74,879 | 0.010 |
| से | 4,821 | 44,687 | 0.016 |
| का | 4,806 | 36,013 | 0.017 |
| और | 4,789 | 47,396 | 0.019 |
| को | 4,788 | 41,178 | 0.019 |
| पर | 4,678 | 31,069 | 0.029 |
| है | 4,461 | 55,111 | 0.050 |
| ने | 4,369 | 32,794 | 0.059 |
| भी | 4,146 | 18,068 | 0.081 |
| लिए | 4,069 | 14,592 | 0.089 |
| हैं | 3,684 | 19,196 | 0.133 |
| किया | 3,675 | 11,436 | 0.134 |
| कि | 3,654 | 17,882 | 0.136 |

**What it shows**
- **The stop words come straight out of the data.** The top 15 by df are all Hindi postpositions, conjunctions and auxiliaries. के is in 4,926 of 5,000 articles, so its idf is almost 0 and it barely affects lnc.ltc even without a stop list. That's why the ranker keeps them instead of removing them: idf already does the job, and phrase queries like "भूकंप के झटके" still match exactly.
- **Zipf's law holds for the frequent words.** Over the 1,000 most frequent words the slope is -0.90, close to the -1 of Zipf's law. The fit over all 83,560 words gives -1.45 because 40,255 words appear only once (names, numbers, typos), which drags the tail down. On the plot the line follows the middle of the curve and overshoots the very top, where a handful of function words take a large share of all tokens.

## Query difficulty

> **Frozen corpus, no judgments yet.** Same 5,000-article corpus. Whether flagged queries really do worse can only be checked once there are judgments.

- **Queries:** the 64 needs queries (R01 to R08, V01 to V08, four forms each), plus a handful of deliberately vague ones
- **How to rerun:** `predict()` in `dhvani/rank/difficulty.py`

| Signal | Needs queries | Vague queries |
|---|---|---|
| Specificity (highest per-word idf) | lowest 1.24, middle half 1.9 to 2.4 | news 0.63, kya hua 0.56, बड़ी खबर 0.81, के में 0.01, india 0.69 |
| Clarity (bits) | 1.11 to 1.82 | news 3.02, के में 1.27 |

**What it shows so far**
- **Specificity separates them cleanly.** Every needs query has a word with idf above 1.2; every vague query stays below 1.0. The threshold sits at 1.0.
- **Clarity doesn't work here.** It stays in a narrow band for real queries, and "news" scores highest of all because its top results are near-identical listing pages. It's kept as a reported number only.
- **6 of the 64 needs queries are flagged** (11 before Viraja's rare-spelling fix), all because only the "any word" stage matched: Hinglish or messy forms of V01, V02, V04, V05 and V08. Those are the forms where the phonetic layer still doesn't find every word, so the flag points at real weak spots.

## Sanity check without judgments

> **Final runs.** Frozen corpus, all 128 queries (32 needs, four forms each), 4 stemming modes x 4 rankers, listing pages pushed down, top 10.

- **How to rerun:** `python -m dhvani.eval.experiments --out data/eval/full`, then `python -m dhvani.eval.sanity`

### Do the four forms of a need find the same articles?
Share of the Hindi form's top 10 that each other form also returns, averaged over the 32 needs.

| Run | Hinglish | Messy | English |
|---|---|---|---|
| none, lnc.ltc | 0.47 | 0.42 | 0.40 |
| none, BM25 | 0.51 | 0.44 | 0.42 |
| none, fusion | **0.52** | 0.45 | 0.42 |
| none, net score | 0.38 | 0.30 | 0.28 |
| light, BM25 | **0.52** | 0.45 | 0.43 |
| light, fusion | 0.51 | **0.48** | **0.44** |
| aggr, fusion | **0.52** | 0.47 | **0.44** |
| auto, BM25 | 0.51 | 0.44 | 0.42 |

English form with translation off vs on (no stemming, net score): **0.22 → 0.28**.

### Do the systems agree with each other? (no stemming)

| Pair | Overlap | Tau |
|---|---|---|
| lnc.ltc vs BM25 | 0.88 | 0.76 |
| lnc.ltc vs fusion | 0.83 | 0.64 |
| BM25 vs net score | 0.56 | 0.48 |
| lnc.ltc vs net score | 0.56 | 0.47 |

**What it shows**
- **About half of the Hindi form's top 10 comes back for the Hinglish form** with BM25 or fusion, and 42 to 44% for English. Words that still don't map to the right Hindi spelling (देल्ही for delhi) or are missing from the dictionary send the search elsewhere.
- **Translation lifts English agreement** from 0.22 to 0.28.
- **BM25 and fusion are the most consistent across forms; the net score is the least** (0.28 to 0.38), the same pattern the judged evaluation shows.
- Earlier runs (64 queries, before listing pages were pushed down) had lower agreement (0.36 to 0.47 for Hinglish); the listing-page fix accounts for most of the gain.

## Learned translations

> **Frozen corpus.** Learned from the 973 Jagran headlines that have an English version appended.

- **How to rerun:** `python -m dhvani.rank.learn_dict`

Cut-off choice, checked against the 180-entry hand-made dictionary (English words both have, and how many the learned pair agrees with exactly):

| Dice at least | Pairs at least | Learned | Also in hand dictionary | Agree exactly |
|---|---|---|---|---|
| 0.3 | 3 | 360 | 47 | 74% |
| 0.4 | 3 | 328 | 40 | 75% |
| **0.5** | **3** | **272** | **31** | **81%** |
| 0.6 | 4 | 132 | 15 | 93% |

Exact agreement undercounts: many "disagreements" are correct variants (women → महिलाएं instead of महिला, traffic → ट्रैफिक, road → रोड, exam → परीक्षाएं). The real errors come from one story dominating a word (students → वृंदावन, captain → स्मित, vote → चोरी from "वोट चोरी").

| | Hand dictionary only | With learned pairs |
|---|---|---|
| English words in our 16 needs that get a translation | 56 of 83 | 64 of 83 |
| English vs Hindi form agreement, BM25 | 0.394 | 0.388 |
| English vs Hindi form agreement, net score | 0.244 | 0.237 |

**What it shows**
- **The corpus can teach itself to translate.** 272 pairs with no outside data, mostly right, many of them names and places a general dictionary wouldn't have.
- **On our needs the effect is flat**, because the hand dictionary was grown from these same needs and the new words (rahul, gandhi, uttarakhand, rajya sabha) were already reachable through Viraja's phonetic layer. The value is for English words outside the needs (airport, compensation, border, challan).
- **Listing pages hide the gain.** For new English queries the top results are often section pages like "अंबाला की सबसे ताज़ा खबर", with or without the learned pairs.

## Page-type quality: pushing listing pages down

> **Frozen corpus.** 64 needs queries, no stemming, top 30 from the parser, then cut to 10.

- **How to rerun:** `python -m dhvani.rank.quality`, then compare `demote()` on and off with `cross_form_agreement()`

| Page type | Pages | Share |
|---|---|---|
| Article | 3,996 | 80% |
| Listing (section, city, live-blog index pages) | 900 | 18% |
| Horoscope | 104 | 2% |

Listing pages were 21% of the top 10 with the net score and 33% with BM25 before this.

Agreement with the Hindi form of the same need (overlap@10), before → after pushing non-articles down:

| Form | Net score | BM25 |
|---|---|---|
| Hinglish | 0.37 → **0.44** | 0.46 → **0.55** |
| Messy | 0.28 → **0.37** | 0.45 → **0.52** |
| English | 0.24 → **0.36** | 0.39 → **0.49** |

**What it shows**
- **A fifth of the corpus wasn't news,** and it was crowding out real articles, most of all for English and Hinglish queries, whose words a listing page is most likely to contain somewhere.
- **A query-independent quality score fixes most of it.** English queries agree with their Hindi form half again as often with the net score (0.24 → 0.36). This is the largest single improvement measured so far.
- **It works without cleaning the corpus,** so every number stays on the same frozen set of articles.

## Pseudo-relevance feedback (Rocchio)

> **Frozen corpus; the metric rows use partial judgments** (Viraja's 459 and Rishit's first 23, 40 queries), so they're early.

- **Setup:** top 5 real articles as feedback, tf x idf vectors, Rocchio alpha 1.0 and beta 0.75, top 5 new terms, net score, no stemming, listing pages pushed down
- **How to rerun:** `feedback_query()` in `dhvani/rank/feedback.py`, or `app/cli.py "query" --prf`

| | Without feedback | With feedback |
|---|---|---|
| Hinglish vs Hindi form agreement | 0.46 | 0.47 |
| Messy vs Hindi form agreement | 0.42 | 0.35 |
| English vs Hindi form agreement | 0.39 | 0.38 |
| P@10 (early, 40 judged queries) | 0.588 | 0.545 |
| MAP (early, 40 judged queries) | 0.217 | 0.206 |

Examples of the terms it adds: "earthquake tremors delhi" → तीव्रता, रिक्टर, महसूस, चमोली; "iyer century" → श्रेयस, वेस्टइंडीज; "smriti mandana captain bani" → हरमनप्रीत, कप्तानी, बीसीसीआई; "farmers worried about snow" → हेक्टेयर, अनुदान, फसल, क्षति.

**What it shows**
- **It helps when the first results are right.** "iyer century" picks up the first name श्रेयस and all of the top 3 become Shreyas Iyer stories.
- **It drifts when they're mixed.** "farmers worried about snow" moves to crop damage in general, and messy queries, whose first results are the weakest, lose the most.
- **On average it doesn't pay off on our needs,** so it's off by default. The judgments will say whether that holds.

## Evaluation on the first 16 needs (fully judged at the time)

> **Complete for these 16 needs.** Frozen corpus (5,000 articles), the 16 needs written first (R and V), every pooled article judged at the time (Rishit 297, Viraja 459 pairs, pooled from the top 10 of every run, listing pages pushed down as in the app), 64 queries (four forms each), k = 10. Grades 1 and 2 count as relevant.

- **How to rerun:** `python -m dhvani.eval.experiments --out data/eval/full` and `python -m dhvani.eval.ltr --dense` (or `scripts/rishit_results.py --dense` for everything)
- MAP and R@10 are low next to P@10 because recall is measured against everything judged relevant in the pool, which is often far more than 10 articles per need.

### Rankers (no stemming)

| Ranker | P@10 | R@10 | MAP | nDCG@10 |
|---|---|---|---|---|
| lnc.ltc | 0.803 | 0.291 | 0.272 | 0.778 |
| **BM25** | **0.830** | **0.300** | **0.281** | 0.799 |
| Fusion (RRF) | 0.811 | 0.295 | 0.277 | **0.800** |
| Net score | 0.722 | 0.265 | 0.243 | 0.729 |

Significance on per-query AP against lnc.ltc (randomization / t-test): BM25 better, p = 0.002 / 0.003; net score worse, p = 0.002 / 0.003; fusion no significant difference, p = 0.31.

### Stemming (net score)

| Stemming | P@10 | MAP | nDCG@10 | vs none (better / worse / same) | p (rand) |
|---|---|---|---|---|---|
| None | 0.722 | 0.243 | 0.729 | | |
| Light | **0.739** | **0.251** | **0.742** | 22 / 9 / 33 | **0.008** |
| Aggressive | 0.734 | 0.250 | 0.741 | 21 / 8 / 35 | **0.017** |
| Auto | 0.723 | 0.244 | 0.732 | 5 / 2 / 57 | 0.12 |

With BM25 and fusion the stemming modes are all within 0.02 of each other.

### Translation (English queries, net score)

| | P@10 | R@10 | MAP | nDCG@10 |
|---|---|---|---|---|
| Translation off | 0.556 | 0.198 | 0.168 | 0.545 |
| **Dictionary translation** | **0.719** | **0.263** | **0.229** | **0.695** |

### Learning to rank (leave-one-need-out, net score's top 30 re-ranked, with the dense feature)

| | MAP | P@10 |
|---|---|---|
| Net score (hand-picked weights) | 0.792 | 0.744 |
| **Learned** | **0.835** | **0.788** |

Learned vs net score: p = 0.07 (randomization), 0.08 (t-test). Average learned weights (standardised features): BM25 +1.06, zone +0.70, dense +0.45, cosine +0.19, PageRank +0.07, first to publish +0.02, real article -0.24, recency -0.33, proximity -0.34, parser stage -0.55.

(These MAP values are higher than in the tables above because they're computed over the 30 re-ranked candidates of each query rather than over the whole judged pool.)

### What it shows
- **BM25 is the best single ranker,** significantly better than lnc.ltc. Its term saturation and length normalisation suit news articles of very different lengths.
- **The hand-tuned net score is significantly worse than plain lnc.ltc.** Its fixed boosts for recency, proximity and the parser stage push relevant articles down. Learning to rank confirms it: those features get negative weight, and BM25, the headline match and dense get the most.
- **Learning the weights recovers the loss:** the learned ranker beats the net score by 0.04 MAP and 0.04 P@10 (p = 0.07, just short of significance with 16 needs).
- **Light stemming helps, significantly** (22 queries better, 9 worse), and aggressive stemming almost as much. Auto barely stems on the frozen corpus, so it equals no stemming.
- **Dictionary translation is the clearest Track 5 result:** English queries gain 0.16 P@10 and 0.06 MAP.
- **Fusion matches BM25** without needing to pick one ranker, and has the best nDCG.

## Final evaluation: all 32 needs

Graphs (regenerate with `python scripts/rishit_figures.py`): precision-recall curves `documentation/figures/rishit-pr-curves.png`, rankers `rishit-rankers.png`, stemming `rishit-stemming.png`, translation `rishit-translation.png`, learning to rank `rishit-ltr.png`.


> **Final, with a limitation: 57% of the pool is judged.** All four needs files are in (R, V, Y, D: 32 needs, 128 queries), and the pool has 1,659 articles from every system on every query (including Dhrithi's fixed auto stemming). The team stopped judging at Rishit 297 of 297, Viraja 296 of 510, Riya 240 of 509 and Dhrithi 120 of 343 pooled articles (129 of her 249 judgments are for articles no system retrieves, so they don't count). Unjudged articles count as not relevant, which lowers every absolute score and makes differences between similar systems harder to detect; comparisons between systems are still fair, since every system is scored against the same judgments. Learning to rank trains only on judged pairs, so it's the result least affected.

- **How to rerun:** `python -m dhvani.eval.experiments --out data/eval/full`, `python -m dhvani.eval.ltr --dense`

### Rankers (no stemming)

| Ranker | P@10 | R@10 | MAP | nDCG@10 |
|---|---|---|---|---|
| lnc.ltc | 0.673 | 0.304 | 0.274 | 0.676 |
| **BM25** | **0.677** | **0.305** | **0.275** | **0.682** |
| Fusion (RRF) | 0.666 | 0.297 | 0.267 | 0.677 |
| Net score | 0.555 | 0.241 | 0.212 | 0.574 |

Against lnc.ltc on per-query AP: net score worse, p < 0.001; BM25 no significant difference (p = 0.52); fusion no significant difference (p = 0.13).

### Stemming (net score)

| Stemming | P@10 | MAP | p vs none (rand) |
|---|---|---|---|
| None | 0.555 | 0.212 | |
| Light | 0.560 | 0.212 | 0.72 |
| Aggressive | 0.557 | 0.212 | 0.99 (26 better, 26 worse, 76 same) |
| Auto (Dhrithi's fixed version) | 0.555 | 0.213 | 0.50 (2 better, 0 worse, 126 same) |

With BM25, light stemming is the best mode (P@10 0.681, MAP 0.277), but no stemming difference is significant on these partial judgments.

### Translation (English queries, net score)

| | P@10 | MAP | nDCG@10 |
|---|---|---|---|
| Translation off | 0.488 | 0.183 | 0.493 |
| **Dictionary translation** | **0.578** | **0.224** | **0.577** |

### Learning to rank (leave-one-need-out, net score's top 30 re-ranked, with the dense feature)

| | MAP | P@10 |
|---|---|---|
| Net score | 0.690 | 0.566 |
| **Learned** | **0.807** | **0.637** |

Learned vs net score: **p < 0.001** (randomization and t-test), on 128 queries from 32 needs. Average learned weights: BM25 +0.98, zone +0.46, dense +0.43, cosine +0.23, PageRank +0.05, first to publish +0.03, real article +0.02, proximity -0.06, recency -0.08, parser stage -0.23.

### What it shows
- **Learning to rank is the clearest win:** +0.12 MAP and +0.07 P@10 over the hand-tuned net score, significant at p < 0.001 across 32 needs. Learning to rank trains only on judged pairs, so it's the result least affected by the missing judgments.
- **BM25, headline match and dense e5 carry the relevance signal;** the hand-picked recency, proximity and parser-stage boosts don't, which is why the net score trails plain lnc.ltc.
- **Translation still clearly helps English queries** (+0.09 P@10, +0.04 MAP).
- **Stemming effects shrink to nothing** on the larger, partly judged set. On the first 16 needs, which were fully judged, light stemming was a small significant gain (p = 0.008), so stemming helps a little on some needs but not reliably across all 32.
- **Auto stemming now changes the index** (82,751 terms against 83,560 with no stemming) but still changes the top 10 for only 2 of 128 queries.

### Stop words: no idf vs idf vs removing them (lnc.ltc, no stemming, 128 queries)

| Setup | P@10 | MAP | nDCG@10 |
|---|---|---|---|
| No idf (lnc.lnc) | 0.620 | 0.241 | 0.610 |
| **idf (lnc.ltc)** | **0.710** | **0.289** | **0.663** |
| Stop words removed (top 15 by df) | 0.710 | 0.289 | 0.663 |

idf is worth +0.09 P@10 and +0.05 MAP. Removing the 15 most common words on top of that changes nothing, because idf already gives them almost zero weight (के has idf 0.006), and keeping them lets phrase queries like "भूकंप के झटके" match exactly.

### Does the difficulty hint predict bad queries? (net score, no stemming, 128 queries)

| | Queries | P@10 | AP |
|---|---|---|---|
| Flagged "low confidence" | 16 | 0.35 | 0.147 |
| Not flagged | 112 | 0.585 | 0.222 |

Flagged queries really do worse: about 40% lower P@10. So the hint works as a warning without any judgments.
