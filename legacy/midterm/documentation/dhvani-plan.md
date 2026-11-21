# Dhvani: our Hindi + Hinglish news search engine

Track 5 · repo `Dhvani-lingual` · Riya, Dhrithi, Viraja, Rishit · 36 hours

## 1. What we're building
We're building a search engine over Hindi newspaper articles that we crawl ourselves. You can search in Hindi, Hinglish, English, or a mix of all three, and these should all bring up the same articles: `कल का मौसम`, `kal ka mausam`, `kal ka mosam`, `weather tomorrow`.

Every search shows three columns side by side: no stemming, stemming, and auto (where the system decides for each word whether stemming is worth it). Each result tells you how it matched (exact, phonetic or translated), shows a snippet with the matching words highlighted, and says "also in: …" when other papers ran the same wire story.

## 2. Data
| What | Where from | What we use it for |
|---|---|---|
| Hindi news, roughly 5k to 12k articles | We crawl Jagran, Navbharat Times, Live Hindustan, Amar Ujala and Aaj Tak. Jansatta and Bhaskar are backups | Our main corpus |
| Aksharantar Hindi (33 MB) | Hugging Face `ai4bharat/Aksharantar` | Learning edit costs (train split) and testing word matching (test split) |
| MUSE English to Hindi dictionary | Facebook MUSE | Handling English words in queries |
| MIRACL-Hindi (optional) | Hugging Face `miracl/miracl` | A standard benchmark that already has relevance judgments |

- **Sites we're not touching:** BBC Hindi, because its robots.txt says no crawling and no building datasets. News18 and NDTV blocked our crawler, so we leave them alone too.
- **How we crawl:** our bot uses its real name, waits 8 seconds between requests to the same site, and stops if it sees a 403 or a captcha. We don't store author names. Article text stays on our laptops, and only code and URLs go on GitHub.

## 3. Who's doing what
How to read this: **Lecture** means stuff from the syllabus (marks for using IR principles). **NEW** means things beyond the syllabus (extra marks). **OURS** means ideas we came up with ourselves (novelty marks).

| | Riya: Crawler | Dhrithi: Text & indexes | Viraja: Hinglish layer | Rishit: Ranking & app |
|---|---|---|---|---|
| Lectures | L20 crawling | L1, L2, L6 | L2, L3 | L6, L7, L8 |
| Track 5 part they own | The regional news corpus | Hindi tokenizing and normalizing, stemming vs no stemming | Soundex-style phonetic matching, name spelling variants | Cross-lingual ranking with the vector space model, stop words and idf on a Hindi corpus |
| NEW | MinHash + LSH, adaptive recrawl | YASS stemmer learned from the corpus, extended-biword phrase index, variable-byte and gamma compression | Query expansion (Rocchio) | BM25, learning-to-rank, PageRank and "first to publish" authority, date-aware "kal", collapsing duplicate stories, dense re-ranker (optional) |
| OURS | A robots.txt checker that fixes the mistakes Python's built-in one makes | Selective stemming (the auto column) | Dhvani-code and learned edit distance | The stemming columns with match-type snippets |
| Lecture-core hours | 15 | 15 | 15 | 15 |
| NEW + OURS hours | 4.5 | 4.5 | 4.5 | 4.5 |
| Shared hours | 7.5 | 7.5 | 7.5 | 7.5 |
| **Total** | **27** | **27** | **27** | **27** |
| Code folder | `dhvani/crawl/` | `dhvani/text/`, `dhvani/index/` | `dhvani/query/` | `dhvani/rank/`, `app/` |
| Shared tooling they build | Pooling script | Nothing extra | Nothing extra | Metrics code, experiment runner and the final experiments |

**Shared hours, same for everyone:** 1 h kickoff and integration, 1.5 h writing 8 information needs, 2 h judging, 1.5 h on your report section, 1.5 h on your video segment.

## 4. The roadmap

### Phase 0 · H0 to H1 · Kickoff (all of us)
- Agree on the 5 shared formats and then don't change them: the article file, analyzer, index, query object and run file.
- Set up the repo with the 4 code folders, install packages, and download Aksharantar and MUSE.
- **We're done when** the formats are written in the README and everyone can run `python -c "import dhvani"`.

### Phase 1 · H1 to H3 · Foundations (nobody waits on anybody)
| Riya | Dhrithi | Viraja | Rishit |
|---|---|---|---|
| robots.txt checker and its tests; frontier skeleton | Normalizer and tokenizer; light stemmer | Language ID; standard Roman spellings; Soundex and Dhvani-code | A sample 20-document index with lnc.ltc running on it; the metrics code |

- **We're done when (H3)** Riya has shared a 300-article sample, Viraja has shared a simple exact-match query stub, and the crawler is running.

### Phase 2 · H3 to H8 · Building the core
| Riya | Dhrithi | Viraja | Rishit |
|---|---|---|---|
| Pulling articles out of JSON-LD; URL normalization; filters | 3 positional indexes with zones and fields; Boolean search, skip pointers, phrases | k-gram index; learned edit distance (aligning and counting) | Heap top-K, zones, proximity, recency; BM25; Streamlit app with the stemming columns |

- **We're done when (H8)** a Hindi query shows no-stem and stem results in the app. The auto column comes later, in Phase 4.

### Phase 3 · H8 to H12 · Plugging it together
| Riya | Dhrithi | Viraja | Rishit |
|---|---|---|---|
| Near-duplicates (shingles + Jaccard); keeping an eye on the crawl | Aggressive stemmer; stem-diff tool | Weighted expansion; context correction; hooking into Rishit's ranker | Cross-lingual layer; query parser; snippets; filters; `--explain`; idf and Zipf analysis |

- **Everyone** writes their 8 information needs by H10, before we tune anything.
- **We're done when (H12)** the Hindi, Hinglish and English versions of the same need all work end to end.

### Phase 4 · H12 to H22 · The extra stuff, plus sleep
- **Sleep in shifts:** Riya and Dhrithi sleep H12 to H17, Viraja and Rishit sleep H17 to H22. Someone is always watching the crawler.

| Riya | Dhrithi | Viraja | Rishit |
|---|---|---|---|
| MinHash + LSH; adaptive recrawl; **freezes the news snapshot at H18** | Selective stemming and the auto column; YASS stemmer; extended-biword index; compression; full index build after H18 | Rocchio query expansion; edit-cost heatmap; names test set | PageRank and authority g(d); date-aware "kal"; collapsing duplicates; champion lists and the recent tier; learning-to-rank features |

- **We're done when (H22)** every version of the system can produce results for all 120 queries.

### Phase 5 · H22 to H28 · Judging and experiments
- **H22 to H25, all of us:** Riya's script builds the pools and everyone judges for about 2 hours.
- **H25 to H28, everyone runs their own results table:**
  - **Riya:** how well near-duplicate detection works at each Jaccard threshold, and exact vs MinHash speed.
  - **Dhrithi:** phrase-query speed; index size with compression.
  - **Viraja:** the phonetic methods compared (word by word and on full queries); with vs without query expansion.
  - **Rishit:** the stemming comparison (no stemming vs light vs aggressive vs YASS vs auto); stop words, idf and Zipf; cross-lingual results; lnc.ltc vs BM25 vs learning-to-rank; authority g(d); "kal"; duplicate collapsing; champion-list speed.
- **We're done when (H28)** every table and graph exists.

### Phase 6 · H28 to H36 · Report, video, submit
- **H28 to H34:** everyone writes their report section and records their video segment. Riya writes the README, and Rishit puts the final report and video together.
- **H34 to H36:** buffer time, last checks, submit.

## 5. Each person's part in detail

### Riya · Crawler and corpus · 27 h
**From the lectures (15 h)**
- A robots.txt checker that follows the actual robots.txt standard. Python's built-in `urllib.robotparser` gets Jagran, Jansatta, Live Hindustan and Amar Ujala wrong, and those cases become our unit tests (2.5 h).
- A Mercator frontier: priority front queues, one back queue per site, and a heap tracking when each site can next be hit (4 h).
- News sitemaps as seeds, re-read every 2 to 6 hours. URL normalization, and skipping video, photo and astrology pages (2.5 h).
- Pulling headline, body, date and section from each page's JSON-LD, with `<p>` tags as a fallback. State and city come from the URL. Also saving each article's links to other crawled articles, so Rishit can run PageRank (2 h).
- Near-duplicate detection: an exact hash plus 4-word shingles and Jaccard, recording `dup_of` (2.5 h).
- Watching the crawl and putting together corpus stats (1.5 h).

**NEW (4.5 h):** MinHash + LSH (2.5 h), adaptive recrawl (2 h).
**Shared tooling:** the pooling script.
**Report:** data and crawling. **Video:** the crawler log, robots.txt tests, duplicates.

### Dhrithi · Text processing and indexes · 27 h
**From the lectures (15 h)**
- A normalizer that handles NFC; nukta, chandrabindu and anusvara (so हिंदी and हिन्दी match); zero-width joiners; the danda; and Devanagari digits. Tokenizing uses `regex` with `[\p{L}\p{M}\p{Nd}]+` (3 h).
- A light stemmer (based on the Ramanathan & Rao 2003 suffix list) and an aggressive one (3 h).
- One positional index per stemmer, with headline and body zones and source, date, state and section fields (4 h).
- Boolean search: smallest postings list first, skip pointers, phrases and proximity (2 h).
- A stem-diff tool (1 h).

**OURS (2.5 h): selective stemming.** Stemming helps some queries and hurts others (we saw this in L2), so Dhrithi lets the system decide for each query word whether to stem it:
- Group words into stem classes, like बारिश and बारिशों → बारिश.
- Split up any class whose words rarely show up in the same articles. This is corpus-based stemming, from Xu & Croft 1998.
- At search time, a word only gets stemmed if its class held together.
- This gives the app its third column, **auto**, next to no stem and stem.

**NEW (5 h):**
- **A stemmer learned from the corpus (YASS, Majumder et al. 2007)** (2 h). It finds suffixes on its own by clustering words that share long prefixes, with no hand-written rules. We compare it with the Ramanathan & Rao stemmer.
- **A Hindi extended-biword phrase index** (1.5 h). In Hindi, postpositions like का, की, के, में, से and पर play the same role as the lecture's in-between "X" words, so a phrase like "भारत का प्रधानमंत्री" becomes one indexed term. Common biwords get indexed directly and everything else uses positional intersection, which is the lecture's combination scheme. We report phrase-query speed for each approach.
- **Index compression** (1.5 h). Variable-byte and gamma coding of the gaps in postings lists, comparing index size and query speed before and after.
- A Stanza Hindi lemmatizer, but only if there's time.

**Report:** tokenizing, stemming, selective stemming. **Video:** the normalization trace, postings, the stem diff, the auto column, and what the stemmers do behind the stemming results.

### Viraja · Hinglish layer · 27 h
**From the lectures (15 h)**
- Language ID for each word. Words that could go either way ("main", "to", "hi") keep both readings, with weights (2 h).
- A standard Roman spelling for every Hindi word, with the rule that drops the final "a" (कमल → kamal) (2 h).
- Four ways of matching spellings, compared head to head (6 h):
  1. plain Levenshtein
  2. the lecture's Soundex
  3. Dhvani-code, our own Soundex for Hindi (मौसम, mausam and mosam all become 585)
  4. a learned edit distance, where each edit costs −log P of that edit, counted from Aksharantar using the Levenshtein backtrace and re-aligned 2 or 3 times
- A k-gram index to find candidates, re-ranked by edit distance. The top 5 weighted variants get added to the query rather than merged in the index (2 h).
- Context correction: for multi-word queries, picking the combination of variants that shows up together most often (2 h).
- A word-level test: accuracy and MRR on the Aksharantar test split (1 h).

**NEW (4.5 h):** Rocchio query expansion using the top 10 results (3 h), plus the edit-cost heatmap and a 50-name test set (1.5 h).
**Report:** phonetic matching and what's new about it. **Video:** the problem, the codes, the candidates, the heatmap, query expansion.

### Rishit · Ranking, cross-lingual and app · 27 h
**From the lectures (15 h)**
- The cross-lingual layer: English words turn into weighted Hindi words using MUSE plus our own 300-word news dictionary (2.5 h).
- lnc.ltc scoring (1 + log10 tf, as corrected on L7 slide 2), zones, proximity, recency g(d), and net score (3.5 h).
- Heap top-K, index elimination, champion lists, and a recent-news tier (2 h).
- A query parser that tries phrase first, then AND, then phonetic AND, then free text (1.5 h).
- Filters for source, state, section and date range (1 h).
- Snippets with the matched words highlighted in different colours (1.5 h).
- The Streamlit app with the stemming columns, plus `cli.py --explain` (3 h).

**NEW (5.5 h):** BM25 (1 h), learning-to-rank (a logistic regression trained on half the judged needs) (2.5 h), date-aware "kal" (1.5 h), and collapsing duplicate stories (0.5 h). The dense re-ranker only happens if there's time.

**NEW (2.5 h): authority ranking**, building on L7's static quality score g(d) and adding PageRank.
- **PageRank on the link graph:** Riya's extractor saves the links between crawled articles, so this needs no extra crawling. PageRank runs with damping 0.85 using power iteration.
- **Credit for publishing first:** in each group of near-duplicate wire stories, the earliest one counts as the original and gets an authority boost. Later copies don't.
- **The final static score:** g(d) = a·recency + b·PageRank + c·original-source flag, scaled to between 0 and 1. Net score = cosine + λ·g(d), just like in L7. Champion lists are sorted by g(d) + tf-idf, as the lecture suggests.
- **Results table:** P@10 and nDCG@10 with recency only, then adding PageRank, then adding the first-to-publish credit.

**Evaluation:**
- The metrics code: P@k, R@k, MAP, nDCG and 11-point PR, checked against the lecture's own examples (MAP has to come out as 0.53).
- Stop words and idf on the Hindi corpus: the idf of the most common words, a Zipf plot, and keeping stop words vs removing them vs leaving it to idf.
- An experiment runner that runs every stemming mode and ranker over every query, writes the run files and builds the tables, including the stemming comparison.
**Report:** ranking, cross-lingual, stop words and idf, evaluation. **Video:** the live demo, `--explain`, cross-lingual, authority, the stemming comparison table.

## 6. Who needs what from whom
| Person | Needs | By when | What to use until then |
|---|---|---|---|
| Riya | Nothing | – | – |
| Dhrithi | An article sample from Riya | H3 | MIRACL-Hindi passages |
| Viraja | The index vocabulary from Dhrithi | H6 | Just Aksharantar |
| Rishit | Dhrithi's index and Viraja's query object | H8 and H12 | A fake 20-document index and the exact-match stub |

## 7. How we evaluate
- 30 information needs, each written 4 ways (Hindi, natural Hinglish, messy Hinglish from Aksharantar's test data, and English). That's 120 queries.
- For each need we pool the top 10 from every version of the system, then judge each article 0, 1 or 2 against the need itself. One judgment covers all 4 ways of asking.
- **Baseline:** plain tf-idf with no normalization.
- **Graphs:** PR curves, idf and Zipf, the edit-cost heatmap, champion-list speed vs quality.

## 8. If we run out of time
Drop things in this order: dense re-ranker, lemmatizer, learning-to-rank, gamma coding, index compression, adaptive recrawl, MIRACL, aggressive stemmer, context correction.

## 9. The video (7 minutes)
| Time | What's on screen | Who |
|---|---|---|
| 0:00 to 0:50 | The problem | Viraja |
| 0:50 to 2:30 | Live demo: 4 ways of asking, all 3 stemming columns, snippets, filters, "kal" | Rishit |
| 2:30 to 3:15 | The crawler, robots.txt tests, duplicates | Riya |
| 3:15 to 4:00 | Normalization, postings, stem diff, selective stemming (the auto column) | Dhrithi |
| 4:00 to 4:45 | Codes, candidates, heatmap, query expansion | Viraja |
| 4:45 to 5:30 | `--explain` scores, cross-lingual, authority ranking | Rishit |
| 5:30 to 7:00 | Results tables | everyone shows their own |
