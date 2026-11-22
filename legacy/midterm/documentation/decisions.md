# Decisions

A running list of the choices we made and why. Newest at the bottom.

## Project

**Track 5, Hindi only to start.** Track 5 fits our team best and lets the tokenizing, stemming, Soundex and spelling-correction lectures do the real work instead of just being a label. We started with Hindi only so we finish something solid in 36 hours. More languages can come later as the roadmap.

**Hindi news as the corpus.** The brief literally suggests a regional news search engine with precision and recall with and without stemming. News also gives us dates, sections and wire-story duplicates, which our ranking can use.

**Crawl only sites that allow it.** We checked robots.txt for every site. BBC Hindi says no crawling and no datasets in its robots.txt, and News18 and NDTV blocked us, so we left all three out. We use an honest bot name and wait 8 seconds between requests to the same site.

**Article text stays local.** News articles are copyrighted, so only code and URLs go on GitHub.

**Stemming shown side by side.** Every search shows no stemming, stemming and auto next to each other. The brief asks for results with and without stemming, and seeing it per query is more convincing than one table.

**English news idea kept small.** Riya suggested crawling English news too. A full English search engine would mean a second analyzer and index, merging scores across languages and translating articles, which is too much for 36 hours. If there's time after H12, we'll use English coverage only as a small "also reported in English" authority signal.

## Team and workflow

**Work split by component.** Riya does the crawler, Dhrithi does text processing and indexes, Viraja does the Hinglish layer, Rishit does ranking and the app. Each person owns one Track 5 requirement, one original idea and one results table, because individual contribution is graded.

**Everyone at 27 hours.** We kept the split even so nobody carries more than the others.

**Shared formats agreed up front.** The article file, analyzer, index, query object and run file are written down in `formats.md` so we can all build at the same time without waiting on each other.

**Own branches, merge at checkpoints.** Everyone commits to their own branch and merges into `main` only at the checkpoints, so `main` always works.

**Sample corpus stays out of git on main.** When merging Riya's branch, `data/news_sample_300.jsonl` was left out because the repo is public and it holds full article text. Her stats reports, which only have counts, are kept.

**One combined .gitignore.** Each branch had its own. On `main` they're merged into one: everything in `data/` is ignored except the placeholder and Riya's stats reports, and built indexes are ignored too.

**Commits as ourselves.** Every commit is under the person who did the work. AI help is declared in the report's AI-use section instead.

## Rishit's ranking code

**A sample index to start with.** Dhrithi's real index won't be ready for a few hours, so I made a small index over 20 hand-written Hindi articles. It has exactly the same methods as the real one, so swapping it out later needs no other changes. I called it a sample index rather than a fake one.

**lnc.ltc as the main scorer.** It's the standard SMART scheme from the lectures. I used 1 + log10(tf) for the tf weight because the lecture 7 slides correct the log(1 + tf) version.

**Heap for the top results.** Picking the top K with a heap is cheaper than sorting every score, as lecture 7 points out.

**Query stub with exact matches.** Viraja's query layer isn't ready yet, so a small stub builds the same query object with exact matches only. The ranker already uses expansion weights, so phonetic and translated variants will plug straight in.

**Net score on top of cosine.** Headline matches count more than body matches, query words close together get a bonus, and newer articles get a boost. Each part is shown separately so we can explain every score in the demo.

**"Now" is the newest article.** Recency is measured from the newest article in the index instead of the real clock, so results don't change depending on when we run them.

**BM25 alongside lnc.ltc.** BM25 is outside the syllabus, so it earns extra marks, and it gives us a strong baseline to compare against. I used the version of idf that never goes negative, so common Hindi words like का don't get penalised.

**Filters applied before picking the top results.** Source, section, state and date filters run before the top K is chosen, so a filtered search still gets its best matches instead of a few leftovers.

**Streamlit for the app.** The frontend isn't graded, and Streamlit gives us three columns, filters and a score breakdown with very little code.

**Matches coloured by type.** Exact, phonetic and translated matches get different colours, so the demo shows where each result came from at a glance.

**Snippets show the best sentence.** Each result shows the first sentence that contains a query word, falling back to the opening words. It's quick, and it shows the reader why the article matched.

**"Only here" tags.** When a result shows up in just one of the three stemming columns, it gets tagged. That makes the effect of stemming easy to spot in the demo without reading every column.

**Score breakdown in the app.** Every result has a dropdown showing how its score was built. The brief asks us to show weights and scores, not just final results, and this does it live.

**Rebase instead of merge for my own branch.** When GitHub had a small edit I'd made on the website, I put my local commits on top of it instead of making a merge commit. The history stays a straight line and is easier to read.

**Apple-style redesign.** The frontend isn't graded, but the demo video is, and a clean interface makes the system easier to follow on screen. We followed Apple's design guidelines: system font with tight tracking on large text, a translucent top bar, one accent colour, a segmented control for the ranking model, and filters tucked into a popover so the main page stays simple. Results are grouped lists like the iPhone Settings app.

**Styling lives inside the app file.** We kept all the styling in `app/streamlit_app.py` instead of adding a Streamlit theme file, so the look is in one place. The catch is that Streamlit's own controls default to red, so a few of them get a small colour shift to blue in CSS.

**An explain mode in the terminal.** The brief wants the video to show postings, weights and scores, not only the final results. `--explain` prints each stage in order so we can walk through one query live. It's in the terminal rather than the app because a plain text dump is easier to read on a screen recording.

**Query parser with a cascade.** Lecture 7 shows one free-text query being run as a phrase first, then as shorter phrases, then as a plain vector space query. We do the same with five stages and stop once we have enough results. An exact phrase match is a much stronger signal than scattered words, so it always ranks above them, and the ranker only decides the order within a stage.

**Phrases stay inside one zone.** A phrase only counts if the words are next to each other within the headline or within the body. Otherwise the last word of a headline and the first word of the body would wrongly count as a phrase.

**AND starts from the rarest word.** Intersecting the shortest postings list first keeps the result small from the start, which is the query optimisation from Lecture 1.

**Cross-lingual layer as dictionary-based query translation.** English words are looked up in an English to Hindi dictionary and their translations go into the same query vector as Hindi words. That keeps cross-lingual search inside the vector space model, which is what Track 5 asks for, instead of translating whole queries with a separate system.

**Weight split across translations.** "rain" becomes बारिश 0.7 and वर्षा 0.3, so a word with several translations doesn't count more than a word with one.

**Phrases first.** "prime minister" and "stock market" mean something different from their single words, so the longest dictionary phrase is matched first.

**Our own news dictionary, MUSE optional.** A small dictionary of common news words lives in the repo so translation works without any download. The bigger MUSE dictionary is merged in automatically if someone downloads it, and our entries win where both have a word.

**Only top-level data/ is ignored.** The old `.gitignore` rule ignored every folder called data, which would have kept the dictionary out of git. It now only ignores the top-level `data/` folder where crawled articles and downloads live.

**Metrics checked against the lecture.** The tests use the lecture's own worked examples: AP of 0.62 and 0.44 giving MAP 0.53, P@5 of 0.6 and R@5 of 0.5. If those numbers match, we can trust the tables built on top of them.

**Judgments per information need.** Each need is written several ways (Hindi, Hinglish, English) and judged once. Every form is scored against the same judged articles, so we can compare how well each form does on equal terms.

**The runner skips what isn't built yet.** Stemming modes that don't exist yet are skipped with a note instead of crashing, so the runner works from day one and fills in as the indexes arrive.

**Stop words come from the data.** Instead of typing in a Hindi stop word list, we rank terms by document frequency. The words with the lowest idf (में, का, की, के) are the stop words, which shows what idf is doing.

**Three ways to handle stop words.** No idf (lnc.lnc) lets every word count fully, idf (lnc.ltc) keeps stop words but pushes them towards zero, and removing them drops them from the query. Comparing the three shows whether removing stop words still matters once idf is in place.

**Feedback words only count in scoring.** Viraja's Rocchio expansion adds words from the top results as extra query tokens tagged `prf`. They aren't words the user typed, so the query parser leaves them out of the phrase and "all words" stages; otherwise a good result could be dropped just for missing a feedback word. They still add to the score and to the "some words" stage, and the app shows them as purple "Feedback" matches.

**Wait for Dhrithi before covering doc_norm.** Dhrithi's index leaves `doc_norm` empty, which lnc.ltc needs. The ranker can work it out from her postings itself, and that change is written, but we're holding it back to give her time to fill it in her own index first. If she doesn't, we commit the fallback. She filled `doc_norm` (same lnc formula), `doc_len`, `links` and `city` in her index, so the fallback was dropped.

**idf uses log10 everywhere.** The lecture slides use log10(N/df), and so does this branch. The base doesn't change any ranking, but the idf numbers in the report should all be on the same scale, so we've asked Dhrithi to switch her `idf()` from the natural log as well.

**Index elimination relaxes instead of returning nothing.** Skipping common words and asking for most of the query words cuts how many articles get scored, which is the point. But on a short or unusual query that rule can leave too few articles, so it loosens one word at a time until there are k. Speed shouldn't cost the user an empty page.

**Champion lists fall back instead of coming up short.** r is fixed when the lists are built, so a query can end up with fewer than k contenders. Instead of returning a short page, the search then scores the full postings, which is Lecture 7's high list then low list. Ordering by weight + g(d) is built in so recency and PageRank can shape the lists later.

**Fresh news first, even over a slightly better older match.** The tiered search stops at the newest tier once it has k results, so an older article with a higher cosine can be left out. For news that's usually what a reader wants, and it means only a fraction of the articles get scored. Searching all tiers gives back the exact ranking when freshness doesn't matter.

**One requirements.txt for everyone.** It lists what every part of the code actually imports, so anyone can set up with one command. Plotting and dataset downloads are marked optional because the search engine runs without them.

**kal defaults to yesterday.** When a query has कल but no other clue, we treat it as yesterday, because news mostly reports what already happened.

**Weather words mean tomorrow.** "kal ka mausam" is almost always about tomorrow's weather, and weather news is mostly forecasts, so weather words count as a future cue.

**The kal boost multiplies the score instead of adding to it.** A flat bonus let an unrelated article jump to the top just because it was published on the right day. Multiplying keeps the boost proportional to how relevant the article already is.

**kal only reorders within a parser stage.** The query parser puts exact-phrase matches above looser ones on purpose. The kal boost respects that and only changes the order inside a stage, so a date boost can never push a stricter match below a looser one.

**Dangling articles spread their vote evenly.** Many news articles don't link to anything else we crawled. In PageRank their score is shared equally across all articles, so it isn't lost and the scores still add up to 1.

**Recency still leads in g(d).** The static score is 0.5 recency, 0.3 PageRank and 0.2 first to publish. For news, how fresh a story is matters most, and the link graph between crawled articles is sparse, so PageRank and the original flag adjust the order rather than dominate it.

**The original gets the credit, not the newest copy.** When several papers run the same wire story, recency alone favoured the latest copy. First-to-publish credit gives the boost to the article the others copied, so the original ranks first.

**Authority is on by default in the app and CLI.** The net score now uses the full g(d) (recency, PageRank and first to publish) unless it's switched off, so the original of a wire story ranks above its copies. The switch and `--no-authority` are there to show the difference side by side.

**Duplicates collapse to the best-ranked copy, not always the original.** Authority already pushes the original up, so usually it's the one kept. If a copy ranks higher for a query, that copy is shown and the original is listed under "also in", so the user sees the best match first.

**Ask for 3k results before collapsing.** Collapsing can remove results, so the ranker returns three times as many and the list is cut to k afterwards. That way the top k still has k different stories.

**Extra explain info stays out of the term scores.** lnc.ltc and BM25 return a plain dictionary of term scores. When kal or collapsing add their own info, the term scores move under "terms" so the app doesn't mistake "also in" for a query word.

**Speed-ups run on lnc.ltc in the app and CLI.** They're shortcuts for cosine scoring, so they're compared against plain lnc.ltc. The app keeps them off by default and shows how many articles were scored, so the saving is visible next to the results.

**Champion list size grows with the corpus.** r is N / 20 with a minimum of 5, so the lists stay useful on the 300-article sample and on the full crawl without retuning.

**Speed-ups are judged by overlap with full search.** For each speed-up the table shows how much of the exact top k it keeps and what share of articles it scored. That's the trade-off Lecture 7 describes, and it doesn't need relevance judgments, so it can be run as soon as the corpus is frozen.

**My information needs come from the real crawl.** Each of R01 to R08 is a story with at least two articles in Riya's 300-article sample, so every need has something to find. Several are the same story in many papers (the earthquake, Char Dham, Shreyas Iyer), which also tests duplicate collapsing, and the English forms test the translation step. Only the queries are in git, not the article text.

**Queries go through the same analyzer as the articles.** Each column's index was built with a different stemming mode, so the query is analyzed with that mode too. The stemming column gets stemmed query terms, and the three columns stay a fair comparison.

**Phonetic candidates come from the unstemmed vocabulary.** Viraja's k-gram index is built over the no-stemming index's words, so a Hinglish word is matched to a real Hindi spelling first and only then stemmed for each column.

**Tiny phonetic variants are dropped.** Variants with weight under 0.05 (like भूखंड for भूकंप) can't change the ranking but would pull unrelated articles into the "any word" stage, so they're left out.

**Letter case is handled on the query side for now.** The index keeps "Iyer" and "iyer" apart, so a Roman query word is matched to every spelling of it in the index. The proper fix is case folding in the normalizer.

**Fallback to the sample index.** If the real indexes aren't built, the app and CLI still run on the 20-article sample, and the tests always use it so they pass the same way on every machine.

**Dictionary grown from the information needs.** Words like earthquake, protest, detained, bypoll and pilgrims were added because the English forms of our needs use them. General news words only; nothing is copied from articles.

**Cluster pruning uses random leaders.** Lecture 7 picks sqrt(N) leaders at random: it's fast, and random picks land where the articles are dense. A fixed seed keeps the clusters the same between runs so results can be compared.

**Cluster pruning falls back like champion lists.** If the closest clusters give fewer than k results, the next-closest leader is added, so a query never comes back with a short page.

**On the sample, champion lists beat cluster pruning.** Cluster pruning scores the fewest articles but keeps only 28% of the exact top 10 with one cluster, while champion lists with r=5 keep 93% for about the same work. News clusters by story are small and specific, so a query's articles are often spread over several clusters. We'll recheck on the full crawl.

**All my results in one file.** `documentation/results/rishit-results.md` has a section per experiment, each saying which corpus, queries, k and date it used. Numbers from the 300-article sample are marked as early, so they don't get mixed up with the final numbers on the frozen corpus.

**Impact-ordered postings stop after a fixed number of articles.** Lecture 7 gives two ways to stop early: after a set number of articles, or when the weight drops below a threshold. On news the weights inside one word's list are close together, so the threshold hardly cuts anything. Stopping after 20 articles per word (or N / 15 on a bigger corpus) is the default; the threshold stays in the speed-ups table for comparison.

**Query words are read in decreasing idf.** The rarest words decide the ranking most, so their lists are read first, as the lecture suggests.

**High/low lists weren't built separately.** Champion lists already fall back to the full postings when they give fewer than k results, which is the high list / low list idea.

**multilingual-e5-small for dense re-ranking.** It's trained for search (not just sentence similarity), covers Hindi, and is small enough to embed the whole corpus on a laptop in seconds to minutes. LaBSE was about four times bigger for no clear gain here.

**Dense re-ranks, it doesn't retrieve.** lnc.ltc, BM25 or the net score pick the top 50 from the index, and dense only re-scores those. The IR part stays in charge of what's a candidate, which is what the assignment asks for, and there's no need for a vector index.

**Half first stage, half dense, both scaled.** e5's cosines are bunched together (about 0.75 to 0.85), so both scores are min-max scaled over the candidates before mixing. 0.5 each is a starting point; learning-to-rank can set it once there are judgments.

**Dense gets the query with Devanagari spellings added.** e5 does well with Hindi and English but not Roman Hindi: on its own "kal ka mausam" scored a cricket article above a weather one. Adding Viraja's confident Devanagari spellings and the translations fixes that. Phonetic spellings under 0.4 are left out because for English words they're often unrelated (farmers → हार्मोन्स).

**Dense is optional.** It needs torch and a model download, so it lives in `requirements-dense.txt`, and the app and CLI hide it when it isn't installed. Nobody else on the team has to install it.

**Working copy of the full crawl without the repeat.** Riya's file has one article saved twice, and Dhrithi's index builder stops on a repeated id. Rather than wait, we build from a local copy that keeps the later of the two. It stays out of git, and the indexes get rebuilt from Riya's cleaned file once she sends it.

**Speed-up numbers redone on the full crawl.** On 300 articles every speed-up looked close to exact because there was so little to skip. On 5,000 the differences are clear, so the full-crawl table is the main one and the 300-article table is kept only for comparison.

**Function words stay in the index.** The top-df words are all Hindi function words with idf near 0, so lnc.ltc already ignores them in practice. Keeping them means phrase queries like "भूकंप के झटके" still match exactly.

**Zipf slope reported over the frequent words.** The fit over every word is pulled down by the 40,255 words that appear once, so the report quotes the slope over the top 1,000 words (-0.90) next to the overall one (-1.45).

**RRF with c = 60.** That's the value from Cormack, Clarke and Buettcher (SIGIR 2009), which works well without tuning. RRF only looks at ranks, so lnc.ltc, BM25, the net score and e5 can be combined without making their scores comparable.

**The net score is one of the fused lists.** It already carries zones, proximity and authority, so fusing it with plain lnc.ltc and BM25 lets those signals count without hand-picking weights between the three.

**Dense only orders what the index found.** In fusion the dense list is the sparse lists' candidates ordered by e5 cosine, so dense never brings in an article that no IR ranker matched.

**Dense is used once, not twice.** With Fusion and Dense both on, dense is one of the fused lists and the separate re-ranking step is skipped.

**My to-do covers only my code.** Team items like the report, the video and the README, and the teammates' own task lists, are tracked by each person. My to-do keeps my code, my novelty with what's done and left, and only the outside things my code is waiting on.

**Experiments use the same query pipeline as the app.** The runner builds each query with `real_index.make_query` for the mode being tested, so the numbers describe the system people actually use, with Viraja's phonetic variants, translation and the right stemming.

**Queries come straight from the needs files.** Each person's needs file has a tsv block, and the runner reads those, so there's no separate query list to keep in sync.

**No judgments, no metrics.** Without judgments the runner writes the run files for pooling and the speed-ups table and stops, instead of printing zeros. The stop word experiment likewise only runs with real judgments.

**MMR with lambda 0.7.** Relevance still leads, and diversity only breaks near-ties between similar articles. Lower values started pulling weak matches into the top 10. It's off by default, since for most queries the plain ranking is what people want.

**MMR similarity from word vectors, not e5.** Plain log-tf vectors from the article text need no model and work for everyone, and they're good enough to spot two papers telling the same story.

**Difficulty from specificity, not clarity.** Clarity is the classic post-retrieval predictor, but on our news crawl vague queries land on near-identical listing pages that look very focused, so clarity ranked "news" as the clearest query. The highest per-word idf separated the needs queries from vague ones cleanly, so the flag uses that plus the parser stage.

**A word's idf comes from its most common strong spelling.** Roman spellings like "kya" are rare in a Devanagari corpus even when the word (क्या) is everywhere, so taking the exact spelling's idf made Hinglish filler look specific.

**The k-gram index knows document frequencies.** Viraja's `KGramIndex.from_index` carries each word's df, so among sound-alike spellings the common word wins ("modi" now goes to मोदी, not मोड़). On the full crawl rare spellings can still win when they're a closer letter match (भूकम्प over भूकंप), which is on her side to tune.

**The frozen corpus is the 5,000-article crawl as it is.** The team froze Riya's full crawl with only its one repeated article removed. HTML leftovers, astrology pages and section pages stay in and are reported as limitations, so every number in the report is on the same fixed set of articles.

**A sanity check before judgments.** The rubric accepts a clear sanity check for a partial system. Agreement between the four forms of a need tests the Track 5 claim directly (do Hindi, Hinglish and English find the same news?), and agreement between systems shows where judgments will matter most.

**Pool per need.** The pool merges every form of a need, so each article is judged once per need, which is how `formats.md` defines judgments and how the metrics read them. It cut 1,254 per-form pairs to 804.

**Judgments in git, one file per person.** Judgments are only ids and grades, so they're safe in a public repo. A file per person means four people judging at the same time never conflict, and `all_judgments()` merges them, keeping the higher grade on a disagreement.

**Judging page inside the app.** It's a second Streamlit page, so it uses the same index and setup as search, and nobody has to install anything else.

**Judgment-free numbers rerun when the query side changes.** Viraja's rare-spelling fix changed which Hindi words Hinglish queries reach, so the sanity check, speed-ups and difficulty numbers were rerun on the same frozen corpus. The results file notes the before and after for agreement.

**Translations learned from the corpus, under the hand-made dictionary.** Jagran's bilingual headlines give free aligned pairs. Dice 0.5 with at least 3 pairs keeps the errors low enough while still adding 272 words; the hand-made dictionary always wins, so a learned mistake can't override a known translation.

**Listing pages handled in ranking, not by removing them.** The corpus is frozen, so the 900 listing pages and 104 horoscopes stay in and get a low query-independent quality score instead. They still appear when nothing else matches (a plain "news" query), but never above a real article.

**Non-articles go after all articles, across parser stages.** A listing page contains almost every word somewhere, so it often reached the "all words" stage while the real articles only reached "some words". Demoting within a stage left it on top, so page type is checked before the parser stage.

**Page types saved as ids, decided with the URL.** The index has no URLs, and many listing headlines are plain ("मौसम", "उम्मीदवार"), so the corpus is classified once with the URL and only the ids of non-articles are kept in the repo.

**Pool rebuilt from the current runs.** The first pool came from runs made before Viraja's rare-spelling fix; 146 of today's top-10 articles weren't in it. It was rebuilt early, while only 23 pairs had been judged (all still in the new pool), so nothing was lost. Riya's per-need pooling gives the identical 641 articles.


**The evaluation runs push listing pages down.** The report should measure the system people use, so the runs demote listing pages like the app (rank 30, demote, keep 10). The pool only grows when runs change, so judgments already made are never thrown away.

**Randomization test as the main significance test.** It makes no assumption about how AP is distributed, which matters with a few dozen queries; the t-test is shown next to it for comparison.

**Logistic regression for learning to rank, tested leave-one-need-out.** With a few hundred judged pairs a handful of weights is all the data supports. Holding out whole needs (not single queries) stops the four forms of a need from leaking into each other's training data.

**One script for the whole part.** `scripts/rishit_results.py` runs the tests, every evaluation and a demo of the queries that show the system best (the same need in Hindi, Hinglish and English, the assignment's "kal ka mausam", date-aware kal, fusion, a low-confidence warning and one --explain run), so the results and the demo can be regenerated in one go before the report and video.

**Rocchio with idf-weighted vectors from real articles only.** Plain tf vectors made function words the top feedback terms, and listing pages in the feedback set pulled in unrelated words, so the feedback articles are the top real articles and their vectors are tf x idf.

**Feedback is off by default.** It helps queries whose first results are already right and hurts the rest; on our needs it lowered P@10 slightly, so it's a switch rather than part of the default pipeline.

**The report shows the net score losing to lnc.ltc.** The final evaluation found the hand-tuned net score significantly worse than plain lnc.ltc; we report it as found, and learning to rank shows which of its parts were miscalibrated (recency, proximity, parser stage).

**The 32-need evaluation is reported as provisional.** Only 57% of the pool is judged, and unjudged articles count as not relevant, so the ranker and stemming numbers are pulled down and blurred. They're kept in a separate provisional section; the 16-need section, judged in full at the time, stays as the complete one until the rest of the pool is judged.

**Judgments count only for pooled articles.** Judgments for articles none of our systems retrieves (129 of Dhrithi's 249) don't affect any metric, since metrics only look at what systems return; they're kept in the file but don't reduce the work left.

**Judging stops at 57% of the pool.** The team decided not to judge further. The 32-need evaluation is reported as final with that limitation stated: absolute scores are lower than with full judging, but every system is scored against the same judgments, so comparisons between them are fair.

**Stop words are kept.** Removing the 15 most common words changed nothing once idf was on, so they stay in the index and phrase queries keep working.
