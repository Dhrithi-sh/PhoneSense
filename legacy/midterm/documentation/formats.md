# Shared formats

These are the 5 handoffs between our parts. Once we agree on them, nobody renames a field without telling the group, because it'll break someone else's code.

## 1. Article file
Riya writes it, Dhrithi reads it. Lives at `data/news.jsonl`, one article per line.

```json
{"doc_id": "jagran_23456789", "url": "https://...", "source": "jagran",
 "section": "weather", "state": "uttar-pradesh", "city": "lucknow",
 "date": "2026-10-06T13:21:43+05:30", "headline": "...", "body": "...",
 "keywords": ["मौसम", "बारिश"], "agency_flag": false,
 "content_hash": "a9f3...", "dup_of": null, "links": ["jagran_23450001"]}
```

- `date` is ISO format in IST.
- `dup_of` is the `doc_id` of the original if this is a near-duplicate, otherwise `null`.
- `links` are other crawled articles this one links to (for PageRank).
- No author names.

### Generation Commands
- **H3 Sample (300 articles):** `python -m dhvani.crawl.crawler --sample` (writes `data/news_sample_300.jsonl`)
- **Full Corpus (5,000 articles):** `python -m dhvani.crawl.crawler --max-articles 5000` (writes `data/news.jsonl`)

## 2. Analyzer
Dhrithi writes it. Everyone uses it, so text gets processed the same way everywhere.

```python
analyze(text, mode) -> [(term, position), ...]
# mode: "none" | "light" | "aggr" | "yass" | "auto"
```

## 3. Index
Dhrithi builds it, Viraja and Rishit read it.

```python
idx = Index.load(mode)
idx.postings(term, zone)  # -> [(doc_id, tf, [positions])], zone is "headline" or "body"
idx.df(term)
idx.N                     # number of articles
idx.vocab                 # all terms (Viraja needs this for matching)
idx.doc_norm[doc_id]      # vector length, for cosine
idx.meta[doc_id]          # source, date, state, section, dup_of
```

## 4. Query object
Viraja builds it, Rishit scores it.

```python
{"raw": "kal ka mosam",
 "tokens": [
   {"surface": "mosam", "script": "roman",
    "lang": {"hi": 0.0, "hinglish": 0.9, "en": 0.1},
    "expansions": [("मौसम", 0.82, "phonetic"), ("मोसम", 0.11, "phonetic")]},
   ...]}
```

The third value in each expansion says where it came from: `exact`, `phonetic` or `xling` (translated from English). The snippets and `--explain` use it.

## 5. Run file and judgments
Rishit writes the runs, Riya pools them, everyone judges.

- **Run file** (standard TREC format): `qid Q0 doc_id rank score run_name`
  e.g. `N07_hinglish Q0 jagran_23456789 1 0.8123 lightstem_dhvani`
- **Judgments:** `need_id 0 doc_id rel`, where rel is 0 (not relevant), 1 (partly) or 2 (fully).
