# Porting from the midterm project

`legacy/midterm/` is a snapshot of our midterm project **Dhvani-lingual**
(<https://github.com/InformationRetrieval-Mid/Dhvani-lingual>, a Hindi/Hinglish news search
engine), cloned with `--depth 1` and with its `.git` removed. It is reference material only:
ruff excludes `legacy/` and pytest does not collect it (it has its own `conftest.py`,
`tests/` and `partwise-tests/`). Its `data/` folder (copyrighted article text) is gitignored.

Nothing is imported from `legacy/`. Logic was re-written into our modules, typed and
documented, and every ported module says "Adapted from our midterm project Dhvani-lingual
(`<original path>`)" in its docstring.

## T03 / T04

| Original file | New file | What changed |
|---------------|----------|--------------|
| `index/positional.py` (`Index`) | `phonesense/index/inverted.py` (`InvertedIndex`) | Same positional postings `(doc, tf, positions)` per zone. Zones `headline`/`body` became fields `title` / `features` / `description` / `text`. Internal int doc ids `0..N-1` with a two-way map to contract `doc_id`s (was string ids). Added `cf`, per-field doc lengths and averages (for BM25), a forward index (`term_vector`), `vocabulary()`. Pickle replaced by a versioned on-disk format (`meta.json`, per-field lexicon + postings) with an analyzer-config hash; `load()` rejects stale indexes. Fields are decoded lazily on load. The midterm stored lnc doc norms in the index; they now live in the TF-IDF retriever so BM25 etc. don't carry them. |
| `index/build.py` | `scripts/build_index.py` | Reads `phones.jsonl` (or the sample) through `phonesense.io`; duplicate-id check kept (now an error, the contract guarantees unique ids). Adds up-to-date check, `--force`, stats JSON with compressed vs uncompressed size. |
| `index/search.py` (`build_skip_pointers`, `intersect_postings`) | `phonesense/index/postings.py` (`skip_pointers`, `intersect`) | Same sqrt(n) skip spacing and skip-aware AND, on `Posting` / int doc ids. Phrase and proximity search from this file are left for T07. |
| — (no compression in the midterm) | `phonesense/index/postings.py` (`vb_encode`, `vb_decode`, `encode_postings`, `decode_postings`) | New: variable-byte encoding (IIR 5.3.1) of doc-id gaps, tfs and position gaps; numpy-vectorized decoder; plain uint32 format kept for the size comparison. |
| `text/analyzer.py` (`analyze(text, mode)`) | `phonesense/index/analyzer.py` (`Analyzer`) | Kept the shape: one analyzer for documents and queries, normalize -> tokenize -> optional stemmer variant, `(term, position)` output. Replaced all Hindi/Hinglish logic (script normalization, Hindi light/aggressive/auto stemmers) with English phone-listing rules: NFKC, unit joining, model-code-preserving tokenizer, built-in stopwords with protected words, Porter / light (plural) stemming that skips tokens with digits. Mode `none`/`light`/`aggr` survives as `stem: false` / `stemmer: light` / `stemmer: porter`. |
| `text/stopwords.py` | `phonesense/index/stopwords.py` | Hindi list replaced by an English list plus a `PROTECTED` set. |
| `dhvani/rank/vsm.py` (`query_vector`, `cosine_scores`, `search`), `index/scoring.py` (`idf`) | `phonesense/retrieval/tfidf.py` (`TfidfRetriever`) | Same lnc.ltc maths (`1 + log10 tf`, `log10(N/df)`, cosine normalization), term-at-a-time accumulators and heap top-k. Dropped: zone summing, query-expansion weights (phonetic/translation), per-term explanations, `use_idf` switch. Ties now broken by `doc_id` ascending (the midterm broke them by doc id descending via `nlargest`). Implements the shared `Retriever` API. |

## Not ported (don't fit English phone listings)

`text/normalize.py`, `text/stem.py`, `text/aggressive_stem.py`, `text/auto_stem.py`
(Hindi), `dhvani/query/{phonetics,roman,aksharantar,langid,names,...}.py` (Soundex,
transliteration), `dhvani/crawl/` (news crawler), recency tiers and `g(d)` in
`dhvani/rank/{scoring,speedups}.py`, `app/` (Streamlit).

## Candidates for later tasks

| Task | Original | Notes |
|------|----------|-------|
| T05 BM25 | `dhvani/rank/bm25.py` | Needs `idx.df`, `idx.postings`, `idx.doc_len`, `idx.avg_doc_len` (all available per field, so BM25F is possible). |
| T07 phrase / proximity | `index/search.py` (`phrase_search`, `proximity_search`), `dhvani/rank/scoring.py` (`smallest_window`) | Positions are in every posting; stopword gaps are kept identically for documents and queries. |
| T07 tolerant retrieval | `dhvani/query/{kgram,editdist}.py` | Use `idx.vocabulary(field)`. |
| T11 evaluation | `dhvani/eval/{metrics,significance}.py` | Adapt to `phonesense.io.read_run` / `read_qrels` and the `Results` schema. |
| Efficiency | `dhvani/rank/speedups.py` (index elimination, champion lists, impact ordering) | Optional extra for the efficiency table. |
