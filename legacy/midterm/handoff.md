# P2 Handoff - Text Processing & Indexes

**Project:** Dhvani  
**Repository:** `Dhvani-lingual`  
**Branch:** `Dhrithi`  
**Owner:** Dhrithi  
**Current milestone:** Phase 1-3 complete + Selective AUTO complete; Phase 4 in progress  
**Latest pushed commit:** `da8ae0c`  
**Test status:** 99 passed, 0 failed

---

# 1. P2 Responsibility

P2 owns the text-processing and indexing layer of Dhvani.

The implemented P2 pipeline currently covers:

- Hindi Unicode normalization
- Hindi/Hinglish tokenization
- No-stemming analysis
- Light stemming
- Aggressive stemming foundation
- Selective/corpus-derived AUTO stemming
- Positional indexing
- Headline/body zones
- Document metadata
- Boolean AND search
- Smallest-postings-list-first optimization
- Skip pointers
- Phrase search
- Proximity search
- JSONL corpus ingestion
- Index serialization/loading
- Query-to-index integration
- Corpus-derived stopword analysis
- IDF calculation
- Document norms and document lengths
- Real-corpus validation

---

# 2. Current Architecture

The analysis pipeline is:

    Raw article/query
          |
          v
      normalize()
          |
          v
       tokenize()
          |
          v
       analyze(mode)
          |
          +---- none
          |
          +---- light
          |
          +---- aggr
          |
          +---- auto
          |
          v
    Positional Index
          |
          v
        Search

The supported analyzer modes are:

    none
    light
    aggr
    auto

The shared analyzer API is:

    analyze(text, mode)

It returns:

    [(term, position), ...]

---

# 3. Text Processing

## 3.1 `text/normalize.py`

Responsible for Unicode normalization.

Current behavior:

- NFC normalization
- Removes zero-width joiner
- Removes zero-width non-joiner
- Normalizes the specified Hindi spelling case
- Collapses repeated whitespace

The normalizer intentionally avoids broad anusvara replacement because generic conversion can corrupt valid Hindi words.

---

## 3.2 `text/tokenize.py`

Uses the `regex` package and Unicode-aware pattern:

    [\p{L}\p{M}\p{Nd}]+

This preserves:

- Devanagari letters
- Combining marks
- Unicode digits

while treating punctuation and whitespace as token boundaries.

---

## 3.3 `text/stem.py`

Implements the light Hindi stemmer.

Entry point:

    stem(word)

The implementation uses the project's light Hindi suffix inventory.

Examples include:

    लड़कियों -> लड़क
    लड़कियाँ -> लड़क
    किताबों -> किताब
    खाना -> खा
    खाता -> खा
    खाती -> खा

The light stemmer can occasionally produce undesirable lexical collisions. This is expected and is one reason AUTO/selective stemming was added.

---

## 3.4 `text/aggressive_stem.py`

Provides the current aggressive suffix-based stemmer.

Entry point:

    stem_aggressive(word)

This uses a larger suffix inventory than the light stemmer.

Important:

**This implementation is NOT YASS.**

It is currently only an experimental aggressive suffix-stripping stemmer. It should not be described as the YASS implementation.

The aggressive stemmer was deliberately not further tuned for the current milestone.

---

# 4. Selective / AUTO Stemming

## 4.1 Purpose

AUTO is a corpus-derived selective stemming mode.

The objective is not to stem every word aggressively. Instead, AUTO should:

1. Identify useful morphological classes from the corpus.
2. Detect classes where different surface forms map to a common light stem.
3. Check for corpus-observed collisions.
4. Keep only conservative candidate classes.
5. Apply stemming only to approved surface forms.

This avoids some false matches introduced by unrestricted light stemming.

---

## 4.2 Files

Relevant files:

    text/auto_stem.py
    scripts/analyze_stems.py
    scripts/analyze_stem_collisions.py
    scripts/analyze_auto_candidates.py

The candidate-generation pipeline is corpus-derived.

The current AUTO implementation uses:

    artifacts/auto_candidates.tsv

and the development corpus:

    data/dev_news.jsonl

These are local generated artifacts and are intentionally not committed to Git.

---

## 4.3 Conservative AUTO behavior

Examples from the current implementation:

    दिक्कतें -> दिक्कत
    दिक्कतों -> दिक्कत

but:

    विधानसभा -> विधानसभा
    विधानसभाओं -> विधानसभाओं

and:

    भारी -> भारी
    भारती -> भारती

This is intentional.

The light stemmer currently produces:

    भारी -> भार
    भारती -> भार

AUTO avoids this collision.

Similarly:

    विधानसभा -> विधानसभ

under light stemming, while AUTO keeps the natural surface form unchanged.

---

## 4.4 AUTO validation

The analyzer sanity test produced:

    NONE :
    [('दिक्कतें', 0), ('दिक्कतों', 1),
     ('विधानसभा', 2), ('विधानसभाओं', 3),
     ('भारी', 4), ('भारती', 5)]

    LIGHT:
    [('दिक्कत', 0), ('दिक्कत', 1),
     ('विधानसभ', 2), ('विधानसभ', 3),
     ('भार', 4), ('भार', 5)]

    AUTO:
    [('दिक्कत', 0), ('दिक्कत', 1),
     ('विधानसभा', 2), ('विधानसभाओं', 3),
     ('भारी', 4), ('भारती', 5)]

This confirms that AUTO is selectively applying useful stemming while avoiding known light-stemming collisions.

---

# 5. Index Implementation

## 5.1 `index/positional.py`

Main index class:

    Index(mode)

Supported modes:

    none
    light
    aggr
    auto

The index stores:

- document IDs
- term frequencies
- positional postings
- headline/body zones
- document metadata
- vocabulary
- document frequency
- document norms
- document lengths
- raw article text

---

## 5.2 Index API

The shared index contract is:

    idx = Index.load(mode)

    idx.postings(term, zone)
    idx.df(term)
    idx.N
    idx.vocab
    idx.doc_norm[doc_id]
    idx.doc_len[doc_id]
    idx.meta[doc_id]
    idx.text[doc_id]

Postings have the form:

    [
        (doc_id, tf, [positions]),
        ...
    ]

Zones:

    headline
    body

Metadata includes:

    source
    date
    state
    city
    section
    dup_of
    links

The positional structure must be preserved because it is required for phrase and proximity search.

---

# 6. Search Implementation

## `index/search.py`

Implemented:

### Boolean AND

    and_search(idx, terms, zone="body")

Only documents containing every query term are returned.

---

### Smallest-postings-first

The search implementation starts with the smallest postings list to reduce the candidate set early.

---

### Skip pointers

Implemented:

    build_skip_pointers()
    intersect_postings()

Skip pointers are used to accelerate postings-list intersections.

---

### Phrase search

Implemented:

    phrase_search(idx, terms, zone="body")

Uses positional information to identify consecutive query terms.

---

### Proximity search

Implemented:

    proximity_search(
        idx,
        terms,
        distance,
        zone="body"
    )

The current implementation treats the first query term as the anchor and checks whether other query terms occur within the supplied absolute positional distance.

---

# 7. Query Integration

## `index/query.py`

Main API:

    search_query(idx, query, mode=None, zone="body")

The function:

1. Validates the analysis mode.
2. Ensures the query mode matches the index mode.
3. Analyzes the raw query.
4. Extracts analyzed terms.
5. Runs Boolean AND search.

Supported modes:

    none
    light
    aggr
    auto

The existing comparison helper:

    search_both(idx_none, idx_light, query)

returns:

    {
        "none": [...],
        "light": [...]
    }

---

# 8. Index Building

## `index/build.py`

The builder reads JSONL articles.

Required fields:

    doc_id
    headline
    body

Relevant metadata fields:

    source
    date
    state
    city
    section
    dup_of
    links

Supported build modes:

    none
    light
    aggr
    auto

Build all supported indexes:

    python -m index.build --input data/dev_news.jsonl --mode all

Build a single mode:

    python -m index.build --input data/dev_news.jsonl --mode auto

Generated `.pkl` files are local build artifacts and should not be committed.

---

# 9. Real Corpus

A 1,000-article Hindi news development corpus from ILSUM-2.0 was converted into the Dhvani JSONL format.

Local corpus:

    data/dev_news.jsonl

The corpus is intentionally not committed to Git.

Earlier index statistics:

| Index | Documents | Vocabulary |
|---|---:|---:|
| none | 1,000 | 27,263 |
| light | 1,000 | 21,104 |
| aggr | 1,000 | 22,086 |

AUTO was subsequently built on the same development corpus.

---

# 10. Real-Corpus Query Validation

Representative results from the 1,000-document corpus:

| Query | NONE | LIGHT | AUTO |
|---|---:|---:|---:|
| बारिश | 110 | 110 | 110 |
| भारी बारिश | 3 | 3 | 3 |
| उत्तर प्रदेश | 7 | 9 | 7 |
| मौसम | 92 | 93 | 92 |
| तेज बारिश | 5 | 7 | 5 |
| लोगों की मौत | 7 | 7 | 7 |

These results demonstrate that AUTO does not simply reproduce LIGHT stemming.

Examples:

    उत्तर प्रदेश
    NONE = 7
    LIGHT = 9
    AUTO = 7

    तेज बारिश
    NONE = 5
    LIGHT = 7
    AUTO = 5

This supports the design goal of conservative selective stemming.

---

# 11. Positional / Phrase Validation

Real-corpus phrase tests were performed on the 1,000-document corpus.

Representative exact phrase results:

    भारी बारिश -> 46 documents
    उत्तर प्रदेश -> 41 documents
    जम्मू कश्मीर -> 27 documents
    तेज बारिश -> 20 documents

Representative proximity tests:

    बारिश + मौसम, distance <= 3 -> 25 documents
    बारिश + संभावना, distance <= 5 -> 15 documents
    उत्तर + प्रदेश, distance <= 5 -> 41 documents

Manual inspection of positional postings confirmed that adjacent terms receive consecutive positions.

---

# 12. Stopword Analysis

Corpus-derived stopword analysis has been implemented.

The conservative candidate set was selected using document frequency.

The current 90%-DF candidate list contains 15 terms:

    के
    में
    है
    की
    से
    को
    कर
    और
    पर
    का
    हैं
    रह
    हो
    ने
    भी

These are currently treated as stopword candidates for analysis/query work.

They remain indexed; removal from the index is not currently performed.

Relevant scripts include:

    scripts/analyze_stopwords.py
    scripts/build_stopword_candidates.py
    scripts/stopword_stats.py

---

# 13. IDF

IDF calculation is implemented in:

    index/scoring.py

Current formula:

    IDF(t) = log10(N / df(t))

where:

    N  = number of documents
    df = document frequency of the term

Examples from the development corpus:

    बारिश -> approximately 2.2073
    मौसम   -> approximately 2.3645
    सरकार  -> approximately 1.0051

Very common terms have IDF values close to zero.

The current scoring module provides:

    idf(index, term)
    idf_table(index, terms)

Final ranked retrieval is not yet implemented.

---

# 14. Document Norms

The index currently stores document norms using the lnc-style document weighting:

    w = 1 + log10(tf)

The norm is calculated across analyzed headline/body terms.

The index also stores analyzed document length.

These values are available through:

    idx.doc_norm
    idx.doc_len

They are intended for later ranked retrieval.

---

# 15. Tests

Current test status:

    99 passed
    0 failed

Run:

    python -m pytest -q

Tests cover:

- normalization
- tokenization
- light stemming
- aggressive stemming
- AUTO stemming
- analyzer modes
- invalid analyzer modes
- index construction
- postings
- document frequency
- document norms
- Boolean search
- skip pointers
- phrase search
- proximity search
- JSONL parsing
- malformed JSON
- missing fields
- metadata
- index building
- aggressive index building
- AUTO index building
- query integration
- none/light comparison

The latest full test run completed with:

    99 passed

---

# 16. Git Status / Repository State

Latest pushed branch:

    Dhrithi

Latest pushed commit:

    da8ae0c

The repository is currently configured so generated development data and serialized indexes are not committed.

`.gitignore` includes:

    __pycache__/
    *.py[cod]
    .pytest_cache/
    data/
    indexes/*.pkl

Do not commit:

    data/
    indexes/*.pkl

The local development corpus and generated indexes can be rebuilt when required.

---

# 17. Phase Status

## Phase 1

**Complete.**

Implemented:

- normalization
- tokenization
- light stemming
- initial analyzer pipeline

---

## Phase 2

**Complete.**

Implemented:

- positional indexes
- headline/body zones
- metadata
- Boolean AND
- smallest-postings-list-first
- skip pointers
- phrase search
- proximity search
- index builder
- query integration

---

## Phase 3

**Core work complete.**

Implemented:

- aggressive stemming foundation
- corpus-derived stopword analysis
- IDF
- document norms
- document lengths
- real-corpus testing
- stem collision analysis supporting selective stemming

Aggressive stemming should remain separate from YASS.

---

## Phase 4

**In progress.**

Completed Phase 4 work:

- selective/corpus-derived AUTO stemming
- AUTO candidate analysis
- AUTO collision analysis
- conservative AUTO mapping
- AUTO analyzer integration
- AUTO index
- real-corpus AUTO validation

Remaining Phase 4 work:

1. YASS implementation
2. Extended-biword / phrase indexing
3. Variable-byte compression
4. Gamma compression
5. Final/full H18 corpus rebuild
6. Any required integration with the final application

---

# 18. YASS

YASS has **not** been implemented yet.

Important:

    aggressive_stem.py != YASS

Do not describe the current aggressive suffix stemmer as YASS.

The YASS implementation should be based on the actual YASS method/reference rather than inventing a YASS-like heuristic.

---

# 19. Extended-Biword Index

Extended-biword indexing is still pending.

The existing positional index and phrase-search implementation should be preserved.

Do not replace positional postings with simple document-ID sets.

The extended-biword implementation should be added without breaking:

    idx.postings(term, zone)
    idx.df(term)
    idx.N
    idx.vocab
    idx.doc_norm
    idx.meta

---

# 20. Compression

Still pending:

- Variable-byte encoding
- Gamma coding

Compression should be implemented as an additional representation/utility rather than destroying the existing readable positional index representation.

The uncompressed index should remain available for debugging and phrase/proximity functionality.

---

# 21. Full Corpus

The current 1,000-document ILSUM corpus is a development/test corpus.

When the final H18 corpus is available:

1. Regenerate the corpus-derived AUTO candidates.
2. Regenerate AUTO mappings.
3. Rebuild all required indexes.
4. Run the complete test suite.
5. Run representative real queries.
6. Verify metadata and positional behavior.

Because AUTO is corpus-derived, its candidate files must correspond to the corpus used to build the AUTO index.

---

# 22. Phase 5 / Evaluation

Final retrieval evaluation is **not P2's responsibility**.

P2 should provide the indexing, analyzer, search primitives, and data required by the application/evaluation owner.

Do not expand P2 scope into final evaluation unless explicitly requested by the team.

---

# 23. Important Shared Contracts

Do not break the analyzer API:

    analyze(text, mode)

Current supported modes:

    none
    light
    aggr
    auto

Do not break the index API:

    Index.load(mode)

    idx.postings(term, zone)
    idx.df(term)
    idx.N
    idx.vocab
    idx.doc_norm
    idx.doc_len
    idx.meta
    idx.text

Postings must remain positional:

    (doc_id, tf, [positions])

The headline/body distinction must also be preserved.

These interfaces are shared with the rest of the Dhvani project.

---

# 24. Handoff Instructions

Before modifying the branch:

    git pull origin Dhrithi

Run tests:

    python -m pytest -q

Expected current result:

    99 passed

Generated artifacts should not be committed:

    data/
    indexes/*.pkl

When modifying shared analyzer/index contracts, coordinate with the other team members.

---

# 25. Current Checkpoint

**P2 has completed the core text-processing and indexing work through Phase 3 and has implemented the selective AUTO stemming portion of Phase 4.**

The current repository checkpoint is:

    da8ae0c

The next major P2 tasks are:

    1. YASS
    2. Extended-biword indexing
    3. Variable-byte compression
    4. Gamma compression
    5. Final H18 corpus rebuild

The branch is currently in a clean checkpoint before continuing Phase 4 development.
