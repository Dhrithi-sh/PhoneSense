# Dhvani P2 TODO

**Owner:** Dhrithi  
**Current status:** Phase 2 complete  
**Current tests:** 83 passed

---

# Phase 2

## COMPLETE

- [x] Hindi normalization
- [x] Hindi/Hinglish tokenization
- [x] None analyzer
- [x] Light analyzer
- [x] Aggressive analyzer foundation
- [x] Positional index
- [x] Headline zone
- [x] Body zone
- [x] Metadata
- [x] Document frequency
- [x] Boolean AND search
- [x] Smallest postings list first
- [x] Skip pointers
- [x] Phrase search
- [x] Proximity search
- [x] JSONL index builder
- [x] Index save/load
- [x] Query integration
- [x] None vs light comparison
- [x] Test suite

**Phase 2 result: 83 passed**

---

# Phase 3 - Stemming, Stop Words and Statistics

## Aggressive stemming

- [ ] Review aggressive suffix inventory
- [ ] Add more morphology-focused tests
- [ ] Compare light vs aggressive output
- [ ] Build stem-difference report

## Stop words

- [ ] Calculate document frequency for all terms
- [ ] Calculate corpus term frequency
- [ ] Calculate IDF
- [ ] Identify high-document-frequency terms
- [ ] Generate corpus-derived stop-word candidates
- [ ] Decide keep/remove/IDF-only strategy
- [ ] Save generated stop-word list

## IDF

- [ ] Implement IDF calculation
- [ ] Add configurable smoothing if required
- [ ] Produce IDF statistics
- [ ] Verify common terms have low IDF
- [ ] Verify rare terms have high IDF

## Zipf

- [ ] Generate term-frequency/rank data
- [ ] Generate Zipf plot
- [ ] Save plot for report
- [ ] Check Hindi news corpus distribution

---

# Phase 4 - Advanced P2 Features

## Selective stemming

- [ ] Build stem classes
- [ ] Measure document overlap within stem classes
- [ ] Split ambiguous classes
- [ ] Implement selective stemming
- [ ] Add `auto` analyzer mode
- [ ] Add auto index
- [ ] Test auto query behavior

## YASS

- [ ] Implement YASS corpus learning
- [ ] Generate YASS stem classes
- [ ] Integrate YASS into analyzer
- [ ] Add `yass` mode
- [ ] Build YASS index
- [ ] Evaluate YASS against light/aggressive

## Extended biword index

- [ ] Implement adjacent word pairs
- [ ] Add postposition-aware pairs
- [ ] Support:
  - [ ] का
  - [ ] की
  - [ ] के
  - [ ] में
  - [ ] से
  - [ ] पर
- [ ] Add phrase retrieval using biwords
- [ ] Compare phrase retrieval speed

## Compression

- [ ] Implement variable-byte encoding
- [ ] Implement gamma coding
- [ ] Compress document gaps
- [ ] Compress positions if practical
- [ ] Measure compressed index size
- [ ] Measure query-time impact

---

# Phase 5 - Evaluation

- [ ] Build 30 information needs
- [ ] Create 4 query forms per need
- [ ] Generate 120 evaluation queries
- [ ] Pool top 10 results
- [ ] Prepare relevance judgments
- [ ] Compare:
  - [ ] none
  - [ ] light
  - [ ] aggressive
  - [ ] YASS
  - [ ] auto
- [ ] Measure precision
- [ ] Measure recall where possible
- [ ] Measure retrieval speed
- [ ] Measure index size
- [ ] Compare phrase search speed
- [ ] Generate PR graphs
- [ ] Generate IDF/Zipf plots
- [ ] Generate edit/stem comparison visualizations

---

# Integration

- [ ] Connect final analyzer to P3 query layer
- [ ] Connect P4 ranking
- [ ] Verify shared query object
- [ ] Verify snippets/highlighting
- [ ] Verify three search columns:
  - [ ] no stem
  - [ ] stem
  - [ ] auto
- [ ] Verify UI displays match type
- [ ] Verify duplicate/wire-story metadata
- [ ] Build indexes from final corpus
- [ ] Run end-to-end Streamlit test

---

# Quality / Cleanup

- [ ] Run full pytest suite after every major change
- [ ] Keep shared interfaces stable
- [ ] Add documentation for every new public API
- [ ] Avoid committing generated indexes unless team agrees
- [ ] Keep crawled article content local
- [ ] Check `.gitignore`
- [ ] Remove accidental `__pycache__` files
- [ ] Keep commits focused