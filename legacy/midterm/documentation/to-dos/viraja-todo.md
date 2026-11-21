# Hinglish Phonetic Matching Layer
**Component:** P3 - Hinglish Layer (Language ID, Phonetic Matching & Query Expansion)
**Deliverable:** `build_query()` → the query object (following `documentation/formats.md`, format 4)

---

## 1. Scope & Matching Rules

### Target Capabilities
One search should hit the same articles whether typed as `कल का मौसम`,
`kal ka mausam`, `kal ka mosam`, or `weather tomorrow`. This layer makes the
*Hinglish* half work:
1. **Language ID** — label each word Hindi / Hinglish / English, with weights.
2. **Canonical Roman spelling** — one deterministic Latin form per Hindi word.
3. **Four phonetic matchers** — plain Levenshtein, lecture Soundex, Dhvani-code, learned edit distance.
4. **k-gram candidate search** — narrow the vocabulary before scoring edits.
5. **Context correction** — pick the variant combination that co-occurs most.
6. **Weighted query expansion** — add top-5 variants to the query (not the index).
7. **Rocchio** — pseudo-relevance feedback on the top-10 results.

### Data Sources
* **Aksharantar (Hindi)** — Hugging Face `ai4bharat/Aksharantar` (~33 MB).
  - **train split:** learn the edit costs for the learned edit distance.
  - **test split:** word-level matching evaluation (accuracy + MRR).
* **Index vocabulary** (`idx.vocab`) — needed by **H6** for the k-gram index; Aksharantar-only until then.
* **50-name test set** — our own labelled Indian names + spelling variants.

### Matching Guidelines
* **Expand, don't pollute:** top-5 weighted variants go into the *query*, never into the index.
* **Shape is frozen:** phonetic / cross-lingual matches are extra entries in each
  token's `expansions` list; the query-object shape in `formats.md` never changes.
* **No test leakage:** edit costs learned on the Aksharantar **train** split only;
  accuracy / MRR reported on the **test** split only.
* **Top-5 cap:** at most five variants per word, weighted by softmax of −distance.

---

## 2. Directory Structure
```text
dhvani/
└── query/
    ├── __init__.py
    ├── tokenize.py           # Query tokenizer (shared [\p{L}\p{M}\p{Nd}]+ pattern)
    ├── langid.py             # Per-word language ID (hi / hinglish / en, weighted)
    ├── roman.py              # Standard Roman spelling + final-schwa drop (कमल→kamal)
    ├── phonetics.py          # Soundex (lecture) + Dhvani-code (our Hindi Soundex)
    ├── editdist.py           # Plain Levenshtein + learned edit distance (−log P, EM re-align)
    ├── kgram.py              # k-gram index over the vocabulary; candidate generation
    ├── match.py              # Run the 4 matchers; re-rank candidates; pick top-5 variants
    ├── context.py            # Context correction for multi-word queries (candidate lattice)
    ├── expand.py             # Weighted expansion into the query object + Rocchio PRF
    └── build.py              # build_query() — assembles the final query object
```

---

## 3. Core Components & Implementation Design

### 1. Per-word Language ID (`langid.py`)
#### The Problem
A single query mixes scripts and languages: `kal ka weather`. We cannot label the
whole query one language; each word must be judged on its own, and words like
`main`, `to`, `hi`, `is`, `me`, `us` are valid English *and* common romanised
Hindi, so forcing a single choice loses recall.

#### Implementation Design
* Devanagari word → certainly Hindi `{hi: 1.0}`.
* Ambiguous word (both English and romanised Hindi) → keep **both** readings at
  `hinglish: 0.5, en: 0.5`.
* Known English → `en: 0.9, hinglish: 0.1`.
* Any other Roman word → treated as romanised Hindi `hinglish: 0.9, en: 0.1`.
* Weights always carry keys `{hi, hinglish, en}` and sum to 1.0. Word lists are
  small seeds in the module, grown as real queries appear.

---

### 2. Standard Roman Spelling (`roman.py`)
#### The Problem
To compare a user's romanised query against Hindi vocabulary we need **one**
canonical Latin spelling per Devanagari word. Hindi also deletes the silent
inherent "a" on a word's last consonant, so कमल is "kamal", not "kamala".

#### Implementation Design
* Deterministic transliteration: long vowels double (`आ→aa`, `ई→ii`, `ऊ→uu`),
  aspirates add "h" (`ख→kh`), matras replace the inherent vowel, virama removes it.
* **Final schwa deletion** — drop the trailing inherent "a" only: `कमल→kamal`,
  `मौसम→mausam`. Medial schwa deletion is out of scope.
* This is a *canonical* form, not how a human types it — the matchers below bridge
  `kaa`↔`ka`. Non-Devanagari input passes through unchanged.

---

### 3. Soundex — Lecture Baseline (`phonetics.py`)
#### How It Works
The classic algorithm from the lecture (Manning IIR §3.4), run on a Roman string:
keep the first letter, map the rest to digits (`bfpv→1`, `cgjkqsxz→2`, `dt→3`,
`l→4`, `mn→5`, `r→6`; vowels/`hwy→0`), collapse runs of the same digit, drop the
zeros, pad to four. `Robert`/`Rupert` → `R163`. Used both as one of the four
matchers and as a quick bucketing key.

---

### 4. Dhvani-code: Our Hindi Soundex (`phonetics.py`)
#### Why It's Feasible
Classic Soundex is tuned for English letters and keeps the first letter, so it
splits variants a Hindi speaker hears as identical. Dhvani-code fixes this and is
**script-agnostic** — it works on Devanagari *and* Roman input.

#### Implementation Design
* Map every consonant — Devanagari or Roman — to a shared phonetic class, drop the
  vowels, collapse doubled consonants.
* Result is a consonant skeleton: `मौसम`, `mausam`, `mosam` all reduce to **`585`**
  (म=5, स=8, म=5); `बारिश` ≡ `baarish` → `968`.
* Roman digraphs (`sh`, `ch`, `kh`, `th`, …) are parsed longest-first so each maps
  to one class. Full skeleton is kept (no 3-digit truncation) for sharper matching.

---

### 5. Learned Edit Distance (`editdist.py`)
#### Why It's Feasible
Plain Levenshtein charges every edit 1.0, but in Hinglish some confusions are far
more likely than others (`au`↔`o`, `v`↔`w`, `ii`↔`i`, a dropped schwa). We learn
those probabilities from Aksharantar and charge likely edits less, so true variants
score closer than coincidental ones.

#### Implementation Design (stochastic edit distance, Ristad & Yianilos 1998)
1. **Align:** for each Aksharantar pair, run a Levenshtein backtrace to get the
   character-level alignment (substitution / insertion / deletion).
2. **Count:** tally every operation over the whole train split.
3. **Cost:** each edit costs the negative log of its smoothed probability:
   $$\text{cost}(a \rightarrow b) = -\log \frac{c(a \rightarrow b) + 1}{\sum_{b'} c(a \rightarrow b') + V}$$
   (add-1 smoothing; `V` = alphabet size). Matches are near-free, rare edits dear.
4. **Re-align (EM):** re-run the alignments using the *new* costs, re-count,
   re-estimate — **2 or 3 passes** until the costs stabilise.
5. **Distance:** weighted-Levenshtein DP using the learned costs.

The learned cost table is also the input to the edit-cost heatmap (§10).

---

### 6. k-gram Candidate Index (`kgram.py`)
#### Why It's Feasible
Scoring a query word against the whole vocabulary by edit distance is too slow. A
k-gram index narrows the field to a handful of candidates first.

#### Implementation Design
* Build a `k=2`/`k=3`-gram → terms postings index over the vocabulary (romanised and
  Devanagari forms both indexed).
* For a query word, take its k-grams and retrieve terms sharing enough of them,
  ranked by **Jaccard overlap** on the k-gram sets:
  $$J(w, c) = \frac{|G_k(w) \cap G_k(c)|}{|G_k(w) \cup G_k(c)|}$$
* Keep the top-N (≈50) candidates and pass them to the matcher for re-ranking.

---

### 7. Four-Matcher Comparison & Re-ranking (`match.py`)
#### How It Works
```text
   query word  ──►  k-gram index  ──►  ~50 candidate terms
                                            │
             ┌──────────────────────────────┼──────────────────────────────┐
             ▼              ▼                ▼                ▼
      Levenshtein      Soundex         Dhvani-code     Learned edit dist
             └──────────────────────────────┬──────────────────────────────┘
                                            ▼
                      re-rank candidates; keep TOP-5 variants
                                            ▼
                     weighted expansions added to the query object
```
* All four matchers run the same way so they can be compared head-to-head.
* The production path re-ranks k-gram candidates by **learned edit distance**; the
  other three are kept for the comparison table.

#### Selection
Keep the **top-5** variants and turn distances into weights with a softmax over
negative distance:
$$w_i = \frac{e^{-d_i / T}}{\sum_{j} e^{-d_j / T}}$$
These become the token's phonetic expansions.

---

### 8. Context Correction (`context.py`)
#### Why It's Feasible
A word's best variant depends on its neighbours. For a multi-word query we pick the
**combination** of per-word variants that co-occurs most often in the corpus.

#### Implementation Design
* Build a candidate **lattice**: each query position holds its top-5 variants.
* Score a path by per-word match weight × neighbour co-occurrence (bigram counts
  from the index postings).
* Run **Viterbi** over the lattice to choose the best joint assignment, then
  redistribute expansion weight toward the chosen combination.

---

### 9. Weighted Expansion & Rocchio (`expand.py`, `build.py`)
#### Expansion
Fill each token's `expansions` list with `(term, weight, source)` entries:
`"exact"` for the surface form and `"phonetic"` for the top-5 matcher variants
(weighted), leaving room for the cross-lingual `"xling"` source. This is where
`build_query` grows from exact-only to full expansion — the object shape is unchanged.

#### Rocchio Pseudo-Relevance Feedback
After a first retrieval, treat the top-10 results as relevant and move the query
vector toward them:
$$\vec{q}_{\text{new}} = \alpha\,\vec{q} + \frac{\beta}{|D_r|}\sum_{d \in D_r}\vec{d} - \frac{\gamma}{|D_{nr}|}\sum_{d \in D_{nr}}\vec{d}$$
with typical `α=1.0, β=0.75, γ=0.15`. Added terms enter as extra weighted
expansions, so nothing downstream changes.

---

### 10. Edit-Cost Heatmap & Name Test Set
#### Heatmap
Render the learned substitution-cost table (§5) as a character-vs-character matrix,
showing which confusions are cheap (`au↔o`, `v↔w`, `ii↔i`). Saved as a figure for
the report and video.

#### Name Test Set
Assemble **50 Indian names** with real spelling variants (e.g. `Lakshmi`/`Laxmi`,
`Siddharth`/`Sidharth`) to measure how well each matcher retrieves the canonical
name from a messy spelling.

---

## 4. Contract Compliance: `documentation/formats.md`
`build_query(raw)` returns the query object the ranker scores (format 4). Exact-only
today; phonetic / xling matches are added as extra `expansions` entries later — same shape.
```json
{
  "raw": "kal ka mosam",
  "tokens": [
    {
      "surface": "mosam",
      "script": "roman",
      "lang": {"hi": 0.0, "hinglish": 0.9, "en": 0.1},
      "expansions": [
        ["mosam", 1.0, "exact"],
        ["मौसम", 0.82, "phonetic"],
        ["मोसम", 0.11, "phonetic"]
      ]
    }
  ]
}
```
* `expansions` entries are `(term, weight, source)` with `source ∈ {exact, phonetic, xling}`.
* `lang` always has keys `{hi, hinglish, en}` and sums to 1.0.
* At most five variants per token; weights are the softmax of −distance.

---

## 5. Test Suite (`partwise-tests/viraja/`)
* **`test_langid.py`:** script detection; Devanagari→hi; ambiguous words keep both readings; weights sum to 1.
* **`test_roman.py`:** final-schwa drop (कमल→kamal), matras, virama, independent vowels.
* **`test_phonetics.py`:** Soundex lecture values (Robert/Rupert→R163); Dhvani-code `585`; cross-script equality.
* **`test_build.py`:** query-object shape; exact-only expansion; script/lang filled in; punctuation stripped.
* **`test_editdist.py`:** a match costs ~0; frequent edits cost less than rare ones; distance is monotonic.
* **`test_kgram.py`:** candidate recall — the true variant is in the k-gram candidate set for a sample of Aksharantar pairs.
* **`test_match.py`:** **the word-level evaluation — accuracy@1 and MRR on the Aksharantar test split**, for all four matchers (the comparison-table source).
* **`test_context.py`:** a two-word query where context flips the best variant vs the per-word choice.
* **`test_names.py`:** retrieval accuracy / MRR on the 50-name set, per matcher.

---

## 6. Milestones & Checklist
* [x] **Phase 1 (H1–H3):**
  - Implement `langid.py`, `roman.py`, and Soundex + Dhvani-code in `phonetics.py`.
  - **Handoff (H3):** ship the exact-match **query stub** via `build_query()`.
* [x] **Phase 2 (H3–H8):**
  - `editdist.py` — plain Levenshtein + learned edit distance (3 EM re-aligns on 50k Aksharantar pairs); `edit_costs.json` shipped.
  - `kgram.py` candidate index (built against the Aksharantar word list; swap to Dhrithi's `idx.vocab` at H6).
  - `match.py` four-matcher comparison + top-5 re-ranking (`weighted_variants`).
  - `aksharantar.py` loader + `evaluate.py`: word-level acc@1 / MRR — dhvani 0.931, soundex 0.912, learned 0.906, levenshtein 0.895.
* [x] **Phase 3 (H8–H12):**
  - `expand.py` — weighted phonetic expansion wired into `build_query` (`source: "phonetic"`).
  - `context.py` — context correction (candidate lattice + Viterbi over co-occurrence).
  - **Integration:** `test_integration.py` scores `build_query` output through Rishit's `vsm.search` (skips until `dhvani/rank/` is present); Hindi `मौसम` and Hinglish `mosam`→मौसम both hit the weather docs.
* [x] **Phase 4 (H12–H22):** *(sleep shift H17–H22)*
  - `rocchio.py` — Rocchio top-10 pseudo-relevance feedback (adds `"prf"` terms).
  - `heatmap.py` — edit-cost heatmap → `documentation/figures/edit_cost_heatmap.png`.
  - `names.py` + `names_testset.tsv` — 50-name set; acc@1 soundex/dhvani 1.000, learned 0.993, levenshtein 0.973.
* [~] **Phase 5 (H22–H28):**
  - [x] Word-level results written up: `documentation/results/viraja-phonetic-results.md` (Aksharantar + 50-name tables).
  - [x] My 8 information needs (4 forms each): `documentation/needs/viraja-needs.md`.
  - [x] Full-query experiment (`experiment.py`): phonetic with vs without expansion through Dhrithi's index + Rishit's ranker — hinglish nDCG 0.000→0.879. Preliminary on the sample; re-run on the full corpus after judging.
  - [ ] Rocchio on/off table (code ready in `rocchio.py`; needs a real-corpus first retrieval).
  - [ ] Judge the pools with the group (~2 h).
* [ ] **Phase 6 (H28–H36):**
  - Write report section (phonetic matching and what's new about it).
  - Record video segment (0:00–0:50 the problem; 4:00–4:45 codes, candidates, heatmap, query expansion).
