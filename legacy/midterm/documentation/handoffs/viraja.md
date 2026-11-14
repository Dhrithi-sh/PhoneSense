# Viraja handoff — the Hinglish layer (`dhvani/query/`)

Living notes on what my part gives the team, what I need back, and the calls I
made at forks in the road. If you rename something I expose, tell me first.

## What I expose right now (Phase 0–4)

All importable as `from dhvani.query.X import ...`:

| Module | Function | What it does |
|---|---|---|
| `langid` | `script(word)` | `"devanagari"` or `"roman"` |
| `langid` | `classify(word)` | `{"hi","hinglish","en"}` weight dict for one word |
| `roman` | `romanize(word)` | canonical Roman spelling of a Devanagari word, with the final-schwa drop (कमल→kamal) |
| `phonetics` | `soundex(word)` | classic lecture Soundex on a Roman string |
| `phonetics` | `dhvani_code(word)` | our Hindi Soundex; script-agnostic (मौसम / mausam / mosam → `585`) |
| `editdist` | `levenshtein(a,b)` | plain unit-cost edit distance |
| `editdist` | `learn_costs(pairs)` / `distance(a,b,table,default)` | learned edit distance (−log P, 2–3 EM re-aligns on Aksharantar) |
| `editdist` | `load_costs(path)` | loads `dhvani/query/edit_costs.json` (shipped, trained on 50k Aksharantar pairs) |
| `kgram` | `KGramIndex(vocab).candidates(word)` | k-gram candidate terms ranked by Jaccard |
| `match` | `weighted_variants(word, index, costs)` | top-5 `(term, weight, "phonetic")` variants — the production path |
| `expand` | `expand_query(query, index, costs)` | appends each token's phonetic variants to its `expansions` list |
| `context` | `correct(query, cooccur)` / `best_path(...)` | context correction: pick the variant combo that co-occurs most (Viterbi over a lattice) |
| `build` | `build_query(raw, index=None, costs=None)` | the **query object** (format 4). No index → exact-only (H3 stub). Pass a `KGramIndex` + costs → tokens also carry `"phonetic"` expansions |
| `rocchio` | `expand_query(query, relevant_vecs)` / `rocchio(...)` | Rocchio PRF: fold terms from the top results back into the query (adds `"prf"` tokens) |
| `names` | `load_names()` / `evaluate(names, costs)` | the 50-name test set + its accuracy@1 / MRR eval |
| `heatmap` | `render(table, default, out)` / `cost_matrix(...)` | edit-cost heatmap PNG from `edit_costs.json` |

Tests: `partwise-tests/viraja/` (**50 passing, 1 skipped** — the skip is the end-to-end ranker test, which runs once `dhvani/rank/` is present). Run `python -m pytest partwise-tests/viraja/`.

### Phase 4 — 50-name test set (150 variant queries)
| matcher | accuracy@1 | MRR |
|---|---|---|
| levenshtein | 0.973 | 0.987 |
| **soundex** | **1.000** | **1.000** |
| **dhvani (ours)** | **1.000** | **1.000** |
| learned | 0.993 | 0.997 |

Phonetic codes nail name variants (Lakshmi/Laxmi, Siddharth/Sidharth). Edit-cost heatmap: `documentation/figures/edit_cost_heatmap.png` (regenerate with `python -m dhvani.query.heatmap`).

### Phase 2 word-level results (Aksharantar test split, 1500 queries / 1156-word vocab)
| matcher | accuracy@1 | MRR |
|---|---|---|
| levenshtein | 0.895 | 0.921 |
| soundex | 0.912 | 0.935 |
| **dhvani (ours)** | **0.931** | **0.947** |
| learned | 0.906 | 0.934 |

Our Dhvani-code tops the table. Learned edit distance's cheapest edits come out as real Hinglish confusions (`q→k`, `z→j`, dropped schwa `a→∅`, `v→w`, `u→o`), rebuild with `python -m dhvani.query.evaluate`.

## Phase 5 so far (word-level done; full-query queued)
- **Word-level results:** `documentation/results/viraja-phonetic-results.md` — the
  four matchers compared on Aksharantar (dhvani 0.931 acc@1, top) and the 50-name
  set (soundex/dhvani 1.000), plus what the learned edit distance learned.
- **My 8 information needs** (4 forms each: hindi / hinglish / messy / english):
  `documentation/needs/viraja-needs.md`, ids `V01`–`V08`, ready to append to the
  group's `queries.tsv`.
- **Full-query experiment (done on the sample):** `dhvani/query/experiment.py`
  runs the eval queries through **Dhrithi's index** + **Rishit's ranker**, exact
  vs phonetic-expanded, scored with `dhvani/eval/metrics.py`. Headline result:
  Hinglish nDCG@10 **0.000 → 0.879** with expansion; Hindi unchanged (0.985).
  Table in `documentation/results/viraja-phonetic-results.md` §4. `test_experiment.py`
  `importorskip`s the three parts, so it skips here and runs once we're merged.
  Re-run `python -m dhvani.query.experiment` on the full corpus + 120 queries
  after judging.
- **Still queued:** Rocchio on/off table (needs a first retrieval over the real
  corpus; `rocchio.py` is ready).
- **Note on Dhrithi's layout:** her modules are top-level `index/` and `text/`
  (not under `dhvani/`), so `experiment.py` imports `index.positional`. If she
  moves them under `dhvani/`, that one import path updates.

## For Rishit

- `build_query(raw)` returns exactly the format-4 object your ranker scores. It
  is meant to **replace your temporary `dhvani/rank/query_stub.py`** — swap
  `from dhvani.rank.query_stub import exact_query` for
  `from dhvani.query.build import build_query` whenever you're ready. Same shape,
  so nothing downstream changes.
- **Phonetic expansion is live (Phase 3).** Call
  `build_query(raw, index=kgram_index, costs=load_costs(...))` and each token
  carries its `"phonetic"` variants alongside the `"exact"` one; call it with no
  args for the old exact-only stub. Object shape is unchanged, so your scoring
  code keeps working either way.
- **Verified end-to-end against your ranker.** `partwise-tests/viraja/test_integration.py`
  builds a query with `build_query` and scores it with `dhvani.rank.vsm.search`
  over your `SampleIndex`: `मौसम` and the Hinglish `mosam` (via phonetic
  expansion to मौसम) both land the weather docs. It `importorskip`s your rank, so
  it skips on my branch and runs once our parts are together.

## What I need from the team

- **From Dhrithi — the index vocabulary (`idx.vocab`), by ~H6.** The k-gram
  index (`KGramIndex`) takes any iterable of terms; Phase 2 is built and tested
  against the **Aksharantar word list** as a stand-in, so I'm **not blocked**.
  When her index lands I point `KGramIndex` at `idx.vocab` and re-run
  `test_kgram.py` — no code change beyond the vocab source.
- **From Rishit (Phase 3) — his `dhvani/rank/`** so I can run `build_query`
  output through his ranker end-to-end.
- **From the group — Rishit's Phase-0 `dhvani/` package should land on `main`.**
  See decision 1 below.

## Decisions I made at forks

1. **`dhvani/__init__.py` on my branch, identical to Rishit's.**
   Phase 0's "everyone can `import dhvani`" skeleton currently lives only on the
   `rishit` branch, not on `main`. Rather than block, I added a
   `dhvani/__init__.py` that is byte-for-byte Rishit's one-liner, plus
   `dhvani/query/__init__.py`. Because the shared file is identical, merging the
   branches causes **no conflict**. Ideal path: Rishit's Phase-0 setup is merged
   to `main`, then I `git pull` and my `__init__.py` is already in step.

2. **My own query tokenizer (`dhvani/query/tokenize.py`) as a stand-in.**
   Dhrithi's analyzer (format 2) isn't importable yet, and I need to tokenize
   *queries* regardless. I use the same agreed pattern `[\p{L}\p{M}\p{Nd}]+`
   (the `regex` module). Documents are still tokenized only by Dhrithi's
   analyzer — never by this.

3. **Romanization uses a fixed "aa/ii/uu" long-vowel scheme** (आ→aa, ई→ii, ऊ→uu;
   aspirates add "h"). So बारिश→`baarish`, का→`kaa`. This is a *canonical*
   spelling, not how a human would type it — the phonetic layer (Soundex /
   Dhvani-code / learned edit distance) bridges `kaa`↔`ka`. I only do the
   **final** schwa drop the plan asks for; medial schwa deletion is out of scope.

4. **Language ID is a small lexicon-based classifier, not trained.** Ambiguous
   words (main/to/hi/is/me/us/do/he/the/or) carry both readings at 0.5/0.5; the
   word lists are seeds in `langid.py`, easy to grow.

5. **Dhvani-code keeps the full consonant skeleton** (no 3-digit truncation like
   classic Soundex) and collapses adjacent duplicates. Variable length is more
   discriminative for a matching key; the digit labels are chosen so म=5, स=8 to
   hit the plan's `585`.

6. **Learned edit distance treats a match as free (cost 0), not −log P(x,x).**
   This makes it a proper distance (identical strings → 0); the learned table
   still governs every substitution/insertion/deletion.

7. **`edit_costs.json` is committed** under `dhvani/query/` (4.6 KB, trained on
   50k Aksharantar pairs). It's a small model artifact, not article text, so the
   matcher works out of the box without anyone re-downloading the 33 MB dataset.
   The dataset itself stays under git-ignored `data/`. Regenerate with
   `python -m dhvani.query.aksharantar` then `python -m dhvani.query.editdist`.

8. **Aksharantar loads from `hin.zip` directly, not `load_dataset(..., "hin")`.**
   The old per-language config was removed from the HF dataset; it now ships one
   zip per language, so `aksharantar.py` pulls `hin.zip` via `huggingface_hub`.

9. **Rocchio terms use a new `"prf"` source tag** (a 4th provenance beyond
   `exact`/`phonetic`/`xling`). **Rishit:** your ranker ignores the source string
   when scoring (`for term, weight, _source in ...`), so this is additive and
   safe; `"prf"` is only for snippet/`--explain` labelling. Flag if you'd rather
   I reuse an existing tag.

10. **PRF terms are appended as new tokens**, never merged into an existing
    token's `expansions`, so the original query tokens stay exactly as built.

11. **50-name set is committed reference data** (`dhvani/query/names_testset.tsv`,
    small, curated — not article text), and the heatmap PNG is committed under
    `documentation/figures/` as a report asset.

12. **Candidate generation now uses Dhvani-code, not just k-grams.** Fixed a bug
    where a name like `iyer` matched the wrong word (`श्रेयस`) because the right
    term `अय्यर` has near-zero k-gram overlap and got truncated from the pool
    before Dhvani-code could rank it. `KGramIndex.phonetic_candidates(word)` now
    always adds same-Dhvani-code terms to the pool, so phonetically-close but
    spelling-distant pairs are never missed.

13. **The exact self-match no longer starves the phonetic variants.** When the
    query word is itself in the index (e.g. "iyer" is an English word in some
    articles), `weighted_variants` used to leave it in the softmax, crushing the
    real Devanagari matches to ~0.0008 so they did nothing. It now excludes the
    *Roman* self-match (it's already the `"exact"` expansion) and gives
    same-Dhvani-code homophones a bonus, so अय्यर/एयर carry real weight. A
    Devanagari term that romanises to the query (कल for "kal") is **not** treated
    as a self-match — it's a real phonetic hit and kept.

15. **Real-corpus audit (83k-term vocab) — four fixes.** Stress-testing against
    the full 5k-article index surfaced issues the 20-doc sample never did:
    - **Speed:** candidate generation now counts shared k-grams first and scores
      Jaccard only on the top overlappers (a common bigram sits in thousands of
      terms). Per-word latency dropped from ~300–2800 ms to ~30–170 ms.
    - **Devanagari-only expansions:** a phonetic variant must be a Devanagari
      term, killing the Roman noise the real vocab is full of (laxmi→laxman,
      iyer→year, kal→kl).
    - **Corpus frequency (df):** `KGramIndex` takes a `df` map; among homophones
      the common corpus word wins (modi→**मोदी** not मॉड, kohli→**कोहली**).
    - **Tiered quality gate:** English-reading words need a *tight*
      transliteration to expand, so weather/earthquake/farmers expand to nothing
      while cricket→क्रिकेट, market→मार्केट and names still work; the gate applies
      to same-code candidates too (earlier they bypassed it).
    **Rishit:** build the k-gram index with `KGramIndex.from_index(load_index("none"))`
    (one line in `dhvani/rank/real_index.py:_phonetic_tools`) so df is carried —
    otherwise homophone ranking falls back to spelling-distance only.

16. **Rare spellings beating common ones — two more full-corpus fixes.**
    - **Homorganic anusvara:** `romanize` (and Dhvani-code) now turn an anusvara
      before a labial into "m" (भूकंप → "bhukamp"), elsewhere "n" (हिंदी). So
      भूकंप (35 docs) now matches "bhukamp" and df picks it over भूकम्प (1 doc).
    - **Soundex-keyed candidates:** `phonetic_candidates` unions Dhvani-code *and*
      Soundex matches, a second net for sound-alikes whose codes differ (delhi /
      दिल्ली share Soundex D400).
    - **Place names protected:** common Indian cities/states (delhi, mumbai,
      bihar…) are in `HINDI_PROTECT`, so they read as Hinglish and expand to their
      Devanagari form instead of being gated as English.
    Known hard cases left for the limitations section: "delhi" (the silent "h" in
    the English spelling inflates the edit distance to दिल्ली — "dilli" works
    perfectly), and "iyer" → एयर ("air", common) over the rarer surname अय्यर —
    genuine homophones that df resolves toward the corpus-common word.

14. **English words are detected properly and not phonetically expanded.** The
    old 40-word English seed meant real English words (farmers, snow, earthquake,
    worried) fell through to "Hinglish" and got phonetic junk (farmers → हामॉन्स).
    Now `langid` uses a bundled wordlist `dhvani/query/english_words.txt` (~48k
    common English words from wordfreq, minus a `HINDI_PROTECT` set so kal/ka/hai/
    chai/mausam stay Hindi). English-dominant tokens are skipped for expansion,
    and a quality gate drops any variant that isn't a genuine phonetic match. The
    wordlist is a static file, so there's **no runtime dependency** (wordfreq was
    only used to build it). Result: hinglish full-query nDCG rose 0.879 → 0.954
    and English stopped injecting junk. English→Hindi stays Rishit's xling job.

## Dependency note for packaging
Runtime + tests need `regex` and `pytest`. Re-training the edit costs needs
`huggingface_hub` (to pull `hin.zip`); the shipped `edit_costs.json` means
teammates don't need it just to run the matcher. The edit-cost **heatmap** needs
`matplotlib` (imported lazily — only `python -m dhvani.query.heatmap` needs it,
nothing else does). We still have no shared `requirements.txt` — someone should
add one and I'll list my deps there.
