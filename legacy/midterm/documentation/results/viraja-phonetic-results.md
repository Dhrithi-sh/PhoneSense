# Phonetic matching results (Viraja)

Word-level evaluation of the Hinglish layer. These are the "phonetic methods
compared, word by word" numbers for the report; the full-query "with vs without
expansion" table (P@10 / nDCG on the real corpus) is added once Dhrithi's index
and the judged pools are ready.

All four matchers share the same pipeline: a k-gram index narrows the vocabulary
to candidates, then each matcher re-ranks them. Soundex and Dhvani-code bucket by
a phonetic key; Levenshtein and the learned edit distance rank by distance.

## 1. Aksharantar word-level test
Query = a human romanisation from the Aksharantar test split; the task is to
retrieve the correct Devanagari word. 1500 queries over a 1156-word vocabulary.

| Matcher | accuracy@1 | MRR |
|---|---|---|
| Levenshtein (plain) | 0.895 | 0.921 |
| Soundex (lecture) | 0.912 | 0.935 |
| **Dhvani-code (ours)** | **0.931** | **0.947** |
| Learned edit distance (ours) | 0.906 | 0.934 |

Reproduce:
```bash
python -m dhvani.query.aksharantar --max-train 50000 --max-test 3000
python -m dhvani.query.editdist data/aksharantar/train_pairs.tsv dhvani/query/edit_costs.json
python -m dhvani.query.evaluate data/aksharantar/test_pairs.tsv dhvani/query/edit_costs.json --limit 1500
```

**Takeaway.** Our Dhvani-code tops the table: a Hindi-aware phonetic key beats
the English-tuned lecture Soundex and plain Levenshtein. The learned edit
distance sits just behind on this clean word-retrieval task.

## 2. 50-name test set
Names are where phonetic matching matters most (Lakshmi / Laxmi, Siddharth /
Sidharth). 50 names, 150 variant queries; retrieve the canonical spelling.

| Matcher | accuracy@1 | MRR |
|---|---|---|
| Levenshtein (plain) | 0.973 | 0.987 |
| **Soundex (lecture)** | **1.000** | **1.000** |
| **Dhvani-code (ours)** | **1.000** | **1.000** |
| Learned edit distance (ours) | 0.993 | 0.997 |

Reproduce:
```bash
python -m dhvani.query.names
```

**Takeaway.** Phonetic codes nail name variants (both perfect here); edit
distance is a hair behind because a few variants are closer in letters to a
different name than to their own.

## 3. What the learned edit distance learned
The cheapest (most common) non-identity edits, from `edit_costs.json` trained on
50k Aksharantar pairs, are exactly the Hinglish confusions we expected:

| Edit | Cost | |
|---|---|---|
| `q → k` | 0.72 | qutub / kutub |
| `z → j` | 0.99 | zindagi / jindagi |
| `a → ∅` | 1.23 | dropped schwa (kamala → kamal) |
| `v → w` | 1.28 | vishnu / wishnu |
| `f → h` | 1.31 | — |
| `u → o` | 1.53 | mausam / mosam |

The full picture is the edit-cost heatmap:
`documentation/figures/edit_cost_heatmap.png` (darker = cheaper = more common).

## 4. Full-query retrieval: with vs without phonetic expansion
End to end through **Dhrithi's index** + **Rishit's ranker**, scored with
`dhvani/eval/metrics.py` against the judged pools. Preliminary run on the sample
set (20 docs, 10 queries); re-run on the full corpus + 120 queries once judging
is done (`python -m dhvani.query.experiment`).

| Query form | P@10 exact | P@10 expanded | nDCG@10 exact | nDCG@10 expanded |
|---|---|---|---|---|
| hindi | 0.240 | 0.240 | 0.985 | 0.985 |
| **hinglish** | 0.000 | **0.233** | 0.000 | **0.954** |
| english | 0.000 | 0.000 | 0.000 | 0.000 |

**Takeaway — this is the whole point of the Hinglish layer.** Devanagari queries
already work and expansion leaves them untouched (nDCG 0.985). Hinglish queries
match **nothing** on exact lookup (Roman text vs a Devanagari index) and phonetic
expansion lifts them to **0.954 nDCG** — the romanised query now reaches the
Hindi articles. English stays at 0 on purpose: the phonetic layer deliberately
does **not** touch English words (they have no Hindi homophone, so expanding them
only adds junk); English is the cross-lingual layer's job, not mine. (P@10 looks
low because each need has only 1–2 relevant docs in the 20-doc sample, so nDCG is
the meaningful metric here.)

## 5. Pending
- With vs without **Rocchio** query expansion (needs a first retrieval over the
  real corpus; code ready in `rocchio.py`).
- Re-run sections 4–5 on the full 5k-article corpus and the full 120 judged
  queries.
