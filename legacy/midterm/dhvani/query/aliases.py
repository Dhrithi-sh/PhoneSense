"""A small hand-curated Roman -> Devanagari lexicon for terms the phonetic matcher
gets wrong on its own.

Two kinds of word defeat pure phonetics:
- irregular English spellings of Hindi words (``delhi`` has a silent "h" that
  inflates its edit distance to दिल्ली), and
- common homophones where corpus frequency points the wrong way (``iyer``
  sounds exactly like एयर "air", which is far more frequent than the surname
  अय्यर).

For these high-value proper nouns we keep an explicit mapping — a standard
query-synonym technique layered on top of the phonetic model. It stays small and
only fires when the mapped term actually exists in the index vocabulary.
"""

ALIASES = {
    "delhi": "दिल्ली",
    "dilli": "दिल्ली",
    "newdelhi": "दिल्ली",
    "iyer": "अय्यर",
    "ayyar": "अय्यर",
}
