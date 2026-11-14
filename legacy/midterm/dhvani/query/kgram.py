"""k-gram index over the vocabulary, for fast candidate generation.

Scoring a query word against every vocabulary term by edit distance is too slow.
A k-gram index narrows the field first: index the character k-grams of each term
(on its canonical Roman form, so Devanagari and Roman queries meet on common
ground), then for a query word return the terms that share the most k-grams,
ranked by Jaccard overlap. The matcher (`match.py`) re-ranks those few candidates
by edit distance.

Vocabulary source: Dhrithi's ``idx.vocab`` once it exists; until then, the
Aksharantar word list. Either way it is just an iterable of terms, so swapping
the source changes nothing here.
"""

from collections import Counter, defaultdict

from dhvani.query.langid import script
from dhvani.query.phonetics import dhvani_code, soundex
from dhvani.query.roman import romanize


def _canonical(word):
    """Canonical Roman form used for indexing and lookup."""
    return romanize(word) if script(word) == "devanagari" else word.lower()


def kgrams(text, k=2):
    """Character k-grams of ``text`` with boundary markers.

    The ``$`` markers make the first/last characters matter, so ``mausam`` and
    ``xmausam`` don't look identical at the edges.
    """
    s = "$" + text + "$"
    if len(s) <= k:
        return {s}
    return {s[i:i + k] for i in range(len(s) - k + 1)}


class KGramIndex:
    """Maps character k-grams to the vocabulary terms that contain them."""

    def __init__(self, vocab, k=2, df=None):
        """``df`` (optional) maps term -> document frequency. When given, the
        matcher prefers common corpus words among homophones (मोदी over मॉड),
        which is the signal that separates a real name from a rare look-alike."""
        self.k = k
        self.postings = defaultdict(set)   # k-gram -> {terms}
        self._grams = {}                   # term -> frozenset(k-grams)
        self._by_code = defaultdict(set)   # Dhvani-code -> {terms}
        self._by_soundex = defaultdict(set)  # Soundex key -> {terms}
        self.canon = {}                    # term -> canonical Roman (cached)
        self.code = {}                     # term -> Dhvani-code (cached)
        self.df = dict(df) if df else {}   # term -> document frequency
        for term in vocab:
            canonical = _canonical(term)
            grams = kgrams(canonical, k)
            self._grams[term] = grams
            self.canon[term] = canonical
            code = dhvani_code(canonical)
            self.code[term] = code
            for g in grams:
                self.postings[g].add(term)
            self._by_code[code].add(term)
            sdx = soundex(canonical)
            if sdx:
                self._by_soundex[sdx].add(term)

    @classmethod
    def from_index(cls, index, k=2):
        """Build from an index exposing ``.vocab`` and ``.df(term)`` (Dhrithi's).

        Carries document frequencies so the matcher prefers common corpus words
        among homophones. Rishit: use this instead of ``KGramIndex(idx.vocab)``
        so names (मोदी, कोहली) win over rare look-alikes.
        """
        vocab = list(index.vocab)
        return cls(vocab, k=k, df={term: index.df(term) for term in vocab})

    def phonetic_candidates(self, word):
        """Terms that sound like ``word`` by Dhvani-code **or** Soundex.

        This is the safety net for phonetically-close but spelling-distant pairs
        whose k-gram overlap is tiny. Dhvani-code catches homophones like
        ``iyer`` / ``अय्यर`` (both ``26``); Soundex adds a second net for cases
        where the codes differ but the sound matches — ``delhi`` / दिल्ली share
        Soundex ``D400`` though their Dhvani-codes differ over the silent "h".
        """
        canonical = _canonical(word)
        out = set()
        code = dhvani_code(canonical)
        if code:
            out |= self._by_code.get(code, set())
        sdx = soundex(canonical)
        if sdx:
            out |= self._by_soundex.get(sdx, set())
        return out

    def candidates(self, word, limit=50):
        """Return ``[(term, jaccard), ...]`` for the ``limit`` best matches.

        Over a real 80k-term vocabulary a common bigram (``ar``, ``sh``) sits in
        thousands of terms, so computing Jaccard against the whole union is far
        too slow. Instead we first *count* how many query k-grams each term shares
        (cheap), keep only the top overlappers, and compute Jaccard just on those.
        """
        qg = kgrams(_canonical(word), self.k)
        shared = Counter()
        for g in qg:
            terms = self.postings.get(g)
            if terms:
                shared.update(terms)
        if not shared:
            return []

        # Score Jaccard only on the best overlappers (bounded work per query).
        best = shared.most_common(max(limit * 5, 200))
        scored = []
        for term, _count in best:
            tg = self._grams[term]
            jaccard = len(qg & tg) / len(qg | tg)
            scored.append((term, jaccard))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:limit]
