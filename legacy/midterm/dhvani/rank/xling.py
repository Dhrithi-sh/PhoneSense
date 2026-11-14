"""Cross-lingual layer: English query words become weighted Hindi terms.

This is dictionary-based query translation inside the vector space model.
An English word in the query is looked up in an English to Hindi dictionary,
and each Hindi translation is added to that word's expansions with source
"xling". The ranker then treats them like any other query term: they get
tf, idf and cosine weights in the Hindi term space, so "weather tomorrow"
is scored against the same Hindi terms as "कल का मौसम".

Choices:
- An English word's weight is split across its translations, so a word with
  three translations doesn't count three times as much as a word with one.
- Multi-word English entries ("prime minister", "stock market") are matched
  as phrases, longest first, and merged into one query token.
- English stop words ("the", "of", "in") are dropped instead of translated.
- The original English word stays as an "exact" expansion. It usually isn't
  in the Hindi index, so the query parser's exact stages miss it and its
  "with variants" stage picks up the translations.

Dictionaries:
- dhvani/rank/data/en_hi_news.tsv, a small news dictionary in the repo.
- dhvani/rank/data/en_hi_learned.tsv, pairs learned from Jagran's bilingual
  headlines by dhvani/rank/learn_dict.py. They only fill gaps: the hand-made
  news dictionary wins wherever both have a word.
- Optionally the MUSE English-Hindi dictionary (Conneau et al. 2018). If
  data/muse/en-hi.txt exists it's merged in under the other two.
"""

from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NEWS_DICT = Path(__file__).resolve().parent / "data" / "en_hi_news.tsv"
MUSE_DICT = REPO / "data" / "muse" / "en-hi.txt"
LEARNED_DICT = Path(__file__).resolve().parent / "data" / "en_hi_learned.tsv"

ENGLISH_STOP_WORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "from", "by", "with", "and", "or",
    "is", "are", "was", "were", "be", "been", "will", "about", "into", "over", "after", "before",
    "what", "when", "where", "which", "who", "how", "this", "that", "these", "those", "its", "it",
}


def load_news_dict(path=NEWS_DICT):
    """{english: [(hindi_term, weight), ...]} with weights summing to 1 per entry.

    A multi-word Hindi translation is split into its words, sharing that
    translation's weight.
    """
    entries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            english, _, rhs = line.partition("\t")
            options = []
            for part in rhs.split("|"):
                hindi, _, w = part.partition(":")
                options.append((hindi.strip(), float(w) if w else None))
            given = sum(w for _, w in options if w is not None)
            free = [h for h, w in options if w is None]
            share = (1.0 - given) / len(free) if free else 0.0
            weighted = defaultdict(float)
            for hindi, w in options:
                words = hindi.split()
                for word in words:
                    weighted[word] += (w if w is not None else share) / len(words)
            entries[english.strip().lower()] = _normalise(weighted)
    return entries


def load_muse_dict(path=MUSE_DICT):
    """MUSE format: one 'english hindi' pair per line. Equal weights per word."""
    path = Path(path)
    if not path.exists():
        return {}
    pairs = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                pairs[parts[0].lower()].append(parts[1])
    return {en: [(hi, 1.0 / len(his)) for hi in his] for en, his in pairs.items()}


def _normalise(weighted):
    total = sum(weighted.values())
    return [(term, w / total) for term, w in weighted.items()] if total else []


class Translator:
    def __init__(self, news_path=NEWS_DICT, muse_path=MUSE_DICT, learned_path=LEARNED_DICT):
        self.dictionary = load_muse_dict(muse_path)
        if learned_path and Path(learned_path).exists():
            self.dictionary.update(load_news_dict(learned_path))   # learned pairs fill gaps
        self.dictionary.update(load_news_dict(news_path))           # hand-made dictionary wins
        self.max_phrase = max((len(k.split()) for k in self.dictionary), default=1)

    def lookup(self, english):
        return self.dictionary.get(english.lower(), [])

    def translate_query(self, query):
        """Return a new query object with Hindi translations added.

        Only Roman-script tokens that could be English are touched; Hindi
        tokens pass through unchanged.
        """
        tokens = query["tokens"]
        out = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            english_prob = tok.get("lang", {}).get("en", 0.0)
            if tok["script"] != "roman" or english_prob <= 0:
                out.append(tok)
                i += 1
                continue

            # Longest dictionary phrase starting here, over consecutive English tokens.
            match_len, translations = 0, []
            for n in range(min(self.max_phrase, len(tokens) - i), 0, -1):
                span = tokens[i:i + n]
                if any(t["script"] != "roman" for t in span):
                    continue
                found = self.lookup(" ".join(t["surface"] for t in span))
                if found:
                    match_len, translations = n, found
                    break

            if not match_len:
                if tok["surface"].lower() in ENGLISH_STOP_WORDS and english_prob >= 0.5:
                    i += 1          # drop "the", "of" ... instead of searching for them
                    continue
                out.append(tok)
                i += 1
                continue

            span = tokens[i:i + match_len]
            surface = " ".join(t["surface"] for t in span)
            expansions = []
            for t in span:   # keep whatever the earlier layers already found
                expansions.extend(t["expansions"])
            have = {term for term, _w, _s in expansions}
            for hindi, w in translations:
                if hindi not in have:
                    expansions.append((hindi, w * english_prob, "xling"))
            out.append({
                "surface": surface,
                "script": "roman",
                "lang": dict(tok.get("lang", {})),
                "expansions": expansions,
            })
            i += match_len
        return {"raw": query["raw"], "tokens": out}


_DEFAULT = None


def translate(query):
    """Translate with the default dictionaries, loaded once."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Translator()
    return _DEFAULT.translate_query(query)
