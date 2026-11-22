"""Builds a query object (see documentation/formats.md) with exact matches only.

This stands in for Viraja's query layer until it's ready. Every word maps to
itself with weight 1.0 and source "exact", so the ranker can be built and
tested against the real query format from day one.
"""

from dhvani.rank.sample_index import tokenize


def _script(word):
    if any("ऀ" <= ch <= "ॿ" for ch in word):
        return "devanagari"
    return "roman"


def exact_query(raw):
    tokens = []
    for word in tokenize(raw):
        script = _script(word)
        lang = {"hi": 1.0, "hinglish": 0.0, "en": 0.0} if script == "devanagari" else {"hi": 0.0, "hinglish": 0.5, "en": 0.5}
        tokens.append({
            "surface": word,
            "script": script,
            "lang": lang,
            "expansions": [(word, 1.0, "exact")],
        })
    return {"raw": raw, "tokens": tokens}
