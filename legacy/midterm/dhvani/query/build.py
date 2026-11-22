"""Building the query object the ranker scores (format 4 in documentation/formats.md).

``build_query`` is Viraja's half of the Viraja -> Rishit handoff. Right now it
does **exact matching only**: every word expands to itself with weight 1.0 and
source "exact". That is the H3 deliverable — a query stub Rishit can score
against the real query format from day one, replacing his temporary
``dhvani/rank/query_stub.py``.

Phonetic expansions (candidates from the k-gram index, re-ranked by learned edit
distance) are added by ``dhvani.query.expand`` when a k-gram ``index`` and edit
``costs`` are passed; cross-lingual ("xling") expansions are Rishit's. The
*shape* of the object does not change, so Rishit's ranker keeps working as we
grow it.
"""

from dhvani.query import langid
from dhvani.query.tokenize import word_tokens


def build_query(raw, index=None, costs=None, k=5):
    """Turn a raw query string into a query object (format 4).

    Returns::

        {"raw": raw,
         "tokens": [
            {"surface": <word>,
             "script": "devanagari" | "roman",
             "lang": {"hi": .., "hinglish": .., "en": ..},
             "expansions": [(<term>, <weight>, <source>), ...]},
            ...]}

    where ``source`` is one of "exact", "phonetic", "xling".

    With no ``index``/``costs`` it is exact-match only (the H3 stub): one
    ``(word, 1.0, "exact")`` expansion per token. Pass a ``KGramIndex`` and a
    learned-cost table ``(table, default)`` and each token also gets its top-``k``
    phonetic variants appended.
    """
    tokens = []
    for word in word_tokens(raw):
        tokens.append({
            "surface": word,
            "script": langid.script(word),
            "lang": langid.classify(word),
            "expansions": [(word, 1.0, "exact")],
        })
    query = {"raw": raw, "tokens": tokens}

    if index is not None and costs is not None:
        from dhvani.query.expand import expand_query
        expand_query(query, index, costs, k=k)
    return query
