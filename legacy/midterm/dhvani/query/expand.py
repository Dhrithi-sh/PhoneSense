"""Weighted query expansion: add phonetic variants to each query token.

Phase 1–2 gave us the pieces (k-gram candidates re-ranked by learned edit
distance); this is where they meet the query object. For each token we ask
``match.weighted_variants`` for the top-k spelling/phonetic variants and append
them to that token's ``expansions`` list as ``(term, weight, "phonetic")``. The
exact surface form stays as ``(surface, 1.0, "exact")``, so the ranker sees the
literal query *plus* its variants — e.g. ``mosam`` also pulls up ``मौसम``.

Cross-lingual (``"xling"``) expansions are Rishit's; we leave room for them and
never overwrite entries another source already added.
"""

from dhvani.query import match as M
from dhvani.query.aliases import ALIASES


def expand_token(token, index, costs, k=5):
    """Append up to ``k`` phonetic variants to one token's ``expansions`` list.

    We no longer skip "English" words up front: a name like "modi" or "kohli" is
    in any English wordlist yet must still reach मोदी / कोहली. Instead the matcher
    only returns Devanagari homophones that pass the quality gate, so a true
    English word with no Hindi homophone (farmers, weather) expands to nothing on
    its own, while names still reach their Devanagari form. Pure numbers and
    single characters have nothing to expand. Variants already present (the exact
    surface, or one added earlier) are skipped so weights never double up.
    """
    surface = token["surface"]
    if len(surface) < 2 or surface.isdigit():
        return token
    seen = {term for term, _w, _src in token["expansions"]}
    for term, weight, source in M.weighted_variants(surface, index, costs, k=k):
        if term not in seen:
            token["expansions"].append((term, round(weight, 4), source))
            seen.add(term)

    # Curated alias: for a few high-value proper nouns the phonetic matcher misses
    # (delhi -> दिल्ली, iyer -> अय्यर), force the correct Devanagari form to the
    # front with full weight, but only if it is actually in the index vocabulary.
    alias = ALIASES.get(surface)
    if alias and index is not None and alias in getattr(index, "canon", {}):
        token["expansions"] = [e for e in token["expansions"] if e[0] != alias]
        exact = [e for e in token["expansions"] if e[2] == "exact"]
        rest = [e for e in token["expansions"] if e[2] != "exact"]
        token["expansions"] = exact + [(alias, 1.0, "phonetic")] + rest
    return token


def expand_query(query, index, costs, k=5):
    """Expand every token of a query object in place, then return it."""
    for token in query["tokens"]:
        expand_token(token, index, costs, k=k)
    return query
