"""Plugs the real pieces into the ranker: Dhrithi's index and Viraja's query layer.

- ``load_index(mode)`` loads Dhrithi's index from ``indexes/<mode>.pkl`` and
  gives it an ``articles`` view of its stored text, so the app, CLI and kal see
  the same thing as with the sample index. If the index files haven't been
  built yet it falls back to the 20-article sample index.
- ``make_query(raw, mode)`` builds the query the way the articles were built:
  Viraja's ``build_query`` (exact words plus phonetic variants from the k-gram
  index over the unstemmed vocabulary), then the cross-lingual layer, then
  every expansion term goes through Dhrithi's analyzer for that index mode, so
  a stemmed index gets stemmed query terms.

Build the indexes first (article text stays local, in ``data/``):

    python -m index.build --input data/news_sample_300.jsonl

Set DHVANI_INDEX=sample to force the sample index (the tests do this so they
don't depend on whatever corpus is on the machine).
"""

import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from dhvani.rank.sample_index import SampleIndex
from dhvani.rank.xling import translate

INDEX_DIR = Path("indexes")
COSTS_PATH = Path(__file__).resolve().parents[1] / "query" / "edit_costs.json"

# Phonetic variants below this weight are dropped: they only add near-misses
# like भूखंड for भूकंप that can never change the ranking.
MIN_PHONETIC_WEIGHT = 0.05


def real_index_available(mode="none"):
    if os.environ.get("DHVANI_INDEX") == "sample":
        return False
    return (INDEX_DIR / f"{mode}.pkl").exists()


@lru_cache(maxsize=None)
def load_index(mode="none"):
    """Dhrithi's index for this mode, or the sample index if it isn't built."""
    if not real_index_available(mode):
        return SampleIndex.load(mode)
    from index.positional import Index

    idx = Index.load(mode)
    idx.articles = idx.text           # {doc_id: {"headline", "body"}}, same shape as the sample index
    idx.lower_forms = _lower_forms(idx.vocab)
    return idx


def is_real(index):
    return not isinstance(index, SampleIndex)


def _lower_forms(vocab):
    """lowercase -> every spelling in the index, so "iyer" also finds "Iyer"."""
    forms = defaultdict(set)
    for term in vocab:
        forms[term.lower()].add(term)
    return forms


@lru_cache(maxsize=None)
def _phonetic_tools():
    from dhvani.query.editdist import load_costs
    from dhvani.query.kgram import KGramIndex

    # from_index carries document frequencies, so among sound-alike words the
    # common one wins (मोदी over मोड़).
    return KGramIndex.from_index(load_index("none")), load_costs(COSTS_PATH)


def _analyze_term(term, mode):
    from text.analyzer import analyze

    analyzed = analyze(term, mode)
    return analyzed[0][0] if analyzed else term


def analyze_query(query, index, mode):
    """Run every expansion through the analyzer for ``mode`` and fix letter case.

    Weights of terms that end up the same (two variants with one stem) are
    merged by keeping the higher one.
    """
    lower = getattr(index, "lower_forms", {})
    for token in query["tokens"]:
        merged = {}
        for term, weight, source in token["expansions"]:
            if source == "phonetic" and weight < MIN_PHONETIC_WEIGHT:
                continue
            stemmed = _analyze_term(term, mode)
            spellings = lower.get(stemmed.lower(), {stemmed}) if stemmed.isascii() else {stemmed}
            for form in spellings:
                if form not in merged or weight > merged[form][0]:
                    merged[form] = (weight, source)
        token["expansions"] = [(t, w, s) for t, (w, s) in merged.items()]
    return query


def make_query(raw, mode="none", phonetic=True, xling=True):
    """The full query pipeline for one index mode."""
    from dhvani.query.build import build_query

    index = load_index(mode)
    if not is_real(index):
        from dhvani.rank.query_stub import exact_query
        query = exact_query(raw)
        return translate(query) if xling else query
    kgram, costs = _phonetic_tools() if phonetic else (None, None)
    query = build_query(raw, index=kgram, costs=costs)
    if xling:
        query = translate(query)
    return analyze_query(query, index, mode)
