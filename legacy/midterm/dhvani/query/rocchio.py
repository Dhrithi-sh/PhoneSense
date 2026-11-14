"""Rocchio query expansion via pseudo-relevance feedback.

After a first retrieval we *assume* the top results are relevant, pull their
strongest terms, and fold them back into the query. Rocchio moves the query
vector toward the relevant centroid (and away from a non-relevant one):

    q_new = α·q + (β/|Dr|)·Σ_{d∈Dr} d − (γ/|Dnr|)·Σ_{d∈Dnr} d

with negative weights truncated to 0 (standard Rocchio). For blind PRF we pass
the top-k docs as ``Dr`` and usually leave ``Dnr`` empty.

Vectors here are plain ``{term: weight}`` dicts, so this module needs neither the
index nor the ranker to be importable. ``document_vectors`` builds the doc
vectors from any index following the format-3 interface (``vocab`` + ``postings``)
when the real retrieval is wired in.
"""

import math
from collections import defaultdict

ALPHA, BETA, GAMMA = 1.0, 0.75, 0.15
ZONES = ("headline", "body")


def rocchio(query_vec, relevant, nonrelevant=(), alpha=ALPHA, beta=BETA, gamma=GAMMA):
    """Return the Rocchio-updated ``{term: weight}`` vector (negatives clipped)."""
    out = defaultdict(float)
    for term, w in query_vec.items():
        out[term] += alpha * w
    if relevant:
        for vec in relevant:
            for term, w in vec.items():
                out[term] += (beta / len(relevant)) * w
    if nonrelevant:
        for vec in nonrelevant:
            for term, w in vec.items():
                out[term] -= (gamma / len(nonrelevant)) * w
    return {term: w for term, w in out.items() if w > 0}


def top_terms(weights, k, exclude=()):
    """The ``k`` highest-weight terms not in ``exclude``, best first."""
    exclude = set(exclude)
    ranked = sorted(
        ((t, w) for t, w in weights.items() if t not in exclude),
        key=lambda tw: (-tw[1], tw[0]),
    )
    return ranked[:k]


def expand_query(query, relevant_vecs, k=10, nonrelevant_vecs=(), alpha=ALPHA, beta=BETA, gamma=GAMMA):
    """Append up to ``k`` PRF terms to ``query`` as new tokens, then return it.

    Each added term becomes its own token with a single expansion
    ``(term, weight, "prf")`` (see the handoff: "prf" is an additive provenance
    tag; the ranker ignores source when scoring, snippets use it for labelling).
    Terms already present in the query are skipped.
    """
    present = {t for tok in query["tokens"] for t, _w, _s in tok["expansions"]}
    qvec = defaultdict(float)
    for tok in query["tokens"]:
        for term, w, _s in tok["expansions"]:
            qvec[term] += w

    updated = rocchio(qvec, relevant_vecs, nonrelevant_vecs, alpha, beta, gamma)
    for term, weight in top_terms(updated, k, exclude=present):
        query["tokens"].append({
            "surface": term,
            "script": "devanagari" if any("ऀ" <= c <= "ॿ" for c in term) else "roman",
            "lang": {"hi": 1.0, "hinglish": 0.0, "en": 0.0},
            "expansions": [(term, round(weight, 4), "prf")],
        })
    return query


def document_vectors(index, doc_ids, zones=ZONES):
    """Build ``{doc_id: {term: log-tf weight}}`` for ``doc_ids`` from an index.

    Uses only the shared index interface (``vocab`` + ``postings(term, zone)``),
    so it works with Dhrithi's index or Rishit's sample index. Used when the real
    retrieval is wired in; unit tests pass hand-made vectors instead.
    """
    wanted = set(doc_ids)
    vecs = {d: defaultdict(float) for d in wanted}
    for term in index.vocab:
        for zone in zones:
            for doc_id, tf, _positions in index.postings(term, zone):
                if doc_id in wanted:
                    vecs[doc_id][term] += tf
    return {d: {t: 1 + math.log10(tf) for t, tf in v.items() if tf > 0} for d, v in vecs.items()}
