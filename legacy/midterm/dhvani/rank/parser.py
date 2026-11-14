"""Query parser: turn one free-text query into a cascade of stricter queries.

Lecture 7's example: for "rising interest rates", first run it as a phrase.
If fewer than K articles contain the phrase, run the shorter phrases
"rising interest" and "interest rates". If there still aren't K, fall back
to the plain vector space query. Matching articles are ranked by score.

Our cascade, from strictest to loosest:

    1. phrase            all words next to each other, in order, exact spellings
    2. sub-phrase        any two neighbouring words next to each other
    3. all words         every word somewhere in the article (Boolean AND)
    4. all words, with variants
                         every word, allowing phonetic or translated variants
    5. any word          free text: at least one word (plain vector space)

We stop as soon as the stages so far have found k articles. An article
found at a stricter stage always ranks above one found at a looser stage;
within a stage, the chosen ranker's score decides.

Phrase and sub-phrase checks use positions from the positional index
(Lecture 2). AND intersects postings starting from the rarest word
(Lecture 1 query optimisation).
"""

from dhvani.rank.bm25 import search_bm25
from dhvani.rank.fusion import search_rrf
from dhvani.rank.scoring import rank
from dhvani.rank.vsm import ZONES, search

STAGES = ("phrase", "sub-phrase", "all words", "all words, with variants", "any word")

STAGE_LABELS = {
    "phrase": "Exact phrase",
    "sub-phrase": "Part of the phrase",
    "all words": "All words",
    "all words, with variants": "All words, with variants",
    "any word": "Some words",
}


def _alternatives(token, variants):
    """Terms that count as this token. Exact only, or every expansion."""
    if variants:
        return {term for term, _w, _s in token["expansions"]}
    exact = {term for term, _w, source in token["expansions"] if source == "exact"}
    return exact or {token["surface"]}


def _positions(index, terms):
    """{doc_id: {zone: set(positions)}} for any of the given terms."""
    found = {}
    for term in terms:
        for zone in ZONES:
            for doc_id, _tf, pos in index.postings(term, zone):
                found.setdefault(doc_id, {z: set() for z in ZONES})[zone].update(pos)
    return found


def phrase_docs(index, token_alts):
    """Articles where the tokens appear consecutively, in order, in one zone."""
    if not token_alts:
        return set()
    per_token = [_positions(index, alts) for alts in token_alts]
    # Only articles containing every token can contain the phrase.
    candidates = set(per_token[0])
    for p in per_token[1:]:
        candidates &= set(p)
    matches = set()
    for doc_id in candidates:
        for zone in ZONES:
            starts = per_token[0][doc_id][zone]
            if any(all(start + i in per_token[i][doc_id][zone] for i in range(1, len(per_token)))
                   for start in starts):
                matches.add(doc_id)
                break
    return matches


def and_docs(index, token_alts):
    """Articles containing every token, intersecting from the rarest first."""
    if not token_alts:
        return set()
    doc_sets = []
    for alts in token_alts:
        docs = set()
        for term in alts:
            for zone in ZONES:
                docs.update(d for d, _tf, _pos in index.postings(term, zone))
        doc_sets.append(docs)
    doc_sets.sort(key=len)          # shortest list first, so the result shrinks fastest
    result = doc_sets[0]
    for docs in doc_sets[1:]:
        if not result:
            break
        result = result & docs
    return result


def _is_feedback(token):
    """Tokens added by pseudo-relevance feedback (Rocchio), not typed by the user."""
    return bool(token["expansions"]) and all(s == "prf" for _t, _w, s in token["expansions"])


def stage_matches(index, query):
    """Run every stage and return [(stage, set_of_doc_ids)] in order.

    Feedback tokens from Rocchio aren't words the user typed, so they don't
    take part in the phrase and AND stages. They still count in "any word"
    and in the ranker's score.
    """
    all_tokens = query["tokens"]
    tokens = [t for t in all_tokens if not _is_feedback(t)] or all_tokens
    exact = [_alternatives(t, variants=False) for t in tokens]
    loose = [_alternatives(t, variants=True) for t in tokens]

    out = []
    if len(tokens) >= 2:
        out.append(("phrase", phrase_docs(index, exact)))
    if len(tokens) >= 3:
        sub = set()
        for i in range(len(exact) - 1):
            sub |= phrase_docs(index, exact[i:i + 2])
        out.append(("sub-phrase", sub))
    out.append(("all words", and_docs(index, exact)))
    out.append(("all words, with variants", and_docs(index, loose)))
    any_word = set()
    for alts in loose + [_alternatives(t, variants=True) for t in all_tokens if _is_feedback(t)]:
        for term in alts:
            for zone in ZONES:
                any_word.update(d for d, _tf, _pos in index.postings(term, zone))
    out.append(("any word", any_word))
    return out


def _score_all(query, index, ranker, doc_filter, static=None, dense=None):
    n = max(index.N, 1)
    if ranker == "rrf":
        results = search_rrf(query, index, k=n, depth=n, doc_filter=doc_filter, static=static, dense=dense)
    elif ranker == "net":
        results = rank(query, index, k=n, doc_filter=doc_filter, static=static)
    elif ranker == "lnc":
        results = search(query, index, k=n, doc_filter=doc_filter)
    else:
        results = search_bm25(query, index, k=n, doc_filter=doc_filter)
    return {doc_id: (score, explain) for doc_id, score, explain in results}


def parse_and_rank(query, index, k=10, ranker="net", doc_filter=None, static=None, dense=None):
    """Rank with the cascade. Returns [(doc_id, score, explain)], best first.

    explain gets a "stage" entry saying which stage first matched the
    article, and a "stages_run" entry listing the stages that were needed.
    """
    scored = _score_all(query, index, ranker, doc_filter, static, dense)

    picked = {}          # doc_id -> stage index where it first matched
    stages_run = []
    for i, (stage, docs) in enumerate(stage_matches(index, query)):
        stages_run.append(stage)
        for doc_id in docs:
            if doc_id in scored and doc_id not in picked:
                picked[doc_id] = i
        if len(picked) >= k:
            break

    ordered = sorted(picked, key=lambda d: (picked[d], -scored[d][0], d))[:k]
    results = []
    for doc_id in ordered:
        score, explain = scored[doc_id]
        # lnc.ltc and BM25 return a plain {term: contribution} dict. Keep the
        # term scores under "terms" so the stage info doesn't look like a term.
        explain = dict(explain) if "terms" in explain else {"terms": dict(explain)}
        stage_name = STAGES[STAGES.index(stages_run[picked[doc_id]])]
        explain["stage"] = stage_name
        explain["stages_run"] = list(stages_run)
        results.append((doc_id, score, explain))
    return results
