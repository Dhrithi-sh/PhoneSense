"""Context correction: pick the variant combination that fits together best.

A word's best spelling variant depends on its neighbours. Take ``kal ka mosam``:
each word has a few candidates from the matcher, and we want the *combination*
that actually co-occurs in the corpus (मौसम with कल, not some rarer homophone).

We lay the per-position candidates out as a lattice and run Viterbi to maximise

    Σ_i [ log(weight_i) + λ · log(1 + cooccur(prev_term, cur_term)) ]

``cooccur(a, b)`` is how often two terms land in the same document. It is
injected as a callable so this works against any source: Dhrithi's index
postings at integration, or a plain dict/counter in tests. Nothing here needs
the real index to be present.
"""

import math

_NEG_INF = float("-inf")


def best_path(candidates_per_position, cooccur, lam=1.0):
    """Viterbi over a candidate lattice; returns the chosen term per position.

    ``candidates_per_position`` is a list (one per query word) of
    ``[(term, weight), ...]``. ``cooccur(a, b)`` returns a co-occurrence count.
    """
    layers = [list(c) for c in candidates_per_position if c]
    if not layers:
        return []

    # Layer 0: score is just the candidate's own log-weight.
    score = {term: math.log(max(w, 1e-12)) for term, w in layers[0]}
    back = [{term: None for term, _w in layers[0]}]

    for i in range(1, len(layers)):
        new_score, new_back = {}, {}
        for term, weight in layers[i]:
            lw = math.log(max(weight, 1e-12))
            best, best_prev = _NEG_INF, None
            for prev_term, prev_score in score.items():
                s = prev_score + lw + lam * math.log(1 + cooccur(prev_term, term))
                if s > best:
                    best, best_prev = s, prev_term
            new_score[term] = best
            new_back[term] = best_prev
        score, _ = new_score, None
        back.append(new_back)

    # Backtrace from the best final term.
    last = max(score, key=score.get)
    path = [last]
    for i in range(len(back) - 1, 0, -1):
        path.append(back[i][path[-1]])
    path.reverse()
    return path


def correct(query, cooccur, lam=1.0):
    """Reorder each token's expansions so the context-chosen variant is first.

    Uses the already-built ``expansions`` of an expanded query as the candidate
    lattice (term + weight per candidate), runs ``best_path``, then moves the
    winning term to the front of each token's ``expansions`` list. The exact
    surface is always kept as a candidate. Returns the query.
    """
    lattice = [[(t, w) for t, w, _s in tok["expansions"]] for tok in query["tokens"]]
    chosen = best_path(lattice, cooccur, lam=lam)
    for token, pick in zip(query["tokens"], chosen):
        exps = token["expansions"]
        for idx, (term, _w, _s) in enumerate(exps):
            if term == pick and idx != 0:
                exps.insert(0, exps.pop(idx))
                break
    return query
