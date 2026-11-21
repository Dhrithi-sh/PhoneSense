"""Phase 5 experiment: does phonetic expansion help full-query retrieval?

Runs the evaluation queries through the ranker twice — once with exact-match
query objects, once with phonetic expansion added (``build_query`` with a k-gram
index + learned costs) — and compares P@10 / nDCG@10 against the judged pools.
This is the "with vs without query expansion" results table.

It wires three teammates' parts together, so their modules are imported lazily
and the script explains what's missing instead of crashing:
  - Dhrithi's index  — ``index.positional.Index`` (built over the sample corpus)
  - Rishit's ranker  — ``dhvani.rank.vsm`` and metrics ``dhvani.eval.metrics``
  - the queries/qrels — Rishit's sample set under ``dhvani/eval/sample/``

Run once everyone's branches are together (or with them pulled locally):

    python -m dhvani.query.experiment
"""

import os

from dhvani.query.build import build_query
from dhvani.query.editdist import load_costs
from dhvani.query.kgram import KGramIndex

SAMPLE_DIR = os.path.join("dhvani", "eval", "sample")
EDIT_COSTS = os.path.join("dhvani", "query", "edit_costs.json")


def build_sample_index(mode="none"):
    """Build Dhrithi's index over Rishit's 20 sample articles (doc_ids match qrels)."""
    from dhvani.rank.sample_index import SAMPLE_ARTICLES
    from index.positional import Index

    index = Index(mode)
    for art in SAMPLE_ARTICLES:
        meta = {k: art.get(k) for k in ("source", "section", "state", "city", "date", "dup_of", "links")}
        index.add_document(art["doc_id"], art["headline"], art["body"], meta)
    return index


def load_queries(path):
    """Read ``queries.tsv`` -> list of ``(qid, need_id, form, query)`` (skips header)."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 4 and parts[0] != "qid":
                rows.append(tuple(parts))
    return rows


def run(index, queries, qrels, costs, kgram=None):
    """Return ``{form: {"exact": (P@10, nDCG@10), "expanded": (...)}, ...}``.

    ``qrels`` maps need_id -> {doc_id: grade}. Each query is ranked twice:
    exact-only, and with phonetic expansion (when ``kgram``/``costs`` given).
    """
    from dhvani.eval.metrics import ndcg_at_k, precision_at_k
    from dhvani.rank import vsm

    kgram = kgram or KGramIndex.from_index(index)
    buckets = {}
    for qid, need_id, form, text in queries:
        rel = qrels.get(need_id, {})
        if not rel:
            continue
        exact = [d for d, _s, _e in vsm.search(build_query(text), index, k=10)]
        expanded = [d for d, _s, _e in vsm.search(build_query(text, index=kgram, costs=costs), index, k=10)]
        b = buckets.setdefault(form, {"exact": [[], []], "expanded": [[], []]})
        b["exact"][0].append(precision_at_k(exact, rel, 10))
        b["exact"][1].append(ndcg_at_k(exact, rel, 10))
        b["expanded"][0].append(precision_at_k(expanded, rel, 10))
        b["expanded"][1].append(ndcg_at_k(expanded, rel, 10))

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        form: {
            which: (avg(vals[0]), avg(vals[1]))
            for which, vals in runs.items()
        }
        for form, runs in buckets.items()
    }


def run_rocchio(index, queries, qrels, costs, kgram=None, k=10, feedback_k=10):
    """Compare phonetic expansion with vs without Rocchio feedback on top.

    Returns ``{"no_rocchio": (P@10, nDCG@10), "rocchio": (...)}`` averaged over all
    queries. The baseline is the phonetically-expanded query; Rocchio then treats
    that run's top ``feedback_k`` docs as relevant and folds their terms back in.
    """
    from dhvani.eval.metrics import ndcg_at_k, precision_at_k
    from dhvani.query import rocchio as R
    from dhvani.rank import vsm

    kgram = kgram or KGramIndex.from_index(index)
    scores = {"no_rocchio": [[], []], "rocchio": [[], []]}
    for qid, need_id, form, text in queries:
        rel = qrels.get(need_id, {})
        if not rel:
            continue
        base = [d for d, _s, _e in vsm.search(build_query(text, index=kgram, costs=costs), index, k=k)]
        feedback = R.document_vectors(index, base[:feedback_k])
        fq = build_query(text, index=kgram, costs=costs)
        R.expand_query(fq, list(feedback.values()), k=10)
        rocchio = [d for d, _s, _e in vsm.search(fq, index, k=k)]
        scores["no_rocchio"][0].append(precision_at_k(base, rel, 10))
        scores["no_rocchio"][1].append(ndcg_at_k(base, rel, 10))
        scores["rocchio"][0].append(precision_at_k(rocchio, rel, 10))
        scores["rocchio"][1].append(ndcg_at_k(rocchio, rel, 10))

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {which: (avg(v[0]), avg(v[1])) for which, v in scores.items()}


def main(argv=None):
    from dhvani.eval.metrics import read_qrels

    queries = load_queries(os.path.join(SAMPLE_DIR, "queries.tsv"))
    qrels = read_qrels(os.path.join(SAMPLE_DIR, "qrels.txt"))
    index = build_sample_index("none")
    costs = load_costs(EDIT_COSTS)
    kgram = KGramIndex.from_index(index)

    results = run(index, queries, qrels, costs, kgram=kgram)
    print(f"{index.N} docs, {len(queries)} queries\n")
    header = f"{'form':<10}{'P@10 exact':>12}{'P@10 exp':>12}{'nDCG exact':>12}{'nDCG exp':>12}"
    print(header)
    print("-" * len(header))
    for form in sorted(results):
        ex, exp = results[form]["exact"], results[form]["expanded"]
        print(f"{form:<10}{ex[0]:>12.3f}{exp[0]:>12.3f}{ex[1]:>12.3f}{exp[1]:>12.3f}")

    roc = run_rocchio(index, queries, qrels, costs, kgram=kgram)
    print(f"\n{'rocchio':<12}{'P@10':>10}{'nDCG@10':>10}")
    print("-" * 32)
    for which in ("no_rocchio", "rocchio"):
        p, n = roc[which]
        print(f"{which:<12}{p:>10.3f}{n:>10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
