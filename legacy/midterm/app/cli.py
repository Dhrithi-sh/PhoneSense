"""Search from the terminal, with an optional step-by-step explanation.

    python app/cli.py "दिल्ली बारिश"
    python app/cli.py "दिल्ली बारिश" --explain
    python app/cli.py "कोहली शतक" --ranker bm25 --k 3 --explain

--explain prints every stage of the pipeline: the query object, the
query vector with its tf, df, idf and weights, the postings each term
touched, how many documents became candidates, the heap's top K, and a
score breakdown for each result.
"""

import argparse
import html
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dhvani.rank.bm25 import B, K1, bm25_scores, doc_lengths, idf as bm25_idf, search_bm25  # noqa: E402
from dhvani.rank.authority import static_scores  # noqa: E402
from dhvani.rank.collapse import collapse_duplicates, collapse_pool  # noqa: E402
from dhvani.rank.dense import DEFAULT_ALPHA, DEFAULT_DEPTH, DenseIndex, SentenceEncoder, dense_available, dense_rerank  # noqa: E402
from dhvani.rank.difficulty import predict  # noqa: E402
from dhvani.rank.diversify import DEFAULT_LAMBDA, diversify  # noqa: E402
from dhvani.rank.feedback import FEEDBACK_DOCS, feedback_query  # noqa: E402
from dhvani.rank.fusion import search_rrf  # noqa: E402
from dhvani.rank.quality import demote  # noqa: E402
from dhvani.rank.kal import apply_kal, kal_intent  # noqa: E402
from dhvani.rank.speedups import (  # noqa: E402
    ChampionLists,
    ClusterPruning,
    ImpactOrdered,
    RecencyTiers,
    search_champions,
    search_clusters,
    search_impact,
    search_index_elimination,
    search_tiered,
)
from dhvani.rank.parser import STAGE_LABELS, parse_and_rank, stage_matches  # noqa: E402
from dhvani.rank.real_index import load_index, make_query  # noqa: E402
from dhvani.rank.scoring import DEFAULT_WEIGHTS, rank  # noqa: E402
from dhvani.rank.vsm import ZONES, cosine_scores, log_tf, query_vector, search  # noqa: E402

RANKERS = ("net", "lnc", "bm25", "rrf")
MAX_POSTINGS_SHOWN = 6


def heading(out, title):
    out.append("")
    out.append(title)
    out.append("-" * len(title))


def table(out, headers, rows):
    """Plain left-aligned table, wide enough for every cell."""
    cols = list(zip(headers, *rows)) if rows else [[h] for h in headers]
    widths = [max(len(str(c)) for c in col) for col in cols]
    out.append("  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    out.append("  ".join("-" * w for w in widths))
    for row in rows:
        out.append("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))


def explain_query(out, query):
    heading(out, "1. Query object")
    rows = []
    for tok in query["tokens"]:
        lang = ", ".join(f"{k} {v:.2f}" for k, v in tok["lang"].items() if v)
        exps = ", ".join(f"{t} ({w:.2f}, {s})" for t, w, s in tok["expansions"])
        rows.append((tok["surface"], tok["script"], lang, exps))
    table(out, ("token", "script", "language", "expands to"), rows)


def explain_query_vector(out, query, index, ranker):
    if ranker == "bm25":
        heading(out, f"2. Query terms (BM25, k1 = {K1}, b = {B})")
        lengths = doc_lengths(index)
        avgdl = sum(lengths.values()) / len(lengths)
        out.append(f"N = {index.N}, average article length = {avgdl:.1f} tokens")
        weights = defaultdict(float)
        for tok in query["tokens"]:
            for term, w, _ in tok["expansions"]:
                weights[term] += w
        rows = []
        for term, w in weights.items():
            df = index.df(term)
            idf = f"{bm25_idf(df, index.N):.4f}" if df else "not in index"
            rows.append((term, f"{w:.2f}", df, idf))
        table(out, ("term", "query weight", "df", "idf = ln(1 + (N - df + 0.5) / (df + 0.5))"), rows)
        return

    heading(out, "2. Query vector (ltc: log tf x idf, then cosine-normalised)")
    out.append(f"N = {index.N}")
    qtf = defaultdict(float)
    for tok in query["tokens"]:
        for term, w, _ in tok["expansions"]:
            qtf[term] += w
    qvec = query_vector(query, index)
    rows = []
    for term, tf in qtf.items():
        df = index.df(term)
        if not df:
            rows.append((term, f"{tf:.2f}", 0, "-", "-", "dropped, not in index"))
            continue
        idf = math.log10(index.N / df)
        tf_w = log_tf(tf) if tf >= 1 else tf
        rows.append((term, f"{tf:.2f}", df, f"{idf:.4f}", f"{tf_w * idf:.4f}", f"{qvec[term]:.4f}"))
    table(out, ("term", "qtf", "df", "idf = log10(N/df)", "tf x idf", "normalised"), rows)


def explain_postings(out, query, index):
    heading(out, "3. Postings touched")
    terms = []
    for tok in query["tokens"]:
        for term, _, _ in tok["expansions"]:
            if term not in terms:
                terms.append(term)
    for term in terms:
        for zone in ZONES:
            plist = index.postings(term, zone)
            if not plist:
                continue
            shown = ", ".join(f"{d} tf={tf} pos={pos}" for d, tf, pos in plist[:MAX_POSTINGS_SHOWN])
            more = f", ... {len(plist) - MAX_POSTINGS_SHOWN} more" if len(plist) > MAX_POSTINGS_SHOWN else ""
            out.append(f"{term} [{zone}] ({len(plist)}): {shown}{more}")


def explain_candidates(out, query, index, ranker, k):
    heading(out, "4. Candidates and top K")
    if ranker == "bm25":
        scores, _ = bm25_scores(query, index)
    else:
        scores, _ = cosine_scores(query, index)
    out.append(f"{len(scores)} of {index.N} articles share at least one query term, so only those get scored.")
    out.append(f"The heap keeps the best {k} instead of sorting all {len(scores)}.")


def explain_parser(out, query, index, k):
    heading(out, "4b. Query parser (strictest stage first, stop once there are k)")
    found = set()
    for stage, docs in stage_matches(index, query):
        new = docs - found
        found |= docs
        out.append(f"{STAGE_LABELS[stage]:<26} {len(docs):>3} articles, {len(new):>3} new, {len(found):>3} so far")
        if len(found) >= k:
            out.append(f"Stopped here: {len(found)} >= k = {k}.")
            break


def explain_result(out, i, doc_id, score, explain, index, ranker, parts=None):
    article = getattr(index, "articles", {}).get(doc_id, {})
    meta = index.meta[doc_id]
    out.append("")
    stage = explain.get("stage")
    out.append(f"#{i}  {doc_id}  score {score:.4f}" + (f"  [{STAGE_LABELS[stage]}]" if stage else ""))
    if article:
        out.append(f"    {html.unescape(article.get('headline', ''))}")
    out.append(f"    {meta.get('source')} · {meta.get('section')} · {(meta.get('date') or '')[:10]}")
    terms = explain.get("terms", explain)
    if ranker == "rrf" and explain.get("rrf"):
        ranks = ", ".join(f"{name} #{r}" if r else f"{name} -" for name, r in explain["rrf"].items())
        out.append(f"    ranks: {ranks}")
        out.append(f"    rrf = sum of 1 / (60 + rank) = {explain['rrf_score']:.4f}")
    elif ranker == "net":
        w = DEFAULT_WEIGHTS
        out.append(f"    cosine            {explain['cosine']:.4f}")
        for term, part in terms.items():
            out.append(f"      {term:<14}  {part:.4f}")
        out.append(f"    + {w['zone']} x zone      {explain['zone']:.4f}")
        out.append(f"    + {w['prox']} x proximity {explain['proximity']:.4f}")
        if explain.get("static"):
            out.append(f"    + {w['recency']} x g(d)      {explain['recency']:.4f}")
            if parts:
                out.append(f"        g(d) = 0.5 x recency {parts['recency']:.4f} + 0.3 x PageRank {parts['pagerank']:.4f}"
                           f" + 0.2 x first to publish {parts['original']:.0f}")
        else:
            out.append(f"    + {w['recency']} x recency   {explain['recency']:.4f}")
        out.append(f"    = net score       {explain['net']:.4f}")
    else:
        label = "BM25 contribution" if ranker == "bm25" else "cosine contribution"
        for term, part in terms.items():
            out.append(f"      {term:<14}  {part:.4f}  ({label})")
    mmr = explain.get("mmr")
    if mmr:
        out.append(f"    mmr: relevance {mmr['relevance']:.2f}, most similar to an earlier pick {mmr['max_similarity']:.2f}")
    dense = explain.get("dense")
    if dense:
        out.append(f"    dense: e5 cosine {dense['cosine']:.4f}, first stage {dense['first_stage']:.4f} -> {dense['score']:.4f}")
    kal = explain.get("kal")
    if kal:
        out.append(f"    x kal boost ({kal['intent']}): x {1 + kal['weight'] * kal['boost']:.2f} = {score:.4f}")
    if explain.get("also_in"):
        out.append(f"    also in: {', '.join(explain['also_in'])} ({', '.join(explain['duplicates'])})")


def run(argv=None):
    parser = argparse.ArgumentParser(description="Search Dhvani from the terminal.")
    parser.add_argument("query", help="what to search for")
    parser.add_argument("--ranker", choices=RANKERS, default="net", help="net (default), lnc, bm25 or rrf (fusion of all of them)")
    parser.add_argument("--k", type=int, default=5, help="how many results to show")
    parser.add_argument("--stem", default="none", help="which index to use: none, light or auto")
    parser.add_argument("--explain", action="store_true", help="print every stage of the pipeline")
    parser.add_argument("--speedup", choices=("elim", "champions", "tiers", "clusters", "impact"),
                        help="score fewer articles: index elimination, champion lists, recent tiers, cluster pruning or impact-ordered postings (lnc.ltc)")
    parser.add_argument("--no-authority", action="store_true",
                        help="net score uses plain recency instead of recency + PageRank + first to publish")
    parser.add_argument("--no-collapse", action="store_true",
                        help="don't merge copies of the same wire story")
    parser.add_argument("--dense", action="store_true",
                        help="re-rank the top 50 with the multilingual e5 model (needs sentence-transformers)")
    parser.add_argument("--keep-listings", action="store_true",
                        help="don't push listing pages and horoscopes below real articles")
    parser.add_argument("--prf", action="store_true",
                        help="pseudo-relevance feedback: add terms from the top articles (Rocchio) and search again")
    parser.add_argument("--diversify", action="store_true",
                        help="re-order with MMR so the top k covers more different stories")
    parser.add_argument("--no-kal", action="store_true",
                        help="don't re-rank kal queries by date")
    parser.add_argument("--no-xling", action="store_true",
                        help="don't translate English words into Hindi")
    parser.add_argument("--no-parser", action="store_true",
                        help="skip the query parser and rank every article that shares a word")
    args = parser.parse_args(argv)

    index = load_index(args.stem)
    query = make_query(args.query, args.stem, xling=not args.no_xling)
    prf_terms = []
    if args.prf:
        n_before = len(query["tokens"])
        query, _feedback = feedback_query(query, index, args.stem, args.ranker if args.ranker != "rrf" else "net")
        prf_terms = [t["expansions"][0][0] for t in query["tokens"][n_before:]]
    out = [f'Query: "{args.query}"   ranker: {args.ranker}   index: {args.stem}   k: {args.k}']
    if prf_terms:
        out.append(f"Feedback (Rocchio, top {FEEDBACK_DOCS} articles) added: {' '.join(prf_terms)}")

    if args.explain:
        explain_query(out, query)
        explain_query_vector(out, query, index, args.ranker)
        explain_postings(out, query, index)
        explain_candidates(out, query, index, args.ranker, args.k)
        if not args.no_parser:
            explain_parser(out, query, index, args.k)

    static, parts = (None, {}) if args.no_authority else static_scores(index)
    pool = args.k if args.no_collapse else collapse_pool(args.k)
    if args.dense:
        pool = max(pool, DEFAULT_DEPTH)
    speed_stats = None
    dense = None
    if args.dense:
        if dense_available():
            dense = DenseIndex(index, SentenceEncoder())
        else:
            out.append("Dense re-ranking needs sentence-transformers: pip install -r requirements-dense.txt")
    if args.speedup == "elim":
        results, speed_stats = search_index_elimination(query, index, k=pool)
    elif args.speedup == "champions":
        champs = ChampionLists(index, r=max(5, index.N // 20), static_scores=static)
        results, speed_stats = search_champions(query, index, champs, k=pool)
    elif args.speedup == "tiers":
        results, speed_stats = search_tiered(query, index, RecencyTiers(index), k=pool)
    elif args.speedup == "clusters":
        results, speed_stats = search_clusters(query, index, ClusterPruning(index), k=pool)
    elif args.speedup == "impact":
        results, speed_stats = search_impact(query, index, ImpactOrdered(index), k=pool, max_docs=max(20, index.N // 15))
    elif not args.no_parser:
        results = parse_and_rank(query, index, k=pool, ranker=args.ranker, static=static, dense=dense)
    elif args.ranker == "rrf":
        results = search_rrf(query, index, k=pool, static=static, dense=dense)
    elif args.ranker == "net":
        results = rank(query, index, k=pool, static=static)
    elif args.ranker == "lnc":
        results = search(query, index, k=pool)
    else:
        results = search_bm25(query, index, k=pool)

    if speed_stats:
        heading(out, f"Speed-up: {speed_stats['method']}") if args.explain else None
        out.append(f"Scored {speed_stats['scored']} of {speed_stats['full_candidates']} articles that share a query word.")

    if dense is not None and args.ranker == "rrf" and not args.speedup:
        if args.explain:
            heading(out, "4b. Fusion")
            out.append("Dense (e5) is one of the fused lists: rrf = sum of 1 / (60 + rank) over lnc.ltc, BM25, net score and dense.")
    elif dense is not None:
        if args.explain:
            heading(out, "4b. Dense re-ranking")
            out.append(f"Top {DEFAULT_DEPTH} re-scored: {1 - DEFAULT_ALPHA} x first stage + {DEFAULT_ALPHA} x e5 cosine (both scaled to 0 to 1).")
        results = dense_rerank(results, query, dense)

    if not args.no_kal:
        intent = kal_intent(query)
        if args.explain and intent:
            heading(out, "4c. Date-aware kal")
            out.append(f"कल here means {intent}, so articles about that day get a boost: score x (1 + 0.5 x boost).")
        results = apply_kal(results, query, index)

    if not args.keep_listings:
        results = demote(results, index)

    if args.diversify:
        if args.explain:
            heading(out, "4d. Diversify (MMR)")
            out.append(f"Next pick = {DEFAULT_LAMBDA} x relevance - {1 - DEFAULT_LAMBDA:.1f} x highest similarity to articles already picked.")
        results = diversify(results, index)

    results = results[:args.k] if args.no_collapse else collapse_duplicates(results, index, k=args.k)

    hint = predict(query, results, index)
    if hint["low_confidence"]:
        out.append(f"Low confidence: {'; '.join(hint['reasons'])}.")
    if args.explain:
        out.append(f"Query difficulty: specificity {hint['specificity']:.2f} (flag below 1.0), "
                   f"scope {hint['scope']:.2f}, clarity {hint['clarity']:.2f} bits")

    heading(out, "5. Results" if args.explain else "Results")
    if not results:
        out.append("No matches.")
    for i, (doc_id, score, explain) in enumerate(results, start=1):
        if args.explain:
            explain_result(out, i, doc_id, score, explain, index,
                           "lnc" if args.speedup else args.ranker, parts.get(doc_id))
        else:
            headline = html.unescape(getattr(index, "articles", {}).get(doc_id, {}).get("headline", ""))
            stage = explain.get("stage")
            tag = f"  [{STAGE_LABELS[stage]}]" if stage else ""
            kal = explain.get("kal")
            if kal and kal["boost"]:
                tag += f"  [kal: {kal['intent']}]"
            if explain.get("also_in"):
                tag += f"  [also in: {', '.join(explain['also_in'])}]"
            if explain.get("page_type", "article") != "article":
                tag += f"  [{explain['page_type']}]"
            out.append(f"{i}. {score:.4f}  {doc_id}  {headline}{tag}")

    text = "\n".join(out)
    print(text)
    return text


if __name__ == "__main__":
    run()
