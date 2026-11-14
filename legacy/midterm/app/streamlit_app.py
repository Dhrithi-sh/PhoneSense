"""Dhvani search app.

Run from the repo root:
    streamlit run app/streamlit_app.py

Every query runs against three indexes side by side: no stemming, stemming,
and auto (stem a word only when it helps). Until the real indexes are ready
all three columns use the sample index, so they'll look the same.

Design follows Apple's guidelines: a translucent bar that content scrolls
under, a centred Spotlight-style search field, a segmented control for the
ranking model, filters one level deeper in a popover, inset grouped lists
for results, instant press feedback, dark mode, and reduced motion,
transparency and contrast support.
"""

import html
import sys
from pathlib import Path

import regex
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dhvani.rank.bm25 import search_bm25  # noqa: E402
from dhvani.rank.filters import field_values, make_filter  # noqa: E402
from dhvani.rank.authority import static_scores  # noqa: E402
from dhvani.rank.collapse import collapse_duplicates, collapse_pool  # noqa: E402
from dhvani.rank.dense import DEFAULT_DEPTH, DenseIndex, SentenceEncoder, dense_available, dense_rerank  # noqa: E402
from dhvani.rank.difficulty import predict  # noqa: E402
from dhvani.rank.diversify import diversify  # noqa: E402
from dhvani.rank.feedback import feedback_query  # noqa: E402
from dhvani.rank.fusion import search_rrf  # noqa: E402
from dhvani.rank.quality import demote  # noqa: E402
from dhvani.rank.kal import apply_kal  # noqa: E402
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
from dhvani.rank.parser import STAGE_LABELS, parse_and_rank  # noqa: E402
from dhvani.rank import real_index  # noqa: E402
from dhvani.rank.sample_index import SampleIndex, tokenize  # noqa: E402
from dhvani.rank.scoring import rank  # noqa: E402
from dhvani.rank.vsm import search  # noqa: E402

MODES = [("none", "No stemming"), ("light", "Stemming"), ("auto", "Auto")]

RANKERS = {"Net score": "net", "lnc.ltc": "lnc", "BM25": "bm25", "Fusion": "rrf"}

RANKER_NOTES = {
    "net": "Cosine similarity with headline, proximity and recency boosts.",
    "lnc": "Plain cosine similarity using SMART lnc.ltc weights.",
    "bm25": "Okapi BM25 with term saturation and length normalisation.",
    "rrf": "Reciprocal rank fusion of lnc.ltc, BM25 and the net score (and dense when it's on).",
}

SUGGESTIONS = ["भूकंप के झटके", "shreyas iyer shatak", "smriti mandhana captain", "chardham yatra record"]

MATCH_LABELS = {"exact": "Exact", "phonetic": "Phonetic", "xling": "Translated", "prf": "Feedback"}

SENTENCE_END = regex.compile(r"(?<=[।.!?])\s+")

STYLE = """
<style>
:root {
  --dv-page: #f5f5f7;
  --dv-group: #ffffff;
  --dv-text: #1d1d1f;
  --dv-secondary: #6e6e73;
  --dv-tertiary: #86868b;
  --dv-accent: #0071e3;
  --dv-separator: rgba(60, 60, 67, 0.14);
  --dv-fill: rgba(118, 118, 128, 0.12);
  --dv-bar: rgba(245, 245, 247, 0.72);
  --dv-row-hover: rgba(0, 0, 0, 0.025);
  --dv-row-press: rgba(0, 0, 0, 0.05);
  --dv-exact: rgba(52, 199, 89, 0.22);
  --dv-phonetic: rgba(0, 122, 255, 0.18);
  --dv-xling: rgba(255, 149, 0, 0.24);
  --dv-prf: rgba(175, 82, 222, 0.2);
  --dv-font: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display",
             "Kohinoor Devanagari", "Noto Sans Devanagari", "Segoe UI", system-ui, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root {
    --dv-page: #000000;
    --dv-group: #1c1c1e;
    --dv-text: #f5f5f7;
    --dv-secondary: #a1a1a6;
    --dv-tertiary: #8e8e93;
    --dv-accent: #2997ff;
    --dv-separator: rgba(84, 84, 88, 0.6);
    --dv-fill: rgba(118, 118, 128, 0.24);
    --dv-bar: rgba(22, 22, 23, 0.72);
    --dv-row-hover: rgba(255, 255, 255, 0.035);
    --dv-row-press: rgba(255, 255, 255, 0.07);
  }
}

/* Page and chrome */
.stApp { background: var(--dv-page); }
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
html, body, [class*="st-"], .stMarkdown, input, textarea, button, select {
  font-family: var(--dv-font) !important;
  -webkit-font-smoothing: antialiased;
}
[data-testid="stIconMaterial"], [data-testid="stIconMaterial"] * { font-family: "Material Symbols Rounded" !important; }
.block-container { padding-top: 6.5rem !important; padding-bottom: 5rem; max-width: 1200px; }

/* Translucent top bar; content scrolls underneath it. */
.dv-bar {
  position: fixed; top: 0; left: 0; right: 0; z-index: 999990; height: 52px;
  display: flex; align-items: center; justify-content: center;
  background: var(--dv-bar);
  -webkit-backdrop-filter: saturate(180%) blur(20px); backdrop-filter: saturate(180%) blur(20px);
}
.dv-bar::after {   /* scroll edge: a soft fade instead of a hard divider */
  content: ""; position: absolute; left: 0; right: 0; bottom: -12px; height: 12px;
  background: linear-gradient(var(--dv-bar), transparent); pointer-events: none;
}
.dv-bar-inner {
  width: 100%; max-width: 1200px; padding: 0 1.5rem;
  display: flex; align-items: center; justify-content: space-between;
}
.dv-wordmark { font-size: 1.15rem; font-weight: 700; letter-spacing: -0.02em; color: var(--dv-text); }
.dv-bar-meta { font-size: 0.78rem; color: var(--dv-tertiary); }
.dv-bar-right { display: flex; align-items: center; gap: 1.1rem; }
.dv-bar-link { font-size: 0.82rem; color: var(--dv-accent, #0071e3) !important; text-decoration: none !important; }
.dv-bar-link:hover { text-decoration: underline !important; }

/* Hero: big type with tight leading and negative tracking, centred. */
.dv-hero { text-align: center; margin: 0 auto 2rem; max-width: 760px; }
.dv-eyebrow { font-size: 0.95rem; font-weight: 600; color: var(--dv-accent); margin-bottom: 0.6rem; }
.dv-title {
  font-size: clamp(2.6rem, 6vw, 4.2rem); font-weight: 700; line-height: 1.04;
  letter-spacing: -0.035em; color: var(--dv-text); margin: 0;
}
.dv-subtitle {
  font-size: 1.3rem; line-height: 1.4; letter-spacing: -0.012em;
  color: var(--dv-secondary); margin: 1rem auto 0; max-width: 34rem;
}

/* Spotlight-style search field */
div[data-testid="stTextInput"] { max-width: 680px; margin: 0 auto; }
div[data-testid="stTextInput"] label { display: none; }
div[data-testid="stTextInputRootElement"] {
  border-radius: 16px !important; border: 1px solid transparent !important;
  background: var(--dv-group) !important;
  box-shadow: 0 1px 1px rgba(0,0,0,0.03), 0 10px 30px rgba(0,0,0,0.07);
  transition: box-shadow 160ms ease-out, border-color 160ms ease-out;
}
div[data-testid="stTextInputRootElement"]:focus-within {
  border-color: var(--dv-accent) !important;
  box-shadow: 0 0 0 4px rgba(0,113,227,0.2), 0 10px 30px rgba(0,0,0,0.07);
}
div[data-testid="stTextInput"] input {
  background: transparent !important; font-size: 1.3rem !important; letter-spacing: -0.012em;
  padding: 1rem 1.25rem 1rem 3.1rem !important; color: var(--dv-text);
}
div[data-testid="stTextInputRootElement"]::before {   /* magnifying glass */
  content: ""; position: absolute; left: 1.15rem; top: 50%; width: 1.15rem; height: 1.15rem;
  transform: translateY(-50%); opacity: 0.45; pointer-events: none; z-index: 1;
  background: no-repeat center/contain url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2386868b' stroke-width='2.4' stroke-linecap='round'><circle cx='10.5' cy='10.5' r='6.5'/><path d='m15.5 15.5 5 5'/></svg>");
}
div[data-testid="stTextInputRootElement"] { position: relative; }

/* Suggestion pills and the segmented control, centred under the search. */
div[data-testid="stElementContainer"]:has(> div[data-testid="stButtonGroup"]) { align-self: center; }
div[data-testid="stButtonGroup"] { display: flex; justify-content: center; }

/* Pills: quiet grey fill, no outline, no red. */
button[data-variant="pills"] {
  background: var(--dv-fill) !important; border: none !important; color: var(--dv-text) !important;
  border-radius: 999px !important; font-size: 0.85rem !important; padding: 0.3rem 0.95rem !important;
  min-height: 0 !important; box-shadow: none !important;
  transition: transform 100ms ease-out, background 120ms ease-out;
}
button[data-variant="pills"]:hover { background: var(--dv-separator) !important; }
button[data-variant="pills"][data-selected="true"] { background: var(--dv-text) !important; color: var(--dv-page) !important; }
button[data-variant="pills"] p { color: inherit !important; }

/* Segmented control: grey track, white "thumb" on the selected item. */
div[role="radiogroup"]:has(> button[data-variant="segmented_control"]) {
  background: var(--dv-fill); border-radius: 9px; padding: 2px; gap: 0 !important;
}
button[data-variant="segmented_control"] {
  background: transparent !important; border: none !important; color: var(--dv-text) !important;
  font-size: 0.85rem !important; font-weight: 500 !important; padding: 0.3rem 1.2rem !important;
  border-radius: 7px !important; box-shadow: none !important; min-height: 0 !important;
  transition: transform 100ms ease-out, background 160ms ease-out;
}
button[data-variant="segmented_control"] p { color: inherit !important; }
button[data-variant="segmented_control"][data-selected="true"] {
  background: var(--dv-group) !important; font-weight: 600 !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 0 0 0.5px rgba(0,0,0,0.04) !important;
}

/* Instant press feedback (respond on press, not on release). */
button:active { transform: scale(0.97); }

/* Popover button for filters */
div[data-testid="stPopover"] button {
  background: var(--dv-fill) !important; border: none !important; border-radius: 999px !important;
  color: var(--dv-accent) !important; font-size: 0.85rem !important; font-weight: 500 !important;
}
div[data-testid="stPopoverBody"] [data-baseweb="tag"],
div[data-testid="stPopoverBody"] [data-testid="stCheckbox"],
div[data-testid="stPopoverBody"] [data-testid="stSlider"] { filter: hue-rotate(207deg) saturate(1.05); }

.dv-note { text-align: center; font-size: 0.82rem; color: var(--dv-tertiary); margin: 0.75rem 0 2.5rem; }
.dv-hint { text-align: center; font-size: 0.82rem; color: var(--dv-secondary); margin: 0.6rem 0 0; }
.dv-ranker-note { text-align: center; font-size: 0.8rem; color: var(--dv-tertiary); margin-top: 0.35rem; }

/* Inset grouped lists, one per stemming mode. */
.dv-section-head {
  display: flex; justify-content: space-between; align-items: baseline;
  padding: 0 1rem 0.45rem; font-size: 0.8rem; color: var(--dv-tertiary);
}
.dv-section-title { font-size: 1.15rem; font-weight: 700; letter-spacing: -0.015em; color: var(--dv-text); }
.dv-group {
  background: var(--dv-group); border-radius: 14px; overflow: hidden;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  animation: dv-in 260ms ease-out both;
}
@keyframes dv-in { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
.dv-row { position: relative; padding: 0.9rem 1rem 0.8rem; transition: background 120ms ease-out; }
.dv-row + .dv-row::before {   /* inset separator, like iOS lists */
  content: ""; position: absolute; top: 0; left: 1rem; right: 0; height: 0.5px; background: var(--dv-separator);
}
.dv-row:hover { background: var(--dv-row-hover); }
.dv-row:active { background: var(--dv-row-press); }
.dv-row-top { display: flex; align-items: center; gap: 0.4rem; font-size: 0.75rem; color: var(--dv-tertiary); }
.dv-source { font-weight: 600; color: var(--dv-secondary); text-transform: capitalize; }
.dv-score { margin-left: auto; font-variant-numeric: tabular-nums; }
.dv-headline { font-size: 1.02rem; font-weight: 600; line-height: 1.38; letter-spacing: -0.012em; color: var(--dv-text); margin: 0.25rem 0 0.3rem; }
.dv-snippet { font-size: 0.9rem; line-height: 1.55; color: var(--dv-secondary); }
.dv-row mark { color: var(--dv-text); padding: 0 0.12em; border-radius: 4px; }
.dv-row mark.exact { background: var(--dv-exact); }
.dv-row mark.phonetic { background: var(--dv-phonetic); }
.dv-row mark.xling { background: var(--dv-xling); }
.dv-row mark.prf { background: var(--dv-prf); }
.dv-chips { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.55rem; }
.dv-chip { font-size: 0.7rem; font-weight: 500; padding: 0.12rem 0.55rem; border-radius: 999px; color: var(--dv-text); }
.dv-chip.exact { background: var(--dv-exact); }
.dv-chip.phonetic { background: var(--dv-phonetic); }
.dv-chip.xling { background: var(--dv-xling); }
.dv-chip.prf { background: var(--dv-prf); }
.dv-stage {
  font-size: 0.66rem; font-weight: 500; color: var(--dv-secondary); background: var(--dv-fill);
  border-radius: 999px; padding: 0.05rem 0.45rem;
}
.dv-only {
  font-size: 0.66rem; font-weight: 600; color: #fff; background: var(--dv-accent);
  border-radius: 999px; padding: 0.05rem 0.45rem;
}

/* Native disclosure for score details */
.dv-row details { margin-top: 0.5rem; }
.dv-row summary {
  list-style: none; cursor: pointer; font-size: 0.75rem; font-weight: 500; color: var(--dv-accent);
  display: inline-flex; align-items: center; gap: 0.25rem; user-select: none;
}
.dv-row summary::-webkit-details-marker { display: none; }
.dv-row summary::after { content: "›"; font-size: 1rem; line-height: 1; transition: transform 160ms ease-out; }
.dv-row details[open] summary::after { transform: rotate(90deg); }
.dv-table { width: 100%; margin-top: 0.45rem; font-size: 0.78rem; border-collapse: collapse; font-variant-numeric: tabular-nums; }
.dv-table td { padding: 0.28rem 0; color: var(--dv-secondary); border-top: 0.5px solid var(--dv-separator); }
.dv-table td:last-child { text-align: right; color: var(--dv-text); }
.dv-table tr.total td { font-weight: 600; color: var(--dv-text); }
.dv-also { font-size: 0.75rem; color: var(--dv-secondary); margin-top: 0.45rem; }
.dv-empty { padding: 1.4rem 1rem; text-align: center; font-size: 0.88rem; color: var(--dv-tertiary); }

/* Accessibility settings */
@media (prefers-reduced-motion: reduce) {
  .dv-group { animation: none; }
  button:active { transform: none; }
  .dv-row summary::after { transition: none; }
}
@media (prefers-reduced-transparency: reduce) {
  .dv-bar { background: var(--dv-page); -webkit-backdrop-filter: none; backdrop-filter: none; }
}
@media (prefers-contrast: more) {
  .dv-group { box-shadow: 0 0 0 1px var(--dv-secondary); }
  .dv-row + .dv-row::before { background: var(--dv-secondary); height: 1px; }
}
</style>
"""


@st.cache_resource
def load_index(mode):
    return real_index.load_index(mode)


@st.cache_resource
def load_static(mode):
    """Authority g(d) for an index: recency + PageRank + first to publish."""
    return static_scores(load_index(mode))


@st.cache_resource
def load_champions(mode):
    index = load_index(mode)
    return ChampionLists(index, r=max(5, index.N // 20), static_scores=load_static(mode)[0])


@st.cache_resource
def load_tiers(mode):
    return RecencyTiers(load_index(mode))


@st.cache_resource
def load_encoder():
    return SentenceEncoder()


@st.cache_resource
def load_dense(mode):
    return DenseIndex(load_index(mode), load_encoder())


@st.cache_resource
def load_clusters(mode):
    return ClusterPruning(load_index(mode))


@st.cache_resource
def load_impact(mode):
    return ImpactOrdered(load_index(mode))


SPEEDUPS = ["Off", "Index elimination", "Champion lists", "Recent tiers", "Cluster pruning", "Impact-ordered postings"]


def run_speedup(name, query, mode, k, doc_filter):
    """lnc.ltc with one of the Lecture 7 speed-ups. Returns (results, stats)."""
    index = load_index(mode)
    if name == "Index elimination":
        return search_index_elimination(query, index, k=k, doc_filter=doc_filter)
    if name == "Champion lists":
        return search_champions(query, index, load_champions(mode), k=k, doc_filter=doc_filter)
    if name == "Recent tiers":
        return search_tiered(query, index, load_tiers(mode), k=k, doc_filter=doc_filter)
    if name == "Cluster pruning":
        return search_clusters(query, index, load_clusters(mode), k=k, doc_filter=doc_filter)
    return search_impact(query, index, load_impact(mode), k=k, max_docs=max(20, index.N // 15), doc_filter=doc_filter)


def run_ranker(ranker, query, index, k, doc_filter, use_parser=True, static=None, dense=None):
    if use_parser:
        return parse_and_rank(query, index, k=k, ranker=ranker, doc_filter=doc_filter, static=static, dense=dense)
    if ranker == "rrf":
        return search_rrf(query, index, k=k, doc_filter=doc_filter, static=static, dense=dense)
    if ranker == "net":
        return rank(query, index, k=k, doc_filter=doc_filter, static=static)
    if ranker == "lnc":
        return search(query, index, k=k, doc_filter=doc_filter)
    return search_bm25(query, index, k=k, doc_filter=doc_filter)


def term_sources(query):
    """Map each expanded term to how it matched: exact, phonetic or xling."""
    sources = {}
    for token in query["tokens"]:
        for term, _weight, source in token["expansions"]:
            sources.setdefault(term, source)
    return sources


def highlight(text, sources):
    """Wrap matched words in tinted marks. Escapes everything else."""
    out = []
    for piece in regex.split(r"([\p{L}\p{M}\p{Nd}]+)", text):
        source = sources.get(piece.lower())
        if source:
            out.append(f'<mark class="{source}">{html.escape(piece)}</mark>')
        else:
            out.append(html.escape(piece))
    return "".join(out)


def snippet(body, sources, max_words=28):
    """The first sentence that contains a matched word, else the opening words."""
    for sentence in SENTENCE_END.split(body):
        if any(tok in sources for tok in tokenize(sentence)):
            return sentence
    words = body.split()
    return " ".join(words[:max_words]) + (" …" if len(words) > max_words else "")


def score_table(explain, terms, parts=None):
    rows = []
    if "cosine" in explain:
        g_label = "Authority g(d)" if explain.get("static") else "Recency g(d)"
        for label, key in (("Cosine", "cosine"), ("Headline match", "zone"),
                           ("Proximity", "proximity"), (g_label, "recency")):
            rows.append(f"<tr><td>{label}</td><td>{explain[key]:.4f}</td></tr>")
        if explain.get("static") and parts:
            for label, key in (("  recency", "recency"), ("  PageRank", "pagerank"), ("  first to publish", "original")):
                rows.append(f"<tr><td>{label}</td><td>{parts[key]:.4f}</td></tr>")
    if "rrf" in explain:
        for name, r in explain["rrf"].items():
            rows.append(f"<tr><td>Rank in {html.escape(name)}</td><td>{'#' + str(r) if r else '-'}</td></tr>")
        rows.append(f'<tr class="total"><td>RRF score</td><td>{explain["rrf_score"]:.4f}</td></tr>')
    for term, value in terms.items():
        rows.append(f"<tr><td>{html.escape(term)}</td><td>{value:.4f}</td></tr>")
    if "net" in explain:
        rows.append(f'<tr class="total"><td>Net score</td><td>{explain["net"]:.4f}</td></tr>')
    mmr = explain.get("mmr")
    if mmr:
        rows.append(f"<tr><td>MMR: similarity to earlier pick</td><td>{mmr['max_similarity']:.2f}</td></tr>")
    dense = explain.get("dense")
    if dense:
        rows.append(f"<tr><td>Dense (e5) cosine</td><td>{dense['cosine']:.4f}</td></tr>")
        rows.append(f'<tr class="total"><td>After dense re-rank</td><td>{dense["score"]:.4f}</td></tr>')
    kal = explain.get("kal")
    if kal:
        rows.append(f'<tr><td>कल boost ({kal["intent"]})</td><td>x {1 + kal["weight"] * kal["boost"]:.2f}</td></tr>')
    return f'<table class="dv-table">{"".join(rows)}</table>'


def result_row(doc_id, score, explain, index, sources, only_here, parts=None):
    article = index.articles.get(doc_id, {"headline": doc_id, "body": ""})
    # Some crawled text still has HTML entities (&#039;); decode them before we escape for display.
    article = {k: html.unescape(article.get(k, "")) for k in ("headline", "body")}
    meta = index.meta[doc_id]
    terms = explain.get("terms", explain)
    place = meta.get("city") or meta.get("state") or ""
    date = (meta.get("date") or "")[:10]
    details = " · ".join(html.escape(x) for x in (meta.get("section"), place, date) if x)
    only = '<span class="dv-only">Only here</span>' if only_here else ""
    stage = explain.get("stage")
    stage_tag = f'<span class="dv-stage">{STAGE_LABELS[stage]}</span>' if stage else ""
    kal = explain.get("kal")
    if kal and kal["boost"]:
        stage_tag += f'<span class="dv-stage">कल · {kal["intent"]}</span>'
    if explain.get("page_type", "article") != "article":
        stage_tag += f'<span class="dv-stage">{"Listing page" if explain["page_type"] == "listing" else "Horoscope"}</span>'
    also = ""
    if explain.get("also_in"):
        also = f'<div class="dv-also">Also in {html.escape(", ".join(explain["also_in"]))}</div>'
    chips = "".join(
        f'<span class="dv-chip {sources.get(t, "exact")}">{html.escape(t)} · {MATCH_LABELS.get(sources.get(t, "exact"), "Match")}</span>'
        for t in terms
    )
    return f"""
    <div class="dv-row">
      <div class="dv-row-top">
        <span class="dv-source">{html.escape(meta.get("source") or "")}</span><span>{details}</span>{stage_tag}{only}
        <span class="dv-score">{score:.3f}</span>
      </div>
      <div class="dv-headline">{highlight(article["headline"], sources)}</div>
      <div class="dv-snippet">{highlight(snippet(article["body"], sources), sources)}</div>
      <div class="dv-chips">{chips}</div>
      {also}
      <details><summary>Score details</summary>{score_table(explain, terms, parts)}</details>
    </div>"""


def speed_note(stats):
    if not stats:
        return ""
    return f" · scored {stats['scored']} of {stats['full_candidates']}"


def use_suggestion():
    choice = st.session_state.get("suggestion")
    if choice:
        st.session_state["q"] = choice


def main():
    st.set_page_config(page_title="Dhvani", page_icon="🔎", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(STYLE, unsafe_allow_html=True)

    base_index = load_index("none")
    st.markdown(
        f"""
        <div class="dv-bar"><div class="dv-bar-inner">
          <span class="dv-wordmark">Dhvani</span>
          <span class="dv-bar-right"><span class="dv-bar-meta">{base_index.N} articles indexed</span>
          <a class="dv-bar-link" href="/judge" target="_self">Judging</a></span>
        </div></div>
        <div class="dv-hero">
          <div class="dv-eyebrow">Hindi news search</div>
          <h1 class="dv-title">Search the way you speak.</h1>
          <p class="dv-subtitle">Type in Hindi, Hinglish or English. Every search runs three ways, so you can see exactly what stemming changes.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.session_state.setdefault("q", SUGGESTIONS[0])
    raw = st.text_input("Search", key="q", placeholder="कल का मौसम  ·  kal ka mausam  ·  weather tomorrow")
    st.pills("Suggestions", SUGGESTIONS, key="suggestion", on_change=use_suggestion, label_visibility="collapsed")

    # Ranking model and filters share one centred row, aligned on their middles.
    with st.container(horizontal=True, horizontal_alignment="center", vertical_alignment="center", gap="small"):
        choice = st.segmented_control("Ranking model", list(RANKERS), default="Net score",
                                      key="ranker", label_visibility="collapsed")
        ranker = RANKERS[choice or "Net score"]
        with st.popover("Filters", use_container_width=False):
            sources = st.multiselect("Newspaper", field_values(base_index, "source"), placeholder="All newspapers")
            sections = st.multiselect("Section", field_values(base_index, "section"), placeholder="All sections")
            states = st.multiselect("State", field_values(base_index, "state"), placeholder="All states")
            k = st.slider("Results per column", 3, 20, 10)
            use_parser = st.toggle("Smart query parsing", value=True,
                                   help="Try the exact phrase first, then all the words, then any word.")
            use_authority = st.toggle("Authority (PageRank + first to publish)", value=True,
                                      help="Net score uses g(d) = recency + PageRank + first-to-publish credit.")
            use_collapse = st.toggle("Collapse duplicate stories", value=True,
                                     help="Show a wire story once, with the other papers that ran it.")
            speedup = st.selectbox("Speed-up", SPEEDUPS,
                                   help="Score fewer articles with a Lecture 7 speed-up (lnc.ltc only).")
            use_dense = dense_available() and st.toggle(
                "Dense re-ranking (e5)", value=False,
                help="Re-rank the top 50 with a multilingual embedding model: half first stage, half meaning.")
            use_demote = st.toggle("Push listing pages down", value=True,
                                   help="Section pages and horoscopes go below real news articles.")
            use_prf = st.toggle("Pseudo-relevance feedback (Rocchio)", value=False,
                                help="Search once, take words from the top 5 articles, and search again.")
            use_mmr = st.toggle("Diversify results (MMR)", value=False,
                                help="Re-order so the top results cover more different stories.")
            use_kal = st.toggle("Date-aware kal", value=True,
                                help="Work out whether कल means yesterday or tomorrow and favour that day.")
            use_xling = st.toggle("Translate English words", value=True,
                                  help="English words also search for their Hindi translations.")
            use_dates = st.toggle("Limit to a date range")
            date_from = date_to = None
            if use_dates:
                date_from = st.date_input("From", key="date_from")
                date_to = st.date_input("To", key="date_to")
    st.markdown(f'<div class="dv-ranker-note">{RANKER_NOTES[ranker]}</div>', unsafe_allow_html=True)

    if isinstance(base_index, SampleIndex):
        st.markdown(
            '<div class="dv-note">Running on the sample index. The three columns will differ once the stemmed indexes are plugged in.</div>',
            unsafe_allow_html=True,
        )

    if not raw.strip():
        st.markdown('<div class="dv-empty">Type something to search.</div>', unsafe_allow_html=True)
        return

    queries = {mode: real_index.make_query(raw, mode, xling=use_xling) for mode, _label in MODES}
    if use_prf:
        queries = {mode: feedback_query(q, load_index(mode), mode, ranker if ranker != "rrf" else "net")[0]
                   for mode, q in queries.items()}
    # Stemmed columns match stemmed terms; the unstemmed ones are still needed to highlight the text.
    sources_map = {mode: {**term_sources(queries["none"]), **term_sources(queries[mode])} for mode, _label in MODES}

    results, speed_stats = {}, {}
    pool = collapse_pool(k) if use_collapse else k
    if use_dense:
        pool = max(pool, DEFAULT_DEPTH)
    for mode, _label in MODES:
        index = load_index(mode)
        doc_filter = make_filter(index, sources, sections, states, date_from, date_to)
        static = load_static(mode)[0] if use_authority else None
        if speedup != "Off":
            results[mode], speed_stats[mode] = run_speedup(speedup, queries[mode], mode, pool, doc_filter)
        else:
            fuse_dense = load_dense(mode) if use_dense and ranker == "rrf" else None
            results[mode] = run_ranker(ranker, queries[mode], index, pool, doc_filter, use_parser, static, fuse_dense)
        if use_dense and not (ranker == "rrf" and speedup == "Off"):
            results[mode] = dense_rerank(results[mode], queries[mode], load_dense(mode))
        if use_kal:
            results[mode] = apply_kal(results[mode], queries[mode], index)
        if use_demote:
            results[mode] = demote(results[mode], index)
        if use_mmr:
            results[mode] = diversify(results[mode], index)
        if use_collapse:
            results[mode] = collapse_duplicates(results[mode], index, k=k)
        else:
            results[mode] = results[mode][:k]

    hint = predict(queries["none"], results["none"], load_index("none"))
    if hint["low_confidence"]:
        st.markdown(f'<div class="dv-hint">Low confidence: {html.escape("; ".join(hint["reasons"]))}. Try more specific words.</div>',
                    unsafe_allow_html=True)

    ids = {mode: {doc_id for doc_id, _, _ in res} for mode, res in results.items()}
    columns = st.columns(len(MODES), gap="medium")
    for col, (mode, label) in zip(columns, MODES):
        others = set().union(*(ids[m] for m in ids if m != mode))
        count = len(results[mode])
        parts = load_static(mode)[1] if use_authority else {}
        if results[mode]:
            rows = "".join(
                result_row(doc_id, score, explain, load_index(mode), sources_map[mode], doc_id not in others,
                           parts.get(doc_id))
                for doc_id, score, explain in results[mode]
            )
        else:
            rows = '<div class="dv-empty">No matches. Try fewer words or another spelling.</div>'
        with col:
            st.markdown(
                f"""
                <div class="dv-section-head"><span class="dv-section-title">{label}</span>
                <span>{count} result{"s" if count != 1 else ""}{speed_note(speed_stats.get(mode))}</span></div>
                <div class="dv-group">{rows}</div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
