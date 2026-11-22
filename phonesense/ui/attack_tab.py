"""Attack tab for the demo (T25, owner C).

Pick a phone, inject fake reviews, and watch how far it moves under each reputation method.
B's main app calls ``render_attack_tab(reviews, titles)`` inside one of its tabs; until that
exists, ``phonesense/ui/attack_app.py`` runs this tab on its own.

Needs ``pip install -r requirements-ui.txt``.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence

import streamlit as st

from phonesense.attacks.report import attack_report, clean_scores
from phonesense.attacks.simulator import pick_targets
from phonesense.schemas import Review
from phonesense.trust.baselines import raw_average
from phonesense.trust.methods import configured_trust

METHOD_LABELS = {
    "raw": "Raw average",
    "bayesian": "Bayesian average",
    "helpful": "Helpful-vote weighted",
    "median": "Median rating",
    "trust": "Reviewer trust",
}
KIND_LABELS = {"bad_mouth": "Bad-mouthing", "ballot_stuff": "Ballot stuffing"}
KIND_HELP = {
    "bad_mouth": "Fake 1-star reviews on a well-rated phone, as a competitor would post.",
    "ballot_stuff": "Fake 5-star reviews on a poorly rated phone, as its seller would post.",
}

# Scoped styles: every class starts with ps- so B's app is unaffected. Colours are
# translucent greys plus one accent, so the same rules read well in light and dark themes.
# Selectors are prefixed with .stMarkdown below, to outrank Streamlit's markdown defaults.
_CSS = """<style>
:root { --ps-accent: #0A84FF; }
.ps-eyebrow { font-size: .78rem; font-weight: 600; letter-spacing: .06em;
  text-transform: uppercase; color: var(--ps-accent); margin: 0 0 .3rem; }
.ps-display { font-size: 2.2rem; font-weight: 700; line-height: 1.08;
  letter-spacing: -.025em; margin: 0; }
.ps-lede { font-size: 1.02rem; line-height: 1.5; opacity: .7; margin: .55rem 0 0;
  max-width: 44rem; }
.ps-note { display: inline-block; font-size: .84rem; line-height: 1.4; margin-top: .9rem;
  padding: .35rem .8rem; border-radius: 999px; background: rgba(255, 159, 10, .16); }
.ps-headline { font-size: 1.3rem; font-weight: 600; line-height: 1.3;
  letter-spacing: -.012em; margin: .2rem 0 .3rem; }
.ps-sub { font-size: .9rem; opacity: .62; margin: 0 0 1rem;
  font-variant-numeric: tabular-nums; }
.ps-card { border-radius: 18px; padding: .6rem 1.25rem;
  background: rgba(127, 127, 127, .07); border: 1px solid rgba(127, 127, 127, .16); }
.ps-row { display: grid; grid-template-columns: 11.5rem 1fr 3rem; align-items: center;
  gap: 1rem; padding: .7rem 0; }
.ps-row + .ps-row { border-top: 1px solid rgba(127, 127, 127, .13); }
.ps-name { font-size: .95rem; font-weight: 500; }
.ps-meta { display: block; font-size: .78rem; opacity: .55; margin-top: .1rem;
  font-variant-numeric: tabular-nums; }
.ps-track { height: .5rem; border-radius: 999px; background: rgba(127, 127, 127, .14);
  overflow: hidden; }
.ps-fill { height: 100%; border-radius: 999px; background: rgba(127, 127, 127, .5); }
.ps-ours .ps-fill { background: var(--ps-accent); }
.ps-ours .ps-name { font-weight: 600; }
.ps-num { text-align: right; font-weight: 600; font-variant-numeric: tabular-nums; }
.ps-tag { font-size: .66rem; font-weight: 600; letter-spacing: .05em; text-transform: uppercase;
  padding: .12rem .4rem; border-radius: 6px; margin-left: .45rem; vertical-align: 1px;
  color: var(--ps-accent); background: rgba(10, 132, 255, .14); }
.ps-legend { font-size: .78rem; opacity: .55; margin: .55rem 0 0; }
.ps-stats { display: flex; flex-wrap: wrap; gap: 2.25rem; margin: .4rem 0 .2rem; }
.ps-stat b { display: block; font-size: 1.55rem; font-weight: 600; letter-spacing: -.015em;
  font-variant-numeric: tabular-nums; }
.ps-stat span { font-size: .8rem; opacity: .6; }
.ps-stat.ps-ours b { color: var(--ps-accent); }
@media (max-width: 640px) {
  .ps-display { font-size: 1.7rem; }
  .ps-row { grid-template-columns: 7.5rem 1fr 2.4rem; gap: .7rem; }
  .ps-stats { gap: 1.4rem; }
}
@media (prefers-contrast: more) {
  .ps-card { border-color: rgba(127, 127, 127, .6); }
  .ps-lede, .ps-sub, .ps-meta, .ps-legend, .ps-stat span { opacity: .9; }
}
</style>"""
_CSS = re.sub(r"(?m)(^\s*|,\s*)\.ps-", r"\1.stMarkdown .ps-", _CSS)


@st.cache_resource(show_spinner="Scoring phones on the clean reviews...")
def _baseline(_reviews: Sequence[Review], data_key: str) -> dict:
    """Clean scores, review counts and the median real reviewer reliability.
    Cached per data set; ``data_key`` identifies it (the reviews themselves are not hashed)."""
    trust = configured_trust(_reviews)
    return {
        "before": clean_scores(_reviews),
        "counts": Counter(r.doc_id for r in _reviews),
        "raw": raw_average(_reviews),
        "median_reliability": statistics.median(trust.reliability.values()),
    }


def _label(doc_id: str, titles: Mapping[str, str], width: int = 60) -> str:
    title = titles.get(doc_id, doc_id)
    return title if len(title) <= width else title[:width - 1] + "…"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _headline(rep, kind: str) -> tuple[str, str]:
    """One sentence that answers the page's question, and a qualifier under it."""
    rows = {r["method"]: r for r in rep.rows}
    raw, trust = rows["raw"]["rank_gain"], rows["trust"]["rank_gain"]
    if rep.n_fake == 0:
        return ("Move the attack size slider to add fake reviews.",
                "With no fake reviews, every method keeps the phone where it is.")
    if raw <= 0 and trust <= 0:
        return "The attack did not move this phone.", ""
    verb = "pushed it down" if kind == "bad_mouth" else "lifted it"
    head = (f"With reviewer trust the fake reviews {verb} {_plural(trust, 'place')}, "
            f"against {raw} with the raw average.")
    if 0 <= trust < raw:
        return head, f"Reviewer trust absorbed {round(100 * (1 - trust / raw))}% of the effect."
    return head, "Reviewer trust did not help against this attack."


def _comparison(rep) -> str:
    """Rank shift per method as quiet horizontal bars; ours carries the accent."""
    top = max([r["rank_gain"] for r in rep.rows] + [1])
    rows = []
    for r in rep.rows:
        ours = r["method"] == "trust"
        width = 100 * max(r["rank_gain"], 0) / top
        tag = '<span class="ps-tag">Ours</span>' if ours else ""
        rows.append(
            f'<div class="ps-row{" ps-ours" if ours else ""}">'
            f'<div><span class="ps-name">{METHOD_LABELS[r["method"]]}{tag}</span>'
            f'<span class="ps-meta">#{r["rank_before"]} → #{r["rank_after"]}</span></div>'
            f'<div class="ps-track"><div class="ps-fill" style="width:{width:.1f}%"></div></div>'
            f'<div class="ps-num">{r["rank_gain"]}</div></div>')
    return f'<div class="ps-card">{"".join(rows)}</div>'


def _reliability(rep, typical: float) -> str:
    stats = [("ps-stat ps-ours", f"{rep.sybil_reliability:.2f}", "Fake accounts"),
             ("ps-stat", "0.50", "A brand-new account"),
             ("ps-stat", f"{typical:.2f}", "Typical real reviewer")]
    return '<div class="ps-stats">' + "".join(
        f'<div class="{cls}"><b>{value}</b><span>{label}</span></div>'
        for cls, value, label in stats) + "</div>"


def render_attack_tab(reviews: Sequence[Review], titles: Mapping[str, str],
                      data_key: str = "reviews", is_toy: bool = False) -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    note = ('<div class="ps-note">Toy data: reviews.jsonl was not found, so these are '
            "seeded synthetic reviews. Numbers here are not results.</div>") if is_toy else ""
    st.markdown(
        '<p class="ps-eyebrow">Fake-review attack</p>'
        # a div, not <h1>: Streamlit's own heading styles would override ours
        '<div class="ps-display" role="heading" aria-level="1">'
        "How far can fake reviews move a phone?</div>"
        '<p class="ps-lede">Pick a phone and add fake reviews from new accounts. Each '
        "reputation method below ranks every phone; compare how far the target moves. "
        f"The real data is never changed.</p>{note}", unsafe_allow_html=True)
    st.write("")

    base = _baseline(reviews, data_key)
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        kind = st.segmented_control("Attack", list(KIND_LABELS), default="bad_mouth",
                                    required=True, format_func=KIND_LABELS.get)
        st.caption(KIND_HELP[kind])
    try:
        targets = pick_targets(reviews, kind, 15, seed=1)
    except ValueError:
        st.error("No phone has enough reviews to be a target.")
        return
    target = right.selectbox(
        "Phone", targets,
        format_func=lambda d: f"{_label(d, titles)}  ·  {base['raw'][d]:.2f} stars, "
                              f"{base['counts'][d]} reviews")
    left, right = st.columns([1, 1.4], gap="large")
    percent = left.slider("Fake reviews, as % of its real reviews", 0, 100, 20, step=5)
    camouflage = right.slider(
        "Camouflage reviews per fake account", 0, 10, 0,
        help="Before attacking, each fake account posts this many reviews on other phones "
             "that copy the average rating, to look trustworthy.")

    seed = 1
    rep = attack_report(reviews, target, kind, percent, seed=seed, camouflage=camouflage,
                        before=base["before"])
    head, qualifier = _headline(rep, kind)
    detail = _plural(rep.n_fake, "fake review") + " from " + _plural(rep.n_fake, "new account")
    if rep.n_cover:
        detail += f", plus {rep.n_cover} camouflage reviews"
    detail += f"  ·  ranks out of {rep.n_phones} phones"

    st.divider()
    st.markdown(f'<p class="ps-headline">{head}</p><p class="ps-sub">{qualifier}'
                f'{"  ·  " if qualifier else ""}{detail}</p>', unsafe_allow_html=True)
    st.markdown(_comparison(rep) + '<p class="ps-legend">Places the attacker gained, with the '
                "phone's rank before and after.</p>", unsafe_allow_html=True)

    if rep.sybil_reliability is not None:
        st.write("")
        st.markdown('<p class="ps-name">How reliable reviewer trust thinks each account is</p>'
                    + _reliability(rep, base["median_reliability"]), unsafe_allow_html=True)

    st.write("")
    with st.expander("All numbers"):
        st.dataframe([{
            "Method": METHOD_LABELS[r["method"]],
            "Stars before": round(r["score_before"], 3),
            "Stars after": round(r["score_after"], 3),
            "Rank before": r["rank_before"], "Rank after": r["rank_after"],
            "Places gained": r["rank_gain"],
        } for r in rep.rows], hide_index=True, width="stretch")
        st.caption("Top 10 phones after the attack")
        cols = st.columns(2)
        for col, method in zip(cols, ("raw", "trust"), strict=True):
            col.dataframe([{"#": i, METHOD_LABELS[method]: _label(d, titles, 40)
                            + ("  ← target" if d == target else ""), "Stars": round(s, 3)}
                           for i, (d, s) in enumerate(rep.leaderboards[method], start=1)],
                          hide_index=True, width="stretch")
