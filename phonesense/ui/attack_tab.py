"""Attack tab for the demo (T25, owner C).

Pick a phone, inject fake reviews, and watch how far it moves under each reputation method.
B's main app calls ``render_attack_tab(reviews, titles)`` inside one of its tabs; until that
exists, ``phonesense/ui/attack_app.py`` runs this tab on its own.

Needs ``pip install -r requirements-ui.txt``.
"""

from __future__ import annotations

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
    "median": "Median",
    "trust": "Reviewer trust (ours)",
}
SHORT_LABELS = {"raw": "Raw", "bayesian": "Bayesian", "helpful": "Helpful-vote",
                "median": "Median", "trust": "Trust (ours)"}
KIND_LABELS = {
    "bad_mouth": "Bad-mouthing: fake 1-star reviews on a good phone",
    "ballot_stuff": "Ballot stuffing: fake 5-star reviews on a weak phone",
}


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


def _movement(rank_before: int, rank_after: int) -> tuple[str, str]:
    """(text, arrow direction) for a rank change; rank 1 is the top."""
    moved = rank_before - rank_after
    if moved == 0:
        return "no change", "off"
    if moved > 0:
        return f"rose {moved} places", "up"
    return f"dropped {-moved} places", "down"


def render_attack_tab(reviews: Sequence[Review], titles: Mapping[str, str],
                      data_key: str = "reviews", is_toy: bool = False) -> None:
    st.subheader("Fake-review attack")
    st.caption("Inject fake reviews on one phone and compare how far it moves under each way "
               "of turning reviews into a reputation score. Fake reviews come from new "
               "accounts and never touch the real data.")
    if is_toy:
        st.warning("Toy data: reviews.jsonl was not found, so this page uses seeded "
                   "synthetic reviews. Numbers here are not results.")

    base = _baseline(reviews, data_key)
    c1, c2 = st.columns(2)
    kind = c1.radio("Attack", list(KIND_LABELS), format_func=KIND_LABELS.get)
    try:
        targets = pick_targets(reviews, kind, 15, seed=1)
    except ValueError:
        st.error("No phone has enough reviews to be a target.")
        return
    target = c2.selectbox(
        "Target phone", targets,
        format_func=lambda d: f"{_label(d, titles)}  ({base['raw'][d]:.2f} stars, "
                              f"{base['counts'][d]} reviews)")
    c3, c4 = st.columns(2)
    percent = c3.slider("Attack size (% of the phone's real reviews)", 0, 100, 20, step=5)
    camouflage = c4.slider(
        "Camouflage reviews per fake account", 0, 10, 0,
        help="Each fake account first posts this many reviews on other phones that copy the "
             "average rating, to look trustworthy before it attacks.")
    with st.expander("Advanced"):
        seed = st.number_input("Random seed", min_value=1, value=1, step=1)

    rep = attack_report(reviews, target, kind, percent, seed=int(seed), camouflage=camouflage,
                        before=base["before"])
    st.markdown(f"**{rep.n_fake} fake reviews** from {rep.n_fake} new accounts"
                + (f", plus {rep.n_cover} camouflage reviews on other phones" if rep.n_cover
                   else "")
                + f". Ranks are out of {rep.n_phones} phones.")

    cols = st.columns(len(rep.rows))
    for col, row in zip(cols, rep.rows, strict=True):
        text, arrow = _movement(row["rank_before"], row["rank_after"])
        col.metric(METHOD_LABELS[row["method"]], f"#{row['rank_after']}", text,
                   delta_color="off", delta_arrow=arrow)

    st.bar_chart([{"method": SHORT_LABELS[r["method"]], "places": r["rank_gain"]}
                  for r in rep.rows], x="method", y="places", sort=False, horizontal=True,
                 x_label="", y_label="places the attacker gained")  # horizontal swaps axes
    st.dataframe([{
        "Method": METHOD_LABELS[r["method"]],
        "Stars before": round(r["score_before"], 3), "Stars after": round(r["score_after"], 3),
        "Rank before": r["rank_before"], "Rank after": r["rank_after"],
        "Places gained by attacker": r["rank_gain"],
    } for r in rep.rows], hide_index=True, width="stretch")

    if rep.sybil_reliability is not None:
        st.markdown(
            f"Reviewer trust rates the fake accounts at **{rep.sybil_reliability:.2f}** "
            f"reliability. A brand-new account starts at 0.50; the median real reviewer is at "
            f"{base['median_reliability']:.2f}.")

    st.markdown("**Top phones after the attack**")
    left, right = st.columns(2)
    for col, method in ((left, "raw"), (right, "trust")):
        col.caption(METHOD_LABELS[method])
        col.dataframe([{"#": i, "Phone": _label(d, titles, 45)
                        + ("  ← target" if d == target else ""), "Stars": round(s, 3)}
                       for i, (d, s) in enumerate(rep.leaderboards[method], start=1)],
                      hide_index=True, width="stretch")
