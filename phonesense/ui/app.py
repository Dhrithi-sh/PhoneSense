"""PhoneSense demo (owner: B, T24; attack tab: C, T25). Run from the repo root:

    streamlit run phonesense/ui/app.py

Needs ``pip install -r requirements-ui.txt`` and data (``scripts/build_data.py``), or
``PHONESENSE_PHONES=path/to/phones.jsonl`` to try it on any phones file.
All search logic is in ``phonesense/ui/engine.py``; this file only draws. Two tabs: *Search*
(systems, fusion and prior toggles, per-result score breakdown) and *Fake-review attack*
(C's ``phonesense/ui/attack_tab.py``, shown when that module is in the build).
"""

from __future__ import annotations

import streamlit as st

from phonesense.config import get_path
from phonesense.llm.client import LLMError
from phonesense.ui.engine import (
    COMPONENTS,
    DEFAULT_PRIOR_WEIGHT,
    FUSION_METHODS,
    LABELS,
    PRIOR_LABELS,
    PRIORS,
    NoDataError,
    SearchEngine,
)

FUSION_LABELS = {"rrf": "RRF (rank-based, k=60)", "linear": "Weighted linear (min-max)"}

# One example per query type of the evaluation set (spec, aspect, model, vague), for the demo.
EXAMPLES = {
    "Spec": "5G phone under $300",
    "Aspect": "phone whose battery lasts two days of heavy use",
    "Model": "Google Pixel 7a",
    "Vague": "best phone for a teenager",
}
CAVEATS = (
    "Data: Amazon Reviews 2023 (Cell Phones), prices in USD, phones picked by a heuristic "
    "filter. Rankings are not shopping advice."
)


@st.cache_resource(show_spinner="Loading phones and building the index...")
def load_engine() -> SearchEngine:
    return SearchEngine.from_default_data()


@st.cache_resource(show_spinner="Loading reviews...")
def load_attack_data(_titles: dict[str, str], key: str):
    """``(reviews, titles, data_key, is_toy)`` for the attack tab: the real reviews when
    ``reviews.jsonl`` exists, otherwise seeded toy reviews (the tab labels them as toy data).
    ``key`` is only the cache key (the phones file in use); ``_titles`` is not hashed."""
    from phonesense import io

    reviews_path = get_path("reviews")
    if reviews_path.exists():
        return io.read_reviews(reviews_path), _titles, str(reviews_path), False
    from phonesense.trust.toy import make_toy_reviews

    reviews = make_toy_reviews(n_phones=300, n_users=4000)
    return reviews, {r.doc_id: f"Toy phone {r.doc_id[-3:]}" for r in reviews}, "toy", True


def breakdown_table(r) -> str:
    """Markdown table: where each selected system ranked this phone and its raw score, then
    the value of each prior (priors have no rank), then the final rank and score."""
    rows = ["| System | Rank | Raw score |", "|---|---:|---:|"]
    for part in r.breakdown:
        if part.kind == "prior":
            rank = "-"
        else:
            rank = "not in top 100" if part.rank is None else str(part.rank)
        score = "-" if part.score is None else f"{part.score:.4f}"
        rows.append(f"| {part.label} | {rank} | {score} |")
    rows.append(f"| **Final (shown above)** | **{r.rank}** | **{r.score:.4f}** |")
    return "\n".join(rows)


def draw_result(r) -> None:
    with st.container(border=True):
        st.markdown(f"**{r.rank}. {r.title}**")
        st.caption(f"{r.brand or 'brand unknown'}  ·  {r.price_text}  ·  {r.rating_text}")
        st.caption(f"score {r.score:.4f}  ·  id {r.doc_id}")
        if r.breakdown:
            with st.expander("Score breakdown"):
                st.markdown(breakdown_table(r))


def render_sidebar(engine: SearchEngine) -> tuple[dict[str, bool], str, float, int]:
    """Draw the controls; returns ``(toggles, fusion_method, prior_weight, k)``."""
    toggles: dict[str, bool] = {}
    with st.sidebar:
        st.header("Systems")
        for key in COMPONENTS:
            reason = engine.unavailable_reason(key)
            toggles[key] = st.checkbox(
                LABELS[key],
                value=(key == "tfidf"),
                disabled=reason is not None,
                help=reason or None,
                key=f"toggle_{key}",
            )
        st.header("Fusion")
        fusion_reason = engine.unavailable_reason("fusion")
        toggles["fusion"] = st.checkbox(
            "Fuse selected systems",
            value=True,
            disabled=fusion_reason is not None,
            help=fusion_reason or "Merge the rankings of every selected system.",
            key="toggle_fusion",
        )
        fusion_method = st.radio(
            "Fusion method",
            FUSION_METHODS,
            format_func=FUSION_LABELS.get,
            disabled=fusion_reason is not None or not toggles["fusion"],
            key="fusion_method",
        )
        st.header("Priors")
        for key in PRIORS:
            reason = engine.unavailable_reason(key)
            toggles[key] = st.checkbox(
                PRIOR_LABELS[key],
                value=False,
                disabled=reason is not None,
                help=reason or "Reorders the top 100 using a [0, 1] score per phone.",
                key=f"toggle_{key}",
            )
        prior_weight = st.slider(
            "Prior weight (lambda)",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_PRIOR_WEIGHT,
            step=0.05,
            disabled=not any(toggles[key] for key in PRIORS),
            help="0 keeps the retrieval order; 1 ranks by the prior alone.",
            key="prior_weight",
        )
        st.header("Display")
        k = st.slider("Results", min_value=5, max_value=20, value=10, step=5)
        st.caption(f"{len(engine):,} phones indexed")
    return toggles, fusion_method, prior_weight, k


def _use_example(text: str) -> None:
    st.session_state["query_text"] = text


def render_guide(engine: SearchEngine, text: str, results) -> None:
    """Button that asks the LLM for a short buying guide about the top results."""
    if not st.button("Write a buying guide for the top results", key="write_guide"):
        return
    try:
        with st.spinner("Writing the guide..."):
            guide = engine.write_guide(text, results)
    except LLMError as exc:
        st.error(f"The guide needs the LLM: {exc}")
        return
    except Exception as exc:  # noqa: BLE001 - never show a stack trace on the demo page
        st.error(f"The guide failed: {type(exc).__name__}: {exc}")
        return
    st.subheader("Buying guide")
    st.caption("Written by an LLM from the listed phone data only. Check any figure before "
               "you rely on it.")
    st.markdown(guide.text)
    for warning in guide.warnings:
        st.warning(warning)


def render_search_tab(engine, toggles, fusion_method, prior_weight, k) -> None:
    text = st.text_input(
        "What are you looking for?",
        placeholder="phone with a great night camera",
        key="query_text",
    )
    cols = st.columns(len(EXAMPLES))
    for col, (kind, example) in zip(cols, EXAMPLES.items(), strict=True):
        col.button(f"{kind}: {example}", on_click=_use_example, args=(example,),
                   use_container_width=True, key=f"example_{kind}")
    if not text.strip():
        st.info("Type a search above, or pick an example, to see ranked phones.")
        return

    try:
        with st.spinner("Searching (the first dense query embeds every phone, so it is slow)..."):
            response = engine.search(text, toggles, k, fusion_method, prior_weight)
    except Exception as exc:  # noqa: BLE001 - show a readable error instead of a stack trace
        st.error(f"The search failed: {type(exc).__name__}: {exc}")
        with st.expander("Details"):
            st.exception(exc)
        return
    for note in response.notes:
        st.warning(note)
    if not response.results:
        st.warning("No phone matched. Try different words.")
        return
    st.subheader(f"Top {len(response.results)} results")
    used = ", ".join(LABELS[u] for u in response.used)
    active = [p for p in PRIORS if toggles.get(p) and not engine.unavailable_reason(p)]
    priors = f"  ·  Priors: {', '.join(PRIOR_LABELS[p] for p in active)}" if active else ""
    st.caption(f"Systems used: {used}{priors}")
    for r in response.results:
        draw_result(r)
    render_guide(engine, text, response.results)
    st.caption(CAVEATS)


def render_attack_tab_or_note(engine: SearchEngine) -> None:
    """Mount C's attack tab, or explain why it is missing from this build."""
    try:
        from phonesense.ui.attack_tab import render_attack_tab

        reviews, titles, data_key, is_toy = load_attack_data(
            {p.doc_id: p.title for p in engine.phones.values()}, str(len(engine))
        )
    except ImportError as exc:
        st.info(f"The fake-review attack tab is not in this build ({exc}).")
        return
    render_attack_tab(reviews, titles, data_key=data_key, is_toy=is_toy)


def main() -> None:
    st.set_page_config(page_title="PhoneSense", page_icon="📱", layout="wide")
    st.title("PhoneSense")
    st.caption("Search smartphones by what you need, e.g. *phone under $300 with good battery*.")

    try:
        engine = load_engine()
    except NoDataError as exc:
        st.error(str(exc))
        st.stop()

    controls = render_sidebar(engine)
    search_tab, attack_tab = st.tabs(["Search", "Fake-review attack"])
    with search_tab:
        render_search_tab(engine, *controls)
    with attack_tab:
        render_attack_tab_or_note(engine)


main()
