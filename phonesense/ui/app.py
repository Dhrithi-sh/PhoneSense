"""PhoneSense demo (owner: B, T24). Run from the repo root:

    streamlit run phonesense/ui/app.py

Needs ``pip install -r requirements-ui.txt`` and data (``scripts/build_data.py``), or
``PHONESENSE_PHONES=path/to/phones.jsonl`` to try it on any phones file.
All logic is in ``phonesense/ui/engine.py``; this file only draws.
"""

from __future__ import annotations

import streamlit as st

from phonesense.ui.engine import COMPONENTS, FUSION_METHODS, LABELS, NoDataError, SearchEngine

FUSION_LABELS = {"rrf": "RRF (rank-based, k=60)", "linear": "Weighted linear (min-max)"}


@st.cache_resource(show_spinner="Loading phones and building the index...")
def load_engine() -> SearchEngine:
    return SearchEngine.from_default_data()


def breakdown_table(r) -> str:
    """Markdown table: where each selected system ranked this phone and its raw score."""
    rows = ["| System | Rank | Raw score |", "|---|---:|---:|"]
    for part in r.breakdown:
        rank = "not in top 100" if part.rank is None else str(part.rank)
        score = "-" if part.score is None else f"{part.score:.4f}"
        rows.append(f"| {part.label} | {rank} | {score} |")
    rows.append(f"| **Final score shown above** | **{r.rank}** | **{r.score:.4f}** |")
    return "\n".join(rows)


def draw_result(r) -> None:
    with st.container(border=True):
        st.markdown(f"**{r.rank}. {r.title}**")
        st.caption(f"{r.brand or 'brand unknown'}  ·  {r.price_text}  ·  {r.rating_text}")
        st.caption(f"score {r.score:.4f}  ·  id {r.doc_id}")
        if r.breakdown:
            with st.expander("Score breakdown"):
                st.markdown(breakdown_table(r))


def main() -> None:
    st.set_page_config(page_title="PhoneSense", page_icon="📱", layout="wide")
    st.title("PhoneSense")
    st.caption("Search smartphones by what you need, e.g. *phone under $300 with good battery*.")

    try:
        engine = load_engine()
    except NoDataError as exc:
        st.error(str(exc))
        st.stop()

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
        st.header("Display")
        k = st.slider("Results", min_value=5, max_value=20, value=10, step=5)
        st.caption(f"{len(engine):,} phones indexed")

    text = st.text_input("What are you looking for?", placeholder="phone with a great night camera")
    if not text.strip():
        st.info("Type a search above to see ranked phones.")
        return

    with st.spinner("Searching (the first dense query embeds every phone, so it is slow)..."):
        response = engine.search(text, toggles, k, fusion_method)
    for note in response.notes:
        st.warning(note)
    if not response.results:
        st.warning("No phone matched. Try different words.")
        return
    st.subheader(f"Top {len(response.results)} results")
    st.caption("Systems used: " + ", ".join(LABELS[u] for u in response.used))
    for r in response.results:
        draw_result(r)


main()
