"""PhoneSense demo (owner: B, T24). Run from the repo root:

    streamlit run phonesense/ui/app.py

Needs ``pip install -r requirements-ui.txt`` and data (``scripts/build_data.py``), or
``PHONESENSE_PHONES=path/to/phones.jsonl`` to try it on any phones file.
All logic is in ``phonesense/ui/engine.py``; this file only draws.
"""

from __future__ import annotations

import streamlit as st

from phonesense.ui.engine import NoDataError, SearchEngine


@st.cache_resource(show_spinner="Loading phones and building the index...")
def load_engine() -> SearchEngine:
    return SearchEngine.from_default_data()


def draw_result(r) -> None:
    with st.container(border=True):
        st.markdown(f"**{r.rank}. {r.title}**")
        st.caption(f"{r.brand or 'brand unknown'}  ·  {r.price_text}  ·  {r.rating_text}")
        st.caption(f"score {r.score:.4f}  ·  id {r.doc_id}")


def main() -> None:
    st.set_page_config(page_title="PhoneSense", page_icon="📱", layout="wide")
    st.title("PhoneSense")
    st.caption("Search smartphones by what you need, e.g. *phone under $300 with good battery*.")

    try:
        engine = load_engine()
    except NoDataError as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.header("Search settings")
        k = st.slider("Results", min_value=5, max_value=20, value=10, step=5)
        st.caption(f"{len(engine):,} phones indexed")

    text = st.text_input("What are you looking for?", placeholder="phone with a great night camera")
    if not text.strip():
        st.info("Type a search above to see ranked phones.")
        return

    results = engine.search(text, k)
    if not results:
        st.warning("No phone matched. Try different words.")
        return
    st.subheader(f"Top {len(results)} results")
    for r in results:
        draw_result(r)


main()
