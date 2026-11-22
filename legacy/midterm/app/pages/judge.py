"""Judging page: mark each pooled article 0, 1 or 2 for its information need.

Opens from the sidebar of the main app (`streamlit run app/streamlit_app.py`).
Judgments are saved after every click to judgments/qrels_<person>.txt, which
goes in git, so the four of us can judge at the same time and merge cleanly.
"""

import html
import re
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dhvani.eval.experiments import read_needs  # noqa: E402
from dhvani.eval.judge import (  # noqa: E402
    GRADES,
    PEOPLE,
    progress,
    read_need_descriptions,
    read_person,
    read_pool,
    save_judgment,
)
from dhvani.rank import real_index  # noqa: E402

# Whose needs are whose, by the letter of the need id.
PREFIX = {"rishit": "R", "viraja": "V", "riya": "Y", "dhrithi": "D"}

STYLE = """
<style>
:root { --j-text: #1d1d1f; --j-secondary: #6e6e73; --j-card: #ffffff; --j-bg: #f5f5f7; --j-mark: rgba(0,113,227,0.14); --j-accent: #0071e3; }
@media (prefers-color-scheme: dark) {
  :root { --j-text: #f5f5f7; --j-secondary: #a1a1a6; --j-card: #1c1c1e; --j-bg: #000000; --j-mark: rgba(41,151,255,0.25); --j-accent: #2997ff; }
}
.j-need { font-size: 1.35rem; font-weight: 600; color: var(--j-text); margin: 0.2rem 0 0.1rem; }
.j-sub { font-size: 0.85rem; color: var(--j-secondary); margin-bottom: 1rem; }
.j-card { background: var(--j-card); border-radius: 14px; padding: 1.1rem 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.j-meta { font-size: 0.78rem; color: var(--j-secondary); margin-bottom: 0.4rem; }
.j-headline { font-size: 1.1rem; font-weight: 600; color: var(--j-text); line-height: 1.4; margin-bottom: 0.5rem; }
.j-body { font-size: 0.92rem; color: var(--j-text); line-height: 1.6; }
.j-card mark { background: var(--j-mark); color: inherit; border-radius: 4px; padding: 0 2px; }
.j-done { text-align: center; padding: 2rem; color: var(--j-secondary); }
</style>
"""

BODY_CHARS = 900


@st.cache_resource
def load_index():
    return real_index.load_index("none")


@st.cache_data
def needs_and_pool():
    queries = read_needs()
    words = {}
    for _qid, need, form, text in queries:
        if form == "hindi":
            words[need] = [w for w in re.findall(r"[\wऀ-ॿ]+", text) if len(w) > 1]
    return read_need_descriptions(), read_pool(), words


def highlight(text, words):
    """Mark whole query words only, and skip short function words like के and में."""
    keep = {w.lower() for w in words if len(w) >= 3}
    out = []
    for piece in re.split(r"([\w\u0900-\u097f]+)", text):
        if piece.lower() in keep:
            out.append(f"<mark>{html.escape(piece)}</mark>")
        else:
            out.append(html.escape(piece))
    return "".join(out)


def main():
    st.set_page_config(page_title="Dhvani judging", page_icon="✅", layout="centered")
    st.markdown(STYLE, unsafe_allow_html=True)
    st.markdown('<a href="/" target="_self" style="font-size:0.85rem;text-decoration:none;">← Back to search</a>',
                unsafe_allow_html=True)
    st.title("Judging")

    descriptions, pool, need_words = needs_and_pool()
    if not pool:
        st.info("No pool yet. Run `python -m dhvani.eval.judge` after the experiment runner.")
        return

    person = st.selectbox("Who's judging?", PEOPLE)
    mine = [n for n in sorted(pool) if n.startswith(PREFIX.get(person, "?"))]
    needs = st.multiselect("Needs", sorted(pool), default=mine or sorted(pool)[:1])
    if not needs:
        return

    qrels = read_person(person)
    done = progress({n: pool[n] for n in needs}, qrels)
    judged = sum(a for a, _ in done.values())
    total = sum(b for _, b in done.values())
    st.progress(judged / total if total else 1.0, text=f"{judged} of {total} judged")

    queue = [(n, d) for n in needs for d in pool[n] if d not in qrels.get(n, {})]
    skipped = st.session_state.setdefault("skipped", [])
    queue = [item for item in queue if item not in skipped] + [item for item in queue if item in skipped]
    if not queue:
        st.markdown('<div class="j-done">All judged. Thank you.</div>', unsafe_allow_html=True)
        return

    need, doc_id = queue[0]
    index = load_index()
    article = getattr(index, "articles", {}).get(doc_id, {})
    meta = index.meta.get(doc_id, {})
    body = article.get("body", "")
    words = need_words.get(need, [])

    st.markdown(f'<div class="j-need">{need}: {html.escape(descriptions.get(need, ""))}</div>'
                f'<div class="j-sub">Does this article answer the need? {done[need][0]} of {done[need][1]} judged for {need}.</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="j-card"><div class="j-meta">{html.escape(meta.get("source", ""))} · {html.escape(meta.get("section") or "")}'
        f' · {html.escape((meta.get("date") or "")[:10])} · {doc_id}</div>'
        f'<div class="j-headline">{highlight(article.get("headline", doc_id), words)}</div>'
        f'<div class="j-body">{highlight(body[:BODY_CHARS], words)}{" …" if len(body) > BODY_CHARS else ""}</div></div>',
        unsafe_allow_html=True,
    )
    st.write("")

    cols = st.columns(4)
    for col, grade in zip(cols, (0, 1, 2)):
        if col.button(f"{grade} · {GRADES[grade]}", key=f"g{grade}", use_container_width=True,
                      type="primary" if grade == 2 else "secondary"):
            save_judgment(person, need, doc_id, grade)
            if (need, doc_id) in skipped:
                skipped.remove((need, doc_id))
            st.rerun()
    if cols[3].button("Skip", use_container_width=True):
        skipped.append((need, doc_id))
        st.rerun()

    st.caption("2 = it's about this need · 1 = partly, mentions it or a closely related story · 0 = not about it. "
               "Saved to judgments/qrels_" + person + ".txt after every click.")


main()
