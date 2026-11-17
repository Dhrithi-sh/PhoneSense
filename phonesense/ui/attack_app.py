"""Standalone page for the attack tab, until B's main demo app exists.

    pip install -r requirements-ui.txt
    streamlit run phonesense/ui/attack_app.py

Uses data/processed/reviews.jsonl and phones.jsonl when present, otherwise seeded toy
reviews (clearly labelled on the page).
"""

from __future__ import annotations

import streamlit as st

from phonesense import io
from phonesense.config import get_path
from phonesense.trust.toy import make_toy_reviews
from phonesense.ui.attack_tab import render_attack_tab


@st.cache_resource(show_spinner="Loading reviews...")
def load_demo_data():
    """(reviews, {doc_id: title}, data_key, is_toy)."""
    reviews_path, phones_path = get_path("reviews"), get_path("phones")
    if reviews_path.exists() and phones_path.exists():
        titles = {p.doc_id: p.title for p in io.read_phones(phones_path)}
        return io.read_reviews(reviews_path), titles, str(reviews_path), False
    reviews = make_toy_reviews(n_phones=300, n_users=4000)
    titles = {r.doc_id: f"Toy phone {r.doc_id[-3:]}" for r in reviews}
    return reviews, titles, "toy", True


st.set_page_config(page_title="PhoneSense attack demo", layout="wide")
st.title("PhoneSense")
reviews, titles, data_key, is_toy = load_demo_data()
render_attack_tab(reviews, titles, data_key=data_key, is_toy=is_toy)
