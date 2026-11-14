"""Retrieval-agnostic Corpus Utility scoring for frontier prioritization."""

from typing import Dict, Any


def compute_corpus_utility(url: str, title: str = "", section: str = "") -> float:
    """Estimate corpus utility score U(u) in [0, 1] based on crawl-time features.

    Does NOT use search queries or downstream relevance labels.
    """
    score = 0.5
    # Favor core news sections
    if section in {"national", "state", "weather", "politics"}:
        score += 0.2
    elif section in {"lifestyle", "entertainment"}:
        score -= 0.1

    # Length of anchor / title signal
    if len(title.strip().split()) >= 4:
        score += 0.1

    return max(0.0, min(1.0, score))
