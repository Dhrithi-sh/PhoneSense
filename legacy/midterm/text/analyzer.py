from text.normalize import normalize
from text.tokenize import tokenize
from text.stem import stem
from text.aggressive_stem import stem_aggressive
from text.auto_stem import stem_auto


SUPPORTED_MODES = {
    "none",
    "light",
    "aggr",
    "auto",
}


def analyze(text: str, mode: str) -> list[tuple[str, int]]:
    """
    Normalize, tokenize, and analyze text.

    Supported modes:
        none  - normalization + tokenization only
        light - existing Hindi light stemmer
        aggr  - experimental aggressive stemmer
        auto  - corpus-derived selective stemming
    """
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported analysis mode: {mode}. "
            f"Supported modes: {sorted(SUPPORTED_MODES)}"
        )

    normalized = normalize(text)
    tokens = tokenize(normalized)

    analyzed = []

    for position, token in enumerate(tokens):
        if mode == "light":
            token = stem(token)

        elif mode == "aggr":
            token = stem_aggressive(token)

        elif mode == "auto":
            token = stem_auto(token)

        analyzed.append(
            (token, position)
        )

    return analyzed