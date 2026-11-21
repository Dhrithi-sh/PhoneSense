STOPWORDS = frozenset(
    {
        "के",
        "में",
        "है",
        "की",
        "से",
        "को",
        "कर",
        "और",
        "पर",
        "का",
        "हैं",
        "रह",
        "हो",
        "ने",
        "भी",
    }
)


def is_stopword(term: str) -> bool:
    """Return True if the analyzed term is an identified stop word."""
    return term in STOPWORDS