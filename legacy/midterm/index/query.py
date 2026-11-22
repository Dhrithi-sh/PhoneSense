from text.analyzer import analyze


SUPPORTED_MODES = {
    "none",
    "light",
    "aggr",
    "auto",
}


def search_query(
    idx,
    query: str,
    mode: str = None,
    zone: str = "body",
):
    if mode is None:
        mode = idx.mode

    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported analysis mode: {mode}. "
            f"Supported modes: {sorted(SUPPORTED_MODES)}"
        )

    if mode != idx.mode:
        raise ValueError(
            f"Query mode '{mode}' does not match "
            f"index mode '{idx.mode}'"
        )

    if zone not in {
        "headline",
        "body",
    }:
        raise ValueError(
            f"Unsupported zone: {zone}"
        )

    analyzed = analyze(
        query,
        mode,
    )

    terms = [
        term
        for term, _ in analyzed
    ]

    if not terms:
        return []

    from index.search import and_search

    return and_search(
        idx,
        terms,
        zone=zone,
    )


def search_both(
    idx_none,
    idx_light,
    query: str,
    zone: str = "body",
):
    return {
        "none": search_query(
            idx_none,
            query,
            mode="none",
            zone=zone,
        ),
        "light": search_query(
            idx_light,
            query,
            mode="light",
            zone=zone,
        ),
    }