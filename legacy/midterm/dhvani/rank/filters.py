"""Field filters over article metadata (Lecture 7: parametric / field indexes).

A field query is treated as a conjunction: an article has to satisfy every
filter that's set. Filters that are left empty don't restrict anything.
"""

from datetime import date, datetime


def _as_date(value):
    if value is None or isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return datetime.fromisoformat(value).date()


def make_filter(index, sources=None, sections=None, states=None, date_from=None, date_to=None):
    """Return doc_filter(doc_id) -> bool, or None if no filter is set."""
    sources = set(sources or [])
    sections = set(sections or [])
    states = set(states or [])
    date_from = _as_date(date_from)
    date_to = _as_date(date_to)

    if not (sources or sections or states or date_from or date_to):
        return None

    def doc_filter(doc_id):
        meta = index.meta[doc_id]
        if sources and meta.get("source") not in sources:
            return False
        if sections and meta.get("section") not in sections:
            return False
        if states and meta.get("state") not in states:
            return False
        if date_from or date_to:
            if not meta.get("date"):
                return False
            published = _as_date(meta["date"])
            if date_from and published < date_from:
                return False
            if date_to and published > date_to:
                return False
        return True

    return doc_filter


def field_values(index, field):
    """Sorted distinct non-empty values of a metadata field, for dropdowns."""
    return sorted({m.get(field) for m in index.meta.values() if m.get(field)})
