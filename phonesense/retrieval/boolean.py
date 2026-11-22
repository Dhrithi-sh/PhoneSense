"""Boolean term retrieval over the inverted index (T07).

    from phonesense.retrieval.boolean import BooleanSearcher
    bs = BooleanSearcher(load_index())
    bs.search("samsung AND 5g AND NOT refurbished")      # -> sorted doc_ids
    bs.search("(pixel OR iphone) AND NOT renewed", field="title")

Syntax: terms, ``AND``, ``OR``, ``NOT`` (upper case) and parentheses. Adjacent terms without
an operator are ANDed. Binary operators have **equal precedence and associate left to right**
(``a OR b AND c`` = ``(a OR b) AND c``); use parentheses to group. Each term goes through the
index analyzer (so ``Phones`` matches ``phone``, ``128GB`` matches ``128gb``); a term that
analyzes to nothing (a stopword) matches every document.

Evaluation: consecutive ANDs are flattened and their operands sorted by size (smallest
postings list first, IIR sec. 1.3); ``x AND NOT y`` is computed as a difference, not via the
complement. Intersection walks two sorted doc-id lists with sqrt(n)-spaced skip pointers
(IIR sec. 2.3); ``skips=False`` gives the plain linear merge used as the efficiency baseline
(``scripts/bench_boolean.py``). Both count comparisons.

Adapted from our midterm project Dhvani-lingual (``index/search.py``: ``and_search``,
``intersect_postings``, ``build_skip_pointers``): smallest-list-first AND and skip-pointer
intersection; the OR / NOT / parentheses parser is new.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import isqrt

from phonesense.index import InvertedIndex

_TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")


@dataclass
class Stats:
    """Comparison counter for intersections (efficiency numbers)."""

    comparisons: int = 0
    skips_taken: int = 0


def intersect_sorted(
    a: Sequence[int], b: Sequence[int], skips: bool = True, stats: Stats | None = None
) -> list[int]:
    """Intersection of two ascending doc-id lists, optionally with sqrt(n) skip pointers.

    A skip pointer sits on every ``step = floor(sqrt(n))``-th position and points ``step``
    ahead; it is followed when its target is still <= the other list's current doc.
    """
    st = stats if stats is not None else Stats()
    sa = max(1, isqrt(len(a))) if skips and len(a) > 1 else 0
    sb = max(1, isqrt(len(b))) if skips and len(b) > 1 else 0
    i = j = 0
    out: list[int] = []
    na, nb = len(a), len(b)
    while i < na and j < nb:
        x, y = a[i], b[j]
        st.comparisons += 1
        if x == y:
            out.append(x)
            i += 1
            j += 1
        elif x < y:
            if sa and i % sa == 0 and i + sa < na and a[i + sa] <= y:
                st.comparisons += 1
                st.skips_taken += 1
                i += sa
            else:
                i += 1
        else:
            if sb and j % sb == 0 and j + sb < nb and b[j + sb] <= x:
                st.comparisons += 1
                st.skips_taken += 1
                j += sb
            else:
                j += 1
    return out


def union_sorted(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """Union of two ascending doc-id lists (linear merge)."""
    return sorted(set(a) | set(b))


def difference_sorted(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """Docs in ``a`` but not in ``b``."""
    bs = set(b)
    return [d for d in a if d not in bs]


# ------------------------------------------------------------------ parser


@dataclass
class Node:
    """AST node: op in {"term", "and", "or", "not"}."""

    op: str
    term: str = ""
    children: list[Node] = field(default_factory=list)


def parse(query: str) -> Node:
    """Parse a Boolean query into a :class:`Node` tree (see module docstring for syntax)."""
    tokens = _TOKEN_RE.findall(query)
    pos = 0

    def peek() -> str | None:
        return tokens[pos] if pos < len(tokens) else None

    def unary() -> Node:
        nonlocal pos
        tok = peek()
        if tok is None:
            raise ValueError(f"unexpected end of query: {query!r}")
        if tok == "NOT":
            pos += 1
            return Node("not", children=[unary()])
        if tok == "(":
            pos += 1
            node = expr()
            if peek() != ")":
                raise ValueError(f"missing ')' in {query!r}")
            pos += 1
            return node
        if tok in ("AND", "OR", ")"):
            raise ValueError(f"unexpected {tok!r} in {query!r}")
        pos += 1
        return Node("term", term=tok)

    def expr() -> Node:
        nonlocal pos
        left = unary()
        while (tok := peek()) is not None and tok != ")":
            if tok in ("AND", "OR"):
                pos += 1
                op = tok.lower()
            else:
                op = "and"  # implicit AND between adjacent terms
            right = unary()
            if left.op == op:  # flatten a AND b AND c
                left.children.append(right)
            else:
                left = Node(op, children=[left, right])
        return left

    if not tokens:
        raise ValueError("empty Boolean query")
    tree = expr()
    if pos != len(tokens):
        raise ValueError(f"unexpected {tokens[pos]!r} in {query!r}")
    return tree


# ------------------------------------------------------------------ evaluation


class BooleanSearcher:
    """Evaluate Boolean queries on one field of an :class:`InvertedIndex`."""

    def __init__(self, index: InvertedIndex, field: str = "text", skips: bool = True) -> None:
        self.index, self.field, self.skips = index, field, skips
        self.stats = Stats()
        self._all = list(range(index.N))

    def _term_docs(self, raw: str, field: str) -> list[int]:
        terms = self.index.analyzer.analyze(raw)
        if not terms:
            return self._all
        lists = sorted(([p.doc for p in self.index.postings(t, field)] for t in terms), key=len)
        out = lists[0]
        for nxt in lists[1:]:
            out = intersect_sorted(out, nxt, self.skips, self.stats)
        return out

    def _eval(self, node: Node, field: str) -> list[int]:
        if node.op == "term":
            return self._term_docs(node.term, field)
        if node.op == "not":
            return difference_sorted(self._all, self._eval(node.children[0], field))
        if node.op == "or":
            out: list[int] = []
            for c in node.children:
                out = union_sorted(out, self._eval(c, field))
            return out
        # AND: positives smallest first, then subtract the NOT operands
        pos = [self._eval(c, field) for c in node.children if c.op != "not"]
        neg = [self._eval(c.children[0], field) for c in node.children if c.op == "not"]
        if not pos:
            pos = [self._all]
        pos.sort(key=len)
        out = pos[0]
        for nxt in pos[1:]:
            if not out:
                break
            out = intersect_sorted(out, nxt, self.skips, self.stats)
        for n in neg:
            out = difference_sorted(out, n)
        return out

    def search_ids(self, query: str, field: str | None = None) -> list[int]:
        """Matching internal doc ids, ascending."""
        return self._eval(parse(query), field or self.field)

    def search(self, query: str, field: str | None = None) -> list[str]:
        """Matching contract doc_ids (in internal-id order)."""
        return [self.index.doc_id(d) for d in self.search_ids(query, field)]

    def and_query(self, terms: Sequence[str], field: str | None = None) -> list[int]:
        """``t1 AND t2 AND ...`` on already-analyzed terms (benchmark helper)."""
        f = field or self.field
        lists = sorted(([p.doc for p in self.index.postings(t, f)] for t in terms), key=len)
        if not lists:
            return []
        out = lists[0]
        for nxt in lists[1:]:
            if not out:
                break
            out = intersect_sorted(out, nxt, self.skips, self.stats)
        return out
