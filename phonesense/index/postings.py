"""Postings, skip pointers and variable-byte compression (T03).

* :class:`Posting` ``(doc, tf, positions)`` is the in-memory posting used by every retriever.
* :func:`skip_pointers` / :func:`intersect` give sqrt(n)-spaced skip pointers and a skip-aware
  AND of two postings lists (for Boolean / phrase search in T07). Adapted from our midterm
  project Dhvani-lingual (``index/search.py``: ``build_skip_pointers``,
  ``intersect_postings``); now typed, on int doc ids and on :class:`Posting`.
* :func:`vb_encode` / :func:`vb_decode` implement variable-byte (VB) encoding as in Manning,
  Raghavan & Schuetze, *Introduction to IR*, section 5.3.1: 7 payload bits per byte, most
  significant group first, and the high bit set on the *last* byte of each number. The
  midterm project had no compression; this is new.
* :func:`encode_postings` / :func:`decode_postings` serialize one postings list as VB ints:
  for each posting ``doc gap, tf, position gaps...`` (the first doc / first position are
  stored as gaps from 0). The plain (uncompressed) alternative stores the same structure as
  fixed-width little-endian uint32 with absolute doc ids and positions.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterable, Sequence
from itertools import accumulate
from math import isqrt
from typing import NamedTuple

import numpy as np


class Posting(NamedTuple):
    """One document in a term's postings list.

    ``doc`` is the internal int doc id (``0..N-1``), ``tf`` the term frequency in the field,
    ``positions`` the sorted token positions of the term in that field (``len == tf``).
    """

    doc: int
    tf: int
    positions: list[int]


# ------------------------------------------------------------------ skip pointers


def skip_pointers(postings: Sequence[Posting]) -> dict[int, int]:
    """Skip pointers ``{i: i + step}`` with ``step = floor(sqrt(n))`` (IIR section 2.3)."""
    n = len(postings)
    if n < 2:
        return {}
    step = max(1, isqrt(n))
    return {i: i + step for i in range(0, n - step, step)}


def intersect(left: Sequence[Posting], right: Sequence[Posting]) -> list[int]:
    """Doc ids present in both postings lists (sorted), using skip pointers."""
    ls, rs = skip_pointers(left), skip_pointers(right)
    i = j = 0
    out: list[int] = []
    while i < len(left) and j < len(right):
        a, b = left[i].doc, right[j].doc
        if a == b:
            out.append(a)
            i += 1
            j += 1
        elif a < b:
            s = ls.get(i)
            i = s if s is not None and left[s].doc <= b else i + 1
        else:
            s = rs.get(j)
            j = s if s is not None and right[s].doc <= a else j + 1
    return out


# ------------------------------------------------------------------ variable-byte codec


def vb_encode_number(n: int) -> bytes:
    """VB-encode one non-negative int (IIR fig. 5.8)."""
    if n < 0:
        raise ValueError(f"VB encoding needs non-negative ints, got {n}")
    groups = [n & 0x7F]
    n >>= 7
    while n:
        groups.append(n & 0x7F)
        n >>= 7
    groups.reverse()
    groups[-1] |= 0x80
    return bytes(groups)


def vb_encode(numbers: Iterable[int]) -> bytes:
    """VB-encode a sequence of non-negative ints into one byte string."""
    out = bytearray()
    for n in numbers:
        if n < 0:
            raise ValueError(f"VB encoding needs non-negative ints, got {n}")
        if n < 0x80:
            out.append(n | 0x80)
        else:
            out += vb_encode_number(n)
    return bytes(out)


def vb_decode(data: bytes) -> list[int]:
    """Decode a VB byte string back into ints (vectorized with numpy)."""
    if not data:
        return []
    b = np.frombuffer(data, dtype=np.uint8)
    ends = np.flatnonzero(b & 0x80)
    if len(ends) == 0 or ends[-1] != len(b) - 1:
        raise ValueError("truncated VB data: last byte has no stop bit")
    starts = np.concatenate(([0], ends[:-1] + 1))
    # number index of each byte, then its 7-bit shift = 7 * (bytes left until the stop byte)
    group = np.repeat(np.arange(len(ends)), ends - starts + 1)
    shift = 7 * (ends[group] - np.arange(len(b)))
    if shift.max(initial=0) > 56:
        return _vb_decode_py(data)  # numbers wider than 63 bits: fall back to Python ints
    vals = (b & 0x7F).astype(np.int64) << shift.astype(np.int64)
    return np.add.reduceat(vals, starts).tolist()


def _vb_decode_py(data: bytes) -> list[int]:
    out, n = [], 0
    for byte in data:
        if byte & 0x80:
            out.append((n << 7) | (byte & 0x7F))
            n = 0
        else:
            n = (n << 7) | byte
    return out


def gaps(sorted_numbers: Sequence[int]) -> list[int]:
    """``[3, 7, 8] -> [3, 4, 1]`` (first value is a gap from 0)."""
    prev, out = 0, []
    for x in sorted_numbers:
        out.append(x - prev)
        prev = x
    return out


def ungaps(gap_list: Sequence[int]) -> list[int]:
    """Inverse of :func:`gaps`."""
    return list(accumulate(gap_list))


# ------------------------------------------------------------------ postings (de)serialization


def _flatten(postings: Sequence[Posting], use_gaps: bool) -> list[int]:
    ints: list[int] = []
    prev_doc = 0
    for p in postings:
        ints.append(p.doc - prev_doc if use_gaps else p.doc)
        ints.append(p.tf)
        ints.extend(gaps(p.positions) if use_gaps else p.positions)
        prev_doc = p.doc
    return ints


def unflatten_from(
    ints: Sequence[int], start: int, count: int, use_gaps: bool
) -> tuple[list[Posting], int]:
    """Read ``count`` postings from a flat int stream at ``start``; return (postings, end)."""
    out: list[Posting] = []
    i, doc = start, 0
    for _ in range(count):
        doc = doc + ints[i] if use_gaps else ints[i]
        tf = ints[i + 1]
        pos = list(ints[i + 2 : i + 2 + tf])
        out.append(Posting(doc, tf, ungaps(pos) if use_gaps else pos))
        i += 2 + tf
    return out, i


def _unflatten(ints: Sequence[int], count: int, use_gaps: bool) -> list[Posting]:
    out, end = unflatten_from(ints, 0, count, use_gaps)
    if end != len(ints):
        raise ValueError(f"postings block has {len(ints) - end} trailing ints")
    return out


def encode_postings(postings: Sequence[Posting], compression: str = "vbyte") -> bytes:
    """Serialize one postings list (``compression`` = ``"vbyte"`` or ``"none"``)."""
    if compression == "vbyte":
        return vb_encode(_flatten(postings, use_gaps=True))
    if compression == "none":
        return array("I", _flatten(postings, use_gaps=False)).tobytes()
    raise ValueError(f"unknown compression {compression!r}")


def decode_postings(data: bytes, count: int, compression: str = "vbyte") -> list[Posting]:
    """Inverse of :func:`encode_postings`; ``count`` = number of postings (the term's df)."""
    if compression == "vbyte":
        return _unflatten(vb_decode(data), count, use_gaps=True)
    if compression == "none":
        ints = array("I")
        ints.frombytes(data)
        return _unflatten(ints.tolist(), count, use_gaps=False)
    raise ValueError(f"unknown compression {compression!r}")
