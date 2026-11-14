"""Streaming review IO (io.iter_reviews / io.write_reviews_stream) for large files."""

import json

import pytest

from phonesense import io
from phonesense.schemas import Review, SchemaError


def test_stream_roundtrip(fixtures, tmp_path):
    reviews = io.read_reviews(fixtures / "reviews.jsonl")
    out = tmp_path / "r.jsonl"
    assert io.write_reviews_stream(out, iter(reviews)) == len(reviews)
    assert list(io.iter_reviews(out)) == reviews
    assert io.read_reviews(out) == reviews
    assert not (tmp_path / "r.jsonl.tmp").exists()


def test_stream_rejects_duplicates_without_partial_file(fixtures, tmp_path):
    r = io.read_reviews(fixtures / "reviews.jsonl")[0]
    out = tmp_path / "r.jsonl"
    with pytest.raises(SchemaError, match="duplicate"):
        io.write_reviews_stream(out, [r, r])
    assert not out.exists() and not (tmp_path / "r.jsonl.tmp").exists()


def test_stream_keeps_old_file_on_error(fixtures, tmp_path):
    reviews = io.read_reviews(fixtures / "reviews.jsonl")
    out = tmp_path / "r.jsonl"
    io.write_reviews_stream(out, reviews)
    before = out.read_text(encoding="utf-8")
    bad = Review(**{**reviews[0].to_dict(), "rating": 9})
    with pytest.raises(SchemaError):
        io.write_reviews_stream(out, [reviews[1], bad])
    assert out.read_text(encoding="utf-8") == before


def test_iter_reviews_rejects_synthetic(fixtures, tmp_path):
    d = io.read_reviews(fixtures / "reviews.jsonl")[0].to_dict()
    d["is_synthetic"] = True
    p = tmp_path / "syn.jsonl"
    p.write_text(json.dumps(d) + "\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="synthetic"):
        list(io.iter_reviews(p))
    assert len(list(io.iter_reviews(p, allow_synthetic=True))) == 1
