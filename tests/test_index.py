import json

import pytest

from phonesense.index import Analyzer, InvertedIndex, Posting, StaleIndexError, intersect

# Expected values below are worked out by hand with Analyzer(stem=False):
#   D0 text: nova0 x5_1 5g2 phone3 5000mah4 battery5 big6 battery7 [a8] phone9 [with10] [a11]
#            big12 screen13                                                    -> 11 terms
#   D1 text: pebble0 mini1 phone2 compact3 camera4 phone5                       -> 6
#   D2 text: zeta0 pro1 camera2 48mp3 camera4                                   -> 5
#   D3 text: nova0 lite1 battery2 saver3 nova4 phone5 nova6 price7              -> 8


@pytest.fixture
def docs(make_phone):
    return [
        make_phone(
            "D0",
            "Nova X5 5G phone",
            ["5000 mAh battery", "big battery"],
            "A phone with a big screen.",
        ),
        make_phone("D1", "Pebble Mini phone", [], "Compact camera phone"),
        make_phone("D2", "Zeta Pro camera", ["48 MP camera"], ""),
        make_phone("D3", "Nova Lite", ["battery saver"], "Nova phone, nova price."),
    ]


@pytest.fixture
def idx(docs):
    return InvertedIndex.build(docs, Analyzer(stem=False))


def test_ids(idx):
    assert idx.N == 4
    assert [idx.doc_id(i) for i in range(4)] == ["D0", "D1", "D2", "D3"]
    assert idx.doc_index("D2") == 2
    with pytest.raises(KeyError):
        idx.doc_index("nope")


def test_postings_df_cf(idx):
    assert idx.postings("phone") == [
        Posting(0, 2, [3, 9]),
        Posting(1, 2, [2, 5]),
        Posting(3, 1, [5]),
    ]
    assert idx.df("phone") == 3 and idx.cf("phone") == 5
    assert idx.postings("nova") == [Posting(0, 1, [0]), Posting(3, 3, [0, 4, 6])]
    assert idx.postings("camera") == [Posting(1, 1, [4]), Posting(2, 2, [2, 4])]
    assert idx.postings("battery") == [Posting(0, 2, [5, 7]), Posting(3, 1, [2])]
    assert idx.df("with") == 0 and idx.postings("with") == [] and idx.cf("with") == 0
    assert idx.df("pro") == 1 and idx.df("lite") == 1 and idx.df("mini") == 1
    assert idx.vocab_size("text") == 18


def test_fields(idx):
    assert idx.postings("phone", "title") == [Posting(0, 1, [3]), Posting(1, 1, [2])]
    assert idx.postings("battery", "features") == [Posting(0, 2, [1, 3]), Posting(3, 1, [0])]
    assert idx.postings("phone", "description") == [
        Posting(0, 1, [1]),
        Posting(1, 1, [2]),
        Posting(3, 1, [1]),
    ]
    assert idx.df("camera", "features") == 1 and idx.df("camera", "title") == 1
    with pytest.raises(KeyError):
        idx.postings("phone", "reviews")


def test_doc_lengths(idx):
    assert idx.doc_lens("text") == [11, 6, 5, 8]
    assert idx.doc_lens("title") == [4, 3, 3, 2]
    assert idx.doc_lens("features") == [4, 0, 2, 2]
    assert idx.doc_lens("description") == [3, 3, 0, 4]
    assert idx.doc_len(3, "text") == 8
    assert idx.avg_doc_len("text") == pytest.approx(7.5)
    assert idx.avg_doc_len("title") == pytest.approx(3.0)
    assert idx.avg_doc_len("features") == pytest.approx(2.0)
    assert idx.avg_doc_len("description") == pytest.approx(2.5)


def test_term_vector(idx):
    assert idx.term_vector(3) == {
        "nova": 3,
        "lite": 1,
        "battery": 1,
        "saver": 1,
        "phone": 1,
        "price": 1,
    }
    assert idx.term_vector(1, "features") == {}
    assert idx.term_vector(2, "title") == {"zeta": 1, "pro": 1, "camera": 1}


def test_postings_sorted_and_consistent(idx):
    for f in idx.fields:
        total = 0
        for t in idx.vocabulary(f):
            pl = idx.postings(t, f)
            assert [p.doc for p in pl] == sorted({p.doc for p in pl})
            assert all(p.tf == len(p.positions) and p.positions == sorted(p.positions) for p in pl)
            total += sum(p.tf for p in pl)
        assert total == sum(idx.doc_lens(f))


def test_fields_subset(docs):
    idx = InvertedIndex.build(docs, Analyzer(stem=False), fields=("title",))
    assert idx.fields == ("title",)
    with pytest.raises(KeyError):
        idx.df("phone", "text")
    with pytest.raises(ValueError):
        InvertedIndex.build(docs, Analyzer(), fields=("body",))


def test_duplicate_doc_id_rejected(make_phone):
    with pytest.raises(ValueError, match="duplicate"):
        InvertedIndex.build([make_phone("X", "a"), make_phone("X", "b")], Analyzer())


@pytest.mark.parametrize("compression", ["vbyte", "none"])
def test_save_load_round_trip(idx, tmp_path, compression):
    idx.save(tmp_path / "ix", compression=compression)
    loaded = InvertedIndex.load(tmp_path / "ix", Analyzer(stem=False))
    assert loaded == idx
    for f in idx.fields:
        assert loaded.avg_doc_len(f) == idx.avg_doc_len(f)
        for d in range(idx.N):
            assert loaded.term_vector(d, f) == idx.term_vector(d, f)
    assert loaded.stats() == idx.stats()


def test_stale_analyzer_rejected(idx, tmp_path):
    idx.save(tmp_path / "ix")
    with pytest.raises(StaleIndexError, match="build_index.py"):
        InvertedIndex.load(tmp_path / "ix", Analyzer(stem=True))


def test_stale_format_rejected(idx, tmp_path):
    idx.save(tmp_path / "ix")
    meta_path = tmp_path / "ix" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["format_version"] = 0
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(StaleIndexError, match="format version"):
        InvertedIndex.load(tmp_path / "ix", Analyzer(stem=False))


def test_missing_index(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_index.py"):
        InvertedIndex.load(tmp_path / "none", Analyzer())


def test_intersect_with_skips():
    left = [Posting(d, 1, [0]) for d in range(0, 200, 2)]
    right = [Posting(d, 1, [0]) for d in (3, 4, 50, 51, 52, 150, 198, 199)]
    assert intersect(left, right) == [4, 50, 52, 150, 198]
    assert intersect(right, left) == [4, 50, 52, 150, 198]
    assert intersect([], left) == []


def test_build_on_fixture_phones(fixtures):
    from phonesense.io import read_phones

    phones = read_phones(fixtures / "phones.jsonl")
    idx = InvertedIndex.build(phones, Analyzer())
    assert idx.N == len(phones)
    assert idx.df("5000mah") >= 1 and idx.df("128gb") >= 1
