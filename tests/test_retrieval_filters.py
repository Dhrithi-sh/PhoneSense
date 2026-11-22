import importlib.util
from pathlib import Path

import pytest

from phonesense import io
from phonesense.index import Analyzer, InvertedIndex
from phonesense.retrieval.bm25 import BM25Retriever
from phonesense.retrieval.filters import (
    FilteredRetriever,
    allowed_docs,
    evaluate,
    matches,
    parse_constraints,
    resolve_filter,
)
from phonesense.retrieval.tfidf import TfidfRetriever
from phonesense.schemas import Query, SchemaError

REPO = Path(__file__).resolve().parent.parent


def leaf(field, op, value):
    return {"field": field, "op": op, "value": value}


@pytest.fixture
def phones(fixtures):
    # B0TEST0001: Nova, $299.99, 8 GB, 128 GB, 5000 mAh, 6.5", android, 5G
    # B0TEST0002: Pebble, $449, 6 GB, 256 GB, 3200 mAh, 5.4", ios, 5G
    # B0TEST0003: brand null, price null, storage 32, android, has_5g null
    return {p.doc_id: p for p in io.read_phones(fixtures / "phones.jsonl")}


@pytest.mark.parametrize(
    "node, expected",
    [
        (leaf("price", "<", 300), {"B0TEST0001"}),
        (leaf("price", "<=", 449), {"B0TEST0001", "B0TEST0002"}),
        (leaf("price", ">", 300), {"B0TEST0002"}),
        (leaf("ram_gb", ">=", 8), {"B0TEST0001"}),
        (leaf("storage_gb", "==", 32), {"B0TEST0003"}),
        (leaf("os", "!=", "ios"), {"B0TEST0001", "B0TEST0003"}),
        (leaf("brand", "in", ["Nova", "Pebble"]), {"B0TEST0001", "B0TEST0002"}),
        (leaf("has_5g", "==", True), {"B0TEST0001", "B0TEST0002"}),
        (leaf("screen_in", "in", [5.4]), {"B0TEST0002"}),
    ],
)
def test_every_op(phones, node, expected):
    assert allowed_docs(phones.values(), node) == expected


def test_nested_and_or_not(phones):
    f = {
        "and": [
            leaf("os", "==", "android"),
            {"or": [leaf("storage_gb", ">=", 256), {"not": leaf("price", ">", 500)}]},
        ]
    }
    # P1: android, price 299.99 -> not(>500) TRUE -> pass. P3: android, storage 32 FALSE,
    # price null -> not(UNKNOWN) = UNKNOWN -> or UNKNOWN -> and UNKNOWN -> fail.
    assert allowed_docs(phones.values(), f) == {"B0TEST0001"}
    assert allowed_docs(phones.values(), {"not": leaf("os", "==", "ios")}) == {
        "B0TEST0001",
        "B0TEST0003",
    }


def test_null_never_matches_even_under_not(phones):
    p3 = phones["B0TEST0003"]
    for node in (
        leaf("price", "<", 1e9),
        leaf("price", "!=", 5),
        leaf("brand", "!=", "Nova"),
        {"not": leaf("price", "<", 100)},
        {"not": leaf("has_5g", "==", True)},
    ):
        assert evaluate(p3, node) is None
        assert not matches(p3, node)
    # but OR with a TRUE child passes (Kleene), as CONTRACT.md 4.4 says
    assert matches(p3, {"or": [leaf("price", "<", 100), leaf("os", "==", "android")]})
    assert evaluate(p3, {"and": [leaf("price", "<", 1), leaf("os", "==", "ios")]}) is False


def test_dict_phone_and_none_filter(phones, phone_dict):
    assert matches(phone_dict, leaf("ram_gb", ">=", 8))
    assert matches(phones["B0TEST0003"], None)
    assert len(allowed_docs(phones.values(), None)) == 3


@pytest.mark.parametrize(
    "bad",
    [
        leaf("price", "~", 3),
        leaf("color", "==", "red"),
        leaf("brand", "<", "A"),
        leaf("has_5g", "==", 1),
        leaf("os", "==", "windows"),
        {"and": []},
        {"not": [leaf("price", "<", 3)]},
        leaf("ram_gb", "in", []),
    ],
)
def test_invalid_filters_rejected(phones, bad):
    with pytest.raises(SchemaError):
        matches(phones["B0TEST0001"], bad)


# ------------------------------------------------------------------ parser


def flat(f):
    if f is None:
        return set()
    leaves = f["and"] if "and" in f else [f]
    return {(x["field"], x["op"], x["value"]) for x in leaves}


@pytest.mark.parametrize(
    "text, expected, residual",
    [
        ("samsung under $300 with 5g", {("price", "<", 300), ("has_5g", "==", True)}, "samsung"),
        ("8gb ram 128gb", {("ram_gb", ">=", 8), ("storage_gb", ">=", 128)}, ""),
        ("compact iphone", {("screen_in", "<=", 6.2), ("os", "==", "ios")}, "iphone"),
        ("big battery phone", set(), "big battery phone"),
        ("phone", set(), "phone"),
        ("budget phone", set(), "budget phone"),
        ("phone below 200 dollars", {("price", "<", 200)}, "phone"),
        ("less than $150", {("price", "<", 150)}, ""),
        ("$200-400 phone", {("price", ">=", 200), ("price", "<=", 400)}, "phone"),
        ("phone between 100 and 250", {("price", ">=", 100), ("price", "<=", 250)}, "phone"),
        ("over $800 flagship", {("price", ">", 800)}, "flagship"),
        ("at least 6 gb ram android", {("ram_gb", ">=", 6), ("os", "==", "android")}, ""),
        ("pixel 256 gb storage", {("storage_gb", ">=", 256)}, "pixel"),
        ("1TB storage", {("storage_gb", ">=", 1024)}, ""),
        ("4gb 64gb", {("ram_gb", ">=", 4), ("storage_gb", ">=", 64)}, ""),
        ("phone with 5,000 mAh battery", {("battery_mah", ">=", 5000)}, "phone"),
        ("6.5 inch motorola", {("screen_in", ">=", 6.3), ("screen_in", "<=", 6.7)}, "motorola"),
        ("screen under 6 inches", {("screen_in", "<=", 6.0)}, "screen"),
        ("small phone ios", {("screen_in", "<=", 6.2), ("os", "==", "ios")}, "phone"),
        ("good camera 4g phone", set(), "good camera 4g phone"),
    ],
)
def test_parser(text, expected, residual):
    f, rest = parse_constraints(text)
    assert flat(f) == expected
    assert rest == residual


def test_parser_output_is_valid_filter(phones):
    f, _ = parse_constraints("android 5g under $350 with 8gb ram 128gb 5000mah 6.5 inch")
    assert len(flat(f)) == 8
    assert allowed_docs(phones.values(), f) == {"B0TEST0001"}


def test_resolve_filter_sources():
    given = Query("q01", "phone under $300", "dev", filters=leaf("price", "<", 250))
    assert resolve_filter(given) == (leaf("price", "<", 250), "phone under $300", "queries.jsonl")
    f, text, src = resolve_filter(Query("q02", "samsung under $300", "dev"))
    assert src == "parser" and text == "samsung" and flat(f) == {("price", "<", 300)}
    assert resolve_filter(Query("q03", "great camera", "dev")) == (None, "great camera", "none")
    # residual empty -> the original text goes to the ranker
    assert resolve_filter(Query("q04", "under $300", "dev"))[1] == "under $300"


# ------------------------------------------------------------------ FilteredRetriever


@pytest.fixture
def collection(make_phone):
    import dataclasses

    from phonesense.schemas import Specs

    rows = [
        ("A", "nova camera phone", 199.0, True),
        ("B", "nova camera camera", 499.0, True),
        ("C", "pebble camera", 250.0, False),
        ("D", "zeta camera phone", None, True),
        ("E", "battery phone", 120.0, True),
    ]
    out = []
    for doc, title, price, g5 in rows:
        p = make_phone(doc, title)
        specs = dataclasses.replace(p.specs, has_5g=g5)
        out.append(dataclasses.replace(p, price=price, specs=specs))
    assert isinstance(out[0].specs, Specs)
    return out


@pytest.mark.parametrize("mode", ["pre", "post"])
def test_filtered_only_allowed(collection, mode):
    idx = InvertedIndex.build(collection, Analyzer())
    base = BM25Retriever(index=idx)
    fr = FilteredRetriever(base, collection, mode=mode)
    q = Query("q01", "camera", "dev", filters=leaf("price", "<", 300))
    res = fr.search(q, k=10)
    assert {d for d, _ in res} == {"A", "C"}  # D has price null -> excluded
    assert fr.log["q01"]["source"] == "queries.jsonl" and fr.log["q01"]["n_allowed"] == 3
    assert fr.name == "bm25_k1.2_b0.75+filters"


def test_prefilter_scores_equal_unfiltered(collection):
    idx = InvertedIndex.build(collection, Analyzer())
    for base in (BM25Retriever(index=idx), TfidfRetriever(index=idx)):
        full = dict(base.search(Query("q01", "camera phone", "dev"), k=10))
        fr = FilteredRetriever(base, collection, mode="pre")
        got = fr.search(Query("q01", "camera phone 5g", "dev"), k=10)  # parser: has_5g
        assert {d for d, _ in got} == {"A", "B", "D", "E"}  # C has has_5g False
        for d, s in got:
            assert s == pytest.approx(full[d])


def test_fewer_than_k_not_padded_and_no_filter(collection):
    idx = InvertedIndex.build(collection, Analyzer())
    fr = FilteredRetriever(BM25Retriever(index=idx), collection)
    assert len(fr.search(Query("q01", "camera", "dev", filters=leaf("price", "<", 150)), 10)) == 0
    plain = fr.search(Query("q02", "camera", "dev"), k=2)
    assert len(plain) == 2 and fr.log["q02"]["source"] == "none"


def test_post_mode_for_retriever_without_allowed(collection):
    class Dense:  # stand-in for a retriever that cannot pre-filter
        name = "dense"

        def search(self, query, k=100):
            return [(p.doc_id, 1.0 / (i + 1)) for i, p in enumerate(collection)][:k]

    with pytest.raises(TypeError):
        FilteredRetriever(Dense(), collection, mode="pre")
    fr = FilteredRetriever(Dense(), collection, mode="post")
    res = fr.search(Query("q01", "x", "dev", filters=leaf("has_5g", "==", True)), k=2)
    assert [d for d, _ in res] == ["A", "B"]


def test_filtered_run_file_and_log(collection, tmp_path):
    from phonesense.retrieval.base import run_retriever

    idx = InvertedIndex.build(collection, Analyzer())
    fr = FilteredRetriever(BM25Retriever(index=idx, name="bm25_tuned"), collection)
    qs = [Query("q01", "camera under $300", "dev"), Query("q02", "camera", "dev")]
    p = run_retriever(fr, qs, runs_dir=tmp_path)
    assert p.name == "bm25_tuned+filters_dev.run"
    io.read_run_entries(p)
    log = fr.write_log(tmp_path / "x.filters.json")
    assert '"parser"' in log.read_text() and '"none"' in log.read_text()


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_subset_analysis():
    mod = _load("run_t06_t07")
    pq = {
        "q01": {"P@10": 0.5, "MAP": 0.5, "nDCG@10": 0.5, "judged@10": 1.0},
        "q02": {"P@10": 0.1, "MAP": 0.1, "nDCG@10": 0.1, "judged@10": 1.0},
    }
    better = {q: {m: v + 0.2 for m, v in d.items()} for q, d in pq.items()}
    sub = mod.subset_analysis(
        {"bm25_tuned_dev": {"per_query": pq}, "bm25_tuned+filters_dev": {"per_query": better}},
        {"q01"},
    )
    row = sub["runs"]["bm25_tuned+filters_dev"]
    assert row["with_constraints"]["n"] == 1
    assert row["with_constraints"]["nDCG@10"] == pytest.approx(0.7)
    assert row["without_constraints"]["MAP"] == pytest.approx(0.3)
    assert "| bm25_tuned+filters_dev | with constraints | 1 |" in mod.subset_markdown(sub)
