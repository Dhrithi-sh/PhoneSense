import importlib.util
import math
import random
import shutil
from pathlib import Path

import pytest

from phonesense.index import Analyzer, InvertedIndex
from phonesense.io import write_qrels, write_queries
from phonesense.retrieval.bm25 import BM25FRetriever, BM25Retriever, idf
from phonesense.schemas import Query

REPO = Path(__file__).resolve().parent.parent


def q(text: str) -> Query:
    return Query(qid="q01", text=text, split="dev")


@pytest.fixture
def three(make_phone):
    return [
        make_phone("d1", "camera camera phone"),
        make_phone("d2", "phone battery"),
        make_phone("d3", "battery battery battery"),
    ]


def test_hand_computed(three):
    """Worked example from the bm25.py docstring (k1 = 1.2, b = 0.75).

    N = 3, avgdl = (3 + 2 + 3) / 3 = 8/3
    idf(camera) = ln(1 + (3 - 1 + .5)/(1 + .5)) = ln(8/3)  = 0.9808293
    idf(phone)  = ln(1 + (3 - 2 + .5)/(2 + .5)) = ln(1.6)  = 0.4700036
    K(d1) = 1.2 * (.25 + .75 * 3 / (8/3)) = 1.3125;  K(d2) = 1.2 * (.25 + .75 * 2/(8/3)) = 0.975
    d1 = 0.9808293 * 2 * 2.2 / 3.3125 + 0.4700036 * 2.2 / 2.3125 = 1.3028373 + 0.4471386
    d2 = 0.4700036 * 2.2 / 1.975 = 0.5235483
    """
    r = BM25Retriever(analyzer=Analyzer(stem=False))
    r.fit(three)
    res = r.search(q("camera phone"), k=10)
    assert [d for d, _ in res] == ["d1", "d2"]
    assert res[0][1] == pytest.approx(1.7499759, abs=1e-6)
    assert res[1][1] == pytest.approx(0.5235483, abs=1e-6)
    assert idf(1, 3) == pytest.approx(math.log(8 / 3))
    assert r.name == "bm25_k1.2_b0.75"


def test_idf_never_negative():
    assert all(idf(df, 10) > 0 for df in range(1, 11))


def test_b_zero_removes_length_normalization(make_phone):
    phones = [
        make_phone("short", "camera"),
        make_phone("long", "camera " + "x y z w " * 10),
        make_phone("other", "battery"),
    ]
    r = BM25Retriever(k1=1.2, b=0.0)
    r.fit(phones)
    s = dict(r.search(q("camera")))
    assert s["short"] == pytest.approx(s["long"])
    r.set_params(1.2, 0.75)
    s = dict(r.search(q("camera")))
    assert s["short"] > s["long"]


def test_k1_controls_saturation(make_phone):
    phones = [
        make_phone("one", "camera a b c d"),
        make_phone("five", "camera " * 5),
        make_phone("x", "battery"),
    ]
    r = BM25Retriever(b=0.0)
    r.fit(phones)

    def ratio(k1):
        r.set_params(k1, 0.0)
        s = dict(r.search(q("camera")))
        return s["five"] / s["one"]

    assert ratio(0.0) == pytest.approx(1.0)  # k1 = 0: binary presence
    assert 1.0 < ratio(0.5) < ratio(1.2) < ratio(3.0) < 5.0  # more k1 -> closer to raw tf


def test_query_tf_modes(three):
    once = BM25Retriever(analyzer=Analyzer(stem=False), query_tf="once")
    once.fit(three)
    assert once.search(q("camera camera")) == once.search(q("camera"))
    k3 = BM25Retriever(analyzer=Analyzer(stem=False), query_tf="k3", k3=8)
    k3.fit(three)
    single, double = k3.search(q("camera"))[0][1], k3.search(q("camera camera"))[0][1]
    assert double / single == pytest.approx((9 * 2 / 10) / (9 * 1 / 9))
    with pytest.raises(ValueError):
        BM25Retriever(query_tf="sum")


def test_ties_oov_and_determinism(make_phone):
    phones = [make_phone(d, "camera phone") for d in ("Z9", "B2", "A1")] + [
        make_phone("X0", "battery")
    ]
    r = BM25Retriever()
    r.fit(phones)
    assert [d for d, _ in r.search(q("camera"))] == ["A1", "B2", "Z9"]
    assert r.search(q("zzzz")) == [] and r.search(q("the of")) == []
    with pytest.raises(RuntimeError):
        BM25Retriever().search(q("x"))


def test_bm25f_single_field_equals_bm25(make_phone):
    rng = random.Random(0)
    words = [f"w{i}" for i in range(15)]
    phones = [
        make_phone(
            f"P{i:02d}",
            " ".join(rng.choices(words, k=rng.randint(1, 8))),
            [" ".join(rng.choices(words, k=rng.randint(0, 20)))],
            " ".join(rng.choices(words, k=rng.randint(0, 30))),
        )
        for i in range(30)
    ]
    idx = InvertedIndex.build(phones, Analyzer())
    for field in ("title", "features", "description"):
        plain = BM25Retriever(0.9, 0.4, field=field, index=idx)
        f = BM25FRetriever({field: 1.0}, 0.9, 0.4, index=idx)
        for query in ("w1 w2", "w3", "w4 w5 w6 w14"):
            a, b = plain.search(q(query)), f.search(q(query))
            assert [d for d, _ in a] == [d for d, _ in b]
            assert [s for _, s in a] == pytest.approx([s for _, s in b])


def test_bm25f_weights_sum_fields(make_phone):
    phones = [
        make_phone("T", "camera", [], "phone"),
        make_phone("D", "phone", [], "camera"),
        make_phone("N", "battery", [], "battery"),
    ]
    idx = InvertedIndex.build(phones, Analyzer())
    f = BM25FRetriever({"title": 3, "description": 1}, index=idx)
    t = BM25Retriever(field="title", index=idx).score_all("camera")
    d = BM25Retriever(field="description", index=idx).score_all("camera")
    got = dict(f.search(q("camera")))
    assert got["T"] == pytest.approx(3 * t[0] + d.get(0, 0.0))
    assert [x for x, _ in f.search(q("camera"))] == ["T", "D"]  # title match wins
    assert f.name.startswith("bm25f_")
    with pytest.raises(ValueError):
        BM25FRetriever({"title": 0})


# ------------------------------------------------------------------ scripts


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("split", ["test", "all"])
def test_tune_refuses_non_dev(split, capsys):
    tune = _load_script("tune_bm25")
    assert tune.main(["--split", split]) == 2
    err = capsys.readouterr().err
    assert "only tunes on the dev queries" in err and "invalidate" in err


def _tiny_collection(make_phone):
    rng = random.Random(1)
    vocab = ["camera", "battery", "screen", "cheap", "samsung", "apple", "pixel", "zoom"]
    phones = [
        make_phone(
            f"P{i:02d}",
            " ".join(rng.choices(vocab, k=4)),
            [" ".join(rng.choices(vocab, k=rng.randint(0, 12)))],
            " ".join(rng.choices(vocab, k=rng.randint(0, 25))),
        )
        for i in range(40)
    ]
    queries = [
        Query(f"q{i + 1:02d}", t, "dev")
        for i, t in enumerate(["camera zoom", "battery cheap", "samsung screen", "apple camera"])
    ]
    queries.append(Query("q05", "pixel", "test"))
    qrels = {
        qq.qid: {p.doc_id: rng.choice([0, 1, 2, 3]) for p in rng.sample(phones, 12)}
        for qq in queries
    }
    return phones, queries, qrels


def test_tune_end_to_end(make_phone, tmp_path, monkeypatch):
    phones, queries, qrels = _tiny_collection(make_phone)
    idx = InvertedIndex.build(phones, Analyzer())
    write_queries(tmp_path / "queries.jsonl", queries)
    write_qrels(tmp_path / "qrels.txt", qrels)
    tune = _load_script("tune_bm25")
    monkeypatch.setattr(tune, "open_index", lambda sample: (idx, False))
    rc = tune.main(
        [
            "--queries",
            str(tmp_path / "queries.jsonl"),
            "--qrels",
            str(tmp_path / "qrels.txt"),
            "--out",
            str(tmp_path / "t.json"),
            "--figures-dir",
            str(tmp_path),
            "--no-write-config",
            "--objective",
            "map",
        ]
    )
    assert rc == 0
    import json

    rep = json.loads((tmp_path / "t.json").read_text())
    assert rep["objective"] == "MAP" and rep["split"] == "dev" and rep["n_queries"] == 4
    assert len(rep["grid"]) == 9 * 11 and len(rep["bm25f"]["grid"]) == 12
    assert rep["best"]["MAP"] == max(r["MAP"] for r in rep["grid"])
    assert (tmp_path / "bm25_tuning_heatmap.png").exists()
    assert (tmp_path / "bm25f_tuning_heatmap.pdf").exists()


def test_write_tuned_keeps_rest_of_config(tmp_path):
    tune = _load_script("tune_bm25")
    cfg = tmp_path / "config.yaml"
    shutil.copy(REPO / "config.yaml", cfg)
    before = cfg.read_text(encoding="utf-8")
    tune.write_tuned({"k1": 0.8, "b": 0.3, "field_weights": {"title": 5.0}}, cfg)
    after = cfg.read_text(encoding="utf-8")
    import yaml

    assert yaml.safe_load(after)["bm25"]["tuned"]["k1"] == 0.8
    assert yaml.safe_load(after)["seed"] == yaml.safe_load(before)["seed"]
    assert after.count("# BEGIN bm25.tuned") == 1 and "# T05 BM25" in after
    tune.write_tuned({"k1": 1.0, "b": 0.5}, cfg)  # idempotent re-write
    assert yaml.safe_load(cfg.read_text(encoding="utf-8"))["bm25"]["tuned"] == {"k1": 1.0, "b": 0.5}
