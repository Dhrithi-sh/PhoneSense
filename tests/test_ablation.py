import importlib.util
import json
from pathlib import Path

import pytest

from phonesense import io
from phonesense.index import Analyzer, InvertedIndex
from phonesense.retrieval.base import run_retriever
from phonesense.retrieval.bm25 import BM25Retriever
from phonesense.retrieval.filters import FilteredRetriever
from phonesense.retrieval.tfidf import TfidfRetriever

REPO = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Reverse:
    """A stand-in 'dense' run: BM25 order reversed (so it differs per query)."""

    def __init__(self, base, name):
        self.base, self.name = base, name

    def search(self, query, k=100):
        res = self.base.search(query, k)
        return [(d, float(i + 1)) for i, (d, _) in enumerate(reversed(res))]


@pytest.fixture
def ablation_env(tmp_path, fixtures):
    phones = io.read_phones(fixtures / "phones.jsonl")
    queries = io.read_queries(fixtures / "queries.jsonl")
    idx = InvertedIndex.build(phones, Analyzer())
    bm25 = BM25Retriever(index=idx, name="bm25_tuned")
    systems = [
        TfidfRetriever(index=idx),
        BM25Retriever(index=idx, name="bm25_default"),
        bm25,
        FilteredRetriever(bm25, phones),
        Reverse(bm25, "dense_minilm"),
    ]
    runs = tmp_path / "runs"
    for split in ("dev", "test"):
        qs = [q for q in queries if q.split == split]
        for r in systems:
            run_retriever(r, qs, run_name=f"{r.name}_{split}", runs_dir=runs)
    return {
        "runs": runs,
        "results": tmp_path / "results",
        "figs": tmp_path / "figs",
        "qrels": fixtures / "sample_qrels.txt",
        "queries": fixtures / "queries.jsonl",
        "phones": fixtures / "phones.jsonl",
    }


def _run(mod, env, *extra):
    return mod.main(
        [
            "--runs-dir",
            str(env["runs"]),
            "--results-dir",
            str(env["results"]),
            "--figures-dir",
            str(env["figs"]),
            "--qrels",
            str(env["qrels"]),
            "--queries",
            str(env["queries"]),
            "--phones",
            str(env["phones"]),
            "--permutations",
            "200",
            *extra,
        ]
    )


def test_ablation_end_to_end(ablation_env):
    mod = _load("ablation")
    assert _run(mod, ablation_env) == 0
    res = ablation_env["results"]
    for name in (
        "ablation_test.md",
        "ablation_test.tex",
        "ablation_dev.md",
        "ablation_dev.tex",
        "ablation_incremental.md",
        "ablation_incremental.tex",
        "failure_cases.md",
    ):
        assert (res / name).exists(), name
    data = json.loads((res / "ablation.json").read_text())
    test_rows = {r["label"]: r for r in data["rows"]["test"]}
    assert test_rows["BM25 tuned"]["status"] == "ok"
    assert test_rows["Dense"]["run"] == "dense_minilm" and test_rows["Dense"]["status"] == "ok"
    assert test_rows["HyDE"]["status"] == "pending: no run"
    assert test_rows["PageRank rerank"]["owner"] == "C"
    assert {p["owner"] for p in data["pending"]} >= {"A", "B", "C"}
    best = data["best_overall"]["run"]
    dev = {r["run"]: r for r in data["rows"]["dev"] if r["status"] == "ok"}
    assert dev[best]["metrics"]["nDCG@10"] == max(r["metrics"]["nDCG@10"] for r in dev.values())
    md = (res / "ablation_test.md").read_text(encoding="utf-8")
    assert "| HyDE | B | Neural | pending: no run (B) |" in md and "MRR" in md
    assert "Best overall:" in md
    assert r"\begin{table*}" in (res / "ablation_test.tex").read_text(encoding="utf-8")
    inc = (res / "ablation_incremental.md").read_text(encoding="utf-8")
    assert "| + filters | bm25_tuned |" in inc
    fc = (res / "failure_cases.md").read_text(encoding="utf-8")
    assert "## q03" in fc and "Why:" in fc
    assert (ablation_env["figs"] / "ablation_bars.png").exists()
    assert (ablation_env["figs"] / "pr_key_systems.pdf").exists()


def test_ablation_without_qrels_all_pending(ablation_env, tmp_path):
    mod = _load("ablation")
    env = dict(ablation_env, qrels=tmp_path / "missing.txt")
    assert _run(mod, env) == 0  # missing qrels: no crash, every row pending
    data = json.loads((env["results"] / "ablation.json").read_text())
    assert data["best_overall"]["run"] is None
    assert all(r["status"].startswith("pending") for r in data["rows"]["test"])


def test_choose_best_refuses_test_results():
    mod = _load("ablation")
    dev = {
        "a": {"split": "dev", "run": "a_dev", "metrics": {"nDCG@10": 0.4}},
        "b": {"split": "dev", "run": "b_dev", "metrics": {"nDCG@10": 0.5}},
    }
    assert mod.choose_best(dev, ["a", "b"], "nDCG@10") == "b"
    with pytest.raises(AssertionError, match="dev only"):
        mod.choose_best({"a": {"split": "test", "run": "a_test", "metrics": {}}}, ["a"], "nDCG@10")


def test_match_bases_patterns():
    mod = _load("ablation")
    avail = {
        "dev": {"rrf_bm25+dense", "rrf_bm25+dense+ppr0.2", "dense_x"},
        "test": {"rrf_bm25+dense"},
    }
    assert mod.match_bases(["rrf_*"], avail, ["*+ppr*"]) == ["rrf_bm25+dense"]
    assert mod.match_bases(["nope_*", "dense_*"], avail) == ["dense_x"]
    assert mod.match_bases(["hyde_*"], avail) == []


def test_report_numbers_from_ablation(ablation_env, tmp_path):
    assert _run(_load("ablation"), ablation_env) == 0
    rn = _load("report_numbers")
    out = tmp_path / "gen"
    assert rn.main(["--results-dir", str(ablation_env["results"]), "--out-dir", str(out),
                    "--no-readme"]) == 0
    names = {"dataset_stats.tex", "index_stats.tex", "efficiency.tex", "bm25_tuning.tex",
             "filters_subset.tex", "numbers.tex", "experiments_retrieval_skeleton.md"}
    assert names <= {p.name for p in out.iterdir()}
    numbers = (out / "numbers.tex").read_text(encoding="utf-8")
    assert r"\newcommand{\bmTunedNDCG}{" in numbers and "% results/ablation.json" in numbers
    assert r"\newcommand{\bmTunedNDCG}{\textit{pending}}" not in numbers
    assert r"\newcommand{\kappaLLMHuman}{\textit{pending}}" in numbers  # no kappa.json here
    skel = (out / "experiments_retrieval_skeleton.md").read_text(encoding="utf-8")
    assert "## Ablation" in skel and "Best overall" in skel


def test_readme_summary_markers(tmp_path):
    rn = _load("report_numbers")
    readme = tmp_path / "README.md"
    readme.write_text("x\n<!-- BEGIN results-summary -->\nold\n<!-- END results-summary -->\ny\n")
    res = tmp_path / "res"
    res.mkdir()
    (res / "ablation_test.md").write_text("| table |\n")
    assert rn.update_readme(res, readme)
    text = readme.read_text()
    assert "| table |" in text and "old" not in text and text.endswith("y\n")
