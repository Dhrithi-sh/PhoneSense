import subprocess
import sys
from pathlib import Path

import pytest

from phonesense import io
from phonesense.eval import evaluate_run, results_table
from phonesense.eval.plots import pr_curve
from phonesense.retrieval.base import run_retriever
from phonesense.retrieval.bm25 import BM25Retriever
from phonesense.retrieval.tfidf import TfidfRetriever
from phonesense.schemas import Results

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def dev_runs(tmp_path, fixtures):
    phones = io.read_phones(fixtures / "phones.jsonl")
    queries = io.read_queries(fixtures / "queries.jsonl", split="dev")
    paths = []
    for r in (TfidfRetriever(), BM25Retriever(name="bm25_default")):
        r.fit(phones)
        paths.append(run_retriever(r, queries, runs_dir=tmp_path / "runs"))
    return paths


def test_end_to_end_results_validate(dev_runs, tmp_path, fixtures):
    out = tmp_path / "results"
    res = evaluate_run(
        dev_runs[0],
        fixtures / "sample_qrels.txt",
        queries_path=fixtures / "queries.jsonl",
        out_dir=out,
    )
    Results.from_dict(res)
    assert res["run"] == "tfidf_lnc.ltc_dev" and res["split"] == "dev"
    assert set(res["per_query"]) == {"q01", "q02"}  # q03 is test
    assert res["meta"]["relevance_threshold"] == 2
    for name in ("P@10", "MAP", "nDCG@10", "judged@10", "RR", "R-prec"):
        assert name in res["metrics"]
        mean = sum(v[name] for v in res["per_query"].values()) / 2
        assert res["metrics"][name] == pytest.approx(mean)
    loaded = io.read_results(out / "tfidf_lnc.ltc_dev.json")
    assert loaded.metrics == res["metrics"]


def test_split_mismatch_and_missing_qrels(dev_runs, tmp_path, fixtures):
    with pytest.raises(ValueError, match="does not match"):
        evaluate_run(
            dev_runs[0],
            fixtures / "sample_qrels.txt",
            "test",
            queries_path=fixtures / "queries.jsonl",
            write=False,
        )
    with pytest.raises(FileNotFoundError):
        evaluate_run(dev_runs[0], tmp_path / "nope.txt", write=False)


def test_excluded_queries_listed(dev_runs, tmp_path, fixtures):
    qrels = tmp_path / "q.txt"
    qrels.write_text("q01 0 B0TEST0001 3\nq02 0 B0TEST0002 1\n")
    res = evaluate_run(dev_runs[0], qrels, queries_path=fixtures / "queries.jsonl", write=False)
    assert list(res["per_query"]) == ["q01"] and res["meta"]["excluded_qids"] == ["q02"]


def test_table_and_plot(dev_runs, tmp_path, fixtures):
    res = [
        evaluate_run(
            p, fixtures / "sample_qrels.txt", queries_path=fixtures / "queries.jsonl", write=False
        )
        for p in dev_runs
    ]
    t = results_table(
        res, ["p@10", "map", "ndcg@10", "judged@10"], baseline="tfidf_lnc.ltc", permutations=500
    )
    assert "| tfidf_lnc.ltc_dev (baseline) |" in t.markdown
    assert "bm25_default_dev" in t.markdown and "judged@10" in t.markdown
    assert r"\begin{tabular}" in t.latex and r"tfidf\_lnc.ltc\_dev" in t.latex
    assert "nDCG@10" in t.significance and "judged@10" not in t.significance
    with pytest.raises(ValueError):
        results_table(res, ["MAP"], baseline="nope")

    qrels = io.read_qrels(fixtures / "sample_qrels.txt")
    runs = {
        Path(p).stem: {q: [d for d, _ in v] for q, v in io.read_run(p).items()} for p in dev_runs
    }
    paths = pr_curve(runs, qrels, "pr_test", qids=["q01", "q02"], out_dir=tmp_path)
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_cli_eval_and_table(dev_runs, tmp_path, fixtures):
    def cli(*args):
        return subprocess.run(
            [sys.executable, "-m", "phonesense.eval.cli", *args],
            cwd=REPO,
            capture_output=True,
            text=True,
        )

    out = tmp_path / "res"
    r = cli(
        "eval",
        "--run",
        *map(str, dev_runs),
        "--qrels",
        str(fixtures / "sample_qrels.txt"),
        "--queries",
        str(fixtures / "queries.jsonl"),
        "--out-dir",
        str(out),
    )
    assert r.returncode == 0, r.stderr
    assert "tfidf_lnc.ltc_dev: P@10=" in r.stdout
    r = cli(
        "table",
        "--results",
        str(out / "*.json"),
        "--metrics",
        "P@10",
        "nDCG@10",
        "--baseline",
        "tfidf_lnc.ltc",
        "--out",
        str(tmp_path / "t.md"),
        "--permutations",
        "200",
    )
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "t.md").exists() and (tmp_path / "t.tex").exists()
    r = cli(
        "compare",
        "--baseline",
        str(dev_runs[0]),
        "--runs",
        str(dev_runs[1]),
        "--qrels",
        str(fixtures / "sample_qrels.txt"),
        "--queries",
        str(fixtures / "queries.jsonl"),
        "--metric",
        "ndcg@10",
        "--permutations",
        "200",
    )
    assert r.returncode == 0, r.stderr
    assert "bm25_default_dev" in r.stdout
    r = cli("eval", "--run", str(dev_runs[0]), "--qrels", str(tmp_path / "missing.txt"))
    assert r.returncode == 1 and "not found" in r.stderr
