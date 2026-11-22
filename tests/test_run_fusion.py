"""scripts/run_fusion.py: file names, fused content, test-once guard (small synthetic runs)."""

import importlib.util
from pathlib import Path

import pytest

from phonesense.io import read_run, write_rankings
from phonesense.retrieval.fusion import reciprocal_rank_fusion, weighted_linear_fusion

A = {"q01": [("d1", 3.0), ("d2", 2.0), ("d3", 1.0)], "q02": [("d4", 1.0)]}
B = {"q01": [("d2", 0.9), ("d4", 0.5), ("d1", 0.1)], "q02": [("d5", 2.0), ("d4", 1.0)]}


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_fusion.py"
    spec = importlib.util.spec_from_file_location("run_fusion_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def setup(tmp_path, fixtures, monkeypatch):
    script = _load()
    monkeypatch.setattr(script, "get_path", lambda key: fixtures / "queries.jsonl")
    # fixtures/queries.jsonl: q01 and q02 are dev, q03 is test
    for split in ("dev", "test"):
        for name, run in (("sysA", A), ("sysB", B)):
            write_rankings(tmp_path / f"{name}_{split}.run", run, f"{name}_{split}", 100)
    return script, tmp_path


def test_run_names_match_what_the_ablation_looks_for():
    f = _load().fused_run_name
    assert f("rrf", ["a", "b"], 5, None, "dev") == "rrf_k5_a+b_dev"
    assert f("linear", ["a", "b"], None, [0.55, 0.45], "test") == "fusion_linear_w55-45_a+b_test"
    assert f("linear", ["a", "b"], None, None, "dev") == "fusion_linear_w50-50_a+b_dev"


def test_rrf_run_contains_the_fused_ranking(setup):
    script, runs = setup
    assert script.main(["--method", "rrf", "--runs", "sysA", "sysB", "--rrf-k", "5",
                        "--split", "dev"], runs_dir=runs) == 0
    got = read_run(runs / "rrf_k5_sysA+sysB_dev.run")
    want = reciprocal_rank_fusion([A["q01"], B["q01"]], k=5)
    assert [d for d, _ in got["q01"]] == [d for d, _ in want]
    assert [s for _, s in got["q01"]] == pytest.approx([s for _, s in want])


def test_linear_run_uses_the_weights(setup):
    script, runs = setup
    assert script.main(["--method", "linear", "--runs", "sysA", "sysB", "--weights", "0.9", "0.1",
                        "--split", "dev"], runs_dir=runs) == 0
    got = read_run(runs / "fusion_linear_w90-10_sysA+sysB_dev.run")
    want = weighted_linear_fusion([A["q01"], B["q01"]], [0.9, 0.1])
    assert [d for d, _ in got["q01"]] == [d for d, _ in want]


def test_test_runs_are_written_once_unless_forced(setup, capsys):
    script, runs = setup
    argv = ["--method", "rrf", "--runs", "sysA", "sysB", "--rrf-k", "5", "--split", "test"]
    assert script.main(argv, runs_dir=runs) == 0
    assert script.main(argv, runs_dir=runs) == 1
    assert "run once" in capsys.readouterr().err
    assert script.main([*argv, "--force"], runs_dir=runs) == 0


def test_bad_arguments_are_reported(setup, capsys):
    script, runs = setup
    assert script.main(["--method", "linear", "--runs", "sysA", "sysB", "--weights", "1",
                        "--split", "dev"], runs_dir=runs) == 1
    assert script.main(["--method", "rrf", "--runs", "sysA", "nope", "--split", "dev"],
                       runs_dir=runs) == 1
    assert "missing" in capsys.readouterr().err
