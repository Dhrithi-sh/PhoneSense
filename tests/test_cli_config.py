import subprocess
import sys
from pathlib import Path

from phonesense import config

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "phonesense.eval.cli", *args],
                          cwd=REPO_ROOT, capture_output=True, text=True)


def test_cli_accepts_fixtures():
    res = run_cli("--run", "tests/fixtures/sample.run",
                  "--qrels", "tests/fixtures/sample_qrels.txt")
    assert res.returncode == 0, res.stderr
    assert "Inputs are valid" in res.stdout and "not implemented yet" in res.stdout


def test_cli_rejects_invalid(tmp_path):
    bad = tmp_path / "bad.run"
    bad.write_text("q01 Q0 a 1 1.0 x_dev\nq01 Q0 a 2 0.5 x_dev\n")
    res = run_cli("--run", str(bad), "--qrels", "tests/fixtures/sample_qrels.txt")
    assert res.returncode == 1 and "duplicate" in res.stderr


def test_config_seed_and_paths():
    assert config.SEED == 42 == config.get_config()["seed"]
    assert config.get_path("phones") == REPO_ROOT / "data" / "processed" / "phones.jsonl"
    assert config.get_path("raw_meta").parent == REPO_ROOT / "data" / "raw"
    assert config.get_config()["dataset"]["category"] == "Cell_Phones_and_Accessories"
