"""Unit tests for TREC run pooling tool (dhvani/eval/pool.py)."""

from pathlib import Path
from dhvani.eval.pool import pool_runs


def test_pool_runs_basic(tmp_path: Path):
    run1 = tmp_path / "run1.txt"
    run2 = tmp_path / "run2.txt"

    run1.write_text(
        "N01 Q0 doc_1 1 0.95 bm25\n"
        "N01 Q0 doc_2 2 0.85 bm25\n"
        "N01 Q0 doc_3 3 0.75 bm25\n"
        "N02 Q0 doc_A 1 0.90 bm25\n",
        encoding="utf-8",
    )
    run2.write_text(
        "N01 Q0 doc_2 1 0.98 dense\n"
        "N01 Q0 doc_4 2 0.88 dense\n"
        "N02 Q0 doc_B 1 0.92 dense\n"
        "N02 Q0 doc_A 2 0.81 dense\n",
        encoding="utf-8",
    )

    pooled = pool_runs([str(run1), str(run2)], top_k=2)

    assert pooled["N01"] == {"doc_1", "doc_2", "doc_4"}
    assert pooled["N02"] == {"doc_A", "doc_B"}


def test_pool_runs_top_k_cutoff(tmp_path: Path):
    run = tmp_path / "run.txt"
    lines = [f"N01 Q0 doc_{i} {i} {1.0 - i * 0.05} run" for i in range(1, 20)]
    run.write_text("\n".join(lines) + "\n", encoding="utf-8")

    pooled = pool_runs([str(run)], top_k=5)
    assert len(pooled["N01"]) == 5
    assert pooled["N01"] == {f"doc_{i}" for i in range(1, 6)}


def test_pool_runs_collapses_surface_forms_into_need_id(tmp_path: Path):
    """Test 1: R01_hi, R01_hinglish, R01_messy, R01_en all collapse into R01 by default."""
    run = tmp_path / "run_forms.txt"
    run.write_text(
        "R01_hi Q0 doc_1 1 0.95 bm25\n"
        "R01_hinglish Q0 doc_2 1 0.92 bm25\n"
        "R01_messy Q0 doc_3 1 0.89 bm25\n"
        "R01_en Q0 doc_4 1 0.86 bm25\n",
        encoding="utf-8",
    )

    pooled = pool_runs([str(run)], top_k=10)
    assert set(pooled.keys()) == {"R01"}
    assert pooled["R01"] == {"doc_1", "doc_2", "doc_3", "doc_4"}


def test_pool_runs_deduplicates_docs_within_need_pool(tmp_path: Path):
    """Test 2: Documents appearing in multiple forms are deduplicated within the R01 pool."""
    run1 = tmp_path / "run_bm25.txt"
    run2 = tmp_path / "run_dense.txt"

    # doc_shared appears across all 4 linguistic forms and both rankers
    run1.write_text(
        "R01_hi Q0 doc_shared 1 0.99 bm25\n"
        "R01_hi Q0 doc_hi_only 2 0.80 bm25\n"
        "R01_hinglish Q0 doc_shared 1 0.97 bm25\n"
        "R01_messy Q0 doc_shared 1 0.94 bm25\n"
        "R01_en Q0 doc_shared 1 0.91 bm25\n"
        "R01_en Q0 doc_en_only 2 0.82 bm25\n",
        encoding="utf-8",
    )
    run2.write_text(
        "R01_hi Q0 doc_shared 1 0.98 dense\n"
        "R01_en Q0 doc_shared 1 0.95 dense\n"
        "R01_en Q0 doc_dense_only 2 0.79 dense\n",
        encoding="utf-8",
    )

    pooled = pool_runs([str(run1), str(run2)], top_k=10)
    assert set(pooled.keys()) == {"R01"}
    # doc_shared must appear only once in the pool set
    assert pooled["R01"] == {"doc_shared", "doc_hi_only", "doc_en_only", "doc_dense_only"}
    assert len(pooled["R01"]) == 4


def test_pool_runs_raw_qid_preserves_surface_forms(tmp_path: Path):
    """Test 3: --raw-qid (per_need=False) preserves the original four surface form qids."""
    run = tmp_path / "run_raw.txt"
    run.write_text(
        "R01_hi Q0 doc_1 1 0.95 bm25\n"
        "R01_hinglish Q0 doc_2 1 0.92 bm25\n"
        "R01_messy Q0 doc_3 1 0.89 bm25\n"
        "R01_en Q0 doc_4 1 0.86 bm25\n",
        encoding="utf-8",
    )

    pooled = pool_runs([str(run)], top_k=10, per_need=False)
    assert set(pooled.keys()) == {"R01_hi", "R01_hinglish", "R01_messy", "R01_en"}
    assert pooled["R01_hi"] == {"doc_1"}
    assert pooled["R01_hinglish"] == {"doc_2"}
    assert pooled["R01_messy"] == {"doc_3"}
    assert pooled["R01_en"] == {"doc_4"}


def test_pool_runs_preserves_queries_without_underscores(tmp_path: Path):
    """Test 4: Queries without underscores such as N01 or topic05 remain unchanged."""
    run = tmp_path / "run_plain.txt"
    run.write_text(
        "N01 Q0 doc_A 1 0.95 bm25\n"
        "N02 Q0 doc_B 1 0.90 bm25\n"
        "topic10 Q0 doc_C 1 0.85 bm25\n",
        encoding="utf-8",
    )

    pooled = pool_runs([str(run)], top_k=10, per_need=True)
    assert set(pooled.keys()) == {"N01", "N02", "topic10"}
    assert pooled["N01"] == {"doc_A"}
    assert pooled["N02"] == {"doc_B"}
    assert pooled["topic10"] == {"doc_C"}


def test_pool_runs_queries_tsv_mapping_precedence(tmp_path: Path):
    """Test 5: --queries queries.tsv mapping takes precedence over fallback qid.split('_')[0]."""
    queries_tsv = tmp_path / "queries.tsv"
    queries_tsv.write_text(
        "qid\tneed_id\tform\tquery\n"
        "custom_alpha_hi\tNEED_SUPER_1\thindi\tबारिश\n"
        "custom_beta_en\tNEED_SUPER_1\tenglish\train\n"
        "custom_gamma_hi\tNEED_SUPER_2\thindi\tचुनाव\n",
        encoding="utf-8",
    )

    run = tmp_path / "run_mapped.txt"
    run.write_text(
        "custom_alpha_hi Q0 doc_1 1 0.95 bm25\n"
        "custom_beta_en Q0 doc_2 1 0.90 bm25\n"
        "custom_gamma_hi Q0 doc_3 1 0.85 bm25\n",
        encoding="utf-8",
    )

    # Without queries file, fallback would split on '_' giving 'custom' for all three
    pooled_fallback = pool_runs([str(run)], top_k=10)
    assert set(pooled_fallback.keys()) == {"custom"}

    # With queries file, explicit TSV mapping takes precedence
    pooled_mapped = pool_runs([str(run)], top_k=10, queries_file=str(queries_tsv))
    assert set(pooled_mapped.keys()) == {"NEED_SUPER_1", "NEED_SUPER_2"}
    assert pooled_mapped["NEED_SUPER_1"] == {"doc_1", "doc_2"}
    assert pooled_mapped["NEED_SUPER_2"] == {"doc_3"}


def test_pool_output_format_and_cli(tmp_path: Path):
    """Test 6: Output remains exactly 'need_id 0 doc_id 0' in the expected formats.md Format 5."""
    import sys
    from unittest.mock import patch
    from dhvani.eval.pool import main

    run = tmp_path / "run_cli.txt"
    run.write_text(
        "R01_hi Q0 doc_101 1 0.95 bm25\n"
        "R01_en Q0 doc_102 1 0.90 bm25\n"
        "R02_hi Q0 doc_201 1 0.85 bm25\n",
        encoding="utf-8",
    )
    out_file = tmp_path / "judgments_pool.txt"

    test_args = ["pool.py", "--runs", str(run), "--top-k", "10", "--out", str(out_file)]
    with patch.object(sys, "argv", test_args):
        main()

    assert out_file.exists()
    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3

    # Format 5 invariant: need_id 0 doc_id 0
    assert lines[0] == "R01 0 doc_101 0"
    assert lines[1] == "R01 0 doc_102 0"
    assert lines[2] == "R02 0 doc_201 0"

