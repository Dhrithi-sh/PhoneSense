from dhvani.eval.experiments import (
    SAMPLE_DIR,
    read_queries,
    results_rows,
    run_experiments,
    stemming_comparison,
    wins_and_losses,
)
from dhvani.eval.metrics import evaluate, read_qrels, read_run
from dhvani.rank.sample_index import SampleIndex


def sample():
    return read_queries(SAMPLE_DIR / "queries.tsv"), read_qrels(SAMPLE_DIR / "qrels.txt")


# --- experiment runner -------------------------------------------------------

def test_runner_writes_one_run_file_per_mode_and_ranker(tmp_path):
    queries, qrels = sample()
    results, _, skipped = run_experiments(queries, qrels, modes=("none", "light"),
                                          rankers=("lnc", "bm25"), out_dir=tmp_path)
    assert skipped == []
    files = sorted(p.name for p in (tmp_path / "runs").iterdir())
    assert files == ["light_bm25.txt", "light_lnc.txt", "none_bm25.txt", "none_lnc.txt"]
    assert set(read_run(tmp_path / "runs" / "none_lnc.txt")) <= {q[0] for q in queries}


def test_runner_skips_modes_that_are_not_built(tmp_path):
    queries, qrels = sample()

    def loader(mode):
        return SampleIndex.load(mode) if mode == "none" else None

    results, _, skipped = run_experiments(queries, qrels, loader=loader, modes=("none", "yass"),
                                          rankers=("lnc",), out_dir=tmp_path)
    assert skipped == ["yass"]
    assert set(results) == {("none", "lnc")}


def test_runner_numbers_match_evaluate_by_hand(tmp_path):
    queries, qrels = sample()
    results, _, _ = run_experiments(queries, qrels, modes=("none",), rankers=("bm25",), out_dir=tmp_path)
    run = read_run(tmp_path / "runs" / "none_bm25.txt")
    by_query = {qid: qrels[need] for qid, need, _f, _ in queries}
    _, means = evaluate({q: run.get(q, []) for q in by_query}, by_query, k=10)
    assert abs(results[("none", "bm25")]["all"]["MAP"] - means["MAP"]) < 1e-12


def test_results_split_by_query_form(tmp_path):
    queries, qrels = sample()
    results, _, _ = run_experiments(queries, qrels, modes=("none",), rankers=("net",), out_dir=tmp_path)
    groups = results[("none", "net")]
    assert {"all", "hindi", "hinglish", "english"} <= set(groups)
    # Hindi matches exactly and English works through translation.
    assert groups["hindi"]["MAP"] > 0.5
    assert groups["english"]["MAP"] > 0.5


def test_stemming_comparison_and_wins_losses(tmp_path):
    queries, qrels = sample()
    results, per_query_ap, _ = run_experiments(queries, qrels, modes=("none", "light"),
                                               rankers=("net",), out_dir=tmp_path)
    table = stemming_comparison(results, "net")
    assert [row[0] for row in table] == ["none", "light"]
    # The sample index has no stemming, so light can't win or lose anything yet.
    wins, losses, ties, _ = wins_and_losses(per_query_ap, "net")["light"]
    assert (wins, losses, ties) == (0, 0, len(queries))
    assert len(results_rows(results)) == 2 * 4


def test_speedup_table_has_every_method_and_sane_numbers():
    from dhvani.eval.experiments import speedup_table
    queries, _ = sample()
    rows = speedup_table(queries, SampleIndex.load(), k=3, champion_r=(1, 50))
    names = [r[0] for r in rows]
    assert names == ["index elimination", "champion lists, r=1", "champion lists, r=50", "recent-news tiers",
                     "cluster pruning, b=1", "cluster pruning, b=3", "impact-ordered, first 20",
                     "impact-ordered, first 50", "impact-ordered, weight >= 0.5 x best"]
    for _name, scored, kept in rows:
        assert 0.0 < scored <= 1.0 and 0.0 <= kept <= 1.0
    big_r = dict((n, (s, k)) for n, s, k in rows)["champion lists, r=50"]
    assert big_r == (1.0, 1.0)     # r bigger than any postings list = exact search


def test_needs_files_give_every_query_in_four_forms():
    from dhvani.eval.experiments import read_needs
    rows = read_needs()
    assert len(rows) >= 64 and len({r[0] for r in rows}) == len(rows)       # qids are unique
    r01 = [r for r in rows if r[1] == "R01"]
    assert {r[2] for r in r01} == {"hindi", "hinglish", "messy", "english"}
    assert all(r[3] for r in rows)


def test_needs_reader_only_reads_tsv_blocks(tmp_path):
    from dhvani.eval.experiments import read_needs
    (tmp_path / "x-needs.md").write_text(
        "| Need | What |\n|---|---|\n\n```tsv\nqid\tneed_id\tform\tquery\nX01_hi\tX01\thindi\tदिल्ली बारिश\n```\n"
        "```python\nnot\ta\tquery\trow\n```\n", encoding="utf-8")
    assert read_needs(tmp_path) == [("X01_hi", "X01", "hindi", "दिल्ली बारिश")]


def test_default_loader_uses_the_sample_in_tests():
    from dhvani.eval.experiments import default_loader
    assert isinstance(default_loader("none"), SampleIndex)


def test_query_builder_runs_the_analyzer_for_the_mode():
    from dhvani.eval.experiments import build_query, build_query_no_xling
    with_xling = build_query("delhi rain", "none")
    without = build_query_no_xling("delhi rain", "none")
    sources = lambda q: {s for t in q["tokens"] for _w, _x, s in t["expansions"]}
    assert "xling" in sources(with_xling) and "xling" not in sources(without)
