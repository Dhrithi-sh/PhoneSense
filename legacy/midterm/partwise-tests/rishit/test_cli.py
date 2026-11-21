import importlib.util
from pathlib import Path

CLI_PATH = Path(__file__).resolve().parents[2] / "app" / "cli.py"
spec = importlib.util.spec_from_file_location("dhvani_cli", CLI_PATH)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def test_plain_search_lists_results(capsys):
    text = cli.run(["मौसम", "--ranker", "lnc", "--k", "3"])
    assert "Results" in text
    assert "1. " in text and "nbt_2001" in text
    assert "Query vector" not in text  # no explanation unless asked


def test_explain_shows_every_stage():
    text = cli.run(["दिल्ली बारिश", "--explain", "--k", "2"])
    for section in ("1. Query object", "2. Query vector", "3. Postings touched",
                    "4. Candidates and top K", "5. Results"):
        assert section in text
    assert "idf = log10(N/df)" in text
    assert "pos=[" in text            # positions from the postings
    assert "= net score" in text      # full breakdown for the net ranker


def test_explain_bm25_shows_bm25_idf():
    text = cli.run(["कोहली शतक", "--ranker", "bm25", "--explain", "--k", "2"])
    assert "k1 = 1.2" in text and "b = 0.75" in text
    assert "BM25 contribution" in text


def test_unknown_word_is_marked():
    text = cli.run(["xyz", "--explain"])
    assert "dropped, not in index" in text
    assert "No matches." in text


def test_kal_query_shows_its_direction():
    text = cli.run(["कल बारिश", "--k", "2"])
    assert "[kal: tomorrow]" in text


def test_kal_explain_has_its_own_step():
    text = cli.run(["कल बारिश", "--k", "1", "--explain"])
    assert "4c. Date-aware kal" in text
    assert "x kal boost (tomorrow)" in text


def test_no_kal_turns_it_off():
    text = cli.run(["कल बारिश", "--k", "2", "--no-kal"])
    assert "kal:" not in text


def test_wire_story_is_collapsed_with_also_in():
    text = cli.run(["रेल लाइन उद्घाटन", "--k", "3"])
    assert "[also in: amarujala]" in text
    assert text.count("रेल लाइन का उद्घाटन") == 1


def test_no_collapse_shows_both_copies():
    text = cli.run(["रेल लाइन उद्घाटन", "--k", "3", "--no-collapse"])
    assert "aajtak_5004" in text and "amarujala_3004" in text


def test_authority_breakdown_in_explain():
    text = cli.run(["रेल लाइन उद्घाटन", "--k", "1", "--explain"])
    assert "x g(d)" in text and "PageRank" in text and "first to publish" in text


def test_no_authority_uses_plain_recency():
    text = cli.run(["रेल लाइन उद्घाटन", "--k", "1", "--explain", "--no-authority"])
    assert "x recency" in text and "PageRank" not in text


def test_speedups_report_how_many_were_scored():
    for name in ("elim", "champions", "tiers", "clusters", "impact"):
        text = cli.run(["दिल्ली बारिश", "--k", "3", "--speedup", name])
        assert "Scored" in text and "articles that share a query word" in text


def test_rrf_ranker_shows_ranks_in_each_list():
    text = cli.run(["दिल्ली बारिश", "--k", "2", "--ranker", "rrf", "--explain"])
    assert "ranks: lnc.ltc #" in text and "rrf = sum of 1 / (60 + rank)" in text


def test_diversify_flag_adds_the_mmr_step():
    text = cli.run(["दिल्ली बारिश", "--k", "3", "--diversify", "--explain"])
    assert "Diversify (MMR)" in text and "mmr: relevance" in text


def test_vague_query_gets_a_low_confidence_line():
    assert "Low confidence" in cli.run(["में के", "--k", "2"])
    assert "Low confidence" not in cli.run(["कोहली शतक", "--k", "2"])


def test_keep_listings_flag_is_accepted():
    assert "Results" in cli.run(["दिल्ली बारिश", "--k", "2", "--keep-listings"])


def test_prf_flag_reports_the_added_terms():
    assert "Feedback (Rocchio" in cli.run(["दिल्ली बारिश", "--k", "2", "--prf"])
