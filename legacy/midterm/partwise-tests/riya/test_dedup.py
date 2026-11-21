"""Unit tests and evaluation benchmark for Deduplication & Story Lineage Clustering (Task 6).

Benchmarks precision and recall across thresholds on the 100 labeled Hindi news pairs
stored in partwise-tests/riya/fixtures/dedup_pairs_100.json.
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List

import pytest

from dhvani.crawl.config import JACCARD_THRESHOLD, SHINGLE_SIZE, TEMPORAL_WINDOW_HOURS
from dhvani.crawl.dedup import (
    MinHashLSH,
    cluster_articles,
    cluster_corpus,
    compute_content_hash,
    compute_minhash,
    generate_shingles,
    is_agency_story,
    jaccard_similarity,
    parse_iso_datetime,
    within_temporal_window,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
LABELED_PAIRS_FIXTURE = FIXTURES_DIR / "dedup_pairs_100.json"


# ---------------------------------------------------------------------------
# Unit Tests: Primitives & Functions
# ---------------------------------------------------------------------------

def test_compute_content_hash():
    """Verify whitespace normalization and deterministic MD5 hashing."""
    text1 = "  प्रधानमंत्री   नरेंद्र मोदी ने आज \n\t वाराणसी का दौरा किया।  "
    text2 = "प्रधानमंत्री नरेंद्र मोदी ने आज वाराणसी का दौरा किया।"
    text3 = "उत्तर प्रदेश के मुख्यमंत्री योगी आदित्यनाथ ने विकास कार्यों का निरीक्षण किया।"

    h1 = compute_content_hash(text1)
    h2 = compute_content_hash(text2)
    h3 = compute_content_hash(text3)

    assert len(h1) == 32
    assert h1 == h2, "Whitespace variations must produce identical MD5 hashes"
    assert h1 != h3, "Distinct texts must produce distinct hashes"


def test_generate_shingles():
    """Verify sliding 4-word shingles, short text fallback, and empty handling."""
    hindi_text = "मौसम विभाग ने अगले 48 घंटों में भारी बारिश का अलर्ट जारी किया"
    shingles = generate_shingles(hindi_text, n=4)

    # 13 words -> 13 - 4 + 1 = 10 shingles
    assert len(shingles) == 10
    assert "मौसम विभाग ने अगले" in shingles
    assert "भारी बारिश का अलर्ट" in shingles
    assert "बारिश का अलर्ट जारी" in shingles

    # Short sentence fallback (< 4 words)
    short_text = "भारत माता की"
    short_shingles = generate_shingles(short_text, n=4)
    assert short_shingles == {"भारत माता की"}

    # Empty text
    assert generate_shingles("", n=4) == set()
    assert generate_shingles("   ", n=4) == set()


def test_jaccard_similarity():
    """Verify set-based Jaccard similarity coefficient calculations."""
    set_a = {"a", "b", "c", "d"}
    set_b = {"b", "c", "d", "e"}
    set_c = {"x", "y", "z"}

    # Exact identity
    assert jaccard_similarity(set_a, set_a) == 1.0

    # 3 in common, union size 5 -> 3/5 = 0.6
    assert abs(jaccard_similarity(set_a, set_b) - 0.6) < 1e-6

    # Completely disjoint
    assert jaccard_similarity(set_a, set_c) == 0.0

    # Empty sets
    assert jaccard_similarity(set(), set_a) == 0.0
    assert jaccard_similarity(set(), set()) == 0.0


def test_is_agency_story():
    """Verify detection of PTI, ANI, Bhasha, Univarta wire signatures."""
    pti_text = "नई दिल्ली, 6 अक्टूबर (भाषा) उच्चतम न्यायालय ने आज वायु प्रदूषण पर सुनवाई की।"
    ani_text = "Varanasi (ANI): Prime Minister Narendra Modi inaugurated development projects."
    univarta_text = "लखनऊ, वार्ता: मुख्यमंत्री ने राहत शिविरों का जायजा लिया।"
    clean_text = "दैनिक जागरण ब्यूरो: स्थानीय खेल प्रतियोगिता में युवाओं ने उत्साह दिखाया।"

    assert is_agency_story(pti_text) is True
    assert is_agency_story(ani_text) is True
    assert is_agency_story(univarta_text) is True
    assert is_agency_story(clean_text) is False


def test_within_temporal_window():
    """Verify +/- 24-hour publication window filtering."""
    t_base = "2026-10-06T12:00:00+05:30"
    t_plus_2h = "2026-10-06T14:00:00+05:30"
    t_plus_23h = "2026-10-07T11:00:00+05:30"
    t_plus_25h = "2026-10-07T13:00:00+05:30"
    t_plus_5d = "2026-10-11T12:00:00+05:30"

    assert within_temporal_window(t_base, t_plus_2h, window_hours=24) is True
    assert within_temporal_window(t_base, t_plus_23h, window_hours=24) is True
    assert within_temporal_window(t_base, t_plus_25h, window_hours=24) is False
    assert within_temporal_window(t_base, t_plus_5d, window_hours=24) is False

    # Defensive fallback for invalid/missing dates
    assert within_temporal_window(None, t_base, window_hours=24) is True
    assert within_temporal_window("invalid-date", t_base, window_hours=24) is True


def test_minhash_lsh_indexing_and_query():
    """Verify MinHash signature determinism and LSH bucket collisions."""
    text_1 = "प्रधानमंत्री नरेंद्र मोदी ने आज उत्तर प्रदेश के वाराणसी में 1500 करोड़ रुपये की विकास परियोजनाओं का लोकार्पण किया।"
    text_2 = "वाराणसी: प्रधानमंत्री नरेंद्र मोदी ने आज वाराणसी में 1500 करोड़ रुपये की विकास परियोजनाओं का लोकार्पण किया।"
    text_3 = "क्रिकेट विश्व कप के मुकाबले में भारतीय टीम ने ऑस्ट्रेलिया को छह विकेट से पराजित कर ऐतिहासिक जीत दर्ज की।"

    sh_1 = generate_shingles(text_1)
    sh_2 = generate_shingles(text_2)
    sh_3 = generate_shingles(text_3)

    sig_1 = compute_minhash(sh_1)
    sig_2 = compute_minhash(sh_2)
    sig_3 = compute_minhash(sh_3)

    assert len(sig_1) == 64
    assert len(sig_2) == 64

    # MinHash signatures must be deterministic
    assert compute_minhash(sh_1) == sig_1

    lsh = MinHashLSH(num_perm=64, bands=16, rows=4)
    lsh.insert("doc_1", sig_1)
    lsh.insert("doc_2", sig_2)
    lsh.insert("doc_3", sig_3)

    # Similar documents (doc_1 and doc_2) should collide in >= 1 bucket
    candidates_1 = lsh.query(sig_1)
    assert "doc_1" in candidates_1
    assert "doc_2" in candidates_1
    assert "doc_3" not in candidates_1

    candidate_pairs = lsh.get_candidate_pairs()
    assert ("doc_1", "doc_2") in candidate_pairs
    assert ("doc_1", "doc_3") not in candidate_pairs
    assert ("doc_2", "doc_3") not in candidate_pairs


# ---------------------------------------------------------------------------
# Integration Tests: Story Clustering & Lineage
# ---------------------------------------------------------------------------

def test_cluster_articles_lineage():
    """Verify canonical root assignment: earliest story is root, later copies point to root."""
    articles = [
        {
            "doc_id": "pti_wire_1",
            "date": "2026-10-06T10:00:00+05:30",
            "body": "सुप्रीम कोर्ट ने दिल्ली एनसीआर में वायु प्रदूषण पर सख्त रुख अपनाया और सभी संबंधित राज्यों को निर्देश दिए।",
            "links": ["other_doc"],
        },
        {
            "doc_id": "jagran_copy_1",
            "date": "2026-10-06T11:30:00+05:30",
            "body": "सुप्रीम कोर्ट ने दिल्ली एनसीआर में वायु प्रदूषण पर सख्त रुख अपनाया और सभी संबंधित राज्यों को सख्त निर्देश दिए। भाषा",
            "links": ["pti_wire_1"],
        },
        {
            "doc_id": "amarujala_copy_1",
            "date": "2026-10-06T13:00:00+05:30",
            "body": "सुप्रीम कोर्ट ने दिल्ली एनसीआर में वायु प्रदूषण पर सख्त रुख अपनाया और सभी संबंधित राज्यों को निर्देश दिए।",
            "links": [],
        },
        {
            "doc_id": "distinct_sports_1",
            "date": "2026-10-06T10:30:00+05:30",
            "body": "भारतीय क्रिकेट टीम ने धर्मशाला में शानदार खेल का प्रदर्शन करते हुए मैच जीत लिया।",
            "links": ["pti_wire_1"],
        },
    ]

    clustered = cluster_articles(articles, jaccard_threshold=0.70, temporal_window_hours=24, use_lsh=False)
    doc_map = {art["doc_id"]: art for art in clustered}

    # Earliest wire article is canonical cluster root
    assert doc_map["pti_wire_1"]["dup_of"] is None

    # Later syndicated reprints point to the earliest root
    assert doc_map["jagran_copy_1"]["dup_of"] == "pti_wire_1"
    assert doc_map["amarujala_copy_1"]["dup_of"] == "pti_wire_1"

    # Distinct article remains a singleton
    assert doc_map["distinct_sports_1"]["dup_of"] is None


def test_prune_in_corpus_links():
    """Verify that in-corpus link resolution removes dangling IDs and self-links."""
    articles = [
        {
            "doc_id": "doc_a",
            "date": "2026-10-06T10:00:00+05:30",
            "body": "पहला समाचार लेख सामग्री।",
            "links": ["doc_b", "nonexistent_doc_999", "doc_a"],
        },
        {
            "doc_id": "doc_b",
            "date": "2026-10-06T11:00:00+05:30",
            "body": "दूसरा समाचार लेख सामग्री।",
            "links": ["nonexistent_doc_123"],
        },
    ]

    clustered = cluster_articles(articles, prune_links=True)
    doc_map = {art["doc_id"]: art for art in clustered}

    # doc_a: should retain doc_b, prune nonexistent_doc_999 and self-link doc_a
    assert doc_map["doc_a"]["links"] == ["doc_b"]

    # doc_b: should have empty links
    assert doc_map["doc_b"]["links"] == []


# ---------------------------------------------------------------------------
# Benchmark & Evaluation Deliverable: 100 Labeled Pairs
# ---------------------------------------------------------------------------

def test_labeled_100_pairs_benchmark():
    """Evaluate Precision, Recall, and F1 across thresholds on 100 labeled Hindi pairs.
    
    Verifies the calibration table for the final evaluation report.
    Fixture is loaded from external JSON file rather than hardcoding in test code.
    """
    assert LABELED_PAIRS_FIXTURE.exists(), f"Missing fixture file: {LABELED_PAIRS_FIXTURE}"

    with open(LABELED_PAIRS_FIXTURE, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    assert len(pairs) == 100, f"Expected 100 benchmark pairs, found {len(pairs)}"

    thresholds = [0.60, 0.65, 0.70, 0.75, 0.80]
    calibration_metrics = {}

    for th in thresholds:
        tp, fp, tn, fn = 0, 0, 0, 0
        for p in pairs:
            ground_truth = p["label"]
            art_a = p["article_a"]
            art_b = p["article_b"]

            in_window = within_temporal_window(art_a.get("date"), art_b.get("date"), TEMPORAL_WINDOW_HOURS)
            chash_a = compute_content_hash(art_a["body"])
            chash_b = compute_content_hash(art_b["body"])

            if not in_window:
                pred = 0
            elif chash_a == chash_b:
                pred = 1
            else:
                sh_a = generate_shingles(art_a["body"], n=SHINGLE_SIZE)
                sh_b = generate_shingles(art_b["body"], n=SHINGLE_SIZE)
                sim = jaccard_similarity(sh_a, sh_b)
                pred = 1 if sim >= th else 0

            if pred == 1 and ground_truth == 1:
                tp += 1
            elif pred == 1 and ground_truth == 0:
                fp += 1
            elif pred == 0 and ground_truth == 0:
                tn += 1
            elif pred == 0 and ground_truth == 1:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        calibration_metrics[th] = {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1
        }

    # Verify that the standard threshold (0.70) achieves target performance
    m_070 = calibration_metrics[0.70]
    assert m_070["precision"] >= 0.95, f"Expected precision >= 0.95 at J=0.70, got {m_070['precision']}"
    assert m_070["recall"] >= 0.90, f"Expected recall >= 0.90 at J=0.70, got {m_070['recall']}"
    assert m_070["f1"] >= 0.95, f"Expected F1 >= 0.95 at J=0.70, got {m_070['f1']}"

    # Verify precision is monotonically non-decreasing with increasing threshold
    assert calibration_metrics[0.80]["precision"] >= calibration_metrics[0.60]["precision"]


def test_minhash_vs_exact_throughput():
    """Benchmark execution throughput of MinHash LSH vs pairwise comparison."""
    # Generate 40 synthetic articles
    base_text = "उत्तर प्रदेश के लखनऊ में आयोजित कार्यक्रम में विकास योजनाओं का शिलान्यास किया गया।"
    articles = []
    for i in range(40):
        body = base_text if i % 4 == 0 else f"{base_text} खंड संख्या {i} में विस्तृत समीक्षा की गई।"
        articles.append({
            "doc_id": f"doc_{i}",
            "body": body,
            "date": "2026-10-06T12:00:00+05:30",
        })

    # MinHash LSH execution
    t0 = time.perf_counter()
    clustered_lsh = cluster_articles(articles, use_lsh=True)
    t_lsh = time.perf_counter() - t0

    # Pairwise execution
    t1 = time.perf_counter()
    clustered_exact = cluster_articles(articles, use_lsh=False)
    t_exact = time.perf_counter() - t1

    # Both must identify the same number of duplicate linkages
    lsh_dupes = sum(1 for a in clustered_lsh if a.get("dup_of") is not None)
    exact_dupes = sum(1 for a in clustered_exact if a.get("dup_of") is not None)
    assert lsh_dupes == exact_dupes, "MinHash LSH must match exact clustering duplicate count"
    assert t_lsh < 2.0, "LSH execution must complete within 2 seconds"


def test_cli_batch_clustering(tmp_path: Path):
    """Test JSONL batch processing with cluster_corpus."""
    input_file = tmp_path / "sample_in.jsonl"
    output_file = tmp_path / "sample_out.jsonl"

    test_articles = [
        {
            "doc_id": "test_1",
            "url": "https://example.com/1",
            "source": "jagran",
            "date": "2026-10-06T10:00:00+05:30",
            "headline": "पहला लेख",
            "body": "भारतीय अंतरिक्ष अनुसंधान संगठन ने श्रीहरिकोटा के सतीश धवन अंतरिक्ष केंद्र से अपने नए पृथ्वी अवलोकन उपग्रह का प्रक्षेपण किया। वैज्ञानिकों की पूरी टीम को बधाई दी गई।",
            "links": ["test_2"],
        },
        {
            "doc_id": "test_2",
            "url": "https://example.com/2",
            "source": "nbt",
            "date": "2026-10-06T11:00:00+05:30",
            "headline": "दूसरा लेख",
            "body": "भारतीय अंतरिक्ष अनुसंधान संगठन ने श्रीहरिकोटा के सतीश धवन अंतरिक्ष केंद्र से अपने नए पृथ्वी अवलोकन उपग्रह का सफल प्रक्षेपण किया। वैज्ञानिकों की पूरी टीम को बधाई दी गई। पीटीआई",
            "links": ["test_1"],
        },
    ]

    with open(input_file, "w", encoding="utf-8") as f:
        for art in test_articles:
            f.write(json.dumps(art, ensure_ascii=False) + "\n")

    stats = cluster_corpus(input_file, output_file, jaccard_threshold=0.70)

    assert stats["total_articles"] == 2
    assert stats["duplicates_identified"] == 1
    assert stats["unique_articles"] == 1

    # Verify output file
    results = []
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line.strip()))

    assert len(results) == 2
    res_map = {r["doc_id"]: r for r in results}
    assert res_map["test_1"]["dup_of"] is None
    assert res_map["test_2"]["dup_of"] == "test_1"
