#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comprehensive Crawler & Corpus Pipeline Test Runner (Part 1 - Riya).

Runs all test suites across the 10 crawler subsystems in `partwise-tests/riya/`:
1.  Robots.txt Parser (`test_robots.py`)
2.  URL Normalizer & Route Filters (`test_normalizer.py`)
3.  Mercator Frontier & Politeness (`test_frontier.py`)
4.  Sitemap Discovery & Index Feeds (`test_sitemap.py`)
5.  Article & Metadata Extractor (`test_extractor.py`)
6.  Adaptive Recrawler & Burst Detection (`test_recrawl.py`)
7.  Crawler Core Pipeline (`test_crawler.py`)
8.  Deduplication & Story Clustering (`test_dedup.py`)
9.  Contract & Invariant Compliance (`test_format_compliance.py`)
10. Shared TREC Run Pooling Tool (`test_pool.py`)

Usage:
    python scripts/run_crawler_tests.py
    python scripts/run_crawler_tests.py -v       # Verbose mode
    python scripts/run_crawler_tests.py -q       # Quiet / concise mode
"""

import argparse
from pathlib import Path
import sys
import time
import pytest

SUBSYSTEMS = [
    ("test_robots.py", "RFC 9309 Robots Parser & Wildcards"),
    ("test_normalizer.py", "URL Normalization & Route Filtering"),
    ("test_frontier.py", "Mercator Frontier & 8.0s Politeness"),
    ("test_sitemap.py", "Sitemap & Google News Index Parser"),
    ("test_extractor.py", "Article Extractor & Schema Contract"),
    ("test_recrawl.py", "Adaptive Recrawler & Burst Detection"),
    ("test_crawler.py", "Crawler Pipeline Execution & Backoff"),
    ("test_dedup.py", "Deduplication & Story Clustering"),
    ("test_format_compliance.py", "Format Compliance & Invariant Audit"),
    ("test_pool.py", "Shared TREC Run Pooling Tool"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Run all Part 1 (Riya - Crawler & Corpus) test suites with subsystem summary"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print detailed test case output"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Minimal quiet output"
    )
    parser.add_argument(
        "-k", "--keyword", type=str, default="", help="Filter tests by expression/keyword"
    )
    args = parser.parse_args()

    # Reconfigure stdout for Unicode / UTF-8 terminals
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    project_root = Path(__file__).resolve().parent.parent.parent
    tests_dir = Path(__file__).resolve().parent

    print("=" * 76)
    print("   DHVANI-LINGUAL: CRAWLER & CORPUS PIPELINE TEST SUITE (PART 1)")
    print("=" * 76)
    print(f"Test Directory: {tests_dir}")
    print(f"Total Subsystem Modules: {len(SUBSYSTEMS)}\n")

    overall_start = time.perf_counter()
    results = []
    total_passed = 0
    total_failed = 0

    for filename, description in SUBSYSTEMS:
        test_file = tests_dir / filename
        if not test_file.exists():
            results.append((filename, description, "SKIPPED", 0.0))
            continue

        pytest_args = [str(test_file)]
        if args.verbose:
            pytest_args.append("-v")
        else:
            pytest_args.append("-q")

        if args.keyword:
            pytest_args.extend(["-k", args.keyword])

        sub_start = time.perf_counter()
        exit_code = pytest.main(pytest_args)
        sub_duration = time.perf_counter() - sub_start

        status = "PASSED" if exit_code == 0 else f"FAILED (Code {exit_code})"
        if exit_code == 0:
            total_passed += 1
        else:
            total_failed += 1

        results.append((filename, description, status, sub_duration))

    overall_duration = time.perf_counter() - overall_start

    # Print Executive Summary Table
    print("\n" + "=" * 76)
    print(f"{'Subsystem / Test Module':<28} | {'Description':<33} | {'Status':<6} | {'Time':<6}")
    print("-" * 76)
    for filename, description, status, sub_time in results:
        status_symbol = "[OK]" if "PASSED" in status else "[FAIL]"
        print(f"{filename:<28} | {description[:33]:<33} | {status_symbol:<6} | {sub_time:5.2f}s")
    print("=" * 76)

    # Final Summary Banner
    print(f"\nCompleted in {overall_duration:.2f} seconds.")
    print(f"Subsystems Passed: {total_passed}/{len(SUBSYSTEMS)}")

    if total_failed == 0:
        print("\n  ALL 10 CRAWLER SUBSYSTEMS VERIFIED AND FULLY PASSING!\n")
        return 0
    else:
        print(f"\n  {total_failed} SUBSYSTEM(S) FAILED. Please review output above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
