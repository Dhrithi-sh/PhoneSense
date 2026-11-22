"""Build the inverted index (T03) and write index statistics.

    python scripts/build_index.py            # phones.jsonl -> data/index/  (= make index)
    python scripts/build_index.py --sample   # sample -> data/index/sample/  (= make index-sample)
    python scripts/build_index.py --force      # rebuild even if the saved index is up to date

Stats go to results/index_stats.json (results/index_stats_sample.json for --sample): N, vocab
size, postings and average doc length per field, build/load time, and the on-disk size with
VB compression vs plain uint32 postings.

The index is skipped (not rebuilt) when it already exists with the same format version,
analyzer config and an unchanged source file, unless --force is given.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from phonesense.config import get_path
from phonesense.index import Analyzer, InvertedIndex, index_path
from phonesense.index.inverted import FORMAT_VERSION, dir_size, file_fingerprint
from phonesense.io import read_phones


def _up_to_date(out: Path, analyzer: Analyzer, source: dict) -> bool:
    try:
        meta = InvertedIndex.read_meta(out)
    except FileNotFoundError:
        return False
    return (
        meta.get("format_version") == FORMAT_VERSION
        and meta.get("analyzer_hash") == analyzer.config_hash()
        and meta.get("source") == source
    )


def _size_summary(sizes: dict[str, int]) -> dict[str, int]:
    post = sum(v for k, v in sizes.items() if k.endswith(".post"))
    lex = sum(v for k, v in sizes.items() if k.endswith(".lex.json"))
    return {
        "postings": post,
        "lexicon": lex,
        "meta": sizes["meta.json"],
        "total": sum(sizes.values()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--sample", action="store_true", help="index data/processed/sample/phones.jsonl"
    )
    ap.add_argument("--force", action="store_true", help="rebuild even if up to date")
    ap.add_argument("--phones", type=Path, help="override the input phones.jsonl")
    ap.add_argument("--out", type=Path, help="override the index directory")
    args = ap.parse_args(argv)

    phones_path = args.phones or (
        get_path("sample_dir") / "phones.jsonl" if args.sample else get_path("phones")
    )
    out = args.out or index_path(args.sample)
    stats_path = get_path("index_stats")
    if args.sample:
        stats_path = stats_path.with_name(stats_path.stem + "_sample.json")

    if not phones_path.exists():
        hint = "python scripts/build_sample.py" if args.sample else "python scripts/build_data.py"
        print(f"error: {phones_path} not found. Build the data first: {hint}", file=sys.stderr)
        return 1

    analyzer = Analyzer.from_config()
    source = file_fingerprint(phones_path)
    if not args.force and _up_to_date(out, analyzer, source):
        print(f"index at {out} is up to date (use --force to rebuild); stats: {stats_path}")
        return 0

    t0 = time.perf_counter()
    phones = read_phones(phones_path)
    read_s = time.perf_counter() - t0
    print(f"read {len(phones)} phones from {phones_path} in {read_s:.1f}s; indexing...")

    idx = InvertedIndex.build(phones, analyzer)
    t1 = time.perf_counter()
    sizes = idx.save(out, compression="vbyte", source=source)
    save_s = time.perf_counter() - t1
    with tempfile.TemporaryDirectory() as tmp:
        raw_sizes = idx.save(Path(tmp) / "raw", compression="none", source=source)

    t2 = time.perf_counter()
    loaded = InvertedIndex.load(out, analyzer)
    loaded.materialize()
    load_s = time.perf_counter() - t2
    if loaded != idx:
        print("error: saved index does not round-trip", file=sys.stderr)
        return 1

    comp, raw = _size_summary(sizes), _size_summary(raw_sizes)
    stats = {
        "index": "sample" if args.sample else "full",
        "phones_path": str(phones_path),
        "index_dir": str(out),
        "format_version": FORMAT_VERSION,
        "analyzer": analyzer.config_dict(),
        "analyzer_hash": analyzer.config_hash(),
        **idx.stats(),
        "build_seconds": round(idx.build_seconds or 0.0, 3),
        "save_seconds": round(save_s, 3),
        "load_seconds": round(load_s, 3),
        "size_bytes": {
            "vbyte": comp,
            "uncompressed_uint32": raw,
            "on_disk_total": dir_size(out),
        },
        "postings_compression_ratio": round(raw["postings"] / max(comp["postings"], 1), 3),
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print(f"built index in {stats['build_seconds']}s -> {out}")
    print(f"  N={idx.N}  vocab={stats['vocab_size']}")
    print(f"  avg_doc_len={stats['avg_doc_len']}")
    print(
        f"  postings on disk: {comp['postings']:,} B (VB) vs {raw['postings']:,} B (uint32), "
        f"ratio {stats['postings_compression_ratio']}x; total on disk {comp['total']:,} B"
    )
    print(f"  load (decode) time {load_s:.2f}s; stats -> {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
