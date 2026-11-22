"""Deduplication Engine: Exact Hash, 4-word Shingles, Jaccard, MinHash + LSH, and Story Lineage Clustering."""

import argparse
import binascii
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
import logging
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from dhvani.crawl.config import (
    JACCARD_THRESHOLD,
    MINHASH_BANDS,
    MINHASH_NUM_PERM,
    MINHASH_ROWS,
    SHINGLE_SIZE,
    TEMPORAL_WINDOW_HOURS,
    WIRE_AGENCY_KEYWORDS,
)

logger = logging.getLogger("dhvani.dedup")

# Prime modulus for universal hashing (2^31 - 1, Mersenne prime)
_PRIME = 2147483647


def _generate_minhash_coefficients(num_perm: int = MINHASH_NUM_PERM) -> Tuple[List[int], List[int]]:
    """Precomputed deterministic random coefficients for MinHash (seed=42)."""
    rng = random.Random(42)
    a_coeffs = [rng.randint(1, _PRIME - 1) for _ in range(num_perm)]
    b_coeffs = [rng.randint(0, _PRIME - 1) for _ in range(num_perm)]
    return a_coeffs, b_coeffs


_A_COEFFS, _B_COEFFS = _generate_minhash_coefficients(MINHASH_NUM_PERM)


def compute_content_hash(text: str) -> str:
    """Compute MD5 hash of normalized, whitespace-stripped text."""
    normalized = " ".join(text.strip().split())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def generate_shingles(text: str, n: int = SHINGLE_SIZE) -> Set[str]:
    """Generate n-word sliding shingles from Hindi text."""
    words = text.strip().split()
    if not words:
        return set()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Calculate Jaccard similarity coefficient between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def is_agency_story(text: str) -> bool:
    """Detect if text contains syndicated wire agency keywords (PTI, ANI, Bhasha, etc.)."""
    for kw in WIRE_AGENCY_KEYWORDS:
        if kw in text:
            return True
    return False


def compute_minhash(shingles: Set[str], num_perm: int = MINHASH_NUM_PERM) -> Tuple[int, ...]:
    """Compute MinHash signature of length num_perm over a set of shingles."""
    if not shingles:
        return tuple([0] * num_perm)

    # Use CRC32 for deterministic, platform-independent 32-bit hashing
    shingle_hashes = [binascii.crc32(s.encode("utf-8")) & 0x7FFFFFFF for s in shingles]

    sig = [_PRIME] * num_perm
    for sh in shingle_hashes:
        for i in range(num_perm):
            val = (_A_COEFFS[i] * sh + _B_COEFFS[i]) % _PRIME
            if val < sig[i]:
                sig[i] = val
    return tuple(sig)


class MinHashLSH:
    """Locality Sensitive Hashing (LSH) index for MinHash signatures.
    
    Partitions signature into b bands of r rows. Candidate collisions occur when
    two signatures match exactly in all r rows of at least one band.
    """

    def __init__(
        self,
        num_perm: int = MINHASH_NUM_PERM,
        bands: int = MINHASH_BANDS,
        rows: int = MINHASH_ROWS,
    ):
        if bands * rows != num_perm:
            raise ValueError(f"bands * rows ({bands} * {rows} = {bands * rows}) must equal num_perm ({num_perm})")
        self.num_perm = num_perm
        self.bands = bands
        self.rows = rows
        # Buckets: map (band_index, tuple_of_row_hashes) -> set of doc_ids
        self.buckets: Dict[Tuple[int, Tuple[int, ...]], Set[str]] = defaultdict(set)
        self.doc_signatures: Dict[str, Tuple[int, ...]] = {}

    def insert(self, doc_id: str, signature: Tuple[int, ...]) -> None:
        """Insert a document ID and its MinHash signature into LSH buckets."""
        self.doc_signatures[doc_id] = signature
        for b in range(self.bands):
            start = b * self.rows
            band_tuple = signature[start : start + self.rows]
            band_key = (b, band_tuple)
            self.buckets[band_key].add(doc_id)

    def query(self, signature: Tuple[int, ...]) -> Set[str]:
        """Retrieve candidate doc_ids that share at least one band bucket."""
        candidates: Set[str] = set()
        for b in range(self.bands):
            start = b * self.rows
            band_tuple = signature[start : start + self.rows]
            band_key = (b, band_tuple)
            candidates.update(self.buckets.get(band_key, ()))
        return candidates

    def get_candidate_pairs(self) -> Set[Tuple[str, str]]:
        """Extract all unique candidate pairs (doc_a, doc_b) colliding in >= 1 bucket."""
        candidate_pairs: Set[Tuple[str, str]] = set()
        for doc_ids in self.buckets.values():
            if len(doc_ids) > 1:
                sorted_ids = sorted(doc_ids)
                for i in range(len(sorted_ids)):
                    for j in range(i + 1, len(sorted_ids)):
                        candidate_pairs.add((sorted_ids[i], sorted_ids[j]))
        return candidate_pairs


def parse_iso_datetime(date_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO-8601 timestamp string into timezone-aware datetime object."""
    if not date_str:
        return None
    try:
        clean = date_str.replace("Z", "+00:00")
        return datetime.fromisoformat(clean)
    except (ValueError, TypeError):
        return None


def within_temporal_window(
    date_a: Optional[str],
    date_b: Optional[str],
    window_hours: int = TEMPORAL_WINDOW_HOURS,
) -> bool:
    """Check if two publication dates fall within the specified window in hours.
    
    If either date cannot be parsed, returns True to avoid discarding genuine duplicates.
    """
    if not date_a or not date_b:
        return True
    dt_a = parse_iso_datetime(date_a)
    dt_b = parse_iso_datetime(date_b)
    if dt_a is None or dt_b is None:
        return True

    diff_seconds = abs((dt_a - dt_b).total_seconds())
    return diff_seconds <= (window_hours * 3600)


class DisjointSet:
    """Disjoint Set Union (Union-Find) with path compression for connected components."""

    def __init__(self, elements: List[str]):
        self.parent = {elem: elem for elem in elements}

    def find(self, elem: str) -> str:
        if self.parent[elem] != elem:
            self.parent[elem] = self.find(self.parent[elem])
        return self.parent[elem]

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a

    def get_components(self) -> Dict[str, List[str]]:
        components = defaultdict(list)
        for elem in self.parent:
            root = self.find(elem)
            components[root].append(elem)
        return dict(components)


def cluster_articles(
    articles: List[Dict[str, Any]],
    jaccard_threshold: float = JACCARD_THRESHOLD,
    temporal_window_hours: int = TEMPORAL_WINDOW_HOURS,
    use_lsh: bool = True,
    prune_links: bool = True,
) -> List[Dict[str, Any]]:
    """Cluster articles into duplicate groups and assign canonical lineage (dup_of).
    
    For each cluster of near-duplicates (Jaccard >= threshold within temporal window):
      - Canonical root (earliest published) keeps dup_of = None (JSON null).
      - Later syndicated copies set dup_of = root["doc_id"].
      - Singletons keep dup_of = None.
    
    Optionally prunes in-corpus hyperlinks so 'links' only references valid doc_ids.
    """
    if not articles:
        return []

    # Map articles by doc_id
    doc_map: Dict[str, Dict[str, Any]] = {art["doc_id"]: art for art in articles}
    doc_ids = list(doc_map.keys())

    # Precompute shingles and content hashes
    shingles_map: Dict[str, Set[str]] = {}
    hash_to_docs: Dict[str, List[str]] = defaultdict(list)

    for art in articles:
        did = art["doc_id"]
        body = art.get("body", "")
        shingles_map[did] = generate_shingles(body)

        chash = art.get("content_hash") or compute_content_hash(body)
        art["content_hash"] = chash
        hash_to_docs[chash].append(did)

    candidate_pairs: Set[Tuple[str, str]] = set()

    # Fast-path: Add all exact content_hash duplicate pairs
    for chash, docs in hash_to_docs.items():
        if len(docs) > 1:
            for i in range(len(docs)):
                for j in range(i + 1, len(docs)):
                    p = tuple(sorted([docs[i], docs[j]]))
                    candidate_pairs.add(p)

    # Candidate pair generation via MinHash LSH or pairwise comparison
    if use_lsh and len(articles) >= 10:
        lsh = MinHashLSH()
        for did in doc_ids:
            sig = compute_minhash(shingles_map[did])
            lsh.insert(did, sig)
        candidate_pairs.update(lsh.get_candidate_pairs())
    else:
        # Pairwise comparison for small corpora
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                candidate_pairs.add(tuple(sorted([doc_ids[i], doc_ids[j]])))

    # Graph union-find for clustering
    dsu = DisjointSet(doc_ids)

    for doc_a_id, doc_b_id in candidate_pairs:
        art_a = doc_map[doc_a_id]
        art_b = doc_map[doc_b_id]

        # Check temporal window constraint (+/- 24h)
        if not within_temporal_window(art_a.get("date"), art_b.get("date"), temporal_window_hours):
            continue

        # Check exact hash fast path
        if art_a.get("content_hash") == art_b.get("content_hash") and art_a.get("content_hash"):
            dsu.union(doc_a_id, doc_b_id)
            continue

        # Check Jaccard similarity on 4-word shingles
        sim = jaccard_similarity(shingles_map[doc_a_id], shingles_map[doc_b_id])
        if sim >= jaccard_threshold:
            dsu.union(doc_a_id, doc_b_id)

    # Extract clusters and assign canonical root
    components = dsu.get_components()

    fallback_date = datetime.max.replace(tzinfo=timezone(timedelta(hours=5, minutes=30)))

    for cluster_members in components.values():
        if len(cluster_members) == 1:
            # Singleton: not a duplicate of anything
            doc_map[cluster_members[0]]["dup_of"] = None
        else:
            # Sort cluster members by publication date ascending, tie-breaking by doc_id
            def sort_key(did: str):
                art = doc_map[did]
                parsed_dt = parse_iso_datetime(art.get("date")) or fallback_date
                return (parsed_dt, did)

            sorted_members = sorted(cluster_members, key=sort_key)
            root_id = sorted_members[0]

            # Earliest is canonical root
            doc_map[root_id]["dup_of"] = None

            # All other members point to the canonical root
            for non_root_id in sorted_members[1:]:
                doc_map[non_root_id]["dup_of"] = root_id

    # Optional: prune in-corpus links to only include existing doc_ids (excluding self-links)
    if prune_links:
        all_ids = set(doc_map.keys())
        for art in articles:
            raw_links = art.get("links", [])
            if isinstance(raw_links, list):
                art["links"] = [
                    lk for lk in raw_links
                    if lk in all_ids and lk != art["doc_id"]
                ]

    return articles


def cluster_corpus(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    jaccard_threshold: float = JACCARD_THRESHOLD,
    temporal_window_hours: int = TEMPORAL_WINDOW_HOURS,
) -> Dict[str, Any]:
    """Read articles from JSONL file, cluster them, and stream back to output JSONL."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    articles: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                articles.append(json.loads(line))

    total_articles = len(articles)
    clustered = cluster_articles(
        articles,
        jaccard_threshold=jaccard_threshold,
        temporal_window_hours=temporal_window_hours,
        use_lsh=True,
        prune_links=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    duplicate_count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for art in clustered:
            if art.get("dup_of") is not None:
                duplicate_count += 1
            f.write(json.dumps(art, ensure_ascii=False) + "\n")

    stats = {
        "total_articles": total_articles,
        "duplicates_identified": duplicate_count,
        "unique_articles": total_articles - duplicate_count,
        "duplicate_ratio_pct": (duplicate_count / total_articles * 100.0) if total_articles else 0.0,
        "input_path": str(input_path),
        "output_path": str(output_path),
    }

    logger.info("Clustering completed: %d total, %d duplicates (%.1f%%)",
                total_articles, duplicate_count, stats["duplicate_ratio_pct"])
    return stats


def main():
    parser = argparse.ArgumentParser(description="Dhvani Article Deduplication and Story Lineage Clustering")
    parser.add_argument("--input", type=str, required=True, help="Path to input articles JSONL")
    parser.add_argument("--output", type=str, required=True, help="Path to output clustered JSONL")
    parser.add_argument("--threshold", type=float, default=JACCARD_THRESHOLD, help="Jaccard similarity cutoff (default 0.70)")
    parser.add_argument("--window-hours", type=int, default=TEMPORAL_WINDOW_HOURS, help="Temporal comparison window in hours (default 24)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    stats = cluster_corpus(args.input, args.output, args.threshold, args.window_hours)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
