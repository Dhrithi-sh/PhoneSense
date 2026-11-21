"""Mercator Frontier: Front Priority Queues, Per-Host Back Queues, and Politeness Min-Heap."""

import asyncio
import heapq
import random
import time
from collections import deque
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from dhvani.crawl.config import FRONT_QUEUE_PROBABILITIES, PER_HOST_DELAY
from dhvani.crawl.filters import should_skip_url
from dhvani.crawl.normalizer import normalize_url


class MercatorFrontier:
    """Two-tier Mercator crawl frontier isolating priority from per-host politeness.

    Architecture:
    1. Front Queues (Q0..Q3): Priority queues sampled via biased random selection.
       - Q0: Bursts & breaking news (60%)
       - Q1: Fresh live sitemaps (25%)
       - Q2: In-article hyperlinks (10%)
       - Q3: Archive backfill (5%)
    2. Back Queues: Per-host FIFO queues isolating target news domains.
    3. Politeness Min-Heap: Tracks (next_allowed_time, host) strictly enforcing
       the 8.0s per-host delay while allowing concurrent fetches across hosts.
    """

    def __init__(
        self,
        per_host_delay: float = PER_HOST_DELAY,
        max_back_queue_size: int = 10,
        time_func: Callable[[], float] = time.time,
        sleep_func: Optional[Callable[[float], Coroutine[Any, Any, None]]] = None,
    ):
        self.per_host_delay = per_host_delay
        self.max_back_queue_size = max_back_queue_size
        self.time_func = time_func
        self.sleep_func = sleep_func or asyncio.sleep

        # Front Queues (Priority)
        self.front_queues: Dict[str, deque] = {
            "Q0": deque(),
            "Q1": deque(),
            "Q2": deque(),
            "Q3": deque(),
        }

        # Back Queues (Per-Host FIFO queues for politeness isolation)
        self.back_queues: Dict[str, deque] = {}

        # Politeness Min-Heap: entries of (ready_time, entry_id, host)
        self.heap: List[Tuple[float, int, str]] = []
        self.hosts_in_heap: Set[str] = set()
        self._entry_count: int = 0

        # In-flight hosts currently executing a request
        self.in_flight_hosts: Set[str] = set()

        # Last completed request timestamp per host
        self.last_request_time: Dict[str, float] = {}

        # Seen URL deduplication set
        self.seen_urls: Set[str] = set()

    @staticmethod
    def extract_host(url: str) -> str:
        """Extract clean hostname from URL."""
        if not url:
            return ""
        parsed = urlparse(url)
        return parsed.netloc.lower()

    def add_url(self, url: str, priority_tier: str = "Q1") -> bool:
        """Enqueue URL into the specified priority front queue if valid and not seen.

        Returns True if URL was accepted and enqueued, False otherwise.
        """
        if not url:
            return False

        clean_url = normalize_url(url)
        if not clean_url or clean_url in self.seen_urls:
            return False

        if should_skip_url(clean_url):
            return False

        tier = priority_tier if priority_tier in self.front_queues else "Q1"

        self.seen_urls.add(clean_url)
        self.front_queues[tier].append(clean_url)
        self._refill_back_queues()
        return True

    def add_urls(self, urls: List[str], priority_tier: str = "Q1") -> int:
        """Batch-enqueue URLs into specified front queue. Returns count of added URLs."""
        added = 0
        for u in urls:
            if self.add_url(u, priority_tier=priority_tier):
                added += 1
        return added

    def _select_front_queue(self) -> Optional[str]:
        """Select a non-empty front queue according to configured priority weights."""
        non_empty = [q for q in ("Q0", "Q1", "Q2", "Q3") if len(self.front_queues[q]) > 0]
        if not non_empty:
            return None
        if len(non_empty) == 1:
            return non_empty[0]

        weights = [FRONT_QUEUE_PROBABILITIES.get(q, 0.05) for q in non_empty]
        return random.choices(non_empty, weights=weights, k=1)[0]

    def _schedule_host(self, host: str) -> None:
        """Schedule a host on the politeness min-heap if eligible and not already pending."""
        if not host:
            return
        if host in self.hosts_in_heap or host in self.in_flight_hosts:
            return
        if not self.back_queues.get(host):
            return

        now = self.time_func()
        last_time = self.last_request_time.get(host, 0.0)
        ready_time = max(now, last_time + self.per_host_delay)

        self._entry_count += 1
        heapq.heappush(self.heap, (ready_time, self._entry_count, host))
        self.hosts_in_heap.add(host)

    def _refill_back_queues(self) -> None:
        """Transfer URLs from front priority queues into per-host back queues.

        Keeps each host back queue populated up to max_back_queue_size without
        causing starvation or head-of-line blocking.
        """
        # Fast exit: if all existing back queues are full and all primary sources are discovered,
        # there are zero available slots across the frontier.
        hungry_hosts = {h for h, bq in self.back_queues.items() if len(bq) < self.max_back_queue_size}
        if self.back_queues and not hungry_hosts and len(self.back_queues) >= 5:
            return

        for tier in ("Q0", "Q1", "Q2", "Q3"):
            q = self.front_queues[tier]
            if not q:
                continue

            max_checks = len(q)
            checks = 0
            consecutive_skips = 0

            while q and checks < max_checks:
                # If all known back queues are satisfied and all sources discovered, terminate early
                if not hungry_hosts and len(self.back_queues) >= 5:
                    break

                # If we have rotated past all currently known full hosts consecutively without placing a URL, break
                if consecutive_skips >= max(len(self.back_queues), 10):
                    break

                candidate_url = q[0]
                host = self.extract_host(candidate_url)
                if not host:
                    q.popleft()
                    checks += 1
                    continue

                if host not in self.back_queues:
                    self.back_queues[host] = deque()
                    hungry_hosts.add(host)

                if len(self.back_queues[host]) < self.max_back_queue_size:
                    url = q.popleft()
                    self.back_queues[host].append(url)
                    self._schedule_host(host)
                    consecutive_skips = 0
                    if len(self.back_queues[host]) >= self.max_back_queue_size:
                        hungry_hosts.discard(host)
                else:
                    # Host back queue is currently full: rotate URL to back of queue
                    # so other hosts are not starved behind it.
                    q.rotate(-1)
                    consecutive_skips += 1

                checks += 1

    async def get_next_url(self) -> Optional[Tuple[str, str]]:
        """Retrieve the next ready URL adhering to per-host delay constraints.

        Returns (url, host) ready for fetching, or None if frontier has no ready URLs.
        """
        self._refill_back_queues()

        while self.heap:
            ready_time, _, host = self.heap[0]

            # Discard stale host entries with empty back queues
            if not self.back_queues.get(host):
                heapq.heappop(self.heap)
                self.hosts_in_heap.discard(host)
                continue

            now = self.time_func()
            wait_seconds = ready_time - now
            if wait_seconds > 0:
                await self.sleep_func(wait_seconds)

            # Pop eligible host from heap
            heapq.heappop(self.heap)
            self.hosts_in_heap.discard(host)

            # Mark host as in-flight
            self.in_flight_hosts.add(host)

            # Pop next URL from host's back queue
            url = self.back_queues[host].popleft()

            # Refill back queues to replace popped slot
            self._refill_back_queues()

            return url, host

        return None

    def complete_request(
        self, host: str, completion_time: Optional[float] = None, delay: Optional[float] = None
    ) -> None:
        """Mark in-flight request as finished and reschedule host for next eligible window."""
        t = completion_time if completion_time is not None else self.time_func()
        self.last_request_time[host] = t
        self.in_flight_hosts.discard(host)

        eff_delay = delay if delay is not None else self.per_host_delay

        # Reschedule host on heap first if more URLs remain and not already on heap
        if self.back_queues.get(host) and host not in self.hosts_in_heap:
            ready_time = t + eff_delay
            self._entry_count += 1
            heapq.heappush(self.heap, (ready_time, self._entry_count, host))
            self.hosts_in_heap.add(host)

        # Refill back queues for all eligible hosts
        self._refill_back_queues()

    def is_empty(self) -> bool:
        """Check if all queues and in-flight operations are completely drained."""
        front_empty = all(len(q) == 0 for q in self.front_queues.values())
        back_empty = all(len(q) == 0 for q in self.back_queues.values())
        return front_empty and back_empty and len(self.heap) == 0 and len(self.in_flight_hosts) == 0

    def size(self) -> int:
        """Total number of URLs currently waiting in front and back queues."""
        front_count = sum(len(q) for q in self.front_queues.values())
        back_count = sum(len(q) for q in self.back_queues.values())
        return front_count + back_count

    def num_seen(self) -> int:
        """Total number of unique URLs processed or enqueued."""
        return len(self.seen_urls)

    def queue_sizes(self) -> Dict[str, Any]:
        """Get snapshot of URL counts across front queues and active back queues."""
        return {
            "Q0": len(self.front_queues["Q0"]),
            "Q1": len(self.front_queues["Q1"]),
            "Q2": len(self.front_queues["Q2"]),
            "Q3": len(self.front_queues["Q3"]),
            "heap_entries": len(self.heap),
            "in_flight": list(self.in_flight_hosts),
            "back_queues": {h: len(q) for h, q in self.back_queues.items() if len(q) > 0},
        }

