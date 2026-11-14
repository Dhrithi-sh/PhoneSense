"""Dhvani Regional News Crawler (P1).

Orchestrates sitemap feed ingestion, RFC 9309 robots checking, Mercator frontier
priority scheduling, polite per-host rate limiting, resilient HTTP downloading,
3-tier article extraction, and streaming JSONL storage.
"""

import argparse
import asyncio
from collections import defaultdict
from datetime import datetime
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import httpx

from dhvani.crawl.config import (
    ARTICLES_FILE,
    BACKOFF_FACTOR,
    DATA_DIR,
    DEFAULT_HEADERS,
    DEFAULT_TIMEOUT,
    MAX_RETRIES,
    PER_HOST_DELAY,
    PRIMARY_SOURCES,
    PROJECT_ROOT,
    SAMPLE_ARTICLES_FILE,
    USER_AGENT,
)
from dhvani.crawl.extractor import ArticleExtractor, extract_section_from_url, validate_article_schema
from dhvani.crawl.filters import should_skip_url
from dhvani.crawl.frontier import MercatorFrontier
from dhvani.crawl.normalizer import normalize_url
from dhvani.crawl.recrawl import AdaptiveRecrawler
from dhvani.crawl.robots import RobotsParser
from dhvani.crawl.sitemap import SitemapParser


def setup_crawler_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """Configure dual stream (console) and persistent file logging for crawler metrics."""
    crawler_logger = logging.getLogger("dhvani.crawler")
    crawler_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in crawler_logger.handlers
    ):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        crawler_logger.addHandler(console_handler)

    file_path = log_file or (DATA_DIR / "crawler.log")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not any(isinstance(h, logging.FileHandler) for h in crawler_logger.handlers):
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        crawler_logger.addHandler(file_handler)

    return crawler_logger


logger = setup_crawler_logging()

# Mapping of known domains to standardized source keys
SOURCE_MAP: Dict[str, str] = {
    "jagran.com": "jagran",
    "www.jagran.com": "jagran",
    "navbharattimes.indiatimes.com": "nbt",
    "livehindustan.com": "livehindustan",
    "www.livehindustan.com": "livehindustan",
    "amarujala.com": "amarujala",
    "www.amarujala.com": "amarujala",
    "aajtak.in": "aajtak",
    "www.aajtak.in": "aajtak",
    "jansatta.com": "jansatta",
    "www.jansatta.com": "jansatta",
    "bhaskar.com": "bhaskar",
    "www.bhaskar.com": "bhaskar",
}

# Signatures indicating genuine bot challenge or CAPTCHA interception screens
CAPTCHA_SIGNATURES = [
    "cf-challenge-running",
    "challenge-platform",
    "cf-browser-verification",
    "captcha-delivery.com",
    "px-captcha",
    "distil_r_captcha",
    "Attention Required! | Cloudflare",
    "<title>Just a moment...</title>",
]


def get_source_key(url: str) -> str:
    """Resolve source identifier from URL domain."""
    if not url:
        return "unknown"
    netloc = urlparse(url).netloc.lower()
    if netloc in SOURCE_MAP:
        return SOURCE_MAP[netloc]
    for domain, key in SOURCE_MAP.items():
        if domain in netloc:
            return key
    # Fallback to base domain name
    parts = netloc.split(".")
    return parts[-2] if len(parts) >= 2 else netloc


def is_captcha_challenge(html: str) -> bool:
    """Check if HTML content indicates an anti-bot CAPTCHA / challenge interception."""
    if not html:
        return False
    # Legitimate news articles containing structured metadata or article containers
    # are never anti-bot challenge screens.
    if "schema.org" in html and ("NewsArticle" in html or "BlogPosting" in html):
        return False
    if "<article" in html:
        return False
    html_sample = html[:5000]
    return any(sig.lower() in html_sample.lower() for sig in CAPTCHA_SIGNATURES)


class NewsCrawler:
    """Asynchronous regional news crawler orchestrator."""

    def __init__(
        self,
        frontier: Optional[MercatorFrontier] = None,
        robots_parser: Optional[RobotsParser] = None,
        sitemap_parser: Optional[SitemapParser] = None,
        extractor: Optional[ArticleExtractor] = None,
        max_articles: int = 5000,
        is_sample_mode: bool = False,
        output_path: Optional[Path] = None,
        sample_output_path: Optional[Path] = None,
        stats_file: Optional[Path] = None,
        active_sources: Optional[List[str]] = None,
        recrawler: Optional[AdaptiveRecrawler] = None,
        time_func=time.time,
    ):
        self.is_sample_mode = is_sample_mode
        self.max_articles = 300 if is_sample_mode else max_articles

        self.output_path = Path(output_path or (SAMPLE_ARTICLES_FILE if is_sample_mode else ARTICLES_FILE))
        self.sample_output_path = Path(sample_output_path or SAMPLE_ARTICLES_FILE)
        if stats_file:
            self.stats_file = Path(stats_file)
        elif output_path and Path(output_path) not in (ARTICLES_FILE, SAMPLE_ARTICLES_FILE):
            self.stats_file = Path(output_path).parent / "crawl_stats.md"
        else:
            self.stats_file = PROJECT_ROOT / "documentation" / "results" / "riya-corpus-results" / "crawl_stats.md"

        self.frontier = frontier or MercatorFrontier(per_host_delay=PER_HOST_DELAY)
        self.robots = robots_parser or RobotsParser(user_agent=USER_AGENT)
        self.sitemap_parser = sitemap_parser or SitemapParser()
        self.extractor = extractor or ArticleExtractor()
        self.time_func = time_func
        self.recrawler = recrawler or AdaptiveRecrawler(time_func=self.time_func)

        self.active_sources = active_sources or list(PRIMARY_SOURCES.keys())

        # Progress & Content metrics
        self.saved_articles_count: int = 0
        self.articles_by_source: Dict[str, int] = defaultdict(int)
        self.articles_by_section: Dict[str, int] = defaultdict(int)
        self.articles_by_state: Dict[str, int] = defaultdict(int)
        self.articles_by_city: Dict[str, int] = defaultdict(int)
        self.articles_by_language: Dict[str, int] = defaultdict(int)
        self.total_body_chars: int = 0
        self.total_body_words: int = 0
        self.seen_doc_ids: Set[str] = set()
        self.sample_records: List[Dict[str, Any]] = []
        self._sample_snapshot_saved: bool = False
        self.schema_validation_failures: int = 0
        self.total_links_harvested: int = 0

        # Host health and politeness controls
        self.disabled_hosts: Set[str] = set()
        self.host_delays: Dict[str, float] = {}
        self.hosts_backed_off: Set[str] = set()
        self.http_status_counts: Dict[int, int] = defaultdict(int)
        self.total_http_requests: int = 0

        # Adaptive recrawling & frontier priority metrics
        self.sitemap_polls_count: Dict[str, int] = defaultdict(int)
        self.sitemap_304_count: Dict[str, int] = defaultdict(int)
        self.urls_enqueued_by_tier: Dict[str, int] = defaultdict(int)
        self.burst_events_observed: List[Dict[str, Any]] = []
        self.active_burst_categories_seen: Set[str] = set()

        self.start_time: float = 0.0
        self.end_time: float = 0.0

    def get_host_delay(self, host: str) -> float:
        """Get the current scheduling delay for host (default 8.0s or backed-off)."""
        return self.host_delays.get(host, self.frontier.per_host_delay)

    async def bootstrap_seeds(self, client: httpx.AsyncClient) -> int:
        """Discover and enqueue initial URLs from sitemaps of configured sources."""
        total_enqueued = 0
        logger.info("Bootstrapping seeds across sources: %s", ", ".join(self.active_sources))

        urls_by_source: Dict[str, List[str]] = defaultdict(list)

        for source_key in self.active_sources:
            source_cfg = PRIMARY_SOURCES.get(source_key)
            if not source_cfg:
                continue

            domain = source_cfg.get("domain", "")
            robots_url = source_cfg.get("robots_url", "")
            sitemap_urls = list(source_cfg.get("sitemaps", []))

            # Fetch robots.txt to discover additional sitemaps
            if robots_url:
                try:
                    r_resp = await client.get(robots_url, timeout=DEFAULT_TIMEOUT)
                    if r_resp.status_code == 200:
                        self.robots.set_cached_rules(domain, r_resp.text)
                        discovered_sitemaps = self.robots.extract_sitemaps(r_resp.text)
                        for s_url in discovered_sitemaps:
                            if s_url not in sitemap_urls:
                                sitemap_urls.append(s_url)
                except Exception as e:
                    logger.warning("Could not fetch robots.txt for %s: %s", domain, e)

            # Fetch and parse discovered sitemaps
            for s_url in sitemap_urls:
                self.sitemap_polls_count[source_key] += 1
                try:
                    s_resp = await client.get(s_url, timeout=DEFAULT_TIMEOUT)
                    s_headers = getattr(s_resp, "headers", {})
                    self.recrawler.cache_response_headers(s_url, s_headers)
                    if s_resp.status_code == 200:
                        content = s_resp.text
                        if self.sitemap_parser.is_sitemap_index(content):
                            child_sitemaps = self.sitemap_parser.parse_sitemap_index(content)
                            # Sample up to 3 child sitemaps per index to seed without overwhelming
                            for child_url, _ in child_sitemaps[:3]:
                                self.sitemap_polls_count[source_key] += 1
                                try:
                                    c_resp = await client.get(child_url, timeout=DEFAULT_TIMEOUT)
                                    c_headers = getattr(c_resp, "headers", {})
                                    self.recrawler.cache_response_headers(child_url, c_headers)
                                    if c_resp.status_code == 200:
                                        entries = self.sitemap_parser.parse_sitemap(c_resp.text)
                                        for entry in entries:
                                            urls_by_source[source_key].append(entry.url)
                                except Exception as ce:
                                    logger.debug("Failed child sitemap %s: %s", child_url, ce)
                        else:
                            entries = self.sitemap_parser.parse_sitemap(content)
                            for entry in entries:
                                urls_by_source[source_key].append(entry.url)
                except Exception as se:
                    logger.warning("Failed sitemap %s: %s", s_url, se)

        # Initialize baseline check timestamp and nominal rate (1.0 URL/hour) for bootstrapped sources
        now = self.time_func()
        for source_key in self.active_sources:
            if urls_by_source[source_key]:
                self.recrawler.last_checked[source_key] = now
                if source_key not in self.recrawler.change_rates:
                    self.recrawler.change_rates[source_key] = 1.0

        # Round-robin interleave URLs across sources into Frontier with dynamic priority tier
        max_source_urls = max((len(u) for u in urls_by_source.values()), default=0)
        per_source_seed_cap = 500 if self.is_sample_mode else 3000
        for idx in range(min(max_source_urls, per_source_seed_cap)):
            for source_key in self.active_sources:
                if idx < len(urls_by_source[source_key]):
                    candidate = urls_by_source[source_key][idx]
                    tier = self.recrawler.get_priority_tier(url=candidate)
                    if self.frontier.add_url(candidate, priority_tier=tier):
                        total_enqueued += 1
                        self.urls_enqueued_by_tier[tier] += 1

        logger.info(
            "Seed bootstrap complete. Enqueued %d initial URLs interleaved across %d sources.",
            total_enqueued,
            len(urls_by_source),
        )
        return total_enqueued

    async def fetch_article_html(
        self, client: httpx.AsyncClient, url: str, host: str
    ) -> Tuple[Optional[str], Optional[int], Optional[str]]:
        """Fetch article page over HTTP with retry backoff on transient errors.

        Returns: (html_text, status_code, error_message)
        """
        retries = 0
        backoff = 1.0

        while retries <= MAX_RETRIES:
            try:
                response = await client.get(url, timeout=DEFAULT_TIMEOUT, follow_redirects=True)
                status = response.status_code
                self.total_http_requests += 1
                self.http_status_counts[status] += 1

                if status == 200:
                    html_text = response.text
                    if is_captcha_challenge(html_text):
                        return None, 403, "CAPTCHA challenge detected"
                    return html_text, 200, None

                if status == 429:
                    self.hosts_backed_off.add(host)
                    return None, 429, "Too Many Requests"

                if status == 403:
                    return None, 403, "Forbidden"

                if 500 <= status < 600:
                    retries += 1
                    if retries <= MAX_RETRIES:
                        await asyncio.sleep(backoff)
                        backoff *= BACKOFF_FACTOR
                        continue
                    return None, status, f"Server Error {status}"

                # Non-retriable 4xx
                return None, status, f"Client Error {status}"

            except (httpx.TimeoutException, httpx.NetworkError) as ne:
                self.total_http_requests += 1
                self.http_status_counts[0] += 1
                retries += 1
                if retries <= MAX_RETRIES:
                    await asyncio.sleep(backoff)
                    backoff *= BACKOFF_FACTOR
                    continue
                return None, 0, f"Network exception: {ne}"
            except Exception as e:
                self.total_http_requests += 1
                self.http_status_counts[0] += 1
                return None, 0, f"Unexpected fetch error: {e}"

        return None, 0, "Retries exhausted"

    def harvest_links(self, links: List[str]) -> int:
        """Filter, normalize, and enqueue discovered in-body links into Frontier Q2."""
        added = 0
        self.total_links_harvested += len(links)
        for link in links:
            if not link:
                continue
            clean_link = normalize_url(link)
            if not clean_link or clean_link in self.frontier.seen_urls:
                continue
            if should_skip_url(clean_link):
                continue
            if self.frontier.add_url(clean_link, priority_tier="Q2"):
                added += 1
                self.urls_enqueued_by_tier["Q2"] += 1
        return added

    def write_record(self, record: Dict[str, Any], file_handle) -> None:
        """Stream a validated record to disk and manage H3 sample snapshots."""
        line = json.dumps(record, ensure_ascii=False)
        file_handle.write(line + "\n")
        file_handle.flush()

        self.seen_doc_ids.add(record["doc_id"])

        # Track first 300 records for the sample deliverable
        if not self.is_sample_mode and len(self.sample_records) < 300:
            self.sample_records.append(record)
            if len(self.sample_records) == 300 and not self._sample_snapshot_saved:
                self._save_sample_snapshot()

    def _save_sample_snapshot(self) -> None:
        """Save retained first 300 articles to sample output file."""
        self.sample_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.sample_output_path, "w", encoding="utf-8") as f:
            for rec in self.sample_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._sample_snapshot_saved = True
        logger.info("Successfully exported H3 sample snapshot: %s (300 articles)", self.sample_output_path)

    def print_progress(self) -> None:
        """Display live progress dashboard on console and record milestone health to log."""
        elapsed = max(self.time_func() - self.start_time, 0.001)
        rate = (self.saved_articles_count / elapsed) * 60.0
        pct = (self.saved_articles_count / self.max_articles) * 100.0 if self.max_articles else 0.0

        source_stats = " | ".join(f"{s}: {count}" for s, count in sorted(self.articles_by_source.items()))
        total_304 = sum(self.sitemap_304_count.values())
        logger.info(
            "Progress: %d/%d (%.1f%%) | %.1f art/min | Frontier: %d queued | Sources: [%s] | 304s: %d",
            self.saved_articles_count,
            self.max_articles,
            pct,
            rate,
            self.frontier.size(),
            source_stats,
            total_304,
        )

        if self.saved_articles_count % 100 == 0 and self.saved_articles_count > 0:
            elapsed_min = elapsed / 60.0
            reqs = self.total_http_requests
            rps = reqs / elapsed if elapsed > 0 else 0.0
            status_summary = ", ".join(f"{st}: {c}" for st, c in sorted(self.http_status_counts.items()))
            rates_summary = ", ".join(f"{s}: {self.recrawler.get_rate(s):.2f}/h" for s in sorted(self.recrawler.change_rates.keys()))
            logger.info(
                "--- Milestone Health (%d articles | %.1f min) --- Reqs: %d (%.2f req/s) | HTTP: [%s] | Rates: [%s] | Backed-off hosts: %d",
                self.saved_articles_count,
                elapsed_min,
                reqs,
                rps,
                status_summary,
                rates_summary,
                len(self.hosts_backed_off),
            )

    async def poll_source_sitemaps(self, client: httpx.AsyncClient, source_key: str) -> int:
        """Conditionally poll sitemaps for a source and enqueue new URLs with burst prioritization."""
        source_cfg = PRIMARY_SOURCES.get(source_key)
        if not source_cfg:
            return 0

        sitemap_urls = list(source_cfg.get("sitemaps", []))
        new_enqueued = 0
        total_new_urls = 0
        now = self.time_func()

        for s_url in sitemap_urls:
            headers = self.recrawler.get_conditional_headers(s_url)
            self.sitemap_polls_count[source_key] += 1
            try:
                s_resp = await client.get(s_url, headers=headers, timeout=DEFAULT_TIMEOUT)
                resp_headers = getattr(s_resp, "headers", {})
                self.recrawler.cache_response_headers(s_url, resp_headers)

                if s_resp.status_code == 304:
                    self.sitemap_304_count[source_key] += 1
                    logger.debug("Sitemap %s: 304 Not Modified", s_url)
                elif s_resp.status_code == 200:
                    content = s_resp.text
                    if self.sitemap_parser.is_sitemap_index(content):
                        child_sitemaps = self.sitemap_parser.parse_sitemap_index(content)
                        for child_url, _ in child_sitemaps[:3]:
                            self.sitemap_polls_count[source_key] += 1
                            try:
                                c_headers = self.recrawler.get_conditional_headers(child_url)
                                c_resp = await client.get(child_url, headers=c_headers, timeout=DEFAULT_TIMEOUT)
                                c_resp_headers = getattr(c_resp, "headers", {})
                                self.recrawler.cache_response_headers(child_url, c_resp_headers)
                                if c_resp.status_code == 304:
                                    self.sitemap_304_count[source_key] += 1
                                    logger.debug("Child sitemap %s: 304 Not Modified", child_url)
                                elif c_resp.status_code == 200:
                                    entries = self.sitemap_parser.parse_sitemap(c_resp.text)
                                    # Newly observed/published URLs (not yet seen in frontier)
                                    new_cands = [e.url for e in entries if e.url not in self.frontier.seen_urls]
                                    total_new_urls += len(new_cands)
                                    for cand in new_cands:
                                        tier = self.recrawler.get_priority_tier(url=cand)
                                        if self.frontier.add_url(cand, priority_tier=tier):
                                            new_enqueued += 1
                                            self.urls_enqueued_by_tier[tier] += 1
                            except Exception as ce:
                                logger.debug("Failed child sitemap poll %s: %s", child_url, ce)
                    else:
                        entries = self.sitemap_parser.parse_sitemap(content)
                        # Newly observed/published URLs (not yet seen in frontier)
                        new_cands = [e.url for e in entries if e.url not in self.frontier.seen_urls]
                        total_new_urls += len(new_cands)
                        for cand in new_cands:
                            tier = self.recrawler.get_priority_tier(url=cand)
                            if self.frontier.add_url(cand, priority_tier=tier):
                                new_enqueued += 1
                                self.urls_enqueued_by_tier[tier] += 1
            except Exception as e:
                logger.warning("Error re-polling sitemap %s: %s", s_url, e)

        # Update source velocity using the count of newly observed/published URLs across the poll cycle
        # (or delta_n = 0 if all sitemaps returned 304 or yielded zero new URLs)
        self.recrawler.update_rate(source_key, delta_n=total_new_urls, current_time=now)

        return new_enqueued

    async def crawl(self, client: Optional[httpx.AsyncClient] = None) -> int:
        """Main crawl loop executing until target count is reached or frontier drains."""
        self.start_time = self.time_func()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        close_client = False
        if client is None:
            client = httpx.AsyncClient(
                headers=DEFAULT_HEADERS,
                timeout=DEFAULT_TIMEOUT,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
            close_client = True

        try:
            # Seed frontier
            await self.bootstrap_seeds(client)

            # In sample mode, overwrite previous sample runs for a clean deliverable;
            # in normal mode, append to support resuming long crawls.
            file_mode = "w" if self.is_sample_mode else "a"
            with open(self.output_path, file_mode, encoding="utf-8") as out_f:
                while self.saved_articles_count < self.max_articles and not self.frontier.is_empty():
                    now = self.time_func()
                    # Adaptive recrawling: check if any source is due for sitemap re-checking
                    for src in self.active_sources:
                        if src in self.recrawler.last_checked and self.recrawler.should_poll(src, current_time=now):
                            await self.poll_source_sitemaps(client, src)

                    next_item = await self.frontier.get_next_url()
                    if next_item is None:
                        if self.frontier.is_empty():
                            logger.info("Frontier completely drained.")
                            break
                        await asyncio.sleep(0.5)
                        continue

                    url, host = next_item

                    # Skip disabled hosts (403 / CAPTCHA)
                    if host in self.disabled_hosts:
                        self.frontier.complete_request(host, delay=self.get_host_delay(host))
                        continue

                    # Check robots.txt compliance
                    if not self.robots.can_fetch(url):
                        logger.debug("Robots disallowed URL: %s", url)
                        self.frontier.complete_request(host, delay=self.get_host_delay(host))
                        continue

                    # Fetch and extract
                    eff_delay = self.get_host_delay(host)
                    try:
                        html, status, err = await self.fetch_article_html(client, url, host)

                        if status == 429:
                            # Exponential backoff on rate-limit
                            curr_delay = self.get_host_delay(host)
                            eff_delay = min(curr_delay * BACKOFF_FACTOR, 64.0)
                            self.host_delays[host] = eff_delay
                            logger.warning("Host %s returned 429. Increasing delay to %.1fs", host, eff_delay)

                        elif status == 403 or (err and "CAPTCHA" in err):
                            logger.warning("Host %s intercepted (%s: %s). Disabling for crawl session.", host, status, err)
                            self.disabled_hosts.add(host)

                        elif status == 200 and html:
                            # Reset delay if host is healthy
                            if host in self.host_delays:
                                self.host_delays[host] = self.frontier.per_host_delay
                                eff_delay = self.frontier.per_host_delay

                            source_key = get_source_key(url)
                            article = self.extractor.extract(html, url=url, source_slug=source_key)

                            if article and article["doc_id"] not in self.seen_doc_ids:
                                if validate_article_schema(article):
                                    self.write_record(article, out_f)
                                    self.saved_articles_count += 1
                                    self.articles_by_source[source_key] += 1
                                    if article.get("section"):
                                        self.articles_by_section[article["section"]] += 1
                                    if article.get("state"):
                                        self.articles_by_state[article["state"]] += 1
                                    if article.get("city"):
                                        self.articles_by_city[article["city"]] += 1
                                    if article.get("language"):
                                        self.articles_by_language[article["language"]] += 1

                                    body_text = article.get("body", "")
                                    self.total_body_chars += len(body_text)
                                    self.total_body_words += len(body_text.split())

                                    # Record article for category burst tracking
                                    if article.get("section"):
                                        sec = article["section"]
                                        self.recrawler.record_article(sec, timestamp=self.time_func())
                                        if self.recrawler.is_bursting(sec, current_time=self.time_func()):
                                            if sec not in self.active_burst_categories_seen:
                                                self.active_burst_categories_seen.add(sec)
                                                score = self.recrawler.get_burst_score(sec, current_time=self.time_func())
                                                self.burst_events_observed.append({
                                                    "section": sec,
                                                    "timestamp": self.time_func(),
                                                    "burst_score": score,
                                                })
                                                logger.info("TOPICAL BURST DETECTED: Section '%s' surged with score %.2f", sec, score)

                                    # Harvest links
                                    if article.get("links"):
                                        self.harvest_links(article["links"])

                                    if self.saved_articles_count % 10 == 0 or self.saved_articles_count == self.max_articles:
                                        self.print_progress()
                                else:
                                    self.schema_validation_failures += 1
                                    logger.debug("Schema invalid for %s", url)

                    finally:
                        # Politeness guarantee: always reschedule host
                        self.frontier.complete_request(host, delay=eff_delay)

            # Snapshot save if crawl finished early without reaching 300
            if not self.is_sample_mode and not self._sample_snapshot_saved and self.sample_records:
                self._save_sample_snapshot()

            logger.info("Crawl session finished. Total valid articles saved: %d", self.saved_articles_count)
            return self.saved_articles_count

        finally:
            self.end_time = self.time_func()
            try:
                self.generate_crawl_report()
            except Exception as re:
                logger.warning("Could not generate end-of-crawl report: %s", re)
            if close_client:
                await client.aclose()

    def generate_crawl_report(self, output_path: Optional[Path] = None) -> str:
        """Compile and persist comprehensive Markdown crawl summary and statistics report."""
        target_path = Path(output_path or self.stats_file)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        end_t = self.end_time or self.time_func()
        start_t = self.start_time or end_t
        elapsed = max(end_t - start_t, 0.001)

        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        art_per_min = (self.saved_articles_count / elapsed) * 60.0
        art_per_sec = self.saved_articles_count / elapsed
        req_per_sec = self.total_http_requests / elapsed
        pct = (self.saved_articles_count / self.max_articles * 100.0) if self.max_articles else 0.0
        yield_pct = (self.saved_articles_count / self.total_http_requests * 100.0) if self.total_http_requests else 0.0

        avg_chars = (self.total_body_chars / self.saved_articles_count) if self.saved_articles_count else 0
        avg_words = (self.total_body_words / self.saved_articles_count) if self.saved_articles_count else 0

        # Source breakdown
        source_rows = []
        for s_key in sorted(self.active_sources):
            cnt = self.articles_by_source.get(s_key, 0)
            s_pct = (cnt / self.saved_articles_count * 100.0) if self.saved_articles_count else 0.0
            domain = PRIMARY_SOURCES.get(s_key, {}).get("domain", s_key)
            rate = self.recrawler.get_rate(s_key)
            interval_m = self.recrawler.get_polling_interval(s_key) / 60.0
            source_rows.append(f"| {s_key} | {domain} | {cnt:,} | {s_pct:.1f}% | {rate:.2f} | {interval_m:.1f}m |")

        # Section breakdown (top 15)
        sorted_sections = sorted(self.articles_by_section.items(), key=lambda x: x[1], reverse=True)
        section_rows = []
        for sec, cnt in sorted_sections[:15]:
            sec_pct = (cnt / self.saved_articles_count * 100.0) if self.saved_articles_count else 0.0
            section_rows.append(f"| {sec} | {cnt:,} | {sec_pct:.1f}% |")
        if not section_rows:
            section_rows.append("| Universal / General | 0 | 0.0% |")

        # Geographic breakdown
        state_count = sum(self.articles_by_state.values())
        city_count = sum(self.articles_by_city.values())
        null_geo = self.saved_articles_count - state_count
        state_pct = (state_count / self.saved_articles_count * 100.0) if self.saved_articles_count else 0.0
        city_pct = (city_count / self.saved_articles_count * 100.0) if self.saved_articles_count else 0.0
        null_geo_pct = (null_geo / self.saved_articles_count * 100.0) if self.saved_articles_count else 0.0

        state_rows = []
        for st, cnt in sorted(self.articles_by_state.items(), key=lambda x: x[1], reverse=True):
            st_pct = (cnt / state_count * 100.0) if state_count else 0.0
            state_rows.append(f"| {st} | {cnt:,} | {st_pct:.1f}% |")
        if not state_rows:
            state_rows.append("| None identified | 0 | 0.0% |")

        city_rows = []
        for ct, cnt in sorted(self.articles_by_city.items(), key=lambda x: x[1], reverse=True)[:15]:
            ct_pct = (cnt / city_count * 100.0) if city_count else 0.0
            city_rows.append(f"| {ct} | {cnt:,} | {ct_pct:.1f}% |")
        if not city_rows:
            city_rows.append("| None identified | 0 | 0.0% |")

        # HTTP Status codes
        status_names = {
            200: "200 OK (Successful Fetch)",
            304: "304 Not Modified (Cached Feed)",
            403: "403 Forbidden (Blocked / WAF)",
            429: "429 Too Many Requests (Rate Limited)",
            500: "500 Internal Server Error",
            502: "502 Bad Gateway",
            503: "503 Service Unavailable",
            0: "Network Timeout / DNS Error",
        }
        status_rows = []
        for st, cnt in sorted(self.http_status_counts.items()):
            label = status_names.get(st, f"HTTP {st}")
            st_pct = (cnt / self.total_http_requests * 100.0) if self.total_http_requests else 0.0
            status_rows.append(f"| {st} | {label} | {cnt:,} | {st_pct:.1f}% |")
        if not status_rows:
            status_rows.append("| — | No HTTP requests dispatched | 0 | 0.0% |")

        # Host Delays
        delay_rows = []
        known_hosts = set(self.host_delays.keys()) | {PRIMARY_SOURCES.get(s, {}).get("domain", "") for s in self.active_sources}
        for h in sorted(h for h in known_hosts if h):
            delay = self.get_host_delay(h)
            backed_off = "Yes" if h in self.hosts_backed_off else "No"
            disabled = "Yes (403/CAPTCHA)" if h in self.disabled_hosts else "No (Active)"
            delay_rows.append(f"| {h} | {delay:.1f}s | {backed_off} | {disabled} |")
        if not delay_rows:
            delay_rows.append("| — | 8.0s | No | No (Active) |")

        # Adaptive Recrawling
        recrawl_rows = []
        tot_polls = sum(self.sitemap_polls_count.values())
        tot_304 = sum(self.sitemap_304_count.values())
        for s_key in sorted(self.active_sources):
            polls = self.sitemap_polls_count.get(s_key, 0)
            c304 = self.sitemap_304_count.get(s_key, 0)
            savings = (c304 / polls * 100.0) if polls else 0.0
            rate = self.recrawler.get_rate(s_key)
            tau = self.recrawler.get_polling_interval(s_key)
            recrawl_rows.append(f"| {s_key} | {polls:,} | {c304:,} | {savings:.1f}% | {rate:.2f} URLs/h | {tau/60.0:.1f} min ({tau:.0f}s) |")

        # Priority Tiers
        q0 = self.urls_enqueued_by_tier.get("Q0", 0)
        q1 = self.urls_enqueued_by_tier.get("Q1", 0)
        q2 = self.urls_enqueued_by_tier.get("Q2", 0)
        q3 = self.urls_enqueued_by_tier.get("Q3", 0)
        tot_enq = q0 + q1 + q2 + q3

        # Burst events
        burst_rows = []
        if self.burst_events_observed:
            for b in self.burst_events_observed:
                burst_rows.append(f"| {b.get('section', 'unknown')} | {b.get('burst_score', 0.0):.2f} | Burst triggered |")
        else:
            burst_rows.append("| None observed | — | Steady publication flow |")

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report_md = f"""# Dhvani Regional News Crawler — Execution & Corpus Statistics

**Generated At:** {now_str}  
**Target Corpus Limit:** {self.max_articles:,} articles  
**Corpus Output File:** `{self.output_path}`  
**H3 Snapshot File:** `{self.sample_output_path}` (First 300 articles)  

---

## 1. Executive Performance Summary

| Metric | Measured Value | Note |
|---|---|---|
| **Valid Articles Saved** | **{self.saved_articles_count:,}** ({pct:.1f}% of target) | 100% compliant with `formats.md` schema |
| **Total Crawl Duration** | **{hours}h {minutes}m {seconds}s** ({elapsed:.1f} seconds) | Active single-threaded asynchronous runtime |
| **Throughput (Articles)** | **{art_per_min:.2f} articles/min** ({art_per_sec:.2f} art/sec) | Overlapped across 5 distinct news domains |
| **Total HTTP Requests** | **{self.total_http_requests:,}** | Article GETs + Sitemap polling |
| **Network Request Rate** | **{req_per_sec:.2f} requests/sec** | Strictly polite (≥8.0s per host enforced) |
| **Article Extraction Yield** | **{yield_pct:.1f}%** | Saved articles / Total HTTP requests |
| **Schema Validation Errors** | **{self.schema_validation_failures}** | Dropped due to missing body/headline/fields |
| **Average Article Length** | **{avg_words:.0f} words** ({avg_chars:.0f} characters) | Hindi prose content |
| **In-Body Links Harvested** | **{self.total_links_harvested:,}** | Candidate links extracted for PageRank |

---

## 2. Source Balance & Representation

| Source | Domain | Articles Crawled | Share (%) | Publication Velocity | Polling Interval |
|---|---|---|---|---|---|
""" + "\n".join(source_rows) + f"""

---

## 3. Topical Section Distribution (Top 15 Categories)

| Topical Section | Articles Crawled | Share (%) |
|---|---|---|
""" + "\n".join(section_rows) + f"""

---

## 4. Geographic Coverage & District Bureau Mapping

| Geographic Category | Articles | Share (%) | Description |
|---|---|---|---|
| **Articles with State Identified** | **{state_count:,}** | **{state_pct:.1f}%** | Mapped to Hindi-belt state |
| **Articles with District/City** | **{city_count:,}** | **{city_pct:.1f}%** | Mapped to specific district center |
| **Universal / National / State-Wide** | **{null_geo:,}** | **{null_geo_pct:.1f}%** | National, international, cricket, editorial |

### State Distribution
| State | Articles Crawled | Share of Tagged (%) |
|---|---|---|
""" + "\n".join(state_rows) + f"""

### Top District Centers & Cities
| District / City | Articles Crawled | Share of City-Tagged (%) |
|---|---|---|
""" + "\n".join(city_rows) + f"""

---

## 5. Network Health & HTTP Status Codes

| Status Code | Description | Occurrences | Percentage (%) |
|---|---|---|---|
""" + "\n".join(status_rows) + f"""

### Politeness Delays & Host Health
| Host Domain | Scheduling Delay | Backoff Triggered? | Status |
|---|---|---|---|
""" + "\n".join(delay_rows) + f"""

---

## 6. Adaptive Recrawling Performance & Bandwidth Optimization

| News Source | Sitemap Polls | 304 Not Modified | 304 Savings Ratio | Est. Velocity ($\\lambda_s$) | Calculated Interval ($\\tau_s$) |
|---|---|---|---|---|---|
""" + "\n".join(recrawl_rows) + f"""

**Overall Bandwidth Savings:** **{tot_304:,}** of **{tot_polls:,}** sitemap requests returned HTTP 304 Not Modified (**{(tot_304/tot_polls*100.0) if tot_polls else 0.0:.1f}% bandwidth reduction**).

---

## 7. Mercator Frontier Priority Queue Dynamics

| Priority Tier | Refill Bias | Description | URLs Enqueued | Share (%) |
|---|---|---|---|---|
| **$Q_0$ (Burst Surge)** | 60% | Breaking news & category burst events | {q0:,} | {(q0/tot_enq*100.0) if tot_enq else 0.0:.1f}% |
| **$Q_1$ (Sitemap Seeds)** | 25% | Routine sitemap discovery feeds | {q1:,} | {(q1/tot_enq*100.0) if tot_enq else 0.0:.1f}% |
| **$Q_2$ (Hyperlinks)** | 10% | Discovered in-article PageRank hyperlinks | {q2:,} | {(q2/tot_enq*100.0) if tot_enq else 0.0:.1f}% |
| **$Q_3$ (Archives)** | 5% | Deep archive pagination feeds | {q3:,} | {(q3/tot_enq*100.0) if tot_enq else 0.0:.1f}% |
| **Total URLs Enqueued** | — | — | **{tot_enq:,}** | 100.0% |

---

## 8. Topical Surge & Burst Events Observed

| Topical Category | Peak Burst Score | Event Status |
|---|---|---|
""" + "\n".join(burst_rows) + f"""

---

## 9. Downstream Processing Checklist

After crawling completes, run the following pipeline stages:
1. **Deduplication & Story Clustering (Task 6):**
   ```bash
   python -m dhvani.crawl.dedup --input data/news.jsonl --output data/news.jsonl
   ```
2. **Corpus Invariant Verification:**
   ```bash
   pytest partwise-tests/riya/test_format_compliance.py -v
   ```
3. **Downstream Handoff:**
   `data/news.jsonl` is ready for indexing (Dhrithi / P2).
"""

        with open(target_path, "w", encoding="utf-8") as f:
            f.write(report_md)

        logger.info("Comprehensive crawl report generated: %s", target_path)
        return report_md


def main():
    """CLI Entry point for Dhvani Regional News Crawler."""
    parser = argparse.ArgumentParser(description="Dhvani Regional News Crawler (P1)")
    parser.add_argument("--sample", action="store_true", help="Crawl 300 articles for H3 handoff")
    parser.add_argument("--max-articles", type=int, default=5000, help="Target article limit (default: 5000)")
    parser.add_argument("--sources", type=str, default="", help="Comma-separated subset of sources to crawl")
    parser.add_argument("--output", type=str, default="", help="Custom output JSONL path")
    parser.add_argument("--log-file", type=str, default="", help="Custom log file path (default: data/crawler.log)")
    parser.add_argument("--stats-file", type=str, default="", help="Custom markdown stats file (default: documentation/results/riya-corpus-results/crawl_stats.md)")
    args = parser.parse_args()

    if args.log_file:
        setup_crawler_logging(Path(args.log_file))

    active_sources = [s.strip() for s in args.sources.split(",") if s.strip()] or None
    output_path = Path(args.output) if args.output else None
    stats_path = Path(args.stats_file) if args.stats_file else None

    crawler = NewsCrawler(
        max_articles=args.max_articles,
        is_sample_mode=args.sample,
        output_path=output_path,
        stats_file=stats_path,
        active_sources=active_sources,
    )

    try:
        asyncio.run(crawler.crawl())
    except KeyboardInterrupt:
        logger.info("Crawl manually stopped by user. Progress saved.")
        crawler.generate_crawl_report()


if __name__ == "__main__":
    main()

