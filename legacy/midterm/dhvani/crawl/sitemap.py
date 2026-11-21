"""Sitemap fetcher and XML parser for standard XML, Google News, and index sitemaps."""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

from dhvani.crawl.config import DEFAULT_HEADERS, DEFAULT_TIMEOUT, PRIMARY_SOURCES
from dhvani.crawl.filters import should_skip_url
from dhvani.crawl.normalizer import normalize_url


@dataclass
class SitemapEntry:
    """Represents a single parsed URL entry from a sitemap feed."""

    url: str
    lastmod: Optional[str] = None
    publication_date: Optional[str] = None
    title: Optional[str] = None
    keywords: Optional[str] = None
    publication_name: Optional[str] = None
    language: Optional[str] = None
    changefreq: Optional[str] = None
    priority: Optional[float] = None
    is_news: bool = False


class SitemapParser:
    """Parses standard XML sitemaps (<urlset>), Google News sitemaps (<news:news>),

    and sitemap index files (<sitemapindex>).
    """

    def __init__(self, filter_urls: bool = True, normalize: bool = True):
        self.filter_urls = filter_urls
        self.normalize = normalize

    @staticmethod
    def _clean_tag(tag: str) -> str:
        """Strip XML namespace prefix from tag name."""
        return tag.split("}")[-1] if "}" in tag else tag

    def is_sitemap_index(self, xml_content: str) -> bool:
        """Check whether the given XML document represents a sitemap index (<sitemapindex>)."""
        if not xml_content or not xml_content.strip():
            return False
        try:
            root = ET.fromstring(xml_content.strip())
            return self._clean_tag(root.tag).lower() == "sitemapindex"
        except (ET.ParseError, ValueError):
            # Fallback regex if XML declaration causes minor parsing error
            return bool(re.search(r"<\s*sitemapindex[\s>]", xml_content, re.IGNORECASE))

    def parse_sitemap_index(self, xml_content: str) -> List[Tuple[str, Optional[str]]]:
        """Extract child sitemap (loc, lastmod) pairs from a <sitemapindex> document."""
        if not xml_content or not xml_content.strip():
            return []

        sitemaps: List[Tuple[str, Optional[str]]] = []
        try:
            root = ET.fromstring(xml_content.strip())
            for child in root:
                if self._clean_tag(child.tag).lower() == "sitemap":
                    loc: Optional[str] = None
                    lastmod: Optional[str] = None
                    for sub in child:
                        sub_tag = self._clean_tag(sub.tag).lower()
                        if sub_tag == "loc" and sub.text:
                            loc = sub.text.strip()
                        elif sub_tag == "lastmod" and sub.text:
                            lastmod = sub.text.strip()

                    if loc:
                        sitemaps.append((loc, lastmod))
        except (ET.ParseError, ValueError):
            # Regex fallback for imperfect XML feeds
            matches = re.findall(
                r"<\s*sitemap\s*>.*?<\s*loc\s*>([^<]+)<\s*/\s*loc\s*>.*?(?:<\s*lastmod\s*>([^<]+)<\s*/\s*lastmod\s*>)?.*?<\s*/\s*sitemap\s*>",
                xml_content,
                re.DOTALL | re.IGNORECASE,
            )
            for m in matches:
                loc = m[0].strip() if m[0] else ""
                lastmod = m[1].strip() if len(m) > 1 and m[1] else None
                if loc:
                    sitemaps.append((loc, lastmod))

        return sitemaps

    def parse_sitemap(self, xml_content: str) -> List[SitemapEntry]:
        """Extract article URLs and metadata from a standard or Google News <urlset> sitemap."""
        if not xml_content or not xml_content.strip():
            return []

        entries: List[SitemapEntry] = []
        try:
            root = ET.fromstring(xml_content.strip())
            for url_node in root:
                if self._clean_tag(url_node.tag).lower() != "url":
                    continue

                raw_loc: Optional[str] = None
                lastmod: Optional[str] = None
                changefreq: Optional[str] = None
                priority: Optional[float] = None

                # News fields
                is_news = False
                pub_date: Optional[str] = None
                title: Optional[str] = None
                keywords: Optional[str] = None
                pub_name: Optional[str] = None
                language: Optional[str] = None

                for child in url_node:
                    child_tag = self._clean_tag(child.tag).lower()

                    if child_tag == "loc" and child.text:
                        raw_loc = child.text.strip()
                    elif child_tag == "lastmod" and child.text:
                        lastmod = child.text.strip()
                    elif child_tag == "changefreq" and child.text:
                        changefreq = child.text.strip()
                    elif child_tag == "priority" and child.text:
                        try:
                            priority = float(child.text.strip())
                        except ValueError:
                            pass
                    elif child_tag == "news":
                        is_news = True
                        for news_child in child:
                            nc_tag = self._clean_tag(news_child.tag).lower()
                            if nc_tag == "publication":
                                for pub_child in news_child:
                                    pc_tag = self._clean_tag(pub_child.tag).lower()
                                    if pc_tag == "name" and pub_child.text:
                                        pub_name = pub_child.text.strip()
                                    elif pc_tag == "language" and pub_child.text:
                                        language = pub_child.text.strip()
                            elif nc_tag == "publication_date" and news_child.text:
                                pub_date = news_child.text.strip()
                            elif nc_tag == "title" and news_child.text:
                                title = news_child.text.strip()
                            elif nc_tag == "keywords" and news_child.text:
                                keywords = news_child.text.strip()

                if not raw_loc:
                    continue

                final_url = normalize_url(raw_loc) if self.normalize else raw_loc
                if not final_url:
                    continue

                if self.filter_urls and should_skip_url(final_url):
                    continue

                entries.append(
                    SitemapEntry(
                        url=final_url,
                        lastmod=lastmod,
                        publication_date=pub_date,
                        title=title,
                        keywords=keywords,
                        publication_name=pub_name,
                        language=language,
                        changefreq=changefreq,
                        priority=priority,
                        is_news=is_news,
                    )
                )

        except (ET.ParseError, ValueError):
            # Regex fallback extraction on XML parse errors
            url_blocks = re.findall(r"<\s*url\s*>(.*?)<\s*/\s*url\s*>", xml_content, re.DOTALL | re.IGNORECASE)
            for block in url_blocks:
                loc_match = re.search(r"<\s*loc\s*>([^<]+)<\s*/\s*loc\s*>", block, re.IGNORECASE)
                if not loc_match:
                    continue

                raw_loc = loc_match.group(1).strip()
                final_url = normalize_url(raw_loc) if self.normalize else raw_loc
                if not final_url or (self.filter_urls and should_skip_url(final_url)):
                    continue

                lastmod_match = re.search(r"<\s*lastmod\s*>([^<]+)<\s*/\s*lastmod\s*>", block, re.IGNORECASE)
                title_match = re.search(r"<\s*[^:>]*:?title\s*>([^<]+)<\s*/\s*[^:>]*:?title\s*>", block, re.IGNORECASE)
                date_match = re.search(r"<\s*[^:>]*:?publication_date\s*>([^<]+)<\s*/\s*[^:>]*:?publication_date\s*>", block, re.IGNORECASE)

                entries.append(
                    SitemapEntry(
                        url=final_url,
                        lastmod=lastmod_match.group(1).strip() if lastmod_match else None,
                        publication_date=date_match.group(1).strip() if date_match else None,
                        title=title_match.group(1).strip() if title_match else None,
                        is_news=bool(title_match or date_match),
                    )
                )

            # If no closed </url> blocks were found (e.g. truncated stream), extract direct <loc> tags
            if not entries:
                for loc_match in re.finditer(r"<\s*loc\s*>([^<]+)<\s*/\s*loc\s*>", xml_content, re.IGNORECASE):
                    raw_loc = loc_match.group(1).strip()
                    final_url = normalize_url(raw_loc) if self.normalize else raw_loc
                    if final_url and (not self.filter_urls or not should_skip_url(final_url)):
                        entries.append(SitemapEntry(url=final_url))

        return entries

    def discover_archive_sitemaps(self, index_xml: str, year: Optional[int] = None) -> List[str]:
        """Discover archive and backfill sitemap URLs from a sitemap index feed.

        Matches historical archives containing years (e.g. 2024, 2025, 2026) or
        archive path keywords.
        """
        all_sitemaps = self.parse_sitemap_index(index_xml)
        archive_urls: List[str] = []

        year_pattern = str(year) if year else r"\d{4}"
        archive_regex = re.compile(rf"(?:archive|sitemap[-_]{year_pattern}|/{year_pattern}/|{year_pattern}[-_]\d{{2}})", re.IGNORECASE)

        for loc, _ in all_sitemaps:
            if archive_regex.search(loc) or (not year and "archive" in loc.lower()):
                archive_urls.append(loc)

        return archive_urls

    def fetch_sitemap(
        self,
        url: str,
        etag: Optional[str] = None,
        last_modified: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> Tuple[int, Optional[str], Optional[str], Optional[str]]:
        """Fetch a sitemap over HTTP supporting conditional request headers (ETag, If-Modified-Since).

        Returns: (status_code, content_text, new_etag, new_last_modified).
        If unchanged (HTTP 304), content_text is None.
        """
        headers = dict(DEFAULT_HEADERS)
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified

        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            status = response.status_code

            if status == 304:
                return (304, None, etag, last_modified)

            if status == 200:
                new_etag = response.headers.get("ETag")
                new_last_mod = response.headers.get("Last-Modified")
                return (200, response.text, new_etag, new_last_mod)

            return (status, None, None, None)

        except requests.RequestException:
            return (0, None, None, None)

    @staticmethod
    def get_configured_seeds() -> List[str]:
        """Return the initial seed sitemap URLs across configured primary sources."""
        seeds: List[str] = []
        for info in PRIMARY_SOURCES.values():
            seeds.extend(info.get("sitemaps", []))
        return seeds

