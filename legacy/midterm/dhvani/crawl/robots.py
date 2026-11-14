"""Robots.txt parser and per-host cache.

Follows RFC 9309 rules with pattern matching (wildcard * and end-of-path $)
and longest-prefix precedence, addressing standard urllib.robotparser shortcomings
on target regional news sites.
"""

import logging
import re
from typing import Dict, List, NamedTuple, Optional, Tuple
from urllib.parse import urlparse
import requests

from dhvani.crawl.config import DEFAULT_HEADERS, DEFAULT_TIMEOUT, USER_AGENT

logger = logging.getLogger(__name__)


class Rule(NamedTuple):
    pattern: str
    allow: bool
    length: int
    regex: re.Pattern


class HostRules:
    """Parsed rules and sitemaps for a single host."""

    def __init__(self):
        self.rules: List[Rule] = []
        self.sitemaps: List[str] = []

    def can_fetch(self, path: str) -> bool:
        """Evaluate path against rules using RFC 9309 longest-match precedence.

        If no rule matches, the path is allowed by default.
        """
        if not self.rules:
            return True

        matching_rules: List[Rule] = [r for r in self.rules if r.regex.search(path)]
        if not matching_rules:
            return True

        # Sort by pattern length descending (longest match wins).
        # On tie, Allow (allow=True) takes precedence over Disallow.
        best_rule = max(matching_rules, key=lambda r: (r.length, r.allow))
        return best_rule.allow


def pattern_to_regex(pattern: str) -> re.Pattern:
    """Convert robots.txt path pattern to compiled regex per RFC 9309.

    - * matches 0 or more characters (.*)
    - $ at end matches end of path ($)
    - Defaults to prefix match
    """
    pattern = pattern.strip()
    if not pattern:
        return re.compile(r"^.*")

    # If pattern does not begin with / or *, anchor to start of path
    regex = "^"
    i = 0
    n = len(pattern)
    while i < n:
        c = pattern[i]
        if c == "*":
            regex += ".*"
        elif c == "$" and i == n - 1:
            regex += "$"
        else:
            regex += re.escape(c)
        i += 1

    return re.compile(regex)


class RobotsParser:
    """Robots.txt manager that fetches, caches, and checks URLs per host."""

    def __init__(self, user_agent: str = USER_AGENT, timeout: float = DEFAULT_TIMEOUT):
        self.user_agent = user_agent
        self.user_agent_lower = user_agent.lower()
        self.timeout = timeout
        self.cache: Dict[str, HostRules] = {}

    def _get_host(self, url_or_host: str) -> str:
        """Extract netloc host from URL or return string as-is."""
        if "://" in url_or_host:
            parsed = urlparse(url_or_host)
            return parsed.netloc.lower()
        return url_or_host.lower()

    def parse_content(self, content: str, target_user_agent: Optional[str] = None) -> HostRules:
        """Parse raw robots.txt content into a HostRules object.

        Finds rule blocks for target_user_agent or falls back to '*'.
        Extracts all Sitemap: declarations.
        """
        ua_token = (target_user_agent or self.user_agent).lower()
        lines = content.splitlines()

        host_rules = HostRules()

        # Phase 1: Read all user-agent groups
        # Groups format: Dict[agent_name, List[Tuple[allow: bool, pattern: str]]]
        groups: Dict[str, List[Tuple[bool, str]]] = {}
        current_agents: List[str] = []

        for line in lines:
            # Strip comments
            line = line.split("#", 1)[0].strip()
            if not line:
                continue

            if ":" not in line:
                continue

            directive, value = line.split(":", 1)
            directive = directive.strip().lower()
            value = value.strip()

            if directive == "sitemap" and value:
                host_rules.sitemaps.append(value)
            elif directive == "user-agent":
                agent_name = value.lower()
                current_agents.append(agent_name)
                if agent_name not in groups:
                    groups[agent_name] = []
            elif directive in ("allow", "disallow") and current_agents:
                if not value and directive == "disallow":
                    # Disallow: with empty path means allow everything
                    for agent in current_agents:
                        groups[agent].append((True, "/"))
                elif value:
                    is_allow = (directive == "allow")
                    for agent in current_agents:
                        groups[agent].append((is_allow, value))

        # Phase 2: Select the best user-agent group
        # Look for exact / substring match for our bot, fallback to '*'
        selected_rules: List[Tuple[bool, str]] = []
        matched_agent = None

        for agent in groups:
            if agent != "*" and (agent in ua_token or ua_token in agent):
                matched_agent = agent
                selected_rules = groups[agent]
                break

        if not selected_rules and "*" in groups:
            selected_rules = groups["*"]

        # Phase 3: Compile into Rule instances
        for allow, pattern in selected_rules:
            try:
                regex = pattern_to_regex(pattern)
                host_rules.rules.append(
                    Rule(pattern=pattern, allow=allow, length=len(pattern), regex=regex)
                )
            except Exception as e:
                logger.warning("Failed to compile pattern '%s': %s", pattern, e)

        return host_rules

    def fetch_and_parse(self, host: str, scheme: str = "https") -> HostRules:
        """Fetch robots.txt for a host, parse and cache it."""
        host = self._get_host(host)
        if host in self.cache:
            return self.cache[host]

        robots_url = f"{scheme}://{host}/robots.txt"
        try:
            resp = requests.get(
                robots_url,
                headers=DEFAULT_HEADERS,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                host_rules = self.parse_content(resp.text)
            elif resp.status_code in (401, 403):
                # Forbidden robots.txt means all disallowed (RFC 9309 section 3.2.1)
                host_rules = HostRules()
                host_rules.rules.append(Rule(pattern="/", allow=False, length=1, regex=re.compile(r"^/")))
            else:
                # 404 / 410 or other errors mean allow all
                host_rules = HostRules()
        except Exception as e:
            logger.warning("Error fetching robots.txt from %s: %s. Defaulting to allow.", robots_url, e)
            host_rules = HostRules()

        self.cache[host] = host_rules
        return host_rules

    def can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched by our crawler."""
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if not host:
            return True

        if host not in self.cache:
            self.fetch_and_parse(host, scheme=parsed.scheme or "https")

        host_rules = self.cache[host]
        path_and_query = parsed.path or "/"
        if parsed.query:
            path_and_query += f"?{parsed.query}"

        return host_rules.can_fetch(path_and_query)

    def get_sitemaps(self, host_or_url: str) -> List[str]:
        """Get discovered sitemap URLs for a host."""
        host = self._get_host(host_or_url)
        if host not in self.cache:
            self.fetch_and_parse(host)
        return self.cache.get(host, HostRules()).sitemaps

    def extract_sitemaps(self, content: str) -> List[str]:
        """Extract sitemaps directly from robots.txt content string."""
        return self.parse_content(content).sitemaps

    def set_cached_rules(self, host: str, content: str) -> None:
        """Inject raw content for testing or offline usage without network request."""
        host = self._get_host(host)
        self.cache[host] = self.parse_content(content)

    def clear_cache(self) -> None:
        """Clear cached host rules."""
        self.cache.clear()
