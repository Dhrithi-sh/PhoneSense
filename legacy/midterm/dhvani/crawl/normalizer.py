"""URL Normalization and Canonicalization."""

import re
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse


def normalize_url(raw_url: str) -> str:
    """Normalize URL by stripping tracking parameters, normalizing scheme,

    resolving AMP paths, and stripping trailing slashes.
    """
    if not raw_url or not raw_url.strip():
        return ""
    # Normalize scheme & lower host
    parsed = urlparse(raw_url.strip())
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()

    # Strip default ports (:80 for http, :443 for https)
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    elif scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]

    # Drop tracking query params
    clean_params = {}
    if parsed.query:
        for k, v in parse_qs(parsed.query).items():
            if not k.startswith("utm_") and k not in {"fbclid", "ref", "amp_js_v", "amp"}:
                clean_params[k] = v[0]

    query_str = urlencode(clean_params) if clean_params else ""

    # Map AMP to canonical path
    path = parsed.path
    path = re.sub(r"^/amp(/|$)", "/", path)
    path = re.sub(r"/amp/?$", "", path)
    if path.endswith("/") and len(path) > 1:
        path = path[:-1]

    return urlunparse((scheme, netloc, path, "", query_str, ""))
