"""Article and Metadata Extractor from JSON-LD and HTML DOM adhering to formats.md."""

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from dhvani.crawl.dedup import compute_content_hash, is_agency_story
from dhvani.crawl.normalizer import normalize_url

# Indian Standard Time (IST) timezone (+05:30)
IST = timezone(timedelta(hours=5, minutes=30))

# Recognized Hindi-belt states and path aliases
INDIAN_STATES: Dict[str, str] = {
    "uttar-pradesh": "uttar-pradesh",
    "uttarpradesh": "uttar-pradesh",
    "up": "uttar-pradesh",
    "bihar": "bihar",
    "haryana": "haryana",
    "madhya-pradesh": "madhya-pradesh",
    "madhyapradesh": "madhya-pradesh",
    "mp": "madhya-pradesh",
    "rajasthan": "rajasthan",
    "delhi": "delhi",
    "ncr": "delhi",
    "uttarakhand": "uttarakhand",
    "jharkhand": "jharkhand",
    "himachal-pradesh": "himachal-pradesh",
    "himachal": "himachal-pradesh",
    "chhattisgarh": "chhattisgarh",
    "punjab": "punjab",
}

# Prominent Hindi-belt regional cities and district centers
INDIAN_CITIES: Set[str] = {
    # Uttar Pradesh
    "lucknow", "kanpur", "patna", "varanasi", "prayagraj", "allahabad", "meerut", "agra",
    "noida", "ghaziabad", "bareilly", "aligarh", "moradabad", "gorakhpur", "ayodhya",
    "muzaffarnagar", "jhansi", "mathura", "basti", "firozabad", "saharanpur", "rampur",
    "shahjahanpur", "budaun", "bijnor", "sambhal", "amroha", "hardoi", "fatehpur",
    "raebareli", "sitapur", "lakhimpur", "bahraich", "gonda", "balrampur", "siddharthnagar",
    "deoria", "kushinagar", "mau", "azamgarh", "ballia", "ghazipur", "jaunpur", "mirzapur",
    "sonbhadra", "sultanpur", "amethi", "pratapgarh", "kaushambi", "banda", "chitrakoot",
    "hamirpur", "mahoba", "lalitpur", "jalaun", "mainpuri", "etawah", "kannauj", "auraiya",
    "farrukhabad", "kasganj", "hathras", "etah", "pilibhit", "barabanki", "unnao", "baghpat",
    "shamli", "hapur", "bulandshahr",
    # Delhi & NCR
    "delhi", "new-delhi",
    # Madhya Pradesh
    "bhopal", "indore", "gwalior", "jabalpur", "ujjain", "ratlam", "sagar", "rewa", "satna",
    "dewas", "burhanpur", "khandwa", "morena", "bhind", "shivpuri", "vidisha", "chhindwara",
    "damoh", "mandsaur", "neemuch", "singrauli", "sehore", "hoshangabad",
    # Rajasthan
    "jaipur", "jodhpur", "udaipur", "kota", "bikaner", "ajmer", "bhilwara", "alwar",
    "bharatpur", "sikar", "pali", "churu", "jhunjhunu", "tonk", "dausa", "chittorgarh",
    "bundi", "baran", "jhalawar", "sawai-madhopur", "sriganganagar", "hanumangarh",
    # Bihar
    "gaya", "bhagalpur", "muzaffarpur", "darbhanga", "purnea", "begusarai", "katihar",
    "munger", "chhapra", "bettiah", "saharsa", "sasaram", "hajipur", "dehri", "siwan",
    "motihari", "nawada", "buxar", "kishanganj", "sitamarhi", "jamui", "madhubani",
    "samastipur", "arrah",
    # Jharkhand
    "ranchi", "jamshedpur", "dhanbad", "bokaro", "deoghar", "hazaribagh", "giridih",
    "ramgarh", "medininagar", "chaibasa",
    # Chhattisgarh
    "raipur", "bilaspur", "durg", "bhilai", "korba", "rajnandgaon", "jagdalpur", "ambikapur",
    # Uttarakhand
    "dehradun", "haridwar", "roorkee", "haldwani", "rudrapur", "kashipur", "rishikesh",
    "pithoragarh", "almora", "nainital", "mussoorie", "kotdwar", "chamoli", "tehri",
    # Himachal Pradesh
    "shimla", "kangra", "dharamshala", "solan", "mandi", "kullu", "manali", "una", "chamba",
    # Haryana & Punjab
    "chandigarh", "gurugram", "gurgaon", "faridabad", "panipat", "ambala", "rohtak", "hissar",
    "karnal", "sonipat", "panchkula", "sirsa", "bhiwani", "bahadurgarh", "jind", "kaithal",
    "rewari", "palwal", "yamunanagar", "kurukshetra",
}

# Deterministic mapping from Hindi-belt regional cities to their respective states
CITY_TO_STATE: Dict[str, str] = {
    # Uttar Pradesh
    "lucknow": "uttar-pradesh", "kanpur": "uttar-pradesh", "varanasi": "uttar-pradesh",
    "prayagraj": "uttar-pradesh", "allahabad": "uttar-pradesh", "meerut": "uttar-pradesh",
    "agra": "uttar-pradesh", "noida": "uttar-pradesh", "ghaziabad": "uttar-pradesh",
    "bareilly": "uttar-pradesh", "aligarh": "uttar-pradesh", "moradabad": "uttar-pradesh",
    "gorakhpur": "uttar-pradesh", "ayodhya": "uttar-pradesh", "muzaffarnagar": "uttar-pradesh",
    "jhansi": "uttar-pradesh", "mathura": "uttar-pradesh", "basti": "uttar-pradesh",
    "firozabad": "uttar-pradesh", "saharanpur": "uttar-pradesh", "rampur": "uttar-pradesh",
    "shahjahanpur": "uttar-pradesh", "budaun": "uttar-pradesh", "bijnor": "uttar-pradesh",
    "sambhal": "uttar-pradesh", "amroha": "uttar-pradesh", "hardoi": "uttar-pradesh",
    "fatehpur": "uttar-pradesh", "raebareli": "uttar-pradesh", "sitapur": "uttar-pradesh",
    "lakhimpur": "uttar-pradesh", "bahraich": "uttar-pradesh", "gonda": "uttar-pradesh",
    "balrampur": "uttar-pradesh", "siddharthnagar": "uttar-pradesh", "deoria": "uttar-pradesh",
    "kushinagar": "uttar-pradesh", "mau": "uttar-pradesh", "azamgarh": "uttar-pradesh",
    "ballia": "uttar-pradesh", "ghazipur": "uttar-pradesh", "jaunpur": "uttar-pradesh",
    "mirzapur": "uttar-pradesh", "sonbhadra": "uttar-pradesh", "sultanpur": "uttar-pradesh",
    "amethi": "uttar-pradesh", "pratapgarh": "uttar-pradesh", "kaushambi": "uttar-pradesh",
    "banda": "uttar-pradesh", "chitrakoot": "uttar-pradesh", "hamirpur": "uttar-pradesh",
    "mahoba": "uttar-pradesh", "lalitpur": "uttar-pradesh", "jalaun": "uttar-pradesh",
    "mainpuri": "uttar-pradesh", "etawah": "uttar-pradesh", "kannauj": "uttar-pradesh",
    "auraiya": "uttar-pradesh", "farrukhabad": "uttar-pradesh", "kasganj": "uttar-pradesh",
    "hathras": "uttar-pradesh", "etah": "uttar-pradesh", "pilibhit": "uttar-pradesh",
    "barabanki": "uttar-pradesh", "unnao": "uttar-pradesh", "baghpat": "uttar-pradesh",
    "shamli": "uttar-pradesh", "hapur": "uttar-pradesh", "bulandshahr": "uttar-pradesh",
    # Delhi & NCR
    "delhi": "delhi", "new-delhi": "delhi",
    # Madhya Pradesh
    "bhopal": "madhya-pradesh", "indore": "madhya-pradesh", "gwalior": "madhya-pradesh",
    "jabalpur": "madhya-pradesh", "ujjain": "madhya-pradesh", "ratlam": "madhya-pradesh",
    "sagar": "madhya-pradesh", "rewa": "madhya-pradesh", "satna": "madhya-pradesh",
    "dewas": "madhya-pradesh", "burhanpur": "madhya-pradesh", "khandwa": "madhya-pradesh",
    "morena": "madhya-pradesh", "bhind": "madhya-pradesh", "shivpuri": "madhya-pradesh",
    "vidisha": "madhya-pradesh", "chhindwara": "madhya-pradesh", "damoh": "madhya-pradesh",
    "mandsaur": "madhya-pradesh", "neemuch": "madhya-pradesh", "singrauli": "madhya-pradesh",
    "sehore": "madhya-pradesh", "hoshangabad": "madhya-pradesh",
    # Rajasthan
    "jaipur": "rajasthan", "jodhpur": "rajasthan", "udaipur": "rajasthan", "kota": "rajasthan",
    "bikaner": "rajasthan", "ajmer": "rajasthan", "bhilwara": "rajasthan", "alwar": "rajasthan",
    "bharatpur": "rajasthan", "sikar": "rajasthan", "pali": "rajasthan", "churu": "rajasthan",
    "jhunjhunu": "rajasthan", "tonk": "rajasthan", "dausa": "rajasthan",
    "chittorgarh": "rajasthan", "bundi": "rajasthan", "baran": "rajasthan",
    "jhalawar": "rajasthan", "sawai-madhopur": "rajasthan", "sriganganagar": "rajasthan",
    "hanumangarh": "rajasthan",
    # Bihar
    "patna": "bihar", "gaya": "bihar", "bhagalpur": "bihar", "muzaffarpur": "bihar",
    "darbhanga": "bihar", "purnea": "bihar", "begusarai": "bihar", "katihar": "bihar",
    "munger": "bihar", "chhapra": "bihar", "bettiah": "bihar", "saharsa": "bihar",
    "sasaram": "bihar", "hajipur": "bihar", "dehri": "bihar", "siwan": "bihar",
    "motihari": "bihar", "nawada": "bihar", "buxar": "bihar", "kishanganj": "bihar",
    "sitamarhi": "bihar", "jamui": "bihar", "madhubani": "bihar", "samastipur": "bihar",
    "arrah": "bihar",
    # Jharkhand
    "ranchi": "jharkhand", "jamshedpur": "jharkhand", "dhanbad": "jharkhand",
    "bokaro": "jharkhand", "deoghar": "jharkhand", "hazaribagh": "jharkhand",
    "giridih": "jharkhand", "ramgarh": "jharkhand", "medininagar": "jharkhand",
    "chaibasa": "jharkhand",
    # Chhattisgarh
    "raipur": "chhattisgarh", "bilaspur": "chhattisgarh", "durg": "chhattisgarh",
    "bhilai": "chhattisgarh", "korba": "chhattisgarh", "rajnandgaon": "chhattisgarh",
    "jagdalpur": "chhattisgarh", "ambikapur": "chhattisgarh",
    # Uttarakhand
    "dehradun": "uttarakhand", "haridwar": "uttarakhand", "roorkee": "uttarakhand",
    "haldwani": "uttarakhand", "rudrapur": "uttarakhand", "kashipur": "uttarakhand",
    "rishikesh": "uttarakhand", "pithoragarh": "uttarakhand", "almora": "uttarakhand",
    "nainital": "uttarakhand", "mussoorie": "uttarakhand", "kotdwar": "uttarakhand",
    "chamoli": "uttarakhand", "tehri": "uttarakhand",
    # Himachal Pradesh
    "shimla": "himachal-pradesh", "kangra": "himachal-pradesh",
    "dharamshala": "himachal-pradesh", "solan": "himachal-pradesh",
    "mandi": "himachal-pradesh", "kullu": "himachal-pradesh", "manali": "himachal-pradesh",
    "una": "himachal-pradesh", "chamba": "himachal-pradesh",
    # Haryana & Punjab
    "chandigarh": "punjab", "gurugram": "haryana", "gurgaon": "haryana",
    "faridabad": "haryana", "panipat": "haryana", "ambala": "haryana",
    "rohtak": "haryana", "hissar": "haryana", "karnal": "haryana",
    "sonipat": "haryana", "panchkula": "haryana", "sirsa": "haryana",
    "bhiwani": "haryana", "bahadurgarh": "haryana", "jind": "haryana",
    "kaithal": "haryana", "rewari": "haryana", "palwal": "haryana",
    "yamunanagar": "haryana", "kurukshetra": "haryana",
}

# Standard news sections
SECTION_MAPPING: Dict[str, str] = {
    "weather": "weather",
    "mosam": "weather",
    "national": "national",
    "india": "national",
    "desh": "national",
    "state": "state",
    "states": "state",
    "pradesh": "state",
    "politics": "politics",
    "chunav": "politics",
    "rajneeti": "politics",
    "crime": "crime",
    "apradh": "crime",
    "sports": "sports",
    "khel": "sports",
    "cricket": "sports",
    "business": "business",
    "karobar": "business",
    "market": "business",
    "entertainment": "entertainment",
    "bollywood": "entertainment",
    "cinema": "entertainment",
    "education": "education",
    "jobs": "education",
    "rojgar": "education",
    "technology": "technology",
    "tech": "technology",
    "gadgets": "technology",
    "world": "world",
    "videsh": "world",
    "international": "world",
}


def parse_ist_date(date_str: Optional[str]) -> Optional[str]:
    """Parse publication timestamp and convert strictly to ISO-8601 with IST offset (+05:30).

    Returns None if date cannot be parsed.
    """
    if not date_str or not date_str.strip():
        return None

    cleaned = date_str.strip()

    # Try ISO-8601 parsing directly (supports trailing Z and numeric offsets)
    try:
        # Standardize 'Z' to '+00:00' for universal fromisoformat compatibility
        iso_candidate = cleaned.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso_candidate)
        if dt.tzinfo is None:
            # Assume naive timestamp from Indian news site is local IST
            dt = dt.replace(tzinfo=IST)
        else:
            # Convert timezone-aware datetime to IST
            dt = dt.astimezone(IST)
        return dt.strftime("%Y-%m-%dT%H:%M:%S+05:30")
    except ValueError:
        pass

    # Common alternate date formats across Indian portals
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y %H:%M",
        "%d %b %Y, %I:%M %p",
        "%d %b %Y %I:%M %p",
        "%B %d, %Y %I:%M %p",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(cleaned, fmt)
            dt = dt.replace(tzinfo=IST)
            return dt.strftime("%Y-%m-%dT%H:%M:%S+05:30")
        except ValueError:
            continue

    return None


def extract_location_from_url(url: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract (state, city) from URL path.

    Returns (None, None) if not reliably identifiable (no guessing).
    """
    if not url:
        return None, None

    path = urlparse(url).path.lower()
    clean_path = re.sub(r"\.[a-zA-Z0-9]+$", "", path)
    slash_segments = [s.strip() for s in clean_path.split("/") if s.strip()]

    matched_state: Optional[str] = None
    matched_city: Optional[str] = None

    # Compound check for new-delhi
    if "new-delhi" in clean_path or "new_delhi" in clean_path:
        matched_city = "new-delhi"
        matched_state = "delhi"

    # Jagran city slug pattern (e.g. /delhi/new-delhi-city-... or /uttar-pradesh/lucknow-city-...)
    m_jagran = re.search(r"/([a-z-]+)-city-", clean_path)
    if m_jagran and not matched_city:
        cand = m_jagran.group(1).replace("_", "-")
        if cand in INDIAN_CITIES:
            matched_city = cand

    # First check slash segments directly (handles compound names e.g. /uttar-pradesh/)
    for seg in slash_segments:
        normalized_seg = seg.replace("_", "-")
        if normalized_seg in INDIAN_STATES and not matched_state:
            matched_state = INDIAN_STATES[normalized_seg]
        elif normalized_seg in INDIAN_CITIES and not matched_city:
            matched_city = normalized_seg

    # Next check sub-tokens separated by hyphens or underscores (e.g. lucknow-weather-123)
    for seg in slash_segments:
        sub_tokens = [t.strip() for t in re.split(r"[-_]", seg) if t.strip()]
        for token in sub_tokens:
            if token in INDIAN_CITIES and not matched_city:
                matched_city = token
            elif token in INDIAN_STATES and not matched_state:
                matched_state = INDIAN_STATES[token]

    # If city is identified but state is not, deterministically derive state from city
    if matched_city and not matched_state and matched_city in CITY_TO_STATE:
        matched_state = CITY_TO_STATE[matched_city]

    # If state is delhi and city is null, default city to delhi
    if matched_state == "delhi" and not matched_city:
        matched_city = "delhi"

    return matched_state, matched_city



def extract_section_from_url(url: str, metadata_section: Optional[str] = None) -> Optional[str]:
    """Identify normalized section slug from metadata or URL path."""
    if metadata_section:
        ms = metadata_section.lower().strip()
        for k, canonical in SECTION_MAPPING.items():
            if k in ms:
                return canonical

    path = urlparse(url).path.lower()
    segments = [s.strip() for s in re.split(r"[/_-]", path) if s.strip()]

    for seg in segments:
        if seg in SECTION_MAPPING:
            return SECTION_MAPPING[seg]

    return "general"


def generate_doc_id(url: str, source_slug: str) -> str:
    """Generate deterministic doc_id adhering to formats.md (e.g. jagran_23456789)."""
    # Look for numeric article ID in URL path (common across Indian news outlets)
    numeric_match = re.search(r"[-_/](\d{5,12})(?:\.html|\.cms|/|$)", url)
    if numeric_match:
        return f"{source_slug}_{numeric_match.group(1)}"

    # Deterministic fallback hash based on normalized URL
    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
    return f"{source_slug}_{url_hash}"


def validate_article_schema(article: Dict[str, Any]) -> bool:
    """Validate that an extracted record strictly complies with documentation/formats.md.

    Checks:
    - All 14 required fields present
    - Types match contract specification
    - date is strictly ISO-8601 with +05:30 offset
    - dup_of is None or valid doc_id string
    - links is a list of strings
    - Absolutely no author/byline fields exist
    """
    required_keys = {
        "doc_id",
        "url",
        "source",
        "section",
        "state",
        "city",
        "date",
        "headline",
        "body",
        "keywords",
        "agency_flag",
        "content_hash",
        "dup_of",
        "links",
    }

    if not isinstance(article, dict):
        return False

    if set(article.keys()) != required_keys:
        return False

    # String non-empty requirements
    if not isinstance(article["doc_id"], str) or not article["doc_id"].strip():
        return False
    if not isinstance(article["url"], str) or not article["url"].strip():
        return False
    if not isinstance(article["source"], str) or not article["source"].strip():
        return False
    if not isinstance(article["headline"], str) or not article["headline"].strip():
        return False
    if not isinstance(article["body"], str) or not article["body"].strip():
        return False

    # Strict IST date validation (YYYY-MM-DDTHH:MM:SS+05:30)
    date_val = article["date"]
    if not isinstance(date_val, str) or not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\+05:30$", date_val):
        return False

    # Nullable string fields
    if article["section"] is not None and not isinstance(article["section"], str):
        return False
    if article["state"] is not None and not isinstance(article["state"], str):
        return False
    if article["city"] is not None and not isinstance(article["city"], str):
        return False

    # Deduplication and wire fields
    if not isinstance(article["keywords"], list) or not all(isinstance(k, str) for k in article["keywords"]):
        return False
    if not isinstance(article["agency_flag"], bool):
        return False
    if not isinstance(article["content_hash"], str) or len(article["content_hash"]) != 32:
        return False
    if article["dup_of"] is not None and not isinstance(article["dup_of"], str):
        return False
    if not isinstance(article["links"], list) or not all(isinstance(lk, str) for l in article["links"] for lk in (l if isinstance(l, list) else [l])):
        return False

    # Privacy constraint: verify no author fields exist anywhere in record
    forbidden_keys = {"author", "authors", "byline", "creator", "editor", "reporter", "writer"}
    if any(k.lower() in forbidden_keys for k in article.keys()):
        return False

    return True


class ArticleExtractor:
    """Extracts structured news articles adhering to formats.md.

    Prioritizes Schema.org JSON-LD with metadata-only Open Graph fallback
    and HTML DOM full-text body fallback. Guarantees zero author names stored.
    """

    def __init__(self):
        pass

    def extract_article(self, html_content: str, url: str, source_slug: str) -> Optional[Dict[str, Any]]:
        """Alias for extract()."""
        return self.extract(html_content, url=url, source_slug=source_slug)

    def extract(self, html_content: str, url: str, source_slug: str) -> Optional[Dict[str, Any]]:
        """Parse raw HTML and return structured article dict adhering to formats.md,

        or None if required core content (headline/body) cannot be extracted.
        """
        if not html_content or not html_content.strip() or not url:
            return None

        clean_url = normalize_url(url)
        soup = BeautifulSoup(html_content, "html.parser")

        # 1. Tier 1: Schema.org JSON-LD extraction
        jsonld_data = self._extract_jsonld(soup)

        # 2. Tier 2: Open Graph & Meta fallback (metadata only)
        meta_data = self._extract_meta(soup)

        # Resolve Headline (Tier 1 -> Tier 2 -> <h1> fallback)
        headline = (
            jsonld_data.get("headline")
            or meta_data.get("title")
            or self._extract_h1(soup)
        )
        if headline:
            headline = " ".join(headline.strip().split())

        # Resolve Body text
        # Tier 1: JSON-LD articleBody if substantive
        body = jsonld_data.get("articleBody")
        if not body or len(body.strip()) < 80:
            # Tier 3: Fall through to HTML DOM extraction (never substitute og:description)
            body = self._extract_body_from_dom(soup)

        if not headline or not body or len(body.strip()) < 40:
            return None

        body = " ".join(body.strip().split())

        # Resolve Publication Date in strict IST format
        raw_date = (
            jsonld_data.get("datePublished")
            or jsonld_data.get("dateModified")
            or meta_data.get("published_time")
        )
        ist_date = parse_ist_date(raw_date)
        if not ist_date:
            # Fallback to current time in IST
            ist_date = datetime.now(IST).strftime("%Y-%m-%dT%H:%M:%S+05:30")

        # Resolve Location (Strict null if unidentifiable, no guessing)
        state, city = extract_location_from_url(clean_url)

        # Resolve Section slug
        raw_section = jsonld_data.get("articleSection") or meta_data.get("section")
        section = extract_section_from_url(clean_url, raw_section)

        # Resolve Keywords
        keywords = jsonld_data.get("keywords") or meta_data.get("keywords") or []

        # Resolve Links in body
        links = self._extract_body_links(soup, source_slug)

        # Compute content hash and wire agency signature
        content_hash = compute_content_hash(body)
        agency_flag = is_agency_story(body)

        # Generate unique doc_id
        doc_id = generate_doc_id(clean_url, source_slug)

        record = {
            "doc_id": doc_id,
            "url": clean_url,
            "source": source_slug,
            "section": section,
            "state": state,
            "city": city,
            "date": ist_date,
            "headline": headline,
            "body": body,
            "keywords": keywords,
            "agency_flag": agency_flag,
            "content_hash": content_hash,
            "dup_of": None,
            "links": links,
        }

        # Validate schema compliance before returning
        if not validate_article_schema(record):
            return None

        return record

    def _extract_jsonld(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract article metadata from Schema.org JSON-LD scripts."""
        data: Dict[str, Any] = {}

        for script in soup.find_all("script", type=lambda t: t and "ld+json" in t.lower()):
            if not script.string:
                continue
            try:
                parsed = json.loads(script.string.strip())
            except (json.JSONDecodeError, ValueError):
                continue

            candidates = []
            if isinstance(parsed, dict):
                if "@graph" in parsed and isinstance(parsed["@graph"], list):
                    candidates.extend(parsed["@graph"])
                else:
                    candidates.append(parsed)
            elif isinstance(parsed, list):
                candidates.extend(parsed)

            for item in candidates:
                if not isinstance(item, dict):
                    continue
                type_val = item.get("@type", "")
                types = [type_val] if isinstance(type_val, str) else type_val

                if any(t.lower() in ("newsarticle", "article", "blogposting") for t in types if isinstance(t, str)):
                    # Headline
                    if not data.get("headline"):
                        data["headline"] = item.get("headline") or item.get("name")

                    # Body
                    if not data.get("articleBody"):
                        data["articleBody"] = item.get("articleBody")

                    # Dates
                    if not data.get("datePublished"):
                        data["datePublished"] = item.get("datePublished")
                    if not data.get("dateModified"):
                        data["dateModified"] = item.get("dateModified")

                    # Section
                    if not data.get("articleSection"):
                        sec = item.get("articleSection")
                        data["articleSection"] = sec[0] if isinstance(sec, list) and sec else sec

                    # Keywords
                    if not data.get("keywords") and item.get("keywords"):
                        kw = item["keywords"]
                        if isinstance(kw, str):
                            data["keywords"] = [k.strip() for k in kw.split(",") if k.strip()]
                        elif isinstance(kw, list):
                            data["keywords"] = [str(k).strip() for k in kw if str(k).strip()]

        return data

    def _extract_meta(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract fallback metadata from Open Graph and HTML meta tags.

        Notice: og:description is intentionally excluded as a body substitute.
        """
        meta: Dict[str, Any] = {}

        # Title
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            meta["title"] = og_title["content"].strip()
        elif soup.title and soup.title.string:
            # Drop trailing site suffix e.g. " - Dainik Jagran"
            raw_title = soup.title.string.strip()
            meta["title"] = re.split(r"\s*[-|–]\s*", raw_title)[0]

        # Publication Date
        for prop in ("article:published_time", "og:published_time", "publish-date", "date"):
            tag = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
            if tag and tag.get("content"):
                meta["published_time"] = tag["content"].strip()
                break

        # Section
        sec_tag = soup.find("meta", property="article:section") or soup.find("meta", attrs={"name": "section"})
        if sec_tag and sec_tag.get("content"):
            meta["section"] = sec_tag["content"].strip()

        # Keywords
        kw_tag = soup.find("meta", attrs={"name": "keywords"}) or soup.find("meta", property="article:tag")
        if kw_tag and kw_tag.get("content"):
            meta["keywords"] = [k.strip() for k in kw_tag["content"].split(",") if k.strip()]

        return meta

    def _extract_h1(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract headline from primary <h1> tag."""
        h1 = soup.find("h1")
        return h1.get_text().strip() if h1 else None

    def _extract_body_from_dom(self, soup: BeautifulSoup) -> str:
        """Extract substantive article body text from HTML DOM.

        Supports both standard <p> tags and CMS-specific block-level <div> tags
        (e.g. Navbharat Times <div class="Normal">) while decomposing captions,
        image credits, related-content boxes, and advertisements.
        Guarantees no double-counting by extracting only leaf blocks.
        """
        # Create a working copy so caller soup is not mutated
        body_soup = BeautifulSoup(str(soup), "html.parser")

        # Strip standard non-content chrome
        for tag in body_soup.find_all(["script", "style", "noscript", "iframe", "header", "footer", "nav", "aside", "svg"]):
            tag.decompose()

        # Target clearly identified author / byline elements specifically
        author_patterns = re.compile(r"\b(?:byline|author-name|article-byline|author-info|reporter-name|story-byline)\b", re.IGNORECASE)
        for el in body_soup.find_all(class_=author_patterns):
            el.decompose()
        for el in body_soup.find_all(id=re.compile(r"\b(?:byline|article-byline)\b", re.IGNORECASE)):
            el.decompose()

        # Decompose captions, credits, related content, advertisements, and social widgets
        boilerplate_patterns = re.compile(
            r"\b(?:caption|image-caption|photo-caption|photo-credit|credit|related-articles|related-news|read-also|recommended|ad-container|advertisement|ads-box|social-share|share-icons|newsletter|tags-list)\b",
            re.IGNORECASE,
        )
        for el in body_soup.find_all(class_=boilerplate_patterns):
            el.decompose()
        for el in body_soup.find_all(id=boilerplate_patterns):
            el.decompose()

        # Find main article container candidate
        article_container = (
            body_soup.find("article")
            or body_soup.find(attrs={"itemprop": "articleBody"})
            or body_soup.find(class_=re.compile(r"\b(?:article-body|story-content|story-detail|article_content|main-story)\b", re.IGNORECASE))
            or body_soup.find("main")
            or body_soup.body
        )

        if not article_container:
            return ""

        # First, attempt extraction from standard <p> tags
        paragraphs: List[str] = []
        for p in article_container.find_all("p"):
            text = p.get_text().strip()
            if text.startswith("{") and "}" in text[:100]:
                continue
            if len(text) >= 20 and not re.search(r"^(download app|share this|subscribe|copyright|also read)", text, re.IGNORECASE):
                paragraphs.append(text)

        p_body = "\n\n".join(paragraphs)
        if len(p_body) >= 80:
            return p_body

        # Fallback for div-based CMSs (e.g. Navbharat Times <div class="Normal">):
        # Extract from leaf block-level <div> elements that do not contain child block elements
        div_paragraphs: List[str] = []
        for div in article_container.find_all("div"):
            if not div.find(["div", "p"]):
                text = div.get_text().strip()
                if text.startswith("{") and "}" in text[:100]:
                    continue
                if len(text) >= 25 and not re.search(r"^(download app|share this|subscribe|copyright|also read)", text, re.IGNORECASE):
                    div_paragraphs.append(text)

        # Merge blocks without double-counting
        combined_blocks: List[str] = []
        seen_texts: Set[str] = set()
        for block in paragraphs + div_paragraphs:
            if block not in seen_texts:
                seen_texts.add(block)
                combined_blocks.append(block)

        return "\n\n".join(combined_blocks)

    def _extract_body_links(self, soup: BeautifulSoup, current_source: str) -> List[str]:
        """Collect in-body hyperlinks to other articles (for PageRank graph)."""
        links: List[str] = []
        article_el = soup.find("article") or soup.find(class_=re.compile(r"\b(?:article-body|story-content)\b"))

        if not article_el:
            return []

        for a in article_el.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith(("#", "javascript:", "mailto:")):
                continue

            # Look for article ID pattern in linked URL
            numeric_match = re.search(r"[-_/](\d{5,12})(?:\.html|\.cms|/|$)", href)
            if numeric_match:
                links.append(f"{current_source}_{numeric_match.group(1)}")

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for lk in links:
            if lk not in seen:
                seen.add(lk)
                deduped.append(lk)
        return deduped

