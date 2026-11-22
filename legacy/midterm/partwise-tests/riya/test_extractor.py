"""Unit tests for ArticleExtractor and schema validation (dhvani/crawl/extractor.py)."""

import pytest

from dhvani.crawl.extractor import ArticleExtractor, validate_article_schema, parse_ist_date, extract_location_from_url


SAMPLE_JSONLD_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>नागपुर घटना - दैनिक जागरण</title>
  <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@type": "NewsArticle",
    "headline": "नागपुर में नाबालिग लड़की से छेड़छाड़, आरोपी की पिटाई",
    "articleBody": "नागपुर में एक दिल दहला देने वाला मामला सामने आया है जहां भीड़ ने आरोपी को पकड़कर पुलिस के हवाले करने से पहले जमकर पिटाई कर दी। घटना की सूचना मिलते ही पुलिस बल मौके पर पहुंच गया और स्थिति को नियंत्रित किया।",
    "datePublished": "2026-10-06T10:00:00Z",
    "articleSection": "crime",
    "keywords": ["नागपुर", "अपराध", "पुलिस"]
  }
  </script>
</head>
<body>
  <h1>नागपुर में नाबालिग लड़की से छेड़छाड़</h1>
  <p>वेबसाइट फुटर कंटेंट</p>
</body>
</html>
"""

SAMPLE_META_FALLBACK_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>लखनऊ में कल भारी बारिश का अलर्ट - अमर उजाला</title>
  <meta property="og:title" content="लखनऊ में कल भारी बारिश का अलर्ट" />
  <meta property="og:description" content="यह संक्षिप्त विवरण है जो बॉडी नहीं बनना चाहिए।" />
  <meta property="article:published_time" content="2026-10-06T13:21:43+05:30" />
  <meta property="article:section" content="weather" />
  <meta name="keywords" content="मौसम, बारिश, लखनऊ" />
</head>
<body>
  <article class="story-content">
    <div class="byline">विशेष संवाददाता, अमर उजाला</div>
    <p class="authority-guideline">सरकारी मौसम प्राधिकरण के निर्देशानुसार चेतावनी जारी की गई है।</p>
    <p>लखनऊ और आसपास के जिलों में आगामी 24 घंटों के दौरान मूसलाधार बारिश होने की प्रबल संभावना है।</p>
    <p>मौसम विभाग ने नागरिकों से अनावश्यक रूप से घरों से बाहर न निकलने की अपील की है और सतर्क रहने को कहा है।</p>
    <div class="social-share">फेसबुक पर शेयर करें</div>
  </article>
</body>
</html>
"""

SAMPLE_DOM_ONLY_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>बिहार विधानसभा चुनाव को लेकर हलचल तेज</title>
</head>
<body>
  <h1>बिहार विधानसभा चुनाव को लेकर हलचल तेज</h1>
  <div class="article-body">
    <p class="reporter-name">पटना ब्यूरो</p>
    <p>बिहार में आगामी चुनावों को लेकर राजनीतिक दलों ने अपनी रणनीति तेज कर दी है और बैठकों का दौर जारी है।</p>
    <p>विभिन्न दलों के वरिष्ठ नेता जमीनी स्तर पर कार्यकर्ताओं से संवाद स्थापित कर रहे हैं और प्रचार अभियान तैयार कर रहे हैं।</p>
  </div>
</body>
</html>
"""


def test_tier1_jsonld_extraction():
    """Verify that Schema.org JSON-LD is prioritized for core content and dates."""
    extractor = ArticleExtractor()
    url = "https://www.jagran.com/crime/national-nagpur-case-40396762.html"
    res = extractor.extract(SAMPLE_JSONLD_HTML, url=url, source_slug="jagran")

    assert res is not None
    assert res["doc_id"] == "jagran_40396762"
    assert res["source"] == "jagran"
    assert res["section"] == "crime"
    assert res["headline"] == "नागपुर में नाबालिग लड़की से छेड़छाड़, आरोपी की पिटाई"
    assert "नागपुर में एक दिल दहला देने वाला मामला" in res["body"]
    assert res["keywords"] == ["नागपुर", "अपराध", "पुलिस"]

    # UTC 10:00:00Z must be strictly converted to IST 15:30:00+05:30
    assert res["date"] == "2026-10-06T15:30:00+05:30"
    assert validate_article_schema(res) is True


def test_tier2_meta_fallback_without_og_description_as_body():
    """Verify that meta tags recover missing metadata, but og:description is NOT used as body.

    Body text must fall through to Tier 3 HTML DOM paragraphs.
    """
    extractor = ArticleExtractor()
    url = "https://www.amarujala.com/uttar-pradesh/lucknow/weather-update-8888.html"
    res = extractor.extract(SAMPLE_META_FALLBACK_HTML, url=url, source_slug="amarujala")

    assert res is not None
    assert res["headline"] == "लखनऊ में कल भारी बारिश का अलर्ट"
    assert res["section"] == "weather"
    assert res["state"] == "uttar-pradesh"
    assert res["city"] == "lucknow"
    assert res["keywords"] == ["मौसम", "बारिश", "लखनऊ"]

    # Must NOT contain og:description in body
    assert "यह संक्षिप्त विवरण है जो बॉडी नहीं बनना चाहिए" not in res["body"]

    # Body must be recovered from <article> <p> tags
    assert "लखनऊ और आसपास के जिलों में आगामी 24 घंटों के दौरान" in res["body"]
    assert "मौसम विभाग ने नागरिकों से अनावश्यक रूप से" in res["body"]
    assert validate_article_schema(res) is True


def test_targeted_author_removal_and_no_author_in_record():
    """Verify clearly identified author/byline tags are removed while non-author elements are preserved.

    Guarantees no author fields in schema.
    """
    extractor = ArticleExtractor()
    url = "https://www.amarujala.com/uttar-pradesh/lucknow/news-12345.html"
    res = extractor.extract(SAMPLE_META_FALLBACK_HTML, url=url, source_slug="amarujala")

    assert res is not None
    # Author/byline text must NOT appear in body
    assert "विशेष संवाददाता" not in res["body"]
    assert "रमेश कुमार" not in res["body"]

    # Preserved authority content (avoided blind wildcard deletion)
    assert "सरकारी मौसम प्राधिकरण" in res["body"]

    # Record must contain absolutely NO author/byline keys
    forbidden = {"author", "authors", "byline", "creator", "editor", "reporter"}
    for f in forbidden:
        assert f not in res


def test_conservative_location_extraction():
    """Verify state/city are extracted when clear in URL, and set to null when unidentifiable."""
    # Clearly identifiable: UP + Lucknow
    st1, ct1 = extract_location_from_url("https://www.jagran.com/uttar-pradesh/lucknow-news-123.html")
    assert st1 == "uttar-pradesh"
    assert ct1 == "lucknow"

    # Clearly identifiable: Bihar + Patna
    st2, ct2 = extract_location_from_url("https://www.livehindustan.com/bihar/patna/politics.html")
    assert st2 == "bihar"
    assert ct2 == "patna"

    # Unidentifiable geographic route: must be (None, None) rather than guessing
    st3, ct3 = extract_location_from_url("https://www.aajtak.in/technology/news/new-phone-launched-99.html")
    assert st3 is None
    assert ct3 is None

    extractor = ArticleExtractor()
    res = extractor.extract(SAMPLE_DOM_ONLY_HTML, url="https://www.aajtak.in/national/news-555.html", source_slug="aajtak")
    assert res is not None
    assert res["state"] is None
    assert res["city"] is None


def test_strict_ist_date_parsing():
    """Verify all date formats parse into ISO-8601 with +05:30 offset."""
    # UTC with Z
    assert parse_ist_date("2026-10-06T10:00:00Z") == "2026-10-06T15:30:00+05:30"

    # Already IST offset
    assert parse_ist_date("2026-10-06T13:21:43+05:30") == "2026-10-06T13:21:43+05:30"

    # Naive timestamp (assumed IST)
    assert parse_ist_date("2026-10-06 14:00:00") == "2026-10-06T14:00:00+05:30"

    # Common portal string
    assert parse_ist_date("06 Oct 2026, 02:00 PM") == "2026-10-06T14:00:00+05:30"

    # Invalid string returns None
    assert parse_ist_date("invalid-date-string") is None


def test_wire_agency_detection():
    """Verify syndicated wire stories set agency_flag to True."""
    extractor = ArticleExtractor()
    wire_html = """<!DOCTYPE html><html><head><title>Test PTI</title></head>
    <body><h1>पीटीआई रिपोर्ट</h1>
    <p>नई दिल्ली, भाषा। केंद्रीय मंत्रिमंडल ने आज कई अहम प्रस्तावों को मंजूरी दी। पीटीआई के अनुसार यह फैसला महत्वपूर्ण है।</p>
    </body></html>"""

    res = extractor.extract(wire_html, url="https://www.jagran.com/national-1111.html", source_slug="jagran")
    assert res is not None
    assert res["agency_flag"] is True

    # Non-wire story
    normal_html = """<!DOCTYPE html><html><head><title>Test Local</title></head>
    <body><h1>स्थानीय समाचार</h1>
    <p>शहर के मुख्य चौराहे पर आज यातायात व्यवस्था सुचारू रूप से संचालित रही और कोई जाम नहीं लगा।</p>
    </body></html>"""

    res2 = extractor.extract(normal_html, url="https://www.jagran.com/local-2222.html", source_slug="jagran")
    assert res2 is not None
    assert res2["agency_flag"] is False


def test_contract_schema_validation():
    """Verify strict validation against documentation/formats.md."""
    valid_record = {
        "doc_id": "jagran_23456789",
        "url": "https://www.jagran.com/news/weather.html",
        "source": "jagran",
        "section": "weather",
        "state": "uttar-pradesh",
        "city": "lucknow",
        "date": "2026-10-06T13:21:43+05:30",
        "headline": "कल का मौसम",
        "body": "लखनऊ में भारी बारिश की चेतावनी जारी की गई है।",
        "keywords": ["मौसम", "बारिश"],
        "agency_flag": False,
        "content_hash": "a9f3b8214567890abcdef12345678901",
        "dup_of": None,
        "links": ["jagran_23450001"],
    }
    assert validate_article_schema(valid_record) is True

    # Missing required field
    invalid_record = dict(valid_record)
    del invalid_record["content_hash"]
    assert validate_article_schema(invalid_record) is False

    # Invalid date timezone (missing +05:30)
    bad_date = dict(valid_record, date="2026-10-06T13:21:43Z")
    assert validate_article_schema(bad_date) is False

    # Illegal author field inclusion
    author_polluted = dict(valid_record, author="Ramesh Kumar")
    assert validate_article_schema(author_polluted) is False


def test_div_block_extraction_without_double_counting_and_captions():
    """Verify div-based CMS extraction extracts leaf blocks without parent double-counting

    and properly filters out image captions, credits, and related-news blocks.
    """
    html = """<!DOCTYPE html>
    <html>
    <head><title>NBT News Sample</title></head>
    <body>
      <h1>दिल्ली-एनसीआर में प्रदूषण का स्तर बढ़ा</h1>
      <article class="article-body">
        <div class="photo-container">
          <img src="delhi.jpg" />
          <div class="photo-caption">फाइल फोटो: दिल्ली की हवा में धुंध की परत</div>
          <div class="photo-credit">फोटो साभार: पीटीआई</div>
        </div>
        <div class="outer-wrapper">
          <div class="Normal">दिल्ली और राष्ट्रीय राजधानी क्षेत्र में हवा की गुणवत्ता एक बार फिर बेहद खराब श्रेणी में दर्ज की गई है।</div>
          <div class="Normal">केंद्रीय प्रदूषण नियंत्रण बोर्ड के अनुसार वायु गुणवत्ता सूचकांक 350 के पार पहुंच गया है।</div>
        </div>
        <div class="related-news">
          <div class="related-title">यह भी पढ़ें: ग्रैप-3 की पाबंदियां लागू</div>
        </div>
      </article>
    </body>
    </html>"""

    extractor = ArticleExtractor()
    res = extractor.extract(html, url="https://navbharattimes.indiatimes.com/delhi/pollution-update-777.html", source_slug="nbt")

    assert res is not None
    # Captions and related news must be filtered out
    assert "फाइल फोटो" not in res["body"]
    assert "फोटो साभार" not in res["body"]
    assert "यह भी पढ़ें" not in res["body"]

    # Substantive div text must be present
    assert "हवा की गुणवत्ता एक बार फिर बेहद खराब श्रेणी में दर्ज" in res["body"]
    assert "वायु गुणवत्ता सूचकांक 350 के पार पहुंच गया है" in res["body"]

    # Verify no double-counting: sentence must appear EXACTLY ONCE
    count = res["body"].count("हवा की गुणवत्ता एक बार फिर बेहद खराब श्रेणी में दर्ज की गई है।")
    assert count == 1, f"Expected 1 occurrence, found {count} (double-counting occurred!)"

