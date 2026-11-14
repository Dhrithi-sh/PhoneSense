"""A small in-memory index over a hand-written sample corpus.

It follows the Index interface in documentation/formats.md, so the ranking
code can be built and tested before the real index is ready. Swap it for
Dhrithi's index by changing the import; nothing else should need to change.

The articles below are short made-up examples, not copied from any paper.
"""

import math
from collections import Counter, defaultdict

import regex

ZONES = ("headline", "body")

# Same token pattern the real tokenizer uses: letters, combining marks
# (Devanagari vowel signs) and digits. Plain \w would split मौसम apart.
TOKEN = regex.compile(r"[\p{L}\p{M}\p{Nd}]+")

SAMPLE_ARTICLES = [
    {"doc_id": "jagran_1001", "source": "jagran", "section": "weather", "state": "delhi", "city": "delhi",
     "date": "2026-10-05T08:10:00+05:30",
     "headline": "दिल्ली में कल बारिश का अलर्ट",
     "body": "मौसम विभाग ने कल दिल्ली में तेज बारिश का अलर्ट जारी किया है। बारिश के साथ तेज हवा चल सकती है।",
     "links": ["nbt_2001"], "dup_of": None},
    {"doc_id": "nbt_2001", "source": "nbt", "section": "weather", "state": "delhi", "city": "delhi",
     "date": "2026-10-05T09:30:00+05:30",
     "headline": "मौसम अपडेट: दिल्ली में बादल छाए रहेंगे",
     "body": "दिल्ली का मौसम अगले दो दिन बदला रहेगा। मौसम विभाग के अनुसार हल्की बारिश हो सकती है।",
     "links": [], "dup_of": None},
    {"doc_id": "amarujala_3001", "source": "amarujala", "section": "weather", "state": "uttar-pradesh", "city": "lucknow",
     "date": "2026-10-04T18:00:00+05:30",
     "headline": "लखनऊ में गर्मी से राहत, बारिश की संभावना",
     "body": "लखनऊ में आज शाम बारिश की संभावना है। मौसम सुहाना रहेगा और तापमान गिरेगा।",
     "links": ["jagran_1001"], "dup_of": None},
    {"doc_id": "livehindustan_4001", "source": "livehindustan", "section": "weather", "state": "bihar", "city": "patna",
     "date": "2026-10-03T07:45:00+05:30",
     "headline": "पटना में भारी बारिश, कई इलाकों में जलभराव",
     "body": "पटना में रात भर हुई भारी बारिश से कई इलाकों में पानी भर गया। लोगों को परेशानी हुई।",
     "links": [], "dup_of": None},
    {"doc_id": "aajtak_5001", "source": "aajtak", "section": "weather", "state": "maharashtra", "city": "mumbai",
     "date": "2026-10-05T11:00:00+05:30",
     "headline": "मुंबई में मानसून की विदाई, मौसम साफ",
     "body": "मुंबई में मानसून अब लौट रहा है। अगले हफ्ते मौसम साफ रहने की उम्मीद है।",
     "links": ["nbt_2001"], "dup_of": None},
    {"doc_id": "jagran_1002", "source": "jagran", "section": "sports", "state": "", "city": "",
     "date": "2026-10-05T21:00:00+05:30",
     "headline": "भारत ने ऑस्ट्रेलिया को हराया, कोहली का शतक",
     "body": "क्रिकेट मैच में भारत ने ऑस्ट्रेलिया को छह विकेट से हराया। विराट कोहली ने शानदार शतक लगाया।",
     "links": ["aajtak_5002"], "dup_of": None},
    {"doc_id": "aajtak_5002", "source": "aajtak", "section": "sports", "state": "", "city": "",
     "date": "2026-10-05T21:30:00+05:30",
     "headline": "कोहली के शतक से भारत की जीत",
     "body": "विराट कोहली के शतक की मदद से भारत ने क्रिकेट मैच जीत लिया। टीम ने सीरीज में बढ़त बना ली।",
     "links": [], "dup_of": None},
    {"doc_id": "nbt_2002", "source": "nbt", "section": "sports", "state": "", "city": "",
     "date": "2026-10-04T15:00:00+05:30",
     "headline": "महिला क्रिकेट टीम का एलान",
     "body": "विश्व कप के लिए महिला क्रिकेट टीम का एलान हो गया। कप्तान ने टीम पर भरोसा जताया।",
     "links": [], "dup_of": None},
    {"doc_id": "amarujala_3002", "source": "amarujala", "section": "politics", "state": "uttar-pradesh", "city": "lucknow",
     "date": "2026-10-05T10:00:00+05:30",
     "headline": "मुख्यमंत्री ने नई सड़क परियोजना का उद्घाटन किया",
     "body": "लखनऊ में मुख्यमंत्री ने नई सड़क परियोजना का उद्घाटन किया। परियोजना से शहर में जाम कम होगा।",
     "links": [], "dup_of": None},
    {"doc_id": "livehindustan_4002", "source": "livehindustan", "section": "politics", "state": "bihar", "city": "patna",
     "date": "2026-10-04T12:00:00+05:30",
     "headline": "बिहार चुनाव की तैयारी तेज",
     "body": "बिहार में चुनाव से पहले सभी दल तैयारी में जुटे हैं। चुनाव आयोग ने तारीखों पर बैठक की।",
     "links": ["livehindustan_4003"], "dup_of": None},
    {"doc_id": "livehindustan_4003", "source": "livehindustan", "section": "politics", "state": "bihar", "city": "patna",
     "date": "2026-10-05T13:00:00+05:30",
     "headline": "चुनाव आयोग ने बिहार चुनाव की तारीखें घोषित कीं",
     "body": "चुनाव आयोग ने बिहार विधानसभा चुनाव की तारीखें घोषित कर दीं। मतदान तीन चरणों में होगा।",
     "links": [], "dup_of": None},
    {"doc_id": "jagran_1003", "source": "jagran", "section": "politics", "state": "delhi", "city": "delhi",
     "date": "2026-10-03T16:00:00+05:30",
     "headline": "संसद में बजट पर बहस",
     "body": "संसद में आज बजट पर लंबी बहस हुई। विपक्ष ने महंगाई का मुद्दा उठाया।",
     "links": [], "dup_of": None},
    {"doc_id": "nbt_2003", "source": "nbt", "section": "business", "state": "", "city": "",
     "date": "2026-10-05T17:00:00+05:30",
     "headline": "शेयर बाजार में तेजी, सेंसेक्स चढ़ा",
     "body": "शेयर बाजार में आज तेजी रही और सेंसेक्स पांच सौ अंक चढ़ा। निवेशकों को फायदा हुआ।",
     "links": [], "dup_of": None},
    {"doc_id": "aajtak_5003", "source": "aajtak", "section": "business", "state": "", "city": "",
     "date": "2026-10-04T10:00:00+05:30",
     "headline": "पेट्रोल डीजल के दाम स्थिर",
     "body": "आज पेट्रोल और डीजल के दाम में कोई बदलाव नहीं हुआ। महंगाई पर नजर बनी हुई है।",
     "links": [], "dup_of": None},
    {"doc_id": "amarujala_3003", "source": "amarujala", "section": "crime", "state": "uttar-pradesh", "city": "kanpur",
     "date": "2026-10-05T07:00:00+05:30",
     "headline": "कानपुर में चोरी का खुलासा, दो आरोपी गिरफ्तार",
     "body": "कानपुर पुलिस ने चोरी के मामले का खुलासा कर दो आरोपियों को गिरफ्तार किया।",
     "links": [], "dup_of": None},
    {"doc_id": "jagran_1004", "source": "jagran", "section": "education", "state": "delhi", "city": "delhi",
     "date": "2026-10-04T09:00:00+05:30",
     "headline": "दिल्ली के स्कूलों में छुट्टी, बारिश के कारण फैसला",
     "body": "भारी बारिश के अलर्ट के कारण दिल्ली के स्कूलों में कल छुट्टी रहेगी। शिक्षा विभाग ने आदेश जारी किया।",
     "links": ["jagran_1001"], "dup_of": None},
    {"doc_id": "nbt_2004", "source": "nbt", "section": "education", "state": "", "city": "",
     "date": "2026-10-03T11:00:00+05:30",
     "headline": "बोर्ड परीक्षा की तारीखें जारी",
     "body": "बोर्ड ने परीक्षा की तारीखें जारी कर दीं। छात्र अब तैयारी शुरू कर सकते हैं।",
     "links": [], "dup_of": None},
    {"doc_id": "livehindustan_4004", "source": "livehindustan", "section": "entertainment", "state": "", "city": "",
     "date": "2026-10-05T19:00:00+05:30",
     "headline": "नई फिल्म ने पहले दिन रिकॉर्ड कमाई की",
     "body": "शुक्रवार को रिलीज हुई नई फिल्म ने पहले दिन रिकॉर्ड कमाई की। दर्शकों को फिल्म पसंद आई।",
     "links": [], "dup_of": None},
    # A wire story carried by two papers, so duplicate collapsing has something to work on.
    {"doc_id": "aajtak_5004", "source": "aajtak", "section": "national", "state": "", "city": "",
     "date": "2026-10-05T06:00:00+05:30",
     "headline": "प्रधानमंत्री आज करेंगे नई रेल लाइन का उद्घाटन",
     "body": "प्रधानमंत्री आज नई रेल लाइन का उद्घाटन करेंगे। इससे यात्रा का समय कम होगा।",
     "links": [], "dup_of": None},
    {"doc_id": "amarujala_3004", "source": "amarujala", "section": "national", "state": "", "city": "",
     "date": "2026-10-05T06:40:00+05:30",
     "headline": "प्रधानमंत्री करेंगे नई रेल लाइन का उद्घाटन",
     "body": "प्रधानमंत्री आज नई रेल लाइन का उद्घाटन करेंगे। इससे यात्रा का समय कम होगा।",
     "links": ["aajtak_5004"], "dup_of": "aajtak_5004"},
]


def tokenize(text):
    """Split text into lowercase tokens. A stand-in for Dhrithi's analyzer."""
    return [t.lower() for t in TOKEN.findall(text)]


class SampleIndex:
    """Positional inverted index with headline and body zones.

    postings[zone][term] -> list of (doc_id, tf, [positions]), sorted by doc_id.
    """

    def __init__(self, articles=None):
        articles = SAMPLE_ARTICLES if articles is None else articles
        self.postings_by_zone = {zone: defaultdict(list) for zone in ZONES}
        self.meta = {}
        self.doc_norm = {}
        # Full text for display only (headline, snippets). The real system
        # reads this from the article file instead.
        self.articles = {}
        doc_freq = Counter()

        for art in sorted(articles, key=lambda a: a["doc_id"]):
            doc_id = art["doc_id"]
            self.meta[doc_id] = {k: art.get(k) for k in ("source", "section", "state", "city", "date", "dup_of", "links")}
            self.articles[doc_id] = {"headline": art["headline"], "body": art["body"]}
            whole_doc_tf = Counter()
            for zone in ZONES:
                positions = defaultdict(list)
                for pos, term in enumerate(tokenize(art[zone])):
                    positions[term].append(pos)
                for term, pos_list in positions.items():
                    self.postings_by_zone[zone][term].append((doc_id, len(pos_list), pos_list))
                    whole_doc_tf[term] += len(pos_list)
            doc_freq.update(whole_doc_tf.keys())
            # lnc length: log tf weights, no idf, then cosine normalisation.
            self.doc_norm[doc_id] = math.sqrt(sum((1 + math.log10(tf)) ** 2 for tf in whole_doc_tf.values()))

        self._df = dict(doc_freq)
        self.N = len(self.meta)
        self.vocab = sorted(self._df)

    def postings(self, term, zone):
        if zone not in ZONES:
            raise ValueError(f"zone must be one of {ZONES}, got {zone!r}")
        return self.postings_by_zone[zone].get(term, [])

    def df(self, term):
        return self._df.get(term, 0)

    @classmethod
    def load(cls, mode="none"):
        """Matches the real Index.load(mode). The sample has no stemming, so mode is ignored."""
        return cls()
