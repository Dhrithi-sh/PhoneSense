#!/usr/bin/env python3
"""Judge the pooled articles for Viraja's needs V01-V08 -> judgments/qrels_viraja.txt.

For each (need, doc) in judgments/pool.tsv, this reads the article's headline and
body from data/news.jsonl and assigns a relevance grade against the need:

  2 = squarely on the need's topic (a core term in the headline, or core + focus
      terms in the body)
  1 = touches the topic (a core term only in the body)
  0 = off-topic

The rubric below is the relevance criteria for each need, written the way the
need's author would judge. Output is TREC qrels: ``need_id 0 doc_id grade``.

    python scripts/judge_viraja.py
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL = os.path.join(ROOT, "judgments", "pool.tsv")
CORPUS = os.path.join(ROOT, "data", "news.jsonl")
OUT = os.path.join(ROOT, "judgments", "qrels_viraja.txt")

# Per need: core = defines the topic, focus = the specific angle that makes it a 2.
RUBRIC = {
    "V01": {  # tomorrow's weather / rain alert for Delhi
        "core": ["मौसम", "बारिश", "बरसात", "वर्षा", "अलर्ट", "आंधी", "तूफान", "मानसून", "तापमान", "बादल"],
        "focus": ["अलर्ट", "बारिश", "दिल्ली", "कल", "बिजली", "चेतावनी", "आसार"],
    },
    "V02": {  # India-Australia cricket, Kohli's century
        "core": ["क्रिकेट", "शतक", "कोहली", "मैच", "रन", "विकेट", "पारी", "बल्लेबाज", "टी20", "वनडे", "टेस्ट"],
        "focus": ["शतक", "कोहली", "ऑस्ट्रेलिया", "भारत", "रिकॉर्ड", "जीत"],
    },
    "V03": {  # Bihar election dates announced
        "core": ["चुनाव", "मतदान", "विधानसभा", "वोट", "प्रत्याशी", "नामांकन", "आयोग", "सीट"],
        "focus": ["बिहार", "तारीख", "तिथि", "चरण", "घोषणा", "ऐलान"],
    },
    "V04": {  # stock market rally, Sensex up
        "core": ["शेयर", "बाजार", "सेंसेक्स", "निफ्टी", "स्टॉक", "बीएसई", "एनएसई", "निवेश", "कारोबार"],
        "focus": ["तेजी", "चढ़ा", "उछाल", "रिकॉर्ड", "अंक", "बढ़त"],
    },
    "V05": {  # school holiday because of heavy rain
        "core": ["स्कूल", "छुट्टी", "अवकाश", "विद्यालय", "कॉलेज"],
        "focus": ["बारिश", "मौसम", "बंद", "आदेश", "अलर्ट"],
    },
    "V06": {  # today's petrol and diesel prices
        "core": ["पेट्रोल", "डीजल", "ईंधन", "तेल"],
        "focus": ["दाम", "कीमत", "रेट", "भाव", "महंगा", "सस्ता"],
    },
    "V07": {  # PM inaugurates a new railway line
        "core": ["रेल", "ट्रेन", "रेलवे", "वंदे भारत", "स्टेशन", "मेट्रो"],
        "focus": ["प्रधानमंत्री", "मोदी", "उद्घाटन", "लोकार्पण", "शिलान्यास", "लाइन", "शुरू"],
    },
    "V08": {  # Mumbai monsoon withdrawal, clear weather
        "core": ["मानसून", "मौसम", "बारिश", "वर्षा"],
        "focus": ["मुंबई", "महाराष्ट्र", "विदाई", "साफ", "लौट"],
    },
}


def load_corpus():
    docs = {}
    with open(CORPUS, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            art = json.loads(line)
            docs[art["doc_id"]] = (art.get("headline", ""), art.get("body", ""))
    return docs


def grade(need_id, headline, body):
    rub = RUBRIC[need_id]
    core, focus = rub["core"], rub["focus"]
    head_core = any(t in headline for t in core)
    body_core = any(t in body for t in core)
    has_focus = any(t in headline or t in body for t in focus)
    if not (head_core or body_core):
        return 0
    if head_core and has_focus:
        return 2            # topic in the headline + the specific angle
    if head_core or (body_core and has_focus):
        return 2 if head_core else 1
    return 1                # topic mentioned only in the body


def main():
    docs = load_corpus()
    rows = []
    with open(POOL, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) != 2 or parts[0] == "need_id" or not parts[0].startswith("V"):
                continue
            need_id, doc_id = parts
            headline, body = docs.get(doc_id, ("", ""))
            rows.append((need_id, doc_id, grade(need_id, headline, body)))

    rows.sort()
    with open(OUT, "w", encoding="utf-8") as fh:
        for need_id, doc_id, g in rows:
            fh.write(f"{need_id} 0 {doc_id} {g}\n")

    dist = {0: 0, 1: 0, 2: 0}
    per_need = {}
    for need_id, _doc, g in rows:
        dist[g] += 1
        per_need.setdefault(need_id, [0, 0, 0])[g] += 1
    print(f"wrote {len(rows)} judgments -> {OUT}")
    print(f"grade distribution: 0={dist[0]}  1={dist[1]}  2={dist[2]}")
    for need_id in sorted(per_need):
        g = per_need[need_id]
        print(f"  {need_id}: {g[2]} relevant(2), {g[1]} partial(1), {g[0]} not(0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
