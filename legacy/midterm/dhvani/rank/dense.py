"""Dense re-ranking with a multilingual embedding model (beyond the syllabus).

lnc.ltc and BM25 only match words. A dense model maps the query and each
article to vectors where meaning is close even when the words differ, so it
can pick up a relevant article that uses other words for the same thing.

It's used as a re-ranker, not a retriever: the sparse ranker finds the top
results from the index as usual, and only those get re-scored. The final
score mixes the two,

    score = (1 - alpha) x first-stage score + alpha x dense cosine

with both scaled to [0, 1] over the candidates first. e5's cosines all sit in
a narrow band (about 0.75 to 0.85), so without scaling the first stage would
always win. The IR ranking still decides what's a candidate and carries half
the weight by default.

Model: intfloat/multilingual-e5-small (Hindi included). e5 expects
"query: " and "passage: " prefixes. It handles Hindi and English well but not
Roman-script Hindi, so the query text sent to it is the raw query plus the
best Devanagari spelling of each word from Viraja's layer or the
translation, e.g. "kal ka mausam" becomes "kal ka mausam कल का मौसम".

Article vectors depend only on the text, so they're computed once and saved
in data/dense/ (not in git). sentence-transformers is optional; without it
dense_available() is False and the app and CLI hide the option.
"""

import json
from pathlib import Path

MODEL_NAME = "intfloat/multilingual-e5-small"
CACHE_DIR = Path("data/dense")
PASSAGE_CHARS = 1000          # headline + start of the body; e5 reads up to 512 tokens
DEFAULT_ALPHA = 0.5
DEFAULT_DEPTH = 50
MIN_PHONETIC_FOR_DENSE = 0.4   # only confident spellings; weaker ones are often unrelated words


def dense_available():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _is_devanagari(term):
    return any("ऀ" <= ch <= "ॿ" for ch in term)


def dense_query_text(query):
    """Raw query plus the best Devanagari spelling of each word, if it has one.

    Translations are always used. Phonetic spellings only when Viraja's layer
    is fairly sure (weight >= 0.4): English words like "farmers" also get
    phonetic guesses (हार्मोन्स), and those would only confuse the model.
    """
    parts = [query["raw"]]
    for token in query["tokens"]:
        if _is_devanagari(token["surface"]):
            continue
        options = [(w, t) for t, w, s in token["expansions"]
                   if _is_devanagari(t) and (s == "xling" or (s == "phonetic" and w >= MIN_PHONETIC_FOR_DENSE))]
        if options:
            parts.append(max(options)[1])
    return " ".join(parts)


def passage_text(article):
    return (article.get("headline", "") + "\n" + article.get("body", ""))[:PASSAGE_CHARS]


class SentenceEncoder:
    """Wraps the e5 model: encode(texts) -> list of unit-length vectors."""

    def __init__(self, model_name=MODEL_NAME):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return [list(map(float, v)) for v in self.model.encode(texts, normalize_embeddings=True, batch_size=32)]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


class DenseIndex:
    """Article vectors for every article in the index, cached on disk.

    encoder: anything with encode(list_of_texts) -> list of unit vectors.
    """

    def __init__(self, index, encoder, cache_dir=CACHE_DIR):
        self.encoder = encoder
        self.vectors = {}
        name = getattr(encoder, "name", "custom").replace("/", "__")
        cache = Path(cache_dir) / f"{name}.json" if cache_dir else None
        if cache and cache.exists():
            self.vectors = json.loads(cache.read_text(encoding="utf-8"))
        articles = getattr(index, "articles", {})
        missing = [d for d in index.meta if d not in self.vectors]
        if missing:
            texts = ["passage: " + passage_text(articles.get(d, {})) for d in missing]
            for doc_id, vec in zip(missing, encoder.encode(texts)):
                self.vectors[doc_id] = vec
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(self.vectors), encoding="utf-8")

    def similarity(self, query, doc_ids):
        """{doc_id: cosine(query, article)} for the given articles."""
        qvec = self.encoder.encode(["query: " + dense_query_text(query)])[0]
        return {d: _dot(qvec, self.vectors[d]) for d in doc_ids if d in self.vectors}


def _scaled(values):
    """Min-max scale a {doc_id: value} dict to [0, 1]."""
    lo, hi = min(values.values()), max(values.values())
    return {d: (v - lo) / (hi - lo) if hi > lo else 1.0 for d, v in values.items()}


def dense_rerank(results, query, dense, alpha=DEFAULT_ALPHA, depth=DEFAULT_DEPTH):
    """Re-score the top `depth` results with the dense model.

    Like the kal boost, the order only changes within a query-parser stage, so
    an exact-phrase match is never pushed below a looser one. Results past
    `depth` keep their place at the end.
    """
    head, tail = results[:depth], results[depth:]
    if not head:
        return results
    sims = dense.similarity(query, [d for d, _, _ in head])
    first = _scaled({d: s for d, s, _ in head})
    dense_scaled = _scaled({d: sims.get(d, 0.0) for d, _, _ in head})
    out = []
    for doc_id, score, explain in head:
        sim = sims.get(doc_id, 0.0)
        new = (1 - alpha) * first[doc_id] + alpha * dense_scaled[doc_id]
        explain = dict(explain) if "terms" in explain else {"terms": dict(explain)}
        explain["dense"] = {"cosine": sim, "first_stage": score, "alpha": alpha, "score": new}
        out.append((doc_id, new, explain))
    stage_order = {}
    for i, (_d, _s, e) in enumerate(out):
        stage_order.setdefault(e.get("stage"), i)
    out.sort(key=lambda item: (stage_order[item[2].get("stage")], -item[1], item[0]))
    return out + tail
