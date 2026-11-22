"""Search engine behind the demo UI (owner: B, T24). No Streamlit imports: testable on its own.

The UI (``app.py``) only draws; everything that loads data, builds retrievers and turns a
ranking into displayable results lives here.

Systems (toggles in the sidebar), in priority order:

* ``tfidf``   TF-IDF lnc.ltc (always available),
* ``bm25``    BM25 with the k1 and b tuned on dev (``phonesense.retrieval.bm25``, T05),
* ``rocchio`` TF-IDF + Rocchio pseudo-relevance feedback,
* ``hyde``    Rocchio + an LLM-written hypothetical listing (needs a key in ``.env`` or logged
              answers, see ``phonesense/llm/client.py``),
* ``dense``   sentence-embedding cosine search (``phonesense.retrieval.dense``).

Two *modifiers* shape the candidates (A's T06 and T07): ``filters`` parses numeric constraints from
the query ("under $300", "8GB RAM", "5G") and keeps only the phones that satisfy them, for every
selected system (a phone whose value is unknown never passes), and ``phrase`` adds a phrase-match
boost to BM25 (quoted phrases and a few known ones such as "night mode").

Two *priors* can then reorder the top 100 (they never add phones): ``pagerank`` (global
PageRank of the co-review graph, C's T17) and ``trust`` (reviewer-trust phone quality,
C's T19). Both are in [0, 1]; with both on, their mean is used. The final order is
``(1 - lambda) * score_hat + lambda * prior`` (``phonesense.graph.rerank.rerank_with_prior``,
C's T18), where ``score_hat`` is the ranking's own score min-max scaled to [0, 1].

With two or more systems on, their rankings are merged by rank fusion (RRF or linear) when the
fusion switch is on; otherwise the first system in priority order is used and a note says so.
``dense`` and the fusion code live in separate modules that may be missing from a build; they
are imported lazily and reported as unavailable instead of crashing the app.

Data: the phones file is looked up in this order: the ``PHONESENSE_PHONES`` environment
variable, ``paths.phones`` (full data), ``paths.sample_dir/phones.jsonl`` (the early sample).
When no override is given and a saved index exists (``scripts/build_index.py``) it is loaded;
otherwise an index is built in memory.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from phonesense.config import get_config, get_path
from phonesense.index import InvertedIndex, index_path
from phonesense.io import Ranking, read_phones
from phonesense.llm.client import LLMClient, LLMError
from phonesense.retrieval.hyde import HydeGenerator, HydeRetriever
from phonesense.retrieval.rocchio import RocchioRetriever
from phonesense.retrieval.tfidf import TfidfRetriever
from phonesense.schemas import Phone, Query


class NoDataError(FileNotFoundError):
    """No phones file was found; the message says how to create one."""


@dataclass(frozen=True)
class SystemScore:
    """What one system said about one result: its 1-based ``rank`` in its own top list (None if
    the phone was not in it) and its raw ``score`` there."""

    key: str
    label: str
    rank: int | None
    score: float | None
    kind: str = "system"  # "system" (a retriever) or "prior" (a reranking prior; no rank)


@dataclass(frozen=True)
class Result:
    """One search result, ready to draw. ``score`` is the final (possibly fused) score and
    ``breakdown`` lists what each selected system gave this phone."""

    rank: int
    doc_id: str
    title: str
    brand: str | None
    price: float | None
    currency: str
    avg_rating: float | None
    rating_count: int
    score: float
    breakdown: tuple[SystemScore, ...] = ()

    @property
    def price_text(self) -> str:
        return "price unknown" if self.price is None else f"{self.price:,.2f} {self.currency}"

    @property
    def rating_text(self) -> str:
        if self.avg_rating is None:
            return "no ratings"
        return f"{self.avg_rating:.1f}/5 ({self.rating_count:,} ratings)"


COMPONENTS = ("tfidf", "bm25", "rocchio", "hyde", "dense")  # priority order
LABELS = {
    "tfidf": "TF-IDF",
    "bm25": "BM25 (tuned)",
    "rocchio": "Rocchio feedback",
    "hyde": "HyDE expansion",
    "dense": "Dense (embeddings)",
}
PRIORS = ("pagerank", "trust")
PRIOR_LABELS = {"pagerank": "PageRank prior", "trust": "Reviewer-trust prior"}
# prior -> (module, function, data file path key, script that creates the file)
PRIOR_SOURCES = {
    "pagerank": ("phonesense.graph.pagerank", "pagerank_scores", "graph_edges",
                 "scripts/build_graph.py"),
    "trust": ("phonesense.trust.trust", "trust_scores", "reviews", "scripts/build_data.py"),
}
DEFAULT_PRIOR_WEIGHT = 0.2  # lambda; C's rerank: block sweeps 0.1-0.5 on dev
FUSION_METHODS = ("rrf", "linear")
DEPTH = 100  # documents taken from each system before fusing


@dataclass(frozen=True)
class SearchResponse:
    """Results plus what actually ran: ``used`` are the systems, ``notes`` explain fallbacks."""

    results: list[Result]
    used: list[str]
    notes: list[str] = field(default_factory=list)
    filter_summary: str | None = None  # the constraints parsed from the query, if filters ran
    n_allowed: int | None = None  # how many phones pass them


def _tuned_prf(method: str) -> dict:
    """The Rocchio / HyDE settings tuned on dev (``prf.<method>.tuned`` in config.yaml), so the
    demo shows the same systems as the ablation; empty (class defaults) if not configured."""
    tuned = get_config().get("prf", {}).get(method, {}).get("tuned") or {}
    allowed = {"m", "beta", "n_terms"} | ({"delta"} if method == "hyde" else set())
    return {k: v for k, v in tuned.items() if k in allowed}


def describe_filter(node: Mapping) -> str:
    """Readable text for a filter node (CONTRACT 4.4): ``price < 300 AND has_5g == True``."""
    if "field" in node:
        return f"{node['field']} {node['op']} {node['value']}"
    if "not" in node:
        return f"NOT ({describe_filter(node['not'])})"
    key = "and" if "and" in node else "or"
    return f" {key.upper()} ".join(f"({describe_filter(n)})" if ("and" in n or "or" in n) else
                                   describe_filter(n) for n in node[key])


def find_phones_file() -> tuple[Path, bool]:
    """Return ``(path, is_sample)`` of the phones file to use, or raise :class:`NoDataError`."""
    env = os.environ.get("PHONESENSE_PHONES")
    if env:
        path = Path(env)
        if not path.is_file():
            raise NoDataError(f"PHONESENSE_PHONES points to {path}, which does not exist")
        return path, False
    full = get_path("phones")
    if full.is_file():
        return full, False
    sample = get_path("sample_dir") / "phones.jsonl"
    if sample.is_file():
        return sample, True
    raise NoDataError(
        "no phones file found. Build the data first: python scripts/build_data.py "
        "(or point PHONESENSE_PHONES at a phones.jsonl)"
    )


class SearchEngine:
    """Search over a phone collection with switchable systems; returns :class:`Result` rows."""

    def __init__(
        self,
        phones: list[Phone],
        index: InvertedIndex | None = None,
        priors: Mapping[str, Callable[[], Mapping[str, float]]] | None = None,
    ) -> None:
        """``priors`` maps ``pagerank``/``trust`` to a function returning ``{doc_id: [0, 1]}``;
        by default they come from C's modules and data files (loaded lazily, once)."""
        self.phones = {p.doc_id: p for p in phones}
        self._prior_loaders = dict(priors) if priors is not None else None
        self._prior_cache: dict[str, Mapping[str, float]] = {}
        self._phone_list = phones
        self.tfidf = TfidfRetriever()
        if index is not None:
            self.tfidf.fit_index(index)
        else:
            self.tfidf.fit(phones)
        self.rocchio = RocchioRetriever(self.tfidf, **_tuned_prf("rocchio"))
        self._hyde: HydeRetriever | None = None
        self._bm25 = None
        self._phrase = None
        self._allowed_cache: dict[str, set[str]] = {}
        self._dense = None
        self._unavailable: dict[str, str] = {}

    @classmethod
    def from_default_data(cls) -> SearchEngine:
        """Load the phones (see module docstring) and a saved index when one is usable."""
        path, is_sample = find_phones_file()
        phones = read_phones(path)
        index = None
        if not os.environ.get("PHONESENSE_PHONES"):
            saved = index_path(is_sample)
            if (saved / "meta.json").exists():
                try:
                    index = InvertedIndex.load(saved)
                except Exception:  # stale or unreadable index: fall back to building one
                    index = None
        return cls(phones, index)

    def __len__(self) -> int:
        return len(self.phones)

    # ------------------------------------------------------------------ availability

    def unavailable_reason(self, key: str) -> str | None:
        """Why system ``key`` (or ``"fusion"``) cannot be used right now, or None if it can."""
        if key in self._unavailable:
            return self._unavailable[key]
        if key == "dense" and importlib.util.find_spec("phonesense.retrieval.dense") is None:
            return "dense retrieval is not in this build"
        if key == "bm25" and importlib.util.find_spec("phonesense.retrieval.bm25") is None:
            return "BM25 is not in this build"
        if key == "filters" and importlib.util.find_spec("phonesense.retrieval.filters") is None:
            return "Boolean spec filters are not in this build"
        if key == "phrase":
            if importlib.util.find_spec("phonesense.retrieval.positional") is None:
                return "phrase search is not in this build"
            if importlib.util.find_spec("phonesense.retrieval.bm25") is None:
                return "phrase boost needs BM25, which is not in this build"
        if key == "fusion" and importlib.util.find_spec("phonesense.retrieval.fusion") is None:
            return "rank fusion is not in this build"
        if key in PRIORS and importlib.util.find_spec("phonesense.graph.rerank") is None:
            return "prior reranking is not in this build"
        if key in PRIORS and self._prior_loaders is None:  # real loaders: check cheaply first
            module, func, path_key, script = PRIOR_SOURCES[key]
            if importlib.util.find_spec(module) is None or not hasattr(
                importlib.import_module(module), func
            ):
                return f"{PRIOR_LABELS[key].lower()} is not in this build"
            if not get_path(path_key).exists():
                return f"{get_path(path_key).name} not found (run {script})"
        return None

    def _prior(self, key: str) -> Mapping[str, float]:
        """The prior ``{doc_id: [0, 1]}`` for ``key``, computed once and kept."""
        if key not in self._prior_cache:
            if self._prior_loaders is not None:
                self._prior_cache[key] = self._prior_loaders[key]()
            elif key == "pagerank":
                from phonesense.graph.pagerank import pagerank_scores

                self._prior_cache[key] = pagerank_scores(doc_ids=list(self.phones))
            else:
                from phonesense.io import read_reviews
                from phonesense.trust.methods import trust_params
                from phonesense.trust.trust import trust_scores

                reviews = read_reviews(get_path("reviews"))
                self._prior_cache[key] = trust_scores(reviews, **trust_params())
        return self._prior_cache[key]

    def _get_dense(self):
        """Build the dense retriever on first use (embeds all phones; cached on disk)."""
        if self._dense is None:
            module = importlib.import_module("phonesense.retrieval.dense")
            retriever = module.DenseRetriever()
            retriever.fit(self._phone_list)
            self._dense = retriever
        return self._dense

    def _get_bm25(self):
        """BM25 with the k1 and b tuned on dev (``bm25.tuned`` in config.yaml, T05), or the
        textbook defaults if nothing is tuned. Shares our index, so nothing is re-read."""
        if self._bm25 is None:
            module = importlib.import_module("phonesense.retrieval.bm25")
            cfg = get_config().get("bm25", {})
            tuned = cfg.get("tuned")
            params = tuned or cfg.get("default") or {"k1": 1.2, "b": 0.75}
            self._bm25 = module.BM25Retriever(
                params["k1"], params["b"], index=self.tfidf.index,
                name="bm25_tuned" if tuned else "bm25_default",
            )
        return self._bm25

    def _get_phrase(self):
        """BM25 with the phrase-match boost (weight tuned on dev, ``phrase.tuned``)."""
        if self._phrase is None:
            module = importlib.import_module("phonesense.retrieval.positional")
            bm25 = self._get_bm25()
            self._phrase = module.PhraseBoostRetriever(bm25, name=f"{bm25.name}+phrase")
        return self._phrase

    def _allowed(self, filt: Mapping) -> set[str]:
        """doc_ids of the phones that pass ``filt`` (unknown values never pass); cached."""
        import json

        key = json.dumps(filt, sort_keys=True)
        if key not in self._allowed_cache:
            filters = importlib.import_module("phonesense.retrieval.filters")
            self._allowed_cache[key] = filters.allowed_docs(self._phone_list, filt)
        return self._allowed_cache[key]

    def _get_hyde(self) -> HydeRetriever:
        """HyDE with the settings tuned on dev and the LLM the logged texts came from (so queries
        already in ``runs/llm_logs/`` are replayed without any key; new queries need the key of
        that provider in ``.env``)."""
        if self._hyde is None:
            llm = get_config().get("prf", {}).get("hyde", {}).get("llm", {})
            client = LLMClient(llm.get("provider"), llm.get("model"))
            self._hyde = HydeRetriever(HydeGenerator(client), self.tfidf, **_tuned_prf("hyde"))
        return self._hyde

    def _retriever(self, key: str):
        return {
            "tfidf": lambda: self.tfidf,
            "bm25": self._get_bm25,
            "rocchio": lambda: self.rocchio,
            "hyde": self._get_hyde,
            "dense": self._get_dense,
        }[key]()

    # ------------------------------------------------------------------ searching

    def _results(
        self,
        ranking: Ranking,
        systems: Mapping[str, Ranking] | None = None,
        priors: Mapping[str, Mapping[str, float]] | None = None,
        prior_weight: float = 0.0,
        labels: Mapping[str, str] | None = None,
    ) -> list[Result]:
        """Turn a final ranking into rows. ``systems`` ({key: that system's own ranking}) fills
        the per-result breakdown: where each selected system put the phone, and its raw score.
        ``priors`` ({key: {doc_id: value}}) adds one row per prior with its value for the phone."""
        lookup = {
            key: {doc: (pos, score) for pos, (doc, score) in enumerate(r, start=1)}
            for key, r in (systems or {}).items()
        }
        out = []
        for rank, (doc_id, score) in enumerate(ranking, start=1):
            p = self.phones[doc_id]
            parts = tuple(
                SystemScore(key, (labels or LABELS)[key], *lookup[key].get(doc_id, (None, None)))
                for key in lookup
            ) + tuple(
                SystemScore(key, f"{PRIOR_LABELS[key]} (lambda={prior_weight:g})", None,
                            values.get(doc_id), kind="prior")
                for key, values in (priors or {}).items()
            )
            out.append(Result(rank, doc_id, p.title, p.brand, p.price, p.currency,
                              p.avg_rating, p.rating_count, score, parts))
        return out

    def _fuse(self, rankings: list[Ranking], method: str, k: int, used: list[str]) -> Ranking:
        """Fuse with the settings tuned on dev (``fusion`` block of config.yaml): the RRF constant,
        and for linear fusion the tuned weights when exactly BM25 and dense are fused (the pair
        they were tuned for), otherwise equal weights."""
        fusion = importlib.import_module("phonesense.retrieval.fusion")
        cfg = get_config().get("fusion", {})
        if method == "linear":
            tuned = (cfg.get("tuned") or {}).get("linear", {}).get("weights")
            weights = tuned if tuned and used == ["bm25", "dense"] else None
            return fusion.weighted_linear_fusion(rankings, weights, depth=k)
        return fusion.reciprocal_rank_fusion(rankings, int(cfg.get("rrf_k", 60)), depth=k)

    def search(
        self,
        text: str,
        toggles: Mapping[str, bool] | None = None,
        k: int = 10,
        fusion_method: str = "rrf",
        prior_weight: float = DEFAULT_PRIOR_WEIGHT,
    ) -> SearchResponse:
        """Top ``k`` phones for ``text`` using the systems switched on in ``toggles``.

        ``toggles`` maps ``tfidf``/``bm25``/``rocchio``/``hyde``/``dense`` (systems),
        ``fusion``, ``filters``/``phrase`` (modifiers) and ``pagerank``/``trust`` (priors, weight
        ``prior_weight`` = lambda in [0, 1]) to bool; missing keys count as off, except that
        TF-IDF is used when nothing is selected. A system, modifier or prior that cannot run
        (missing module or data, no LLM key) is dropped with a note rather than raising.
        """
        if not 0.0 <= prior_weight <= 1.0:
            raise ValueError(f"prior_weight must be in [0, 1], got {prior_weight}")
        if fusion_method not in FUSION_METHODS:
            raise ValueError(f"fusion_method must be one of {FUSION_METHODS}, got {fusion_method}")
        if not text.strip():
            return SearchResponse([], [])
        toggles = toggles or {"tfidf": True}
        query = Query(qid="q00", text=text, split="dev")
        notes: list[str] = []

        wanted = [c for c in COMPONENTS if toggles.get(c)]
        if not wanted:
            wanted = ["tfidf"]
            notes.append("No system selected, showing TF-IDF.")
        rankings: dict[str, Ranking] = {}
        wants_priors = [p for p in PRIORS if toggles.get(p)]
        depth = DEPTH if len(wanted) > 1 or wants_priors else k

        allowed, filter_summary = self._resolve_filter(query, toggles, notes)
        phrase_on = self._phrase_enabled(toggles, wanted, notes)
        labels = dict(LABELS)
        if phrase_on:
            labels["bm25"] = "BM25 (tuned) + phrase boost"
        # with a filter many hits are dropped, so retrieve deeper and cut after filtering
        fetch = depth
        if allowed is not None:
            filters = importlib.import_module("phonesense.retrieval.filters")
            fetch = max(depth, int(filters.filters_config().get("post_depth", 1000)))

        def restrict(ranking: Ranking) -> Ranking:
            if allowed is None:
                return ranking
            return [(d, sc) for d, sc in ranking if d in allowed][:depth]

        for key in wanted:
            reason = self.unavailable_reason(key)
            if reason is None:
                try:
                    use_phrase = key == "bm25" and phrase_on
                    retriever = self._get_phrase() if use_phrase else self._retriever(key)
                    rankings[key] = restrict(retriever.search(query, fetch))
                    continue
                except ImportError as exc:  # a missing package will not appear mid-session
                    reason = self._unavailable[key] = str(exc)
                except LLMError as exc:  # may be transient (network, quota): retry next search
                    reason = str(exc)
                except Exception as exc:  # noqa: BLE001 - a demo page must not die on one system
                    reason = f"{type(exc).__name__}: {exc}"
            notes.append(f"{LABELS[key]} skipped: {reason}.")
        if not rankings:  # every chosen system failed: fall back so the page is never empty
            rankings["tfidf"] = restrict(self.tfidf.search(query, fetch))
            notes.append("Falling back to TF-IDF.")

        used = list(rankings)
        shown = rankings
        if len(used) == 1:
            final = rankings[used[0]]
        else:
            fusion_reason = self.unavailable_reason("fusion")
            if toggles.get("fusion") and fusion_reason is None:
                final = self._fuse([rankings[u] for u in used], fusion_method, depth, used)
            else:
                why = "fusion is off" if fusion_reason is None else f"{fusion_reason}"
                notes.append(
                    f"{len(used)} systems selected but {why}; showing {LABELS[used[0]]} only."
                )
                used = used[:1]
                shown = {used[0]: rankings[used[0]]}
                final = rankings[used[0]]

        prior_values: dict[str, Mapping[str, float]] = {}
        if wants_priors:
            final, prior_values = self._apply_priors(final, wants_priors, prior_weight, notes)
        if allowed is not None and not final:
            notes.append("No phone passes the filter built from your query.")
        return SearchResponse(
            self._results(final[:k], shown, prior_values, prior_weight, labels),
            used, notes, filter_summary, None if allowed is None else len(allowed),
        )

    def _resolve_filter(
        self, query: Query, toggles: Mapping[str, bool], notes: list[str]
    ) -> tuple[set[str] | None, str | None]:
        """Parse the query's numeric constraints and return ``(allowed doc_ids, summary)``;
        ``(None, None)`` when filters are off, unavailable or the query has no constraint."""
        if not toggles.get("filters"):
            return None, None
        reason = self.unavailable_reason("filters")
        if reason:
            notes.append(f"Spec filters skipped: {reason}.")
            return None, None
        filters = importlib.import_module("phonesense.retrieval.filters")
        filt, _residual, _source = filters.resolve_filter(query)
        if filt is None:
            notes.append("No numeric constraint found in this query, so nothing was filtered.")
            return None, None
        return self._allowed(filt), describe_filter(filt)

    def _phrase_enabled(
        self, toggles: Mapping[str, bool], wanted: list[str], notes: list[str]
    ) -> bool:
        if not toggles.get("phrase"):
            return False
        reason = self.unavailable_reason("phrase")
        if reason:
            notes.append(f"Phrase boost skipped: {reason}.")
            return False
        if "bm25" not in wanted:
            notes.append("Phrase boost applies to BM25: switch BM25 on to use it.")
            return False
        return True

    def _apply_priors(
        self, ranking: Ranking, keys: list[str], weight: float, notes: list[str]
    ) -> tuple[Ranking, dict[str, Mapping[str, float]]]:
        """Reorder ``ranking`` with the mean of the usable priors; returns it and the priors."""
        values: dict[str, Mapping[str, float]] = {}
        for key in keys:
            reason = self.unavailable_reason(key)
            if reason is None:
                try:
                    values[key] = self._prior(key)
                    continue
                except Exception as exc:  # noqa: BLE001 - skip the prior, keep the page alive
                    reason = f"{type(exc).__name__}: {exc}"
            notes.append(f"{PRIOR_LABELS[key]} skipped: {reason}.")
        if not values or weight == 0.0:
            return ranking, values
        combined: dict[str, float] = {}
        for doc_id, _ in ranking:
            have = [v[doc_id] for v in values.values() if doc_id in v]
            if have:
                combined[doc_id] = sum(have) / len(have)
        rerank = importlib.import_module("phonesense.graph.rerank")
        return rerank.rerank_with_prior({"q": ranking}, combined, weight)["q"], values
