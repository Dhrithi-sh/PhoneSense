import math
import pickle
from collections import defaultdict
from pathlib import Path

from text.analyzer import analyze


SUPPORTED_MODES = {
    "none",
    "light",
    "aggr",
    "auto",
}


def _zone_postings():
    return {
        "headline": [],
        "body": [],
    }


class Index:
    INDEX_DIR = Path("indexes")

    def __init__(self, mode: str):
        if mode not in SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported index mode: {mode}. "
                f"Supported modes: {sorted(SUPPORTED_MODES)}"
            )

        self.mode = mode
        self._postings = defaultdict(
            _zone_postings
        )

        self.meta = {}
        self.text = {}
        self.N = 0
        self.vocab = set()
        self.doc_norm = {}
        self.doc_len = {}

    def add_document(
        self,
        doc_id,
        headline,
        body,
        metadata,
    ):
        if doc_id in self.meta:
            raise ValueError(
                f"Document already exists: {doc_id}"
            )

        self.meta[doc_id] = metadata

        self.text[doc_id] = {
            "headline": headline,
            "body": body,
        }

        self.N += 1

        headline_terms = self._add_zone(
            doc_id,
            headline,
            "headline",
        )

        body_terms = self._add_zone(
            doc_id,
            body,
            "body",
        )

        self.doc_len[doc_id] = (
            sum(
                len(positions)
                for positions in headline_terms.values()
            )
            +
            sum(
                len(positions)
                for positions in body_terms.values()
            )
        )

        term_frequencies = defaultdict(int)

        for term, positions in headline_terms.items():
            term_frequencies[term] += len(positions)

        for term, positions in body_terms.items():
            term_frequencies[term] += len(positions)

        norm_squared = 0.0

        for tf in term_frequencies.values():
            weight = 1.0 + math.log10(tf)
            norm_squared += weight * weight

        self.doc_norm[doc_id] = math.sqrt(
            norm_squared
        )

    def _add_zone(
        self,
        doc_id,
        text,
        zone,
    ):
        if zone not in {
            "headline",
            "body",
        }:
            raise ValueError(
                f"Unsupported zone: {zone}"
            )

        analyzed = analyze(
            text,
            self.mode,
        )

        term_positions = defaultdict(list)

        for term, position in analyzed:
            term_positions[term].append(
                position
            )

        for term, positions in term_positions.items():
            self._postings[term][zone].append(
                (
                    doc_id,
                    len(positions),
                    positions,
                )
            )

            self.vocab.add(term)

        return term_positions

    def postings_for(
        self,
        term,
        zone,
    ):
        return self.postings(
            term,
            zone,
        )

    def postings(
        self,
        term,
        zone,
    ):
        if zone not in {
            "headline",
            "body",
        }:
            raise ValueError(
                f"Unsupported zone: {zone}"
            )

        return self._postings.get(
            term,
            {},
        ).get(
            zone,
            [],
        )

    def df(self, term):
        doc_ids = set()

        for zone in (
            "headline",
            "body",
        ):
            for doc_id, _, _ in self.postings(
                term,
                zone,
            ):
                doc_ids.add(doc_id)

        return len(doc_ids)

    def save(self, path=None):
        if path is None:
            self.INDEX_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            path = (
                self.INDEX_DIR
                / f"{self.mode}.pkl"
            )

        else:
            path = Path(path)

            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        with open(
            path,
            "wb",
        ) as file:
            pickle.dump(
                self,
                file,
            )

    @classmethod
    def load(
        cls,
        mode,
    ):
        if mode not in SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported analysis mode: {mode}. "
                f"Supported modes: {sorted(SUPPORTED_MODES)}"
            )

        path = (
            cls.INDEX_DIR
            / f"{mode}.pkl"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Index file not found: {path}"
            )

        with open(
            path,
            "rb",
        ) as file:
            index = pickle.load(file)

        if not isinstance(
            index,
            cls,
        ):
            raise TypeError(
                "Loaded object is not an Index"
            )

        if index.mode != mode:
            raise ValueError(
                f"Index mode mismatch: "
                f"expected {mode}, "
                f"found {index.mode}"
            )

        return index