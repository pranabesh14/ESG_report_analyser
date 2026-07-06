"""
BM25 lexical index. Vector search alone misses exact-term matches
(specific model numbers, regulation names, acronyms) -- BM25 catches those.
"""
from __future__ import annotations

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi


class BM25Store:
    def __init__(self, persist_path: str = "./data/bm25_index.pkl"):
        self._persist_path = Path(persist_path)
        self._bm25: BM25Okapi | None = None
        self._unit_ids: list[str] = []
        self._texts: list[str] = []
        self._load_if_exists()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def build(self, unit_ids: list[str], texts: list[str]) -> None:
        """Full (re)build from the given corpus. NOTE: BM25Okapi has no
        incremental-add API, so ingesting a second document must go through
        add_documents() below, not this method directly, or the first
        document's units silently vanish from lexical search."""
        tokenized = [self._tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized)
        self._unit_ids = unit_ids
        self._texts = texts

    def add_documents(self, unit_ids: list[str], texts: list[str]) -> None:
        """Append new documents' units to the existing corpus and rebuild.
        This is what pipeline.ingest_document() should call -- required for
        multi-document (e.g. multi-year) ingestion to actually accumulate
        rather than each new document overwriting the last."""
        all_ids = self._unit_ids + unit_ids
        all_texts = self._texts + texts
        self.build(all_ids, all_texts)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(self._tokenize(query))
        ranked = sorted(zip(self._unit_ids, scores), key=lambda x: x[1], reverse=True)
        return [(uid, float(score)) for uid, score in ranked[:top_k] if score > 0]

    def save(self) -> None:
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._persist_path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "unit_ids": self._unit_ids, "texts": self._texts}, f)

    def _load_if_exists(self) -> None:
        if self._persist_path.exists():
            with open(self._persist_path, "rb") as f:
                data = pickle.load(f)
            self._bm25 = data["bm25"]
            self._unit_ids = data["unit_ids"]
            self._texts = data.get("texts", [])
