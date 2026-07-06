"""
Vector storage: dual FAISS indexes (raw-text embeddings + contextual
embeddings). Contextual embeddings encode "this chunk relates to entity X,
which also appears on page Y" so semantically-distant-but-graph-linked
chunks can still surface via pure vector search as a fallback.
"""
from __future__ import annotations

import json
import logging
import os
import pickle

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class DualFAISSStore:
    def __init__(self, index_dir: str, dim: int = 384):
        self._index_dir = index_dir
        self._dim = dim
        os.makedirs(index_dir, exist_ok=True)

        self.raw_index = faiss.IndexFlatIP(dim)     # cosine sim via normalized vectors + inner product
        self.contextual_index = faiss.IndexFlatIP(dim)
        self._raw_id_map: list[str] = []            # position -> unit_id
        self._contextual_id_map: list[str] = []

        self._load_if_exists()

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return vectors / norms

    def add_raw(self, unit_ids: list[str], embeddings: list[list[float]]) -> None:
        vecs = self._normalize(np.array(embeddings, dtype="float32"))
        self.raw_index.add(vecs)
        self._raw_id_map.extend(unit_ids)

    def add_contextual(self, unit_ids: list[str], embeddings: list[list[float]]) -> None:
        vecs = self._normalize(np.array(embeddings, dtype="float32"))
        self.contextual_index.add(vecs)
        self._contextual_id_map.extend(unit_ids)

    def search_raw(self, query_embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        return self._search(self.raw_index, self._raw_id_map, query_embedding, top_k)

    def search_contextual(self, query_embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        return self._search(self.contextual_index, self._contextual_id_map, query_embedding, top_k)

    def _search(self, index, id_map: list[str], query_embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        if index.ntotal == 0:
            return []
        q = self._normalize(np.array([query_embedding], dtype="float32"))
        scores, indices = index.search(q, min(top_k, index.ntotal))
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((id_map[idx], float(score)))
        return results

    def save(self) -> None:
        faiss.write_index(self.raw_index, os.path.join(self._index_dir, "raw.index"))
        faiss.write_index(self.contextual_index, os.path.join(self._index_dir, "contextual.index"))
        with open(os.path.join(self._index_dir, "id_maps.json"), "w") as f:
            json.dump({"raw": self._raw_id_map, "contextual": self._contextual_id_map}, f)
        logger.info("Saved FAISS indexes to %s", self._index_dir)

    def _load_if_exists(self) -> None:
        raw_path = os.path.join(self._index_dir, "raw.index")
        ctx_path = os.path.join(self._index_dir, "contextual.index")
        map_path = os.path.join(self._index_dir, "id_maps.json")
        if os.path.exists(raw_path) and os.path.exists(map_path):
            self.raw_index = faiss.read_index(raw_path)
            self.contextual_index = faiss.read_index(ctx_path)
            with open(map_path) as f:
                maps = json.load(f)
            self._raw_id_map = maps["raw"]
            self._contextual_id_map = maps["contextual"]
            logger.info("Loaded existing FAISS indexes from %s", self._index_dir)
