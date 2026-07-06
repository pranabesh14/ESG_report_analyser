"""
Hybrid Graph Retrieval.

Flow: vector search + BM25 -> seed units -> for each seed's entities,
walk the relationship graph N hops -> merge everything with source-aware
scoring. This is the step that actually pulls page 35's projection chart
into context when the query only textually matches page 6.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from pdf_rag_kb.config.settings import RetrievalConfig
from pdf_rag_kb.core.llm_provider import LLMProvider
from pdf_rag_kb.core.schemas import DocUnit, RetrievedEvidence
from pdf_rag_kb.storage.bm25_store import BM25Store
from pdf_rag_kb.storage.relational_store import RelationalStore
from pdf_rag_kb.storage.vector_store import DualFAISSStore

logger = logging.getLogger(__name__)


class HybridGraphRetriever:
    def __init__(
        self, provider: LLMProvider, vector_store: DualFAISSStore,
        bm25_store: BM25Store, relational_store: RelationalStore, config: RetrievalConfig,
    ):
        self._provider = provider
        self._vector = vector_store
        self._bm25 = bm25_store
        self._relational = relational_store
        self._config = config

    def retrieve(self, query: str) -> list[RetrievedEvidence]:
        query_embedding = self._provider.embed([query])[0]

        vector_hits = self._vector.search_contextual(query_embedding, self._config.vector_top_k)
        bm25_hits = self._bm25.search(query, self._config.bm25_top_k)

        evidence_by_unit: dict[str, RetrievedEvidence] = {}

        seed_unit_ids = {uid for uid, _ in vector_hits} | {uid for uid, _ in bm25_hits}
        seed_units = {u.unit_id: u for u in self._relational.get_units_by_ids(list(seed_unit_ids))}

        for uid, score in vector_hits:
            if uid in seed_units:
                self._add_or_boost(evidence_by_unit, seed_units[uid], "vector", score * self._config.weight_vector, 0)
        for uid, score in bm25_hits:
            if uid in seed_units:
                self._add_or_boost(evidence_by_unit, seed_units[uid], "bm25", score * self._config.weight_bm25, 0)

        # Graph expansion from seed units' resolved entities
        self._expand_graph(evidence_by_unit, seed_units)

        ranked = sorted(evidence_by_unit.values(), key=lambda e: e.score, reverse=True)
        return ranked[: self._config.final_context_chunks]

    def _expand_graph(self, evidence_by_unit: dict[str, RetrievedEvidence], seed_units: dict[str, DocUnit]) -> None:
        frontier_entity_ids: set[str] = set()
        for unit in seed_units.values():
            frontier_entity_ids.update(unit.entity_ids)

        visited_entities: set[str] = set()
        hop = 1
        while frontier_entity_ids and hop <= self._config.graph_hops:
            next_frontier: set[str] = set()
            for entity_id in frontier_entity_ids:
                if entity_id in visited_entities:
                    continue
                visited_entities.add(entity_id)

                linked_units = self._relational.get_units_by_entity_across_years(entity_id)[
                    : self._config.graph_max_neighbors_per_hop
                ]
                relationships = self._relational.get_relationships_for_entity(entity_id)
                best_rel = max(relationships, key=lambda r: r.confidence, default=None)

                for linked_unit in linked_units:
                    graph_score = (best_rel.confidence if best_rel else 0.5) * self._config.weight_graph / hop
                    self._add_or_boost(
                        evidence_by_unit, linked_unit, "graph", graph_score, hop, best_rel,
                    )
                    next_frontier.update(linked_unit.entity_ids)
            frontier_entity_ids = next_frontier - visited_entities
            hop += 1

    @staticmethod
    def _add_or_boost(
        evidence_by_unit: dict[str, RetrievedEvidence], unit: DocUnit, source: str,
        score: float, hop_distance: int, relationship=None,
    ) -> None:
        existing = evidence_by_unit.get(unit.unit_id)
        if existing:
            existing.score += score
            # Prefer the shallower/most-direct provenance for display
            if hop_distance < existing.hop_distance:
                existing.hop_distance = hop_distance
                existing.retrieval_source = source
                existing.linking_relationship = relationship
        else:
            evidence_by_unit[unit.unit_id] = RetrievedEvidence(
                unit=unit, retrieval_source=source, score=score,
                hop_distance=hop_distance, linking_relationship=relationship,
            )
