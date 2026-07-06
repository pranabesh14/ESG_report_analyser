"""
Canonical Entity Resolution -- the single most important module in this
system. This is what lets "ESG" on page 6 and "carbon footprint" on page 35
collapse into the same graph node so retrieval can traverse between them.

Strategy (cheapest-first, to control cost):
  1. Exact/normalized string match -> auto-merge.
  2. Embedding similarity above auto_merge_threshold -> auto-merge.
  3. Embedding similarity in the "maybe" band -> LLM disambiguation call.
  4. Below similarity_threshold -> new canonical entity.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

from pdf_rag_kb.config.settings import EntityResolutionConfig
from pdf_rag_kb.core.llm_provider import LLMProvider, LLMProviderError
from pdf_rag_kb.core.schemas import CanonicalEntity, EntityMention, new_id

logger = logging.getLogger(__name__)

_DISAMBIGUATION_PROMPT = """\
Are these two references to the SAME underlying real-world entity/concept,
or DIFFERENT things that merely sound similar?

Reference A: "{name_a}" (type: {type_a})
  Context: {context_a}

Reference B: "{name_b}" (type: {type_b})
  Context: {context_b}

Respond as JSON: {{"same_entity": true|false, "canonical_name": str,
"confidence": float}}
canonical_name should be the clearer/more complete of the two names,
used if same_entity is true.
"""


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _cosine_sim(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom else 0.0


@dataclass
class _PendingMention:
    mention: EntityMention
    embedding: list[float]


class EntityResolver:
    """Stateful across a document (or corpus, if you keep reusing the same
    instance) -- canonical entities accumulate as more mentions arrive.
    """

    def __init__(self, provider: LLMProvider, config: EntityResolutionConfig):
        self._provider = provider
        self._config = config
        self._canonical_entities: dict[str, CanonicalEntity] = {}
        self._entity_embeddings: dict[str, list[float]] = {}  # entity_id -> representative embedding

    def load_existing_state(
        self, entities: list[CanonicalEntity], embeddings: dict[str, list[float]],
    ) -> None:
        """Seed the resolver with entities/embeddings persisted from prior
        ingestion runs (e.g. the 2022/2023 reports), so a new mention like
        "carbon footprint" in the 2024 report can still match against them.
        Without this, resolution silently resets every process restart and
        cross-year linking breaks even though nothing raises an error.
        """
        for entity in entities:
            self._canonical_entities[entity.entity_id] = entity
        for entity_id, embedding in embeddings.items():
            if entity_id in self._canonical_entities:
                self._entity_embeddings[entity_id] = embedding
        logger.info("Seeded EntityResolver with %d existing entities from prior runs", len(entities))

    def get_updated_embeddings(self) -> dict[str, list[float]]:
        """All current representative embeddings, for persisting back to
        storage after an ingestion run (new + updated-by-averaging)."""
        return dict(self._entity_embeddings)

    def resolve_batch(self, mentions: list[EntityMention]) -> dict[str, str]:
        """Resolve a batch of mentions (typically: all mentions from one
        document). Returns mapping mention_id -> canonical entity_id, and
        mutates self._canonical_entities in place.
        """
        if not mentions:
            return {}

        texts = [f"{m.entity_type}: {m.surface_form}. {m.context_snippet}" for m in mentions]
        embeddings = self._provider.embed(texts)
        pending = [_PendingMention(m, e) for m, e in zip(mentions, embeddings)]

        mention_to_entity: dict[str, str] = {}
        for item in pending:
            entity_id = self._resolve_one(item)
            mention_to_entity[item.mention.mention_id] = entity_id

        return mention_to_entity

    def _resolve_one(self, item: _PendingMention) -> str:
        mention = item.mention
        norm_surface = _normalize(mention.surface_form)

        # 1. Exact normalized string match against known aliases
        for entity in self._canonical_entities.values():
            if entity.entity_type != mention.entity_type:
                continue
            if norm_surface in {_normalize(a) for a in entity.aliases}:
                self._attach(entity, mention, item.embedding)
                return entity.entity_id

        # 2/3. Embedding similarity search over existing canonical entities
        best_entity_id, best_sim = None, 0.0
        for entity_id, emb in self._entity_embeddings.items():
            entity = self._canonical_entities[entity_id]
            if entity.entity_type != mention.entity_type:
                continue
            sim = _cosine_sim(item.embedding, emb)
            if sim > best_sim:
                best_entity_id, best_sim = entity_id, sim

        if best_entity_id and best_sim >= self._config.auto_merge_threshold:
            entity = self._canonical_entities[best_entity_id]
            self._attach(entity, mention, item.embedding)
            return entity.entity_id

        if best_entity_id and best_sim >= self._config.similarity_threshold:
            if self._llm_confirms_same(mention, best_entity_id):
                entity = self._canonical_entities[best_entity_id]
                self._attach(entity, mention, item.embedding)
                return entity.entity_id

        # 4. No match -- create new canonical entity
        entity = CanonicalEntity(
            entity_id=new_id("entity"),
            canonical_name=mention.surface_form,
            entity_type=mention.entity_type,
            aliases={mention.surface_form},
            mention_ids=[mention.mention_id],
            doc_ids={mention.doc_id},
        )
        self._canonical_entities[entity.entity_id] = entity
        self._entity_embeddings[entity.entity_id] = item.embedding
        return entity.entity_id

    def _llm_confirms_same(self, mention: EntityMention, candidate_entity_id: str) -> bool:
        entity = self._canonical_entities[candidate_entity_id]
        try:
            result = self._provider.complete_json(
                system="You disambiguate whether two entity references refer to the same "
                       "real-world thing. Be conservative -- when genuinely unsure, say false.",
                prompt=_DISAMBIGUATION_PROMPT.format(
                    name_a=mention.surface_form, type_a=mention.entity_type,
                    context_a=mention.context_snippet,
                    name_b=entity.canonical_name, type_b=entity.entity_type,
                    context_b=", ".join(list(entity.aliases)[:3]),
                ),
                max_tokens=300,
            )
        except LLMProviderError as e:
            logger.warning("Disambiguation call failed, defaulting to no-merge: %s", e)
            return False
        return bool(result.get("same_entity", False)) and float(result.get("confidence", 0)) >= 0.6

    def _attach(self, entity: CanonicalEntity, mention: EntityMention, embedding: list[float]) -> None:
        entity.aliases.add(mention.surface_form)
        entity.mention_ids.append(mention.mention_id)
        entity.doc_ids.add(mention.doc_id)
        # Running average keeps the representative embedding centered as
        # more mentions attach, so later matches compare against the
        # cluster's centroid rather than just the first mention seen.
        old = np.array(self._entity_embeddings[entity.entity_id])
        new = np.array(embedding)
        n = len(entity.mention_ids)
        updated = old + (new - old) / n
        self._entity_embeddings[entity.entity_id] = updated.tolist()

    def get_all_entities(self) -> list[CanonicalEntity]:
        return list(self._canonical_entities.values())
