"""
Relationship Extraction Engine.

Two passes, cheap-first:
  1. Rule-based explicit cross-reference detection ("as shown on page 35",
     "see Figure 4") -- free, very high precision when present.
  2. LLM-based causal/semantic relationship extraction between entities
     that co-occur in a unit, or between a unit's entities and previously
     known entities from earlier in the document.
"""
from __future__ import annotations

import logging
import re

from pdf_rag_kb.core.llm_provider import LLMProvider, LLMProviderError
from pdf_rag_kb.core.schemas import CanonicalEntity, DocUnit, RelationType, Relationship, new_id

logger = logging.getLogger(__name__)

_PAGE_REF_PATTERN = re.compile(
    r"\b(?:see|as (?:shown|described|discussed)(?: on| in)?|refer to)\s+"
    r"(?:page|section|figure|table)\s+(\d+)", re.IGNORECASE,
)

_RELATIONSHIP_PROMPT = """\
Given this text and the list of known entities that appear in or near it,
identify relationships between entities. Only extract relationships that
are clearly stated or strongly implied -- do not speculate.

Text: "{text}"

Entities present: {entities}

Possible predicates: implements (an initiative implements a plan/policy),
projects (data/chart projects a future value of a metric), measures (a
metric measurement), causes, part_of, related_to.

Respond as JSON: {{"relationships": [{{"subject": str, "predicate": str,
"object": str, "confidence": float, "evidence": str}}]}}
subject/object must be exact entity names from the provided list.
evidence should be a short (<20 word) paraphrase, not a direct quote.
"""


class RelationshipExtractor:
    def __init__(self, provider: LLMProvider):
        self._provider = provider

    def extract_explicit_refs(self, unit: DocUnit) -> list[Relationship]:
        """Rule-based pass: catches 'see page X' / 'as shown in Figure Y'."""
        rels = []
        for match in _PAGE_REF_PATTERN.finditer(unit.raw_text or ""):
            target_page = int(match.group(1))
            rels.append(Relationship(
                rel_id=new_id("rel"),
                subject_entity_id=unit.unit_id,  # unresolved unit-level ref; resolved at graph-build time
                predicate=RelationType.REFERENCES_PAGE,
                object_entity_id=f"page:{target_page}",
                source_unit_id=unit.unit_id,
                doc_id=unit.doc_id,
                confidence=0.95,
                evidence_text=match.group(0),
            ))
        return rels

    def extract_semantic(
        self, unit: DocUnit, entities_in_unit: list[CanonicalEntity],
    ) -> list[Relationship]:
        """LLM pass: relationships between entities that co-occur in this unit."""
        if len(entities_in_unit) < 2:
            return []

        text = unit.raw_text or (unit.chart_data.one_line_claim if unit.chart_data else "")
        if not text.strip():
            return []

        entity_names = [e.canonical_name for e in entities_in_unit]
        name_to_id = {e.canonical_name: e.entity_id for e in entities_in_unit}

        try:
            result = self._provider.complete_json(
                system="You extract factual relationships between entities in business/"
                       "technical documents. Skip anything not clearly supported by the text.",
                prompt=_RELATIONSHIP_PROMPT.format(text=text[:2000], entities=entity_names),
                max_tokens=800,
            )
        except LLMProviderError as e:
            logger.warning("Relationship extraction failed for unit %s: %s", unit.unit_id, e)
            return []

        rels = []
        for r in result.get("relationships", []):
            subj_id = name_to_id.get(r.get("subject", ""))
            obj_id = name_to_id.get(r.get("object", ""))
            if not subj_id or not obj_id:
                continue
            try:
                predicate = RelationType(r.get("predicate", "related_to"))
            except ValueError:
                predicate = RelationType.RELATED_TO
            rels.append(Relationship(
                rel_id=new_id("rel"),
                subject_entity_id=subj_id,
                predicate=predicate,
                object_entity_id=obj_id,
                source_unit_id=unit.unit_id,
                doc_id=unit.doc_id,
                confidence=float(r.get("confidence", 0.5)),
                evidence_text=r.get("evidence", ""),
            ))
        return rels
