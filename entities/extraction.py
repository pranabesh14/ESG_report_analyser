"""
Entity Extraction Engine.

Finds entity MENTIONS per unit (not yet resolved/deduplicated -- that's
entities/resolution.py). Keeping extraction and resolution as separate
modules is deliberate: extraction is per-unit and embarrassingly
parallel; resolution needs the full document (or corpus) in view.
"""
from __future__ import annotations

import logging

from pdf_rag_kb.core.llm_provider import LLMProvider, LLMProviderError
from pdf_rag_kb.core.schemas import DocUnit, EntityMention, UnitType, new_id

logger = logging.getLogger(__name__)

_EXTRACT_PROMPT = """\
Extract named entities and key concepts from the following document excerpt.
Focus on: organizations, metrics/KPIs (e.g. "carbon footprint", "ESG score"),
initiatives/programs, products, regulations, and quantifiable targets.
Do not extract generic words (e.g. "the company", "this year") unless they
clearly refer to a specific tracked concept in context.

Section heading: "{heading}"
Text: "{text}"

Respond as JSON: {{"entities": [{{"surface_form": str, "entity_type": str,
"context_snippet": str}}]}}
Use entity_type from: organization, metric, initiative, product, regulation,
target, other. context_snippet should be a short (<20 word) quote-free
paraphrase of how the entity is used here.
"""


class EntityExtractor:
    def __init__(self, provider: LLMProvider):
        self._provider = provider

    def extract(self, unit: DocUnit) -> list[EntityMention]:
        text_for_extraction = self._unit_text(unit)
        if not text_for_extraction.strip():
            return []

        try:
            result = self._provider.complete_json(
                system="You extract structured entities from business/technical documents. "
                       "Be precise and conservative -- skip anything ambiguous.",
                prompt=_EXTRACT_PROMPT.format(heading=unit.section_heading, text=text_for_extraction[:3000]),
                max_tokens=1000,
            )
        except LLMProviderError as e:
            logger.warning("Entity extraction failed for unit %s: %s", unit.unit_id, e)
            return []

        mentions = []
        for e in result.get("entities", []):
            surface_form = e.get("surface_form", "").strip()
            if not surface_form:
                continue
            mentions.append(EntityMention(
                mention_id=new_id("mention"),
                unit_id=unit.unit_id,
                doc_id=unit.doc_id,
                surface_form=surface_form,
                entity_type=e.get("entity_type", "other"),
                context_snippet=e.get("context_snippet", ""),
            ))
        return mentions

    @staticmethod
    def _unit_text(unit: DocUnit) -> str:
        """Build the text to run extraction against, depending on unit type.
        For charts, use the structured claim + axis labels rather than raw
        caption text -- that's where the real entity signal is.
        """
        if unit.unit_type == UnitType.CHART and unit.chart_data:
            cd = unit.chart_data
            parts = [cd.one_line_claim, cd.axis_x_label, cd.axis_y_label]
            series_names = [s.get("name", "") for s in cd.series]
            parts.extend(series_names)
            return ". ".join(p for p in parts if p)
        return unit.raw_text
