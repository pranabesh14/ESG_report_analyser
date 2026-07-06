"""
Contextual Summarization Engine.

Prepends graph-derived context to each unit before embedding, e.g.:
"This chunk describes projected carbon-footprint outcomes for Org X,
following the reduction initiative described on page 6."

This is what lets *pure vector search* (no graph hop needed) sometimes
already surface the cross-page link, and it's the raw material for the
contextual FAISS index.
"""
from __future__ import annotations

from pdf_rag_kb.core.llm_provider import LLMProvider, LLMProviderError
from pdf_rag_kb.core.schemas import CanonicalEntity, DocUnit, Relationship

_CONTEXT_PROMPT = """\
Unit text (page {page}): "{text}"

This unit mentions these entities: {entity_names}
Known relationships involving these entities elsewhere in the document:
{relationships_summary}

Write ONE sentence (<30 words) of context to prepend to this chunk before
embedding, connecting it to the related material described above.

Respond as JSON: {{"context": str}}. If there are no meaningful
relationships, respond with {{"context": ""}}.
"""


class ContextualSummarizer:
    def __init__(self, provider: LLMProvider):
        self._provider = provider

    def summarize(
        self, unit: DocUnit, entities: list[CanonicalEntity], relationships: list[Relationship],
    ) -> str:
        if not entities:
            return unit.raw_text

        rel_lines = []
        for r in relationships[:5]:
            rel_lines.append(f"{r.subject_entity_id} --{r.predicate}--> {r.object_entity_id} ({r.evidence_text})")

        try:
            result = self._provider.complete_json(
                system="You write concise connective context for a document retrieval system. "
                       "Never invent relationships not given to you.",
                prompt=_CONTEXT_PROMPT.format(
                    page=unit.page_number, text=(unit.raw_text or "")[:1000],
                    entity_names=[e.canonical_name for e in entities],
                    relationships_summary="\n".join(rel_lines) or "none",
                ),
                max_tokens=200,
            )
            prefix = result.get("context", "") if isinstance(result, dict) else ""
        except LLMProviderError:
            prefix = ""

        base_text = unit.raw_text or (unit.chart_data.one_line_claim if unit.chart_data else "")
        return f"{prefix} {base_text}".strip() if prefix else base_text
