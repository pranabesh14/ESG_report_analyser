"""
Evidence Aggregator.

Sits between retrieval and generation: filters low-confidence chart/table
extractions, orders evidence so directly-relevant (hop 0) material leads
and graph-linked supporting material follows, and flags weak provenance
so the Citation Generator can hedge appropriately instead of stating
shaky numbers as fact.
"""
from __future__ import annotations

from dataclasses import dataclass

from pdf_rag_kb.core.schemas import RetrievedEvidence, UnitType


@dataclass
class AggregatedEvidence:
    evidence: list[RetrievedEvidence]
    low_confidence_flags: list[str]   # unit_ids whose numeric data should be hedged


class EvidenceAggregator:
    def __init__(self, min_confidence_for_citation: float = 0.55):
        self._min_confidence = min_confidence_for_citation

    def aggregate(self, evidence: list[RetrievedEvidence]) -> AggregatedEvidence:
        deduped: dict[str, RetrievedEvidence] = {}
        for e in evidence:
            if e.unit.unit_id not in deduped or e.score > deduped[e.unit.unit_id].score:
                deduped[e.unit.unit_id] = e

        # Sort by relevance tier first (hop distance, then score), but break
        # ties by doc_year so that when several years' data for the same
        # entity land in the same tier, the generator sees them in
        # chronological order rather than arbitrary retrieval order --
        # this is what actually lets it narrate a trend correctly.
        ordered = sorted(
            deduped.values(),
            key=lambda e: (e.hop_distance, -e.score, e.unit.doc_year if e.unit.doc_year is not None else 9999),
        )

        low_confidence_flags = []
        for e in ordered:
            if e.unit.unit_type == UnitType.CHART and e.unit.chart_data:
                if e.unit.chart_data.extraction_confidence < self._min_confidence:
                    low_confidence_flags.append(e.unit.unit_id)
            if e.unit.unit_type == UnitType.TABLE and e.unit.table_data:
                if e.unit.table_data.extraction_confidence < self._min_confidence:
                    low_confidence_flags.append(e.unit.unit_id)

        return AggregatedEvidence(evidence=ordered, low_confidence_flags=low_confidence_flags)
