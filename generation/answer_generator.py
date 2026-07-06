"""
Citation Generator + final LLM call.

Formats aggregated evidence into a prompt that forces page-level citation
and explicitly tells the model which numeric claims are low-confidence
extractions (so it hedges "approximately" instead of stating a shaky
chart-read number as fact).
"""
from __future__ import annotations

from pdf_rag_kb.core.llm_provider import LLMProvider
from pdf_rag_kb.core.schemas import UnitType
from pdf_rag_kb.retrieval.evidence_aggregator import AggregatedEvidence

_SYSTEM = """\
You are a research assistant answering questions using only the evidence
provided below. Every factual claim must cite the page number it came
from, like [p.6] or [p.35]. Evidence may also carry a year label like
[2023] p.6 -- when evidence spans multiple years, treat this as a
timeseries: state the trend explicitly (direction, magnitude, and
whether it's historical or projected) rather than just listing each
year's figure. If evidence is marked LOW-CONFIDENCE, hedge the claim
(e.g. "approximately", "the chart suggests") rather than stating it as a
precise fact. If the evidence doesn't fully answer the question, say so
explicitly rather than filling gaps from general knowledge.
"""

_ANSWER_PROMPT = """\
Question: {question}

Evidence:
{evidence_block}

Answer the question using only the evidence above, with inline page
citations.
"""


class AnswerGenerator:
    def __init__(self, provider: LLMProvider):
        self._provider = provider

    def generate(self, question: str, aggregated: AggregatedEvidence) -> str:
        evidence_block = self._format_evidence(aggregated)
        prompt = _ANSWER_PROMPT.format(question=question, evidence_block=evidence_block)
        return self._provider.complete_text(_SYSTEM, prompt, max_tokens=1500)

    @staticmethod
    def _format_evidence(aggregated: AggregatedEvidence) -> str:
        lines = []
        for e in aggregated.evidence:
            unit = e.unit
            flag = " [LOW-CONFIDENCE EXTRACTION]" if unit.unit_id in aggregated.low_confidence_flags else ""
            provenance = f"(retrieved via {e.retrieval_source}, hop={e.hop_distance})"

            if unit.unit_type == UnitType.CHART and unit.chart_data:
                cd = unit.chart_data
                content = (
                    f"[CHART{flag}] {cd.one_line_claim} "
                    f"(type={cd.chart_type}, x={cd.axis_x_label}, y={cd.axis_y_label} [{cd.units}], "
                    f"horizon={cd.time_horizon}, series={cd.series})"
                )
            elif unit.unit_type == UnitType.TABLE and unit.table_data:
                td = unit.table_data
                content = f"[TABLE{flag}] headers={td.headers}, rows={td.rows[:10]}"
            else:
                content = unit.raw_text

            year_label = f"[{unit.doc_year}] " if unit.doc_year is not None else ""
            lines.append(f"- {year_label}p.{unit.page_number} {provenance}: {content}")
        return "\n".join(lines)
