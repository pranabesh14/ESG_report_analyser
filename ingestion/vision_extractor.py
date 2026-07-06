"""
Vision Understanding Engine.

This is the highest-leverage part of the whole "correlate the chart with
the text" problem: a chart is useless to the LLM unless it becomes
structured data (axes, series, numbers) rather than a vague caption.
"""
from __future__ import annotations

import logging

from pdf_rag_kb.core.llm_provider import LLMProvider, LLMProviderError
from pdf_rag_kb.core.schemas import ChartData, DocUnit, TableData, UnitType, new_id
from pdf_rag_kb.ingestion.pdf_parser import ExtractedImage

logger = logging.getLogger(__name__)

_CLASSIFY_AND_EXTRACT_PROMPT = """\
You are analyzing an image extracted from a business/technical PDF document.

Nearby caption text (may be empty or irrelevant): "{caption}"

Step 1: Classify the image as one of: "chart", "table_image", "photo", "diagram", "logo_or_decorative".

Step 2: If it is a "chart" (line/bar/pie/projection/trend graph), extract:
- chart_type: e.g. "line", "bar", "projection"
- axis_x_label, axis_y_label, units
- series: list of {{"name": str, "points": [{{"x": ..., "y": ...}}]}} -- read actual values
  off the chart as precisely as you can; if exact values aren't legible, give your best
  estimate and lower the confidence score accordingly.
- time_horizon: e.g. "2024-2030"
- one_line_claim: a single sentence stating the chart's main takeaway with numbers,
  e.g. "Carbon emissions are projected to fall from 120kt to 84kt by 2028."
- extraction_confidence: float 0-1, your confidence in the extracted numbers.

If it is NOT a chart, only return: {{"image_classification": "<type>"}}

Respond as JSON with key "image_classification" always present, and key "chart_data"
(matching the fields above) present only when image_classification == "chart".
"""


class VisionExtractor:
    def __init__(self, provider: LLMProvider):
        self._provider = provider

    def process_image(self, image: ExtractedImage) -> DocUnit:
        """Classify + (if chart) extract structured data. Falls back to a
        low-confidence IMAGE unit on any provider failure -- never crash
        the pipeline on one bad image.
        """
        prompt = _CLASSIFY_AND_EXTRACT_PROMPT.format(caption=image.nearby_caption or "")
        try:
            result = self._provider.complete_vision_json(
                system="You are a precise document-analysis assistant. Never invent data "
                       "you cannot see; lower confidence instead of guessing wildly.",
                prompt=prompt,
                image_bytes=image.image_bytes,
                media_type=image.media_type,
                max_tokens=1500,
            )
        except LLMProviderError as e:
            logger.warning("Vision extraction failed for %s: %s", image.unit_id, e)
            return self._fallback_unit(image)

        classification = result.get("image_classification", "photo")

        if classification == "chart" and "chart_data" in result:
            cd = result["chart_data"]
            chart_data = ChartData(
                chart_type=cd.get("chart_type", ""),
                axis_x_label=cd.get("axis_x_label", ""),
                axis_y_label=cd.get("axis_y_label", ""),
                units=cd.get("units", ""),
                series=cd.get("series", []),
                time_horizon=cd.get("time_horizon", ""),
                one_line_claim=cd.get("one_line_claim", ""),
                extraction_confidence=float(cd.get("extraction_confidence", 0.5)),
            )
            return DocUnit(
                unit_id=image.unit_id, doc_id=image.doc_id, page_number=image.page_number,
                unit_type=UnitType.CHART, bbox=image.bbox,
                raw_text=image.nearby_caption, chart_data=chart_data,
            )

        # Non-chart image: keep a lightweight record (useful for photos/diagrams
        # that might still get referenced, but no structured extraction needed)
        return DocUnit(
            unit_id=image.unit_id, doc_id=image.doc_id, page_number=image.page_number,
            unit_type=UnitType.IMAGE, bbox=image.bbox, raw_text=image.nearby_caption,
        )

    @staticmethod
    def _fallback_unit(image: ExtractedImage) -> DocUnit:
        return DocUnit(
            unit_id=image.unit_id, doc_id=image.doc_id, page_number=image.page_number,
            unit_type=UnitType.IMAGE, bbox=image.bbox, raw_text=image.nearby_caption,
        )
