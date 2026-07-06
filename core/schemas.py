"""
Shared data structures. Every pipeline stage reads/writes these -- keeping
them in one module prevents drift between e.g. what the parser emits and
what the entity extractor expects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import uuid


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class UnitType(str, Enum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TABLE = "table"
    CHART = "chart"
    IMAGE = "image"
    CAPTION = "caption"


@dataclass
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class ChartData:
    """Structured extraction from a chart/graph image. This is the artifact
    that lets the LLM cite actual numbers instead of paraphrasing a caption.
    """
    chart_type: str = ""                 # "line", "bar", "projection", etc.
    axis_x_label: str = ""
    axis_y_label: str = ""
    units: str = ""
    series: list[dict[str, Any]] = field(default_factory=list)  # [{"name": str, "points": [{"x":.., "y":..}]}]
    time_horizon: str = ""
    one_line_claim: str = ""             # e.g. "Emissions projected to drop 30% by 2028"
    extraction_confidence: float = 0.0


@dataclass
class TableData:
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    extraction_confidence: float = 0.0


@dataclass
class DocUnit:
    """One atomic extracted piece of a document: a paragraph, a table, a
    chart, etc. This is the base object everything downstream operates on.
    """
    unit_id: str
    doc_id: str
    page_number: int
    unit_type: UnitType
    bbox: Optional[BoundingBox] = None
    raw_text: str = ""
    chart_data: Optional[ChartData] = None
    table_data: Optional[TableData] = None
    section_heading: str = ""            # nearest enclosing heading, for context
    entity_ids: list[str] = field(default_factory=list)   # filled by entity resolution
    contextual_summary: str = ""         # filled by contextual summarization stage
    doc_year: Optional[int] = None       # e.g. 2023 -- reporting year of the source doc,
                                          # required for trend/timeseries queries across documents


@dataclass
class EntityMention:
    """A single occurrence of an entity in a specific unit, pre-resolution."""
    mention_id: str
    unit_id: str
    doc_id: str
    surface_form: str      # e.g. "carbon footprint", "ESG"
    entity_type: str       # "metric", "organization", "initiative", "concept", etc.
    context_snippet: str = ""


@dataclass
class CanonicalEntity:
    """The resolved, deduplicated entity that multiple mentions map to."""
    entity_id: str
    canonical_name: str
    entity_type: str
    aliases: set[str] = field(default_factory=set)
    mention_ids: list[str] = field(default_factory=list)
    doc_ids: set[str] = field(default_factory=set)


class RelationType(str, Enum):
    IMPLEMENTS = "implements"
    PROJECTS = "projects"
    MEASURES = "measures"
    CAUSES = "causes"
    REFERENCES_PAGE = "references_page"
    PART_OF = "part_of"
    RELATED_TO = "related_to"


@dataclass
class Relationship:
    rel_id: str
    subject_entity_id: str
    predicate: RelationType
    object_entity_id: str
    source_unit_id: str
    doc_id: str
    confidence: float = 0.0
    evidence_text: str = ""   # short snippet justifying the edge, for audit/debug


@dataclass
class RetrievedEvidence:
    """What the Evidence Aggregator hands to the LLM: a unit plus why it
    was retrieved (vector / bm25 / graph-expansion) and its provenance.
    """
    unit: DocUnit
    retrieval_source: str      # "vector" | "bm25" | "graph"
    score: float
    hop_distance: int = 0      # 0 = seed chunk, 1+ = graph-expanded
    linking_relationship: Optional[Relationship] = None
