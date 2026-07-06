"""
Relational storage for units, canonical entities, and relationships.

Deliberately NOT a graph database. A `relationships` table with
(subject_id, predicate, object_id) plus SQL joins covers 1-2 hop traversal
fine at this scale, and keeps ops simple (SQLite for dev, same code path
works against Postgres via the DSN). Migrate to Neo4j only if multi-hop
query patterns genuinely outgrow SQL joins.
"""
from __future__ import annotations

import json
import logging

from sqlalchemy import (Column, Float, Integer, String, Text, create_engine, select)
from sqlalchemy.orm import declarative_base, sessionmaker

from pdf_rag_kb.core.schemas import (CanonicalEntity, DocUnit, Relationship, UnitType)

logger = logging.getLogger(__name__)
Base = declarative_base()


class UnitRow(Base):
    __tablename__ = "units"
    unit_id = Column(String, primary_key=True)
    doc_id = Column(String, index=True)
    page_number = Column(Integer)
    doc_year = Column(Integer, index=True, nullable=True)
    unit_type = Column(String)
    raw_text = Column(Text)
    section_heading = Column(Text)
    contextual_summary = Column(Text)
    entity_ids_json = Column(Text)     # JSON list
    chart_data_json = Column(Text)     # JSON or NULL
    table_data_json = Column(Text)     # JSON or NULL


class EntityEmbeddingRow(Base):
    """Persists the EntityResolver's representative embedding per canonical
    entity so resolution state survives across separate ingest_document()
    calls / process restarts -- required for cross-document (e.g.
    cross-year) entity matching to work at all.
    """
    __tablename__ = "entity_embeddings"
    entity_id = Column(String, primary_key=True)
    embedding_json = Column(Text)   # JSON list[float]


class EntityRow(Base):
    __tablename__ = "entities"
    entity_id = Column(String, primary_key=True)
    canonical_name = Column(String, index=True)
    entity_type = Column(String, index=True)
    aliases_json = Column(Text)
    doc_ids_json = Column(Text)


class RelationshipRow(Base):
    __tablename__ = "relationships"
    rel_id = Column(String, primary_key=True)
    subject_entity_id = Column(String, index=True)
    predicate = Column(String, index=True)
    object_entity_id = Column(String, index=True)
    source_unit_id = Column(String)
    doc_id = Column(String, index=True)
    confidence = Column(Float)
    evidence_text = Column(Text)


class RelationalStore:
    def __init__(self, dsn: str):
        self._engine = create_engine(dsn)
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)

    # -- writes ------------------------------------------------------
    def save_units(self, units: list[DocUnit]) -> None:
        with self._Session() as session:
            for u in units:
                session.merge(UnitRow(
                    unit_id=u.unit_id, doc_id=u.doc_id, page_number=u.page_number,
                    doc_year=u.doc_year,
                    unit_type=u.unit_type.value, raw_text=u.raw_text,
                    section_heading=u.section_heading, contextual_summary=u.contextual_summary,
                    entity_ids_json=json.dumps(u.entity_ids),
                    chart_data_json=json.dumps(u.chart_data.__dict__) if u.chart_data else None,
                    table_data_json=json.dumps(u.table_data.__dict__) if u.table_data else None,
                ))
            session.commit()

    def save_entities(self, entities: list[CanonicalEntity]) -> None:
        with self._Session() as session:
            for e in entities:
                session.merge(EntityRow(
                    entity_id=e.entity_id, canonical_name=e.canonical_name,
                    entity_type=e.entity_type, aliases_json=json.dumps(list(e.aliases)),
                    doc_ids_json=json.dumps(list(e.doc_ids)),
                ))
            session.commit()

    def save_entity_embeddings(self, embeddings: dict[str, list[float]]) -> None:
        """Persist the EntityResolver's representative embedding per entity.
        Must be called after save_entities on every ingestion run, or
        cross-document/cross-year resolution silently stops working on the
        next process restart.
        """
        with self._Session() as session:
            for entity_id, embedding in embeddings.items():
                session.merge(EntityEmbeddingRow(entity_id=entity_id, embedding_json=json.dumps(embedding)))
            session.commit()

    def load_all_entities_with_embeddings(self) -> tuple[list[CanonicalEntity], dict[str, list[float]]]:
        """Reload full resolver state so a fresh EntityResolver can be
        seeded with everything learned from previously-ingested documents.
        """
        with self._Session() as session:
            entity_rows = session.execute(select(EntityRow)).scalars().all()
            embedding_rows = session.execute(select(EntityEmbeddingRow)).scalars().all()

            entities = [
                CanonicalEntity(
                    entity_id=r.entity_id, canonical_name=r.canonical_name, entity_type=r.entity_type,
                    aliases=set(json.loads(r.aliases_json or "[]")),
                    mention_ids=[],
                    doc_ids=set(json.loads(r.doc_ids_json or "[]")),
                )
                for r in entity_rows
            ]
            embeddings = {r.entity_id: json.loads(r.embedding_json) for r in embedding_rows}
            return entities, embeddings

    def save_relationships(self, relationships: list[Relationship]) -> None:
        with self._Session() as session:
            for r in relationships:
                session.merge(RelationshipRow(
                    rel_id=r.rel_id, subject_entity_id=r.subject_entity_id,
                    predicate=r.predicate.value if hasattr(r.predicate, "value") else r.predicate,
                    object_entity_id=r.object_entity_id, source_unit_id=r.source_unit_id,
                    doc_id=r.doc_id, confidence=r.confidence, evidence_text=r.evidence_text,
                ))
            session.commit()

    # -- reads for retrieval ------------------------------------------
    def get_units_by_ids(self, unit_ids: list[str]) -> list[DocUnit]:
        if not unit_ids:
            return []
        with self._Session() as session:
            rows = session.execute(select(UnitRow).where(UnitRow.unit_id.in_(unit_ids))).scalars().all()
            return [self._row_to_unit(r) for r in rows]

    def get_units_by_entity(self, entity_id: str, exclude_unit_id: str = "") -> list[DocUnit]:
        """All units that mention a given canonical entity -- the core
        graph-expansion query."""
        with self._Session() as session:
            rows = session.execute(select(UnitRow)).scalars().all()
            matched = [
                r for r in rows
                if entity_id in json.loads(r.entity_ids_json or "[]") and r.unit_id != exclude_unit_id
            ]
            return [self._row_to_unit(r) for r in matched]

    def get_units_by_entity_across_years(self, entity_id: str) -> list[DocUnit]:
        """Same as get_units_by_entity, but sorted chronologically by
        doc_year. This is the query a trend/timeseries question actually
        needs -- e.g. "how has the carbon footprint metric evolved
        2022-2025" -- so the generator sees the data points in order
        rather than in arbitrary retrieval order.
        """
        units = self.get_units_by_entity(entity_id)
        return sorted(units, key=lambda u: (u.doc_year is None, u.doc_year or 0))

    def get_relationships_for_entity(self, entity_id: str, min_confidence: float = 0.0) -> list[Relationship]:
        with self._Session() as session:
            rows = session.execute(
                select(RelationshipRow).where(
                    (RelationshipRow.subject_entity_id == entity_id) |
                    (RelationshipRow.object_entity_id == entity_id)
                )
            ).scalars().all()
            out = []
            for r in rows:
                if r.confidence is not None and r.confidence < min_confidence:
                    continue
                out.append(Relationship(
                    rel_id=r.rel_id, subject_entity_id=r.subject_entity_id,
                    predicate=r.predicate, object_entity_id=r.object_entity_id,
                    source_unit_id=r.source_unit_id, doc_id=r.doc_id,
                    confidence=r.confidence or 0.0, evidence_text=r.evidence_text or "",
                ))
            return out

    @staticmethod
    def _row_to_unit(row: UnitRow) -> DocUnit:
        from pdf_rag_kb.core.schemas import ChartData, TableData
        chart_data = ChartData(**json.loads(row.chart_data_json)) if row.chart_data_json else None
        table_data = TableData(**json.loads(row.table_data_json)) if row.table_data_json else None
        return DocUnit(
            unit_id=row.unit_id, doc_id=row.doc_id, page_number=row.page_number,
            unit_type=UnitType(row.unit_type), raw_text=row.raw_text or "",
            section_heading=row.section_heading or "", contextual_summary=row.contextual_summary or "",
            entity_ids=json.loads(row.entity_ids_json or "[]"),
            chart_data=chart_data, table_data=table_data, doc_year=row.doc_year,
        )
