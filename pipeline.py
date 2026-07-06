"""
Top-level orchestrator. Wires together every module in the architecture
diagram. Each stage is independently testable/swappable -- this module's
only job is sequencing and passing data between them.
"""
from __future__ import annotations

import logging

from pdf_rag_kb.config.settings import AppConfig, load_config
from pdf_rag_kb.core.llm_provider import get_llm_provider
from pdf_rag_kb.core.schemas import DocUnit, UnitType, new_id
from pdf_rag_kb.entities.contextual_summarizer import ContextualSummarizer
from pdf_rag_kb.entities.extraction import EntityExtractor
from pdf_rag_kb.entities.resolution import EntityResolver
from pdf_rag_kb.generation.answer_generator import AnswerGenerator
from pdf_rag_kb.ingestion.pdf_parser import PDFParser
from pdf_rag_kb.ingestion.vision_extractor import VisionExtractor
from pdf_rag_kb.relationships.extraction import RelationshipExtractor
from pdf_rag_kb.retrieval.evidence_aggregator import EvidenceAggregator
from pdf_rag_kb.retrieval.hybrid_retriever import HybridGraphRetriever
from pdf_rag_kb.storage.bm25_store import BM25Store
from pdf_rag_kb.storage.relational_store import RelationalStore
from pdf_rag_kb.storage.vector_store import DualFAISSStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class RAGKnowledgeBasePipeline:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        self.llm = get_llm_provider(self.config.llm)

        self.pdf_parser_cls = PDFParser
        self.vision_extractor = VisionExtractor(self.llm)
        self.entity_extractor = EntityExtractor(self.llm)
        self.entity_resolver = EntityResolver(self.llm, self.config.entity_resolution)
        self.relationship_extractor = RelationshipExtractor(self.llm)
        self.contextual_summarizer = ContextualSummarizer(self.llm)

        self.relational_store = RelationalStore(self.config.storage.postgres_dsn)
        self.vector_store = DualFAISSStore(self.config.storage.faiss_index_dir)
        self.bm25_store = BM25Store()

        # Seed the resolver with entities/embeddings from documents ingested
        # in prior process runs. Without this, a mention like "carbon
        # footprint" in a newly-ingested 2024 report has nothing to match
        # against and silently becomes a brand-new entity instead of
        # merging with the one built from 2022/2023 -- breaking cross-year
        # trend queries with no error raised anywhere.
        existing_entities, existing_embeddings = self.relational_store.load_all_entities_with_embeddings()
        if existing_entities:
            self.entity_resolver.load_existing_state(existing_entities, existing_embeddings)

        self.retriever = HybridGraphRetriever(
            self.llm, self.vector_store, self.bm25_store, self.relational_store, self.config.retrieval,
        )
        self.evidence_aggregator = EvidenceAggregator(self.config.min_confidence_for_citation)
        self.answer_generator = AnswerGenerator(self.llm)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest_document(self, pdf_path: str, doc_id: str | None = None, doc_year: int | None = None) -> str:
        """doc_year should be the report's fiscal/reporting year (e.g. 2023),
        not the ingestion date. It's required for any cross-document trend
        query -- without it, retrieval has no way to chronologically order
        the same metric pulled from four different annual reports."""
        doc_id = doc_id or new_id("doc")
        logger.info("Ingesting %s as doc_id=%s (doc_year=%s)", pdf_path, doc_id, doc_year)

        parser = self.pdf_parser_cls(doc_id)
        text_units, images = parser.parse(pdf_path)

        chart_units = [self.vision_extractor.process_image(img) for img in images]
        all_units: list[DocUnit] = text_units + chart_units
        for unit in all_units:
            unit.doc_year = doc_year

        all_mentions = []
        mentions_by_unit: dict[str, list] = {}
        for unit in all_units:
            mentions = self.entity_extractor.extract(unit)
            mentions_by_unit[unit.unit_id] = mentions
            all_mentions.extend(mentions)

        mention_to_entity = self.entity_resolver.resolve_batch(all_mentions)
        entities_by_id = {e.entity_id: e for e in self.entity_resolver.get_all_entities()}

        for unit in all_units:
            unit_mentions = mentions_by_unit.get(unit.unit_id, [])
            unit.entity_ids = list({mention_to_entity[m.mention_id] for m in unit_mentions
                                     if m.mention_id in mention_to_entity})

        all_relationships = []
        for unit in all_units:
            all_relationships.extend(self.relationship_extractor.extract_explicit_refs(unit))
            entities_in_unit = [entities_by_id[eid] for eid in unit.entity_ids if eid in entities_by_id]
            all_relationships.extend(self.relationship_extractor.extract_semantic(unit, entities_in_unit))

        for unit in all_units:
            entities_in_unit = [entities_by_id[eid] for eid in unit.entity_ids if eid in entities_by_id]
            rels_for_unit = [r for r in all_relationships if r.source_unit_id == unit.unit_id]
            unit.contextual_summary = self.contextual_summarizer.summarize(unit, entities_in_unit, rels_for_unit)

        self.relational_store.save_units(all_units)
        self.relational_store.save_entities(list(entities_by_id.values()))
        self.relational_store.save_entity_embeddings(self.entity_resolver.get_updated_embeddings())
        self.relational_store.save_relationships(all_relationships)

        self._index_embeddings(all_units)

        logger.info(
            "Ingested doc_id=%s: %d units, %d entities, %d relationships",
            doc_id, len(all_units), len(entities_by_id), len(all_relationships),
        )
        return doc_id

    def _index_embeddings(self, units: list[DocUnit]) -> None:
        indexable = [u for u in units if u.unit_type != UnitType.HEADING and (u.raw_text or u.chart_data)]
        if not indexable:
            return

        raw_texts = [u.raw_text or (u.chart_data.one_line_claim if u.chart_data else "") for u in indexable]
        contextual_texts = [u.contextual_summary or raw_texts[i] for i, u in enumerate(indexable)]
        unit_ids = [u.unit_id for u in indexable]

        raw_embeddings = self.llm.embed(raw_texts)
        contextual_embeddings = self.llm.embed(contextual_texts)

        self.vector_store.add_raw(unit_ids, raw_embeddings)
        self.vector_store.add_contextual(unit_ids, contextual_embeddings)
        self.vector_store.save()

        self.bm25_store.add_documents(unit_ids, raw_texts)
        self.bm25_store.save()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query(self, question: str) -> str:
        evidence = self.retriever.retrieve(question)
        aggregated = self.evidence_aggregator.aggregate(evidence)
        return self.answer_generator.generate(question, aggregated)
