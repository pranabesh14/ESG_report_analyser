# Multimodal PDF RAG Knowledge Base

A modular RAG pipeline for PDFs containing text, images, charts, and tables — built to solve one specific problem: **connecting information that's scattered across pages**.

If page 6 describes a company's ESG position and page 35 shows a projection chart for the resulting carbon reduction, a standard chunk-based RAG treats these as unrelated, because they share almost no vocabulary. This pipeline links them through canonical entity resolution and graph-expansion retrieval, so a query about one surfaces the other automatically.

> **Status**: CLI + Python API only. No web UI.

## Why this exists

Standard RAG (chunk → embed → top-k similarity) breaks down on long structured documents in three specific ways:

1. **Charts become noise.** A projection graph gets reduced to a vague caption instead of the actual numbers on its axes.
2. **Related content stays disconnected.** Two mentions of the same concept in different words ("ESG" vs. "carbon footprint") never get linked, so retrieval can't traverse between them.
3. **Multi-year comparisons don't compose.** Nothing tracks which report-year a figure came from, so "how has this trended since 2022" has no chronological anchor.

This pipeline addresses all three directly — see [Architecture](#architecture) below.

## Key features

- **Structured chart extraction** — charts are parsed into axis labels, units, series, data points, and a one-line claim with a confidence score, not just captioned.
- **Canonical entity resolution** — a cheapest-first cascade (exact match → embedding similarity auto-merge → LLM disambiguation → new entity) collapses synonymous mentions into one graph node, across documents and across process restarts.
- **Hybrid graph retrieval** — vector search + BM25 seed the query, then a 1–2 hop graph expansion pulls in linked content that shares no vocabulary with the query at all.
- **Confidence-aware generation** — low-confidence chart/table extractions are flagged and explicitly hedged in the final answer instead of stated as fact.
- **Cross-year trend support** — documents are tagged with a reporting year; entity resolution state persists to disk, so ingesting four years of reports one at a time still resolves "carbon footprint" to a single entity and returns results in chronological order.
- **Swappable LLM provider** — factory pattern supports Anthropic and Azure OpenAI; each pipeline stage can use a different model if needed.

## Architecture

```
PDFs → Ingestion → Entity & graph layer → Storage → Retrieval → Generation
```

| Phase | Modules |
|---|---|
| **Ingestion** | `ingestion/pdf_parser.py` (layout via PyMuPDF), `ingestion/vision_extractor.py` (charts → structured data) |
| **Entity & graph layer** | `entities/extraction.py`, `entities/resolution.py` (canonical entities), `relationships/extraction.py`, `entities/contextual_summarizer.py` |
| **Storage** | `storage/relational_store.py` (SQLite/Postgres — units, entities, relationship edges), `storage/vector_store.py` (dual FAISS index), `storage/bm25_store.py` |
| **Retrieval** | `retrieval/hybrid_retriever.py` (vector + BM25 + graph expansion), `retrieval/evidence_aggregator.py` |
| **Generation** | `generation/answer_generator.py` (forces page citations, hedges low-confidence data) |
| **Core** | `core/schemas.py` (shared data model), `core/llm_provider.py` (Anthropic / Azure OpenAI factory), `config/settings.py` |
| **Orchestration** | `pipeline.py`, `cli.py` |

Storage is deliberately boring: relationships are a plain `(subject_id, predicate, object_id)` table, not a dedicated graph database — SQLite for dev, same code path works against Postgres.

## Installation

```bash
git clone <this-repo>
cd pdf_rag_kb
pip install -r requirements.txt
```

Set up your LLM provider:

```bash
# Anthropic (default)
export ANTHROPIC_API_KEY=sk-ant-...

# OR Azure OpenAI
export RAG_LLM_PROVIDER=azure_openai
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
```

## Usage

### CLI

```bash
# Ingest documents (doc-year is required for cross-document trend queries)
python -m pdf_rag_kb.cli ingest report_2022.pdf --doc-year 2022
python -m pdf_rag_kb.cli ingest report_2023.pdf --doc-year 2023
python -m pdf_rag_kb.cli ingest report_2024.pdf --doc-year 2024
python -m pdf_rag_kb.cli ingest report_2025.pdf --doc-year 2025

# Query
python -m pdf_rag_kb.cli query "How has the carbon footprint trended from 2022 to 2025?"
```

### Python API

```python
from pdf_rag_kb.pipeline import RAGKnowledgeBasePipeline

pipeline = RAGKnowledgeBasePipeline()
pipeline.ingest_document("report_2023.pdf", doc_year=2023)
answer = pipeline.query("What steps is the org taking to reduce its carbon footprint, and what's the projected impact?")
print(answer)
```

## Configuration

All tunables live in `config/settings.py`, overridable via environment variables — retrieval weights (vector/BM25/graph), entity-resolution similarity thresholds, chunking limits, and storage locations. See the file directly for the full list.

## Known limitations

- **No dedicated table extractor yet.** The schema (`TableData`) is wired through the whole pipeline, but tables currently fall through the vision extractor's generic path. A `camelot`/`pdfplumber`-based extractor is the natural next addition.
- **Anthropic has no embeddings endpoint** — `AnthropicProvider.embed()` falls back to a local `sentence-transformers` model. Swap in Voyage AI or another embeddings provider for production use.
- **No batching/async on LLM calls** — ingestion is sequential per unit. Fine for single-document testing; worth parallelizing for large corpora.
- **No topic/domain tagging** — retrieval can find ESG-specific content via a targeted query, but there's no ingestion-time filter to extract "only ESG data" from a mixed annual report. Would need a lightweight per-unit classification pass.
- **No web UI.** CLI and Python API only.
- **No automated test suite.** Core logic (entity resolution, cross-year linking, BM25 accumulation) has been manually verified with targeted scripts, not a `pytest` suite.

## Roadmap

- [ ] Dedicated table extraction module
- [ ] ESG/topic tagging for filtered extraction
- [ ] Async/batched ingestion for large corpora
- [ ] `pytest` suite covering entity resolution, retrieval, and generation
- [ ] Voyage AI embeddings option

## License

Not yet specified — add a `LICENSE` file before publishing publicly.
