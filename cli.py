"""
Command-line entry point.

Usage:
    python -m pdf_rag_kb.cli ingest path/to/doc.pdf
    python -m pdf_rag_kb.cli query "How will the org reduce its carbon footprint?"
"""
from __future__ import annotations

import argparse
import sys

from pdf_rag_kb.pipeline import RAGKnowledgeBasePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal PDF RAG Knowledge Base")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_p = subparsers.add_parser("ingest", help="Ingest a PDF into the knowledge base")
    ingest_p.add_argument("pdf_path")
    ingest_p.add_argument("--doc-id", default=None)
    ingest_p.add_argument("--doc-year", type=int, default=None,
                           help="Reporting year of the source document (e.g. 2023). "
                                "Required for cross-document trend queries.")

    query_p = subparsers.add_parser("query", help="Ask a question against the knowledge base")
    query_p.add_argument("question")

    args = parser.parse_args()
    pipeline = RAGKnowledgeBasePipeline()

    if args.command == "ingest":
        doc_id = pipeline.ingest_document(args.pdf_path, args.doc_id, args.doc_year)
        print(f"Ingested. doc_id={doc_id}")
    elif args.command == "query":
        answer = pipeline.query(args.question)
        print(answer)


if __name__ == "__main__":
    sys.exit(main())
