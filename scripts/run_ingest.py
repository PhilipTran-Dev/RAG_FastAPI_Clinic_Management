"""CLI entry point to ingest specific documents into the vector store."""
import argparse
from pathlib import Path

from app.core.config import PARSED_MARKDOWN_DIR, RAW_DOCS_DIR, get_settings
from app.core.logging import get_logger
from app.ingestion.pipeline import ingest_directory, ingest_markdown_file

logger = get_logger(__name__)

VALID_SPECIALTIES = {"tim_mach", "da_lieu", "noi_khoa", "chung"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest Markdown documents into Clinic RAG pgvector store"
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Path to a Markdown file (data/parsed_markdown/*.md)",
    )
    parser.add_argument(
        "--specialty",
        type=str,
        default=None,
        choices=sorted(VALID_SPECIALTIES),
        help="Specialty label stored with every chunk",
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        default=None,
        help="Document identifier stored with every chunk",
    )
    parser.add_argument(
        "--audience",
        type=str,
        default="bac_si",
        help="Target audience (default: bac_si)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest every Markdown document under data/parsed_markdown/",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()

    if args.all:
        logger.info("Bulk-ingesting all documents from %s", PARSED_MARKDOWN_DIR)
        results = ingest_directory(settings=settings)
        for result in results:
            logger.info(
                "Ingested %s (%s): %d chunks, %d entities",
                result.doc_id, result.specialty, result.total_chunks, result.entities_found,
            )
        return

    if not args.file:
        raise SystemExit(
            "Provide --file (and --specialty/--doc-id) or use --all for bulk ingestion."
        )
    if not args.file.exists():
        raise SystemExit(f"File not found: {args.file}")
    if not args.specialty or not args.doc_id:
        raise SystemExit("--specialty and --doc-id are required when using --file.")

    result = ingest_markdown_file(
        args.file,
        doc_id=args.doc_id,
        specialty=args.specialty,
        target_audience=args.audience,
        settings=settings,
    )
    logger.info(
        "Ingested %s (%s): %d chunks, %d entities",
        result.doc_id, result.specialty, result.total_chunks, result.entities_found,
    )


if __name__ == "__main__":
    main()