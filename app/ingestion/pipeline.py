"""Ingestion orchestrator: Parse -> Chunk -> Extract -> Embed -> Insert."""
from dataclasses import dataclass
from pathlib import Path

from psycopg.types.json import Jsonb

from app.core.config import PARSED_MARKDOWN_DIR, RAW_DOCS_DIR, Settings, get_settings
from app.core.logging import get_logger
from app.database.connection import get_connection
from app.database.schema import run_migrations
from app.ingestion.embedder import embed_texts
from app.ingestion.entity_extractor import ExtractedEntities, extract_entities
from app.ingestion.splitter import chunk_markdown

logger = get_logger(__name__)

INSERT_SQL = """
    INSERT INTO clinic_knowledge_nodes
        (doc_id, specialty, target_audience, content, dense_embedding, metadata)
    VALUES (%s, %s, %s, %s, %s, %s::jsonb);
"""


@dataclass
class IngestResult:
    doc_id: str
    specialty: str
    total_chunks: int
    entities_found: int


def ingest_markdown_file(
    markdown_path: Path,
    doc_id: str,
    specialty: str,
    target_audience: str = "bac_si",
    settings: Settings | None = None,
) -> IngestResult:
    """Chunk, embed, and insert a single Markdown file into pgvector."""
    settings = settings or get_settings()
    run_migrations(settings)

    markdown_path = Path(markdown_path)
    chunks = chunk_markdown(
        str(markdown_path),
        max_chars=settings.CHUNK_MAX_CHARS,
        overlap=settings.CHUNK_OVERLAP,
    )
    total = len(chunks)
    logger.info("Embedding %d chunks from %s", total, markdown_path.name)
    embeddings = embed_texts(chunks, settings)

    entities_count = 0
    with get_connection(settings) as conn:
        with conn.cursor() as cur:
            for idx, (content, embedding) in enumerate(zip(chunks, embeddings), 1):
                entities: ExtractedEntities = extract_entities(content)
                entities_count += len(entities.icd10_codes) + len(entities.medications)
                metadata = {"entities": entities.as_dict}
                cur.execute(
                    INSERT_SQL,
                    (doc_id, specialty, target_audience, content, embedding, Jsonb(metadata)),
                )
                if idx % 25 == 0 or idx == total:
                    pct = idx / total * 100
                    logger.info("  -> [%5.1f%%] Ingested %d/%d chunks", pct, idx, total)
        conn.commit()  # Persist all chunks to the database

    logger.info("Pipeline complete for %s (%d chunks, %d entities)",
                doc_id, total, entities_count)
    return IngestResult(
        doc_id=doc_id,
        specialty=specialty,
        total_chunks=total,
        entities_found=entities_count,
    )


def ingest_directory(
    raw_docs_dir: Path = RAW_DOCS_DIR,
    parsed_dir: Path = PARSED_MARKDOWN_DIR,
    settings: Settings | None = None,
) -> list[IngestResult]:
    """Ingest every Markdown file in ``parsed_dir`` using the matching PDF stem."""
    settings = settings or get_settings()
    results = []
    for md_path in sorted(parsed_dir.glob("*.md")):
        source = next(
            (f for f in raw_docs_dir.glob("*.pdf") if f.stem == md_path.stem),
            None,
        )
        specialty = _infer_specialty(md_path.stem)
        doc_id = md_path.stem
        logger.info("Ingesting %s (specialty=%s)", md_path.name, specialty)
        results.append(
            ingest_markdown_file(md_path, doc_id, specialty, settings=settings)
        )
    return results


def _infer_specialty(stem: str) -> str:
    if "tim_mach" in stem:
        return "tim_mach"
    if "da_lieu" in stem:
        return "da_lieu"
    if "noi_khoa" in stem:
        return "noi_khoa"
    return "chung"


if __name__ == "__main__":
    ingest_directory()