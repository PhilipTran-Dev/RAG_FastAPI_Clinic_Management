"""Fast, local PDF-to-Markdown extraction using Docling (OCR disabled)."""
from pathlib import Path
from typing import Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from app.core.config import PARSED_MARKDOWN_DIR
from app.core.logging import get_logger
from app.parsers.base import BaseParser

logger = get_logger(__name__)


class DoclingParser(BaseParser):
    """Lightweight parser tuned for sub-minute extraction of text PDFs."""

    name = "docling"

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        super().__init__(output_dir or PARSED_MARKDOWN_DIR)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def parse(self, file_path: Path) -> str:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Missing source file: {file_path}")

        logger.info("Parsing %s (Fast / No-OCR)", file_path)
        result = self._converter.convert(str(file_path))
        markdown_content = result.document.export_to_markdown()
        out_path = self.save(file_path, markdown_content)
        logger.info("Saved markdown to %s", out_path)
        return markdown_content


def main(source_dir: Path, document_stem: str | None = None) -> Path:
    """CLI helper: parse a raw PDF and save Markdown to parsed_markdown/."""
    from app.core.config import RAW_DOCS_DIR

    sources = sorted(source_dir.glob("*.pdf"))
    if document_stem:
        sources = [s for s in sources if s.stem == document_stem]
    if not sources:
        raise SystemExit(f"No PDF documents found in {source_dir}")

    parser = DoclingParser()
    for source in sources:
        parser.parse(source)
    return PARSED_MARKDOWN_DIR


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Docling PDF -> Markdown (No OCR)")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/raw_docs"),
        help="Directory containing source PDFs",
    )
    parser.add_argument("--stem", type=str, default=None, help="Optional file stem to parse")
    args = parser.parse_args()
    main(args.source_dir, args.stem)