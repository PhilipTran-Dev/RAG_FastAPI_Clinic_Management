"""Abstract interface for document parsers."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class BaseParser(ABC):
    """Convert a single source file (e.g. PDF) into plain Markdown text."""

    name: str = "base"

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir

    @abstractmethod
    def parse(self, file_path: Path) -> str:
        """Return the extracted document content as Markdown.

        Implementing classes may also persist the result to ``output_dir``.
        """

    def save(self, file_path: Path, content: str) -> Path:
        """Write ``content`` to ``output_dir`` under the source file's stem."""
        if self.output_dir is None:
            raise ValueError("output_dir not configured for this parser")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / (file_path.stem + ".md")
        out_path.write_text(content, encoding="utf-8")
        return out_path