"""Header-aware recursive Markdown chunker."""
import re
from typing import List

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_HEADING_RE = re.compile(r"\n(?=#{1,3}\s)")


def chunk_markdown(path: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    """Split a Markdown file into overlapping chunks at heading boundaries.

    Each top-level section delimited by a ``#``, ``##`` or ``###`` heading is
    treated as a candidate chunk and further split with ``max_chars - overlap``
    sliding windows when it exceeds ``max_chars``.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    sections = _HEADING_RE.split(text)
    chunks: List[str] = []

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        if len(sec) <= max_chars:
            chunks.append(sec)
        else:
            start = 0
            step = max_chars - overlap
            while start < len(sec):
                chunk = sec[start:start + max_chars]
                chunks.append(chunk)
                start += step

    logger.info("Produced %d chunks from %s", len(chunks), path)
    return chunks