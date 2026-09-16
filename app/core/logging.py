"""Standardized logging setup for the Clinic RAG application."""
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LEVEL = logging.INFO


def setup_logging(level: int = _LEVEL, stream=None) -> None:
    """Configure the root logger once with a consistent format."""
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with the app's default settings."""
    setup_logging()
    return logging.getLogger(name)