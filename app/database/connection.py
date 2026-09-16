"""PostgreSQL + pgvector connection lifecycle management."""
from contextlib import contextmanager
from typing import Iterator

import psycopg
from pgvector.psycopg import register_vector

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def connect(settings: Settings):
    """Open a psycopg connection with pgvector support registered."""
    logger.info("Connecting to PostgreSQL at %s:%s/%s",
                settings.DB_HOST, settings.DB_PORT, settings.DB_NAME)
    conn = psycopg.connect(settings.db_dsn, autocommit=False)
    register_vector(conn)
    return conn


@contextmanager
def get_connection(settings: Settings) -> Iterator[psycopg.Connection]:
    """Context manager that yields a connection and always closes it."""
    conn = connect(settings)
    try:
        yield conn
    finally:
        conn.close()
        logger.debug("PostgreSQL connection closed")


@contextmanager
def get_cursor(settings: Settings) -> Iterator[psycopg.Cursor]:
    """Context manager wrapping connection + cursor, committing on success.

    On an exception the transaction is rolled back and re-raised.
    """
    with get_connection(settings) as conn:
        with conn.cursor() as cur:
            try:
                yield cur
            except Exception:
                conn.rollback()
                raise
            else:
                conn.commit()