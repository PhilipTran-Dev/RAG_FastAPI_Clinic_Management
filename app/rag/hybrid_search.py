"""Hybrid retrieval: full-text (tsvector) + dense vector with reciprocal reranking."""
import math
from dataclasses import dataclass
from typing import List, Optional

from app.core.config import Settings, get_settings
from app.database.connection import get_connection
from app.core.logging import get_logger
from app.rag.retriever import SearchResult, retrieve_context

logger = get_logger(__name__)


@dataclass
class HybridResult:
    result: SearchResult
    dense_rank: int
    fts_rank: int | None
    final_score: float

    @property
    def content(self) -> str:
        return self.result.content

    @property
    def doc_id(self) -> str:
        return self.result.doc_id

    @property
    def specialty(self) -> str:
        return self.result.specialty


def _fts_results(query: str, top_k: int, specialty: Optional[str]) -> List[tuple]:
    sql = """
        SELECT id, doc_id, specialty, content, ts_rank_cd(
            to_tsvector('simple', content),
            plainto_tsquery('simple', %s)
        ) AS rank
        FROM clinic_knowledge_nodes
    """
    params = [query]
    if specialty and specialty != "all":
        sql += " WHERE specialty = %s"
        params.append(specialty)
    sql += " ORDER BY rank DESC NULLS LAST LIMIT %s;"
    params.append(top_k)

    with get_connection(get_settings()) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    return [r for r in rows if r[4] is not None]


def _reciprocal_rank(r: int, k: int = 60) -> float:
    return 1.0 / (k + r)


def hybrid_search(
    query: str,
    top_k: int = 3,
    c: float = 0.5,
    specialty: Optional[str] = None,
    target_audience: Optional[str] = None,
    settings: Settings | None = None,
) -> List[HybridResult]:
    """Run dense + full-text search and fuse results via reciprocal rank fusion."""
    settings = settings or get_settings()
    dense_results = retrieve_context(
        query,
        top_k=top_k * 2,
        specialty=specialty,
        target_audience=target_audience,
        settings=settings,
    )
    fts_results = _fts_results(query, top_k * 2, specialty)

    dense_id_map = {r.id: i for i, r in enumerate(dense_results, 1)}
    fts_id_map = {r[0]: i for i, r in enumerate(fts_results, 1)}

    candidates: dict[int, HybridResult] = {}
    for r in dense_results:
        fts_rank = fts_id_map.get(r.id)
        final_score = c * _reciprocal_rank(dense_id_map[r.id])
        if fts_rank:
            final_score += (1 - c) * _reciprocal_rank(fts_rank)
        candidates[r.id] = HybridResult(
            result=r, dense_rank=dense_id_map[r.id],
            fts_rank=fts_rank, final_score=final_score,
        )

    ranked = sorted(candidates.values(), key=lambda h: h.final_score, reverse=True)
    return ranked[:top_k]