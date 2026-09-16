"""CLI evaluation script for semantic (& hybrid) retrieval quality."""
import argparse

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.hybrid_search import hybrid_search
from app.rag.retriever import retrieve_context

logger = get_logger(__name__)

DEFAULT_QUERIES = [
    "Chỉ định và kỹ thuật nong van hai lá bằng bóng Inoue được thực hiện như thế nào?",
    "Biến chứng và cách xử trí khi đặt bóng đối xung động mạch chủ IABP?",
    "Điều trị vảy nến thể mảng bằng thuốc nào?",
    "Chống chỉ định của thuốc chống đông trong nhồi máu cơ tim?",
]


def run_evaluation(
    query: str,
    top_k: int,
    specialty: str | None,
    use_hybrid: bool,
) -> None:
    settings = get_settings()

    if use_hybrid:
        results = hybrid_search(
            query, top_k=top_k, specialty=specialty, settings=settings
        )
    else:
        results = retrieve_context(
            query, top_k=top_k, specialty=specialty, settings=settings
        )

    print(f"\n=== CÂU HỎI: {query} ===")
    print(f"Phương thức: {'HYBRID (FTS + Dense)' if use_hybrid else 'DENSE VECTOR'}")
    for idx, res in enumerate(results, 1):
        score = f"final_score: {res.final_score:.4f}" if use_hybrid \
            else f"cosine_distance: {res.distance:.4f}"
        print(
            f"\n[Kết quả {idx}] {res.doc_id} | Chuyên khoa: {res.specialty}\n{score}"
        )
        print(f"Nội dung trích đoạn:\n{res.content[:350]}...\n" + "-" * 50)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate semantic/hybrid retrieval against the pgvector store"
    )
    parser.add_argument("--query", type=str, default=None, help="Single query string")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results (default 3)")
    parser.add_argument(
        "--specialty",
        type=str,
        default=None,
        help="Optional specialty filter (tim_mach, da_lieu, noi_khoa)",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Use hybrid (full-text + dense) search instead of dense-only",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    queries = [args.query] if args.query else DEFAULT_QUERIES
    for q in queries:
        run_evaluation(q, args.top_k, args.specialty, args.hybrid)


if __name__ == "__main__":
    main()