"""Healthcheck & chunk inspection for the pgvector knowledge store."""
import sys

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.database.connection import get_connection

logger = get_logger(__name__)


def print_diagnostic_report(settings: Settings) -> None:
    """Print a structured diagnostic report across all ingested specialties."""
    with get_connection(settings) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM clinic_knowledge_nodes;")
            total = cur.fetchone()[0]

            print("=" * 70)
            print(" 🏥 BÁO CÁO CHẨN ĐOÁN CƠ SỞ DỮ LIỆU PGVECTOR")
            print(f"   Host: {settings.DB_HOST} | Port: {settings.DB_PORT} | DB: {settings.DB_NAME}")
            print("=" * 70)
            print(f" • Tổng số chunks: {total}")

            cur.execute("""
                SELECT specialty, doc_id, target_audience, COUNT(*) AS chunk_count
                FROM clinic_knowledge_nodes
                GROUP BY specialty, doc_id, target_audience
                ORDER BY specialty, doc_id;
            """)
            rows = cur.fetchall()
            print("\n--- PHÂN BỐ CHUNKS THEO CHUYÊN KHOA ---")
            for spec, doc, audience, count in rows:
                print(f" • {spec:<12} | {doc:<28} | {audience:<10} | {count} chunks")

            if total == 0:
                print("\n⚠  KHÔNG CÓ DỮ LIỆU - hãy chạy python scripts/run_ingest.py")
                return

            print("\n--- NHẬT KÝ TÓM TẮT 3 BẢN GHI MẪU ---")
            cur.execute("""
                SELECT id, doc_id, specialty, target_audience,
                       LEFT(content, 120) AS excerpt, created_at
                FROM clinic_knowledge_nodes
                ORDER BY id
                LIMIT 3;
            """)
            for idx, row in enumerate(cur.fetchall(), 1):
                print(f"\n[BẢN GHI #{idx}] id={row[0]} | doc={row[1]} | spec={row[2]}")
                print(f"  Đối tượng: {row[3]} | Tạo lúc: {row[5]}")
                print(f"  \"{row[4]}...\"")

            print("\n--- KIỂM TRA CHỈ MỤC ---")
            cur.execute("""
                SELECT indexname FROM pg_indexes
                WHERE tablename = 'clinic_knowledge_nodes'
                ORDER BY indexname;
            """)
            for (index_name,) in cur.fetchall():
                print(f" • {index_name}")


def main() -> None:
    try:
        print_diagnostic_report(get_settings())
    except Exception as exc:  # noqa: BLE001
        logger.error("Healthcheck failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()