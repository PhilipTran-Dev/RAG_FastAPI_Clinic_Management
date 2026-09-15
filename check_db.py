import psycopg
from pgvector.psycopg import register_vector

CONN_STR = "postgresql://postgres:mysecretpassword@localhost:5433/clinic_rag"

with psycopg.connect(CONN_STR) as conn:
    register_vector(conn)
    with conn.cursor() as cur:
        # 1. Đếm tổng số bản ghi hiện có
        cur.execute("SELECT COUNT(*) FROM clinic_knowledge_nodes;")
        total_rows = cur.fetchone()[0]
        print(f"Tổng số bản ghi (chunks) đã lưu: {total_rows}\n")

        # 2. Lấy 3 bản ghi mẫu để xem chi tiết nội dung và vector
        cur.execute("""
            SELECT 
                id, 
                doc_id, 
                specialty, 
                target_audience, 
                content, 
                dense_embedding::text,
                created_at
            FROM clinic_knowledge_nodes 
            LIMIT 3;
        """)
        rows = cur.fetchall()

        for idx, row in enumerate(rows, 1):
            print(f"=== BẢN GHI #{idx} ===")
            print(f"• ID: {row[0]}")
            print(f"• Mã tài liệu: {row[1]}")
            print(f"• Chuyên khoa: {row[2]}")
            print(f"• Đối tượng áp dụng: {row[3]}")
            print(f"• Thời gian tạo: {row[6]}")
            print(f"• Trích đoạn nội dung:\n  \"{row[4][:250]}...\"")
            
            # Cắt xem 5 phần tử đầu tiên của vector 1024 chiều
            vector_preview = row[5][:50] + " ... ]"
            print(f"• Vector nhúng (1024 chiều): {vector_preview}\n")