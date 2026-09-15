import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

# 1. Khởi tạo model BGE-M3
model = SentenceTransformer("BAAI/bge-m3")

# 2. Đặt câu hỏi lâm sàng cần tra cứu trong phác đồ
query = "Chỉ định và kỹ thuật nong van hai lá bằng bóng Inoue được thực hiện như thế nào?"
# query = "Biến chứng và cách xử trí khi đặt bóng đối xung động mạch chủ IABP?"

print(f"Câu hỏi: {query}\n")
query_vector = model.encode(query)

CONN_STR = "postgresql://postgres:mysecretpassword@localhost:5433/clinic_rag"

with psycopg.connect(CONN_STR) as conn:
    register_vector(conn)
    with conn.cursor() as cur:
        # Lấy Top-3 đoạn văn bản liên quan nhất
        cur.execute("""
            SELECT 
                doc_id, 
                specialty, 
                content, 
                dense_embedding <=> %s::vector AS distance
            FROM clinic_knowledge_nodes
            ORDER BY distance ASC
            LIMIT 3;
        """, (query_vector,))
        
        results = cur.fetchall()
        
        print("=== KẾT QUẢ TRÍCH XUẤT TỪ PGVECTOR ===")
        for idx, row in enumerate(results, 1):
            print(f"\n[Kết quả {idx}] - Mã tài liệu: {row[0]} | Chuyên khoa: {row[1]}")
            print(f"Khoảng cách Cosine: {row[3]:.4f} (Càng nhỏ càng chính xác)")
            print(f"Nội dung trích đoạn:\n{row[2][:350]}...\n" + "-"*50)