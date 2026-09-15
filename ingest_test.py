import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

# 1. Tải mô hình BGE-M3 (lần chạy đầu tiên sẽ tự tải khoảng 2.2GB về máy)
print("Đang tải mô hình BGE-M3...")
model = SentenceTransformer("BAAI/bge-m3")

# 2. Dữ liệu y khoa mẫu
sample_docs = [
    {
        "doc_id": "TIM_MACH_01",
        "specialty": "cardiology",
        "target_audience": "bac_si",
        "content": "Xử trí hội chứng vành cấp: Cho bệnh nhân thở oxy nếu SpO2 < 90%, dùng Aspirin liều nạp 150-325mg nhai ngay lập tức và Nitroglycerin ngậm dưới lưỡi."
    },
    {
        "doc_id": "HO_HAP_01",
        "specialty": "pulmonology",
        "target_audience": "bac_si",
        "content": "Cơn hen phế quản cấp tính: Ưu tiên dùng thuốc giãn phế quản tác dụng ngắn SABA dạng khí dung hoặc bình xịt định liều kèm buồng đệm."
    }
]

# 3. Kết nối Database và chèn dữ liệu
CONN_STR = "postgresql://postgres:mysecretpassword@localhost:5433/clinic_rag"

with psycopg.connect(CONN_STR) as conn:
    register_vector(conn)
    with conn.cursor() as cur:
        for doc in sample_docs:
            # Sinh dense vector 1024 chiều
            vector_1024 = model.encode(doc["content"]).tolist()
            
            cur.execute("""
                INSERT INTO clinic_knowledge_nodes (doc_id, specialty, target_audience, content, dense_embedding)
                VALUES (%s, %s, %s, %s, %s)
            """, (doc["doc_id"], doc["specialty"], doc["target_audience"], doc["content"], vector_1024))
            
        conn.commit()
        print("Đã lưu thành công các đoạn phác đồ vào Database!")