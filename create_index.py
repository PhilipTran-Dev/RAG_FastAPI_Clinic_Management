import psycopg

CONN_STR = "postgresql://postgres:mysecretpassword@localhost:5433/clinic_rag"

print("Đang tạo chỉ mục HNSW cho vector nhúng...")
with psycopg.connect(CONN_STR) as conn:
    with conn.cursor() as cur:
        # Tạo index HNSW hỗ trợ cosine distance (<=>)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_clinic_nodes_hnsw 
            ON clinic_knowledge_nodes 
            USING hnsw (dense_embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """)
        conn.commit()
print("Tạo chỉ mục HNSW thành công!")