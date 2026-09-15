import re
import os
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

CONN_STR = "postgresql://postgres:mysecretpassword@localhost:5433/clinic_rag"
MD_PATH = "data/parsed_markdown/phac_do_da_lieu.md"
SPECIALTY = "da_lieu"
DOC_ID = "phac_do_da_lieu"
TARGET_AUDIENCE = "doctor"  # Thêm đối tượng mục tiêu: bác sĩ lâm sàng

print("Đang khởi tạo mô hình nhúng BAAI/bge-m3...")
embed_model = SentenceTransformer("BAAI/bge-m3")

def chunk_markdown(file_path: str, max_chars: int = 1200, overlap: int = 150):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    sections = re.split(r'\n(?=#{1,3}\s)', text)
    chunks = []

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        if len(sec) <= max_chars:
            chunks.append(sec)
        else:
            start = 0
            while start < len(sec):
                end = start + max_chars
                chunk = sec[start:end]
                chunks.append(chunk)
                start += (max_chars - overlap)
    return chunks

print(f"Đang đọc và chunking file {MD_PATH}...")
chunks = chunk_markdown(MD_PATH)
print(f"Tổng số chunks tạo ra: {len(chunks)}")

print("Đang tính toán vector nhúng và nạp vào PostgreSQL...")
with psycopg.connect(CONN_STR) as conn:
    register_vector(conn)
    with conn.cursor() as cur:
        for idx, chunk_content in enumerate(chunks, 1):
            embedding = embed_model.encode(chunk_content)
            
            # Cập nhật thêm cột target_audience vào câu lệnh INSERT
            cur.execute("""
                INSERT INTO clinic_knowledge_nodes (doc_id, specialty, target_audience, content, dense_embedding)
                VALUES (%s, %s, %s, %s, %s);
            """, (DOC_ID, SPECIALTY, TARGET_AUDIENCE, chunk_content, embedding))
            
            if idx % 50 == 0 or idx == len(chunks):
                print(f"  -> Đã nạp {idx}/{len(chunks)} chunks")
        conn.commit()

print("Hoàn tất nạp phác đồ Da liễu vào pgvector!")