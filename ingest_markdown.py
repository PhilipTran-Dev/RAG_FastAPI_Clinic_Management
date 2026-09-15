import os
import re
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

# 1. Cấu hình
MD_FILE_PATH = "data/parsed_markdown/phac_do_tim_mach.md"
CONN_STR = "postgresql://postgres:mysecretpassword@localhost:5433/clinic_rag"

print("Đang khởi động BGE-M3...")
model = SentenceTransformer("BAAI/bge-m3")

def chunk_markdown_by_sections(content: str, max_chunk_size: int = 1000):
    """
    Tách nội dung theo tiêu đề Markdown (# hoặc ##) 
    để giữ trọn vẹn ngữ cảnh từng phần của phác đồ
    """
    # Tách theo các tiêu đề mục lớn/nhỏ
    sections = re.split(r'\n(?=#{1,3}\s)', content)
    chunks = []
    
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        # Nếu section quá dài, chia nhỏ theo đoạn văn
        if len(sec) > max_chunk_size:
            paragraphs = sec.split("\n\n")
            current_chunk = ""
            for p in paragraphs:
                if len(current_chunk) + len(p) < max_chunk_size:
                    current_chunk += p + "\n\n"
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = p + "\n\n"
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
        else:
            chunks.append(sec)
            
    return chunks

def ingest_to_postgres():
    if not os.path.exists(MD_FILE_PATH):
        print(f"Không tìm thấy file: {MD_FILE_PATH}")
        return

    with open(MD_FILE_PATH, "r", encoding="utf-8") as f:
        full_text = f.read()

    print("Đang phân đoạn (chunking) tài liệu...")
    chunks = chunk_markdown_by_sections(full_text)
    print(f"Tổng số chunks tạo thành: {len(chunks)}")

    print("Đang kết nối PostgreSQL và lưu vector...")
    with psycopg.connect(CONN_STR) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for idx, text_chunk in enumerate(chunks):
                # Sinh dense vector 1024 chiều
                vector_1024 = model.encode(text_chunk)
                
                cur.execute("""
                    INSERT INTO clinic_knowledge_nodes (
                        doc_id, 
                        specialty, 
                        target_audience, 
                        content, 
                        dense_embedding
                    )
                    VALUES (%s, %s, %s, %s, %s::vector)
                """, (
                    "QD_3983_BYT_TIM_MACH",
                    "cardiology",
                    "bac_si",
                    text_chunk,
                    vector_1024
                ))
                
                if (idx + 1) % 10 == 0 or (idx + 1) == len(chunks):
                    print(f"Đã lưu: {idx + 1}/{len(chunks)} chunks")
                    
            conn.commit()
            
    print("\n--- HOÀN THÀNH NẠP DỮ LIỆU GIAI ĐOẠN 1 ---")

if __name__ == "__main__":
    ingest_to_postgres()