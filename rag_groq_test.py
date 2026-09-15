import sys
import psycopg
import os
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from groq import Groq
load_dotenv()

# 1. Cấu hình hệ thống
MODEL_ID = "qwen/qwen3.8-27b"
CONN_STR = "postgresql://postgres:mysecretpassword@localhost:5433/clinic_rag"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print("Đang khởi tạo mô hình nhúng BGE-M3 và kết nối Groq...")
embed_model = SentenceTransformer("BAAI/bge-m3")
client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """Bạn là trợ lý y khoa AI hỗ trợ bác sĩ lâm sàng tra cứu phác đồ điều trị của Bộ Y tế.
Nguyên tắc trả lời:
1. CHỈ sử dụng thông tin trong [NGỮ CẢNH PHÁC ĐỒ] được cung cấp.
2. Trình bày rõ ràng theo bố cục: Chỉ định, Chống chỉ định, Quy trình kỹ thuật hoặc Biến chứng/Xử trí.
3. Nếu ngữ cảnh không có thông tin, hãy trả lời thẳng: 'Phác đồ hiện tại không có thông tin này', tuyệt đối không tự bịa đặt.
4. Diễn đạt bằng tiếng Việt y khoa tự nhiên, chuẩn chính tả, dấu câu rõ ràng.
5. Tuyệt đối không dùng từ ngữ dịch thô, không ghép từ Hán-Việt gượng gạo.
6. Xưng hô lịch sự, ân cần và chuyên nghiệp với người bệnh."""



def retrieve_context(query: str, top_k: int = 3):
    query_vector = embed_model.encode(query)
    with psycopg.connect(CONN_STR) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT doc_id, specialty, content, (dense_embedding <=> %s::vector) AS distance
                FROM clinic_knowledge_nodes
                ORDER BY distance ASC
                LIMIT %s;
            """, (query_vector, top_k))
            return cur.fetchall()

def run_chat():
    print("\n" + "="*65)
    print(" 🩺 KHUNG CHAT HỖ TRỢ BÁC SĨ TIM MẠCH (CLINIC RAG)")
    print(f" 🤖 LLM Engine: {MODEL_ID}")
    print(" 💡 Nhập câu hỏi phác đồ cần tra cứu (Gõ 'exit' để thoát)")
    print("="*65 + "\n")

    while True:
        try:
            user_query = input("\nBác sĩ: ").strip()
            if not user_query:
                continue
            if user_query.lower() in ["exit", "quit"]:
                print("Đã kết thúc phiên làm việc.")
                break

            print("\n[Đang truy xuất dữ liệu từ PostgreSQL...]")
            results = retrieve_context(user_query, top_k=3)

            # Tổng hợp ngữ cảnh từ 3 chunks khớp nhất
            context_text = "\n\n---\n\n".join([row[2] for row in results])
            augmented_prompt = f"[NGỮ CẢNH PHÁC ĐỒ]:\n{context_text}\n\n[CÂU HỎI]: {user_query}"

            print("\nTrợ lý AI: ", end="", flush=True)

            # Gọi Groq API và stream trực tiếp từng từ ra màn hình
            completion = client.chat.completions.create(
                model=MODEL_ID,
                temperature=0.1,
                stream=True,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": augmented_prompt}
                ]
            )

            for chunk in completion:
                token = chunk.choices[0].delta.content or ""
                print(token, end="", flush=True)

            print("\n\n" + "-"*50)
            print("Tài liệu trích dẫn:")
            for idx, r in enumerate(results, 1):
                print(f"  [{idx}] {r[0]} ({r[1]}) - Sai số Cosine: {r[3]:.4f}")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nĐã ngắt phiên trò chuyện.")
            break
        except Exception as e:
            print(f"\nLỗi: {e}")

if __name__ == "__main__":
    run_chat()