import streamlit as st
import psycopg
import os
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

# 1. Cấu hình hệ thống
MODEL_CHAT_ID = "openai/gpt-oss-120b"
MODEL_STT_ID = "whisper-large-v3"
CONN_STR = "postgresql://postgres:mysecretpassword@localhost:5433/clinic_rag"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Prompt chuẩn hóa cho Bác sĩ (RAG nghiêm ngặt)
DOCTOR_SYSTEM_PROMPT = """Bạn là trợ lý y khoa AI hỗ trợ bác sĩ lâm sàng tra cứu phác đồ điều trị của Bộ Y tế.
Nguyên tắc trả lời:
1. CHỈ sử dụng thông tin trong [NGỮ CẢNH PHÁC ĐỒ] được cung cấp.
2. Trình bày rõ ràng theo bố cục: Chỉ định, Chống chỉ định, Quy trình kỹ thuật hoặc Biến chứng/Xử trí.
3. Nếu ngữ cảnh không có thông tin, hãy trả lời thẳng: 'Phác đồ hiện tại không có thông tin này', tuyệt đối không tự bịa đặt.
4. Diễn đạt bằng tiếng Việt y khoa tự nhiên, chuẩn chính tả, dấu câu rõ ràng.
5. Tuyệt đối không dùng từ ngữ dịch thô, không ghép từ Hán-Việt gượng gạo."""

# Prompt chuẩn hóa cho Bệnh nhân (Triage phân loại)
TRIAGE_SYSTEM_PROMPT = """Bạn là trợ lý tiếp đón và phân loại triệu chứng tại phòng khám (Medical Triage Assistant).
Nhiệm vụ của bạn là lắng nghe triệu chứng từ người bệnh và:
1. ĐÁNH GIÁ MỨC ĐỘ NGUY CẤP:
   - Cấp cứu (Đỏ): Khó thở dữ dội, đau thắt ngực nghẹt thở, vã mồ hôi, ngất -> Khuyên đến cấp cứu 115 ngay lập tức.
   - Ưu tiên khám sớm (Vàng): Triệu chứng kéo dài nhiều ngày, khó chịu tăng dần.
   - Khám thông thường (Xanh): Triệu chứng nhẹ, định kỳ.
2. GỢI Ý CHUYÊN KHOA PHÙ HỢP: (Tim mạch, Hô hấp, Tiêu hóa, Thần kinh...).
3. TÓM TẮT BỆNH SỬ: Tóm tắt lại ngắn gọn triệu chứng để bệnh nhân trình bày với bác sĩ.

CẢNH BÁO TUYỆT ĐỐI: 
- KHÔNG kết luận bệnh xác định (Ví dụ: Không nói 'Bạn bị nhồi máu cơ tim').
- KHÔNG kê đơn thuốc hoặc chỉ định liều lượng.
- Xưng hô lịch sự, ân cần, chuyên nghiệp và luôn khuyên khám trực tiếp với bác sĩ."""

st.set_page_config(page_title="Clinic AI Portal", page_icon="🏥", layout="wide")

# 2. Khởi tạo tài nguyên dùng chung
@st.cache_resource
def init_resources():
    embedder = SentenceTransformer("BAAI/bge-m3")
    groq_client = Groq(api_key=GROQ_API_KEY)
    return embedder, groq_client

embed_model, client = init_resources()

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

# 3. Thanh điều hướng phân hệ
st.sidebar.title("🏥 Hệ Thống Y Tế Thông Minh")
role = st.sidebar.radio("Chọn phân hệ làm việc:", ["👨‍⚕️ Cổng Bác sĩ (EHR RAG)", "🎙️ Bệnh nhân (Tiếp đón Voice)"])

# ==========================================
# PHÂN HỆ 1: BÁC SĨ TRA CỨU PHÁC ĐỒ (RAG)
# ==========================================
if role == "👨‍⚕️ Cổng Bác sĩ (EHR RAG)":
    st.title("👨‍⚕️ Tra Cứu Phác Đồ Can Thiệp Tim Mạch")
    st.caption(f"Dữ liệu: QĐ 3983/QĐ-BYT | Vector DB: PostgreSQL pgvector | LLM: {MODEL_CHAT_ID}")

    if "doctor_messages" not in st.session_state:
        st.session_state.doctor_messages = []

    for msg in st.session_state.doctor_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 Tài liệu trích dẫn"):
                    for s in msg["sources"]:
                        st.write(s)

    if doc_query := st.chat_input("Nhập câu hỏi tra cứu chỉ định, quy trình, tai biến phác đồ..."):
        st.session_state.doctor_messages.append({"role": "user", "content": doc_query})
        with st.chat_message("user"):
            st.markdown(doc_query)

        with st.chat_message("assistant"):
            with st.status("Đang truy xuất phác đồ từ pgvector...", expanded=False):
                results = retrieve_context(doc_query, top_k=3)
                context_text = "\n\n---\n\n".join([row[2] for row in results])
                augmented_prompt = f"[NGỮ CẢNH PHÁC ĐỒ]:\n{context_text}\n\n[CÂU HỎI]: {doc_query}"

            def stream_doctor_reply():
                completion = client.chat.completions.create(
                    model=MODEL_CHAT_ID,
                    temperature=0.1,
                    stream=True,
                    messages=[
                        {"role": "system", "content": DOCTOR_SYSTEM_PROMPT},
                        {"role": "user", "content": augmented_prompt}
                    ]
                )
                for chunk in completion:
                    token = chunk.choices[0].delta.content or ""
                    yield token

            full_reply = st.write_stream(stream_doctor_reply())

            sources_info = []
            with st.expander("📚 Tài liệu trích dẫn (Top-3)"):
                for idx, r in enumerate(results, 1):
                    info = f"**[{idx}] {r[0]}** ({r[1]}) — Khoảng cách Cosine: `{r[3]:.4f}`"
                    st.markdown(info)
                    sources_info.append(info)

        st.session_state.doctor_messages.append({
            "role": "assistant",
            "content": full_reply,
            "sources": sources_info
        })

# ==========================================
# PHÂN HỆ 2: BỆNH NHÂN SÀNG LỌC TRIỆU CHỨNG (VOICE)
# ==========================================
else:
    st.title("🎙️ Tiếp Đón & Sàng Lọc Triệu Chứng")
    st.caption("Nhận diện giọng nói qua Groq Whisper-large-v3 | Phân loại mức độ ưu tiên khám")

    if "patient_messages" not in st.session_state:
        st.session_state.patient_messages = []

    for msg in st.session_state.patient_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    st.markdown("##### 🎤 Nói vào micro để mô tả cảm giác khó chịu của bạn:")
    audio_value = st.audio_input("Nhấn nút micro để nói (Ví dụ: 'Tôi thấy tức ngực trái 2 hôm nay, thở dốc khi leo cầu thang...')")

    patient_input = ""

    if audio_value is not None:
        if "last_audio" not in st.session_state or st.session_state.last_audio != audio_value:
            st.session_state.last_audio = audio_value
            with st.spinner("Đang nhận diện giọng nói tiếng Việt..."):
                try:
                    transcription = client.audio.transcriptions.create(
                        file=("voice_input.wav", audio_value.read()),
                        model=MODEL_STT_ID,
                        language="vi",
                        prompt="Bệnh nhân khám bệnh, tim mạch, đau thắt ngực, khó thở, huyết áp, nhồi máu cơ tim, hồi hộp, đánh trống ngực, phù chân, chóng mặt, nong van, đặt stent.",
                        response_format="text"
                    )
                    patient_input = transcription.strip()
                except Exception as e:
                    st.error(f"Lỗi nhận diện âm thanh: {e}")

    text_fallback = st.chat_input("Hoặc gõ mô tả triệu chứng tại đây...")
    if text_fallback:
        patient_input = text_fallback

    if patient_input:
        st.session_state.patient_messages.append({"role": "user", "content": patient_input})
        with st.chat_message("user"):
            st.markdown(f"🗣️ **Triệu chứng ghi nhận:** {patient_input}")

        with st.chat_message("assistant"):
            with st.spinner("Đang phân loại và đưa ra hướng dẫn tiếp đón..."):
                completion = client.chat.completions.create(
                    model=MODEL_CHAT_ID,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Bệnh nhân mô tả: {patient_input}"}
                    ]
                )
                triage_reply = completion.choices[0].message.content
                st.markdown(triage_reply)

        st.session_state.patient_messages.append({"role": "assistant", "content": triage_reply})