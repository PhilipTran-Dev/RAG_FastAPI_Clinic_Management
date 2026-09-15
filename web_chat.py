import os
import streamlit as st
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from groq import Groq

# 1. Nạp biến môi trường và cấu hình
load_dotenv()
MODEL_CHAT_ID = "openai/gpt-oss-120b"
MODEL_STT_ID = "whisper-large-v3"
CONN_STR = "postgresql://postgres:mysecretpassword@localhost:5433/clinic_rag"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Prompt chuẩn hóa cho Bác sĩ (Strict RAG)
DOCTOR_SYSTEM_PROMPT = """Bạn là trợ lý y khoa AI hỗ trợ bác sĩ lâm sàng tra cứu phác đồ điều trị của Bộ Y tế.
Nguyên tắc trả lời:
1. CHỈ sử dụng thông tin trong [NGỮ CẢNH PHÁC ĐỒ] được cung cấp.
2. Trình bày rõ ràng theo bố cục: Chỉ định, Chống chỉ định, Quy trình kỹ thuật hoặc Biến chứng/Xử trí.
3. Nếu ngữ cảnh không có thông tin, hãy trả lời thẳng: 'Phác đồ hiện tại không có thông tin này', tuyệt đối không tự bịa đặt.
4. Diễn đạt bằng tiếng Việt y khoa tự nhiên, chuẩn chính tả, dấu câu rõ ràng."""

# Prompt chuẩn hóa tiếp đón bệnh nhân (Triage)
TRIAGE_SYSTEM_PROMPT = """Bạn là trợ lý tiếp đón và phân loại triệu chứng tại phòng khám (Medical Triage Assistant).
Nhiệm vụ của bạn là lắng nghe triệu chứng từ người bệnh và:
1. ĐÁNH GIÁ MỨC ĐỘ NGUY CẤP:
   - Cấp cứu (Đỏ): Khó thở dữ dội, đau thắt ngực, sốc phản vệ, hoại tử/loét da diện rộng cấp tính -> Gọi cấp cứu 115 ngay lập tức.
   - Ưu tiên khám sớm (Vàng): Đau tức ngực gắng sức, tổn thương da lan nhanh, ngứa rát dữ dội, mụn mủ có sốt.
   - Khám thông thường (Xanh): Tái khám định kỳ, ngứa nhẹ, sẩn đỏ rải rác mạn tính.
2. GỢI Ý CHUYÊN KHOA PHÙ HỢP: (Tim mạch, Da liễu, Hô hấp, Tiêu hóa...).
3. TÓM TẮT BỆNH SỬ: Tóm tắt ngắn gọn triệu chứng để bệnh nhân trình bày với bác sĩ.

CẢNH BÁO: Không kết luận tên bệnh xác định, không kê đơn thuốc hoặc liều lượng."""

st.set_page_config(page_title="Clinic AI Portal - Đa Chuyên Khoa", page_icon="🏥", layout="wide")

# 2. Khởi tạo tài nguyên dùng chung
@st.cache_resource
def init_resources():
    embedder = SentenceTransformer("BAAI/bge-m3")
    groq_client = Groq(api_key=GROQ_API_KEY)
    return embedder, groq_client

embed_model, client = init_resources()

def retrieve_context(query: str, top_k: int = 3, specialty: str = None):
    query_vector = embed_model.encode(query)
    with psycopg.connect(CONN_STR) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            if specialty and specialty != "all":
                cur.execute("""
                    SELECT doc_id, specialty, content, (dense_embedding <=> %s::vector) AS distance
                    FROM clinic_knowledge_nodes
                    WHERE specialty = %s
                    ORDER BY distance ASC
                    LIMIT %s;
                """, (query_vector, specialty, top_k))
            else:
                cur.execute("""
                    SELECT doc_id, specialty, content, (dense_embedding <=> %s::vector) AS distance
                    FROM clinic_knowledge_nodes
                    ORDER BY distance ASC
                    LIMIT %s;
                """, (query_vector, top_k))
            return cur.fetchall()

# 3. Điều hướng Sidebar
st.sidebar.title("🏥 Hệ Thống Y Tế Phòng Khám")
role = st.sidebar.radio("Chọn phân hệ:", ["👨‍⚕️ Cổng Bác sĩ (EHR RAG)", "🎙️ Bệnh nhân (Tiếp đón Voice)"])

# ==========================================
# PHÂN HỆ 1: BÁC SĨ (EHR RAG)
# ==========================================
if role == "👨‍⚕️ Cổng Bác sĩ (EHR RAG)":
    st.title("👨‍⚕️ Tra Cứu Phác Đồ Bộ Y Tế (Tim Mạch & Da Liễu)")
    
    # Bộ lọc chuyên khoa
    spec_choice = st.sidebar.selectbox(
        "Lọc chuyên khoa:",
        ["Tất cả chuyên khoa", "Tim mạch (tim_mach)", "Da liễu (da_lieu)"]
    )
    spec_mapping = {
        "Tất cả chuyên khoa": "all",
        "Tim mạch (tim_mach)": "tim_mach",
        "Da liễu (da_lieu)": "da_lieu"
    }
    selected_spec = spec_mapping[spec_choice]

    if "doctor_messages" not in st.session_state:
        st.session_state.doctor_messages = []

    for msg in st.session_state.doctor_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 Tài liệu trích dẫn"):
                    for s in msg["sources"]:
                        st.write(s)

    if doc_query := st.chat_input("Nhập câu hỏi chuyên môn: vảy nến, zona, nong van, chụp mạch vành..."):
        st.session_state.doctor_messages.append({"role": "user", "content": doc_query})
        with st.chat_message("user"):
            st.markdown(doc_query)

        with st.chat_message("assistant"):
            with st.status("Đang truy xuất phác đồ từ pgvector...", expanded=False):
                results = retrieve_context(doc_query, top_k=3, specialty=selected_spec)
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
                    info = f"**[{idx}] {r[0]}** (Khoa: `{r[1]}`) — Khoảng cách Cosine: `{r[3]:.4f}`"
                    st.markdown(info)
                    sources_info.append(info)

        st.session_state.doctor_messages.append({
            "role": "assistant",
            "content": full_reply,
            "sources": sources_info
        })

# ==========================================
# PHÂN HỆ 2: BỆNH NHÂN (VOICE TRIAGE)
# ==========================================
else:
    st.title("🎙️ Tiếp Đón & Sàng Lọc Triệu Chứng")
    st.caption("Nhận diện giọng nói đa chuyên khoa qua Whisper-large-v3")

    if "patient_messages" not in st.session_state:
        st.session_state.patient_messages = []

    for msg in st.session_state.patient_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    audio_value = st.audio_input("Nhấn mic để nói mô tả cảm giác khó chịu hoặc tổn thương...")

    patient_input = ""
    if audio_value is not None:
        if "last_audio" not in st.session_state or st.session_state.last_audio != audio_value:
            st.session_state.last_audio = audio_value
            with st.spinner("Đang nhận diện giọng nói tiếng Việt..."):
                try:
                    # Mở rộng từ điển nhận diện cho cả Tim mạch và Da liễu
                    transcription = client.audio.transcriptions.create(
                        file=("voice_input.wav", audio_value.read()),
                        model=MODEL_STT_ID,
                        language="vi",
                        prompt="Khám bệnh, đau thắt ngực, khó thở, huyết áp, da liễu, nổi mẩn đỏ, ngứa ngáy, mụn nước, dát sẩn, vảy nến, bong vảy, viêm da cơ địa, zona, lở loét.",
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