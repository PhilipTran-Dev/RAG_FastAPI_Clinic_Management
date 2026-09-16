import os
from dotenv import load_dotenv
from llama_parse import LlamaParse

load_dotenv()

LLAMA_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
PDF_PATH = "data/raw_docs/phac_do_dieu_tri_noi_khoa.pdf"
OUTPUT_MD = "data/parsed_markdown/phac_do_dieu_tri_noi_khoa.md"

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"Không tìm thấy file tài liệu tại: {PDF_PATH}")

if not LLAMA_API_KEY:
    raise ValueError("Thiếu LLAMA_CLOUD_API_KEY trong file .env!")

# Prompt hướng dẫn chuyên biệt cho tài liệu y khoa
MEDICAL_PARSING_INSTRUCTION = """
Đây là tài liệu Phác đồ điều trị y tế của Bộ Y tế Việt Nam.
Khi chuyển đổi sang Markdown, hãy tuân thủ nghiêm ngặt các quy tắc sau:
1. Giữ nguyên cấu trúc phân cấp: Phần, Chương, I, II, 1, 2, a, b.
2. Tất cả các bảng danh mục thuốc, bảng liều lượng (mg, ml, liều nạp, liều duy trì) và bảng chỉ số sinh hiệu phải được định dạng chuẩn Markdown Table (| Cột 1 | Cột 2 |). Tuyệt đối không làm gãy hàng/cột.
3. Không bỏ sót các mã chẩn đoán ICD-10, chống chỉ định, liều dùng đặc biệt cho suy gan/suy thận.
4. Giữ nguyên các danh sách gạch đầu dòng triệu chứng, tiêu chuẩn chẩn đoán.
"""

print(f"Đang gửi {PDF_PATH} lên LlamaParse (chế độ tối ưu phác đồ y tế)...")

parser = LlamaParse(
    api_key=LLAMA_API_KEY,
    result_type="markdown",
    language="vi",
    parsing_instruction=MEDICAL_PARSING_INSTRUCTION,
    num_workers=4,            # Xử lý song song 4 worker giúp file 114 trang chạy nhanh gấp 3-4 lần
    page_separator="\n\n---\n\n",  # Đánh dấu ranh giới trang rõ ràng phục vụ bước chunking sau này
    verbose=True
)

# Thực thi bóc tách
documents = parser.load_data(PDF_PATH)

# Lưu kết quả
os.makedirs("data/parsed_markdown", exist_ok=True)
with open(OUTPUT_MD, "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc.text + "\n\n")

print(f"Đã trích xuất thành công! File Markdown được lưu tại: {OUTPUT_MD}")