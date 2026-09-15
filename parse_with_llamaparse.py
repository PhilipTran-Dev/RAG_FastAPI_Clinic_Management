import os
from dotenv import load_dotenv
from llama_parse import LlamaParse

load_dotenv()

LLAMA_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
PDF_PATH = "data/raw_docs/phac_do_da_lieu.pdf"
OUTPUT_MD = "data/parsed_markdown/phac_do_da_lieu.md"

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"Không tìm thấy file tài liệu tại: {PDF_PATH}")

print("Đang gửi tài liệu lên LlamaParse để xử lý bảng biểu và hình ảnh...")

# Cấu hình parser cho tiếng Việt và tối ưu hóa layout y khoa
parser = LlamaParse(
    api_key=LLAMA_API_KEY,
    result_type="markdown",
    language="vi",
    verbose=True
)

# Thực thi parse
documents = parser.load_data(PDF_PATH)

# Lưu kết quả ra Markdown
os.makedirs("data/parsed_markdown", exist_ok=True)
with open(OUTPUT_MD, "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc.text + "\n\n")

print(f"Đã trích xuất thành công! File Markdown được lưu tại: {OUTPUT_MD}")