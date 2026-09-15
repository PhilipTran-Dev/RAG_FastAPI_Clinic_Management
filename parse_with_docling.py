import os
from pathlib import Path
from docling.document_converter import DocumentConverter

# Đường dẫn file
INPUT_PDF = "data/raw_docs/phac_do_tim_mach.pdf"
OUTPUT_DIR = "data/parsed_markdown"

def parse_medical_document(pdf_path: str):
    if not os.path.exists(pdf_path):
        print(f"Lỗi: Không tìm thấy file {pdf_path}. Hãy đặt file PDF vào thư mục.")
        return

    print(f"Đang bóc tách tài liệu: {pdf_path} ... (quá trình này có thể mất 1-2 phút)")
    
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    
    # Xuất toàn bộ nội dung sang Markdown
    markdown_content = result.document.export_to_markdown()
    
    # Lưu ra file .md
    filename = Path(pdf_path).stem + ".md"
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
        
    print(f"Bóc tách hoàn tất! Kết quả đã lưu tại: {output_path}")

if __name__ == "__main__":
    parse_medical_document(INPUT_PDF)