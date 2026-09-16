import os
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# Cấu hình tắt OCR triệt để để bóc tách text siêu tốc
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False
pipeline_options.do_table_structure = True

# Khởi tạo converter với pipeline options đã tắt OCR
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

INPUT_PDF = "data/raw_docs/phac_do_dieu_tri_noi_khoa.pdf"
OUTPUT_DIR = "data/parsed_markdown"

def parse_medical_document(pdf_path: str):
    if not os.path.exists(pdf_path):
        print(f"Lỗi: Không tìm thấy file {pdf_path}. Hãy kiểm tra lại thư mục.")
        return

    print(f"Đang bóc tách tài liệu (chế độ Fast/No-OCR): {pdf_path} ...")
    
    # Sử dụng trực tiếp converter đã cấu hình tắt OCR ở trên
    result = converter.convert(pdf_path)
    
    # Xuất sang Markdown
    markdown_content = result.document.export_to_markdown()
    
    # Lưu ra file .md
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = Path(pdf_path).stem + ".md"
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
        
    print(f"Bóc tách hoàn tất! Kết quả đã lưu tại: {output_path}")

if __name__ == "__main__":
    parse_medical_document(INPUT_PDF)