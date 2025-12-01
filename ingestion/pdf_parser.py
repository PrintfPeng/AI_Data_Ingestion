from __future__ import annotations

"""
pdf_parser.py

หน้าที่:
- เปิดไฟล์ PDF
- ดึงข้อความ (text) ออกจากทุกหน้า
- เก็บพิกัด (bbox) ของแต่ละ block
- สร้าง DocumentMetadata + TextBlock ตาม schema
- คืนค่าเป็น IngestedDocument (ยังไม่มี table / image ในเฟสนี้)
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF

from .schema import (
    DocumentMetadata,
    TextBlock,
    IngestedDocument,
    BBox,
)


def _generate_doc_id(file_path: Path) -> str:
    """
    สร้าง doc_id พื้นฐานจากชื่อไฟล์ (ไม่ต้องซับซ้อนมาก)
    เช่น sample.pdf -> sample
    """
    return file_path.stem


def _extract_text_blocks_from_page(
    pdf_page: fitz.Page,
    doc_id: str,
    page_number: int,
    start_index: int = 0,
) -> List[TextBlock]:
    """
    ดึง text blocks จากหน้าเดียวของ PDF
    ใช้ page.get_text("dict") เพื่อได้ทั้ง text + bbox + font size

    :param pdf_page: fitz.Page
    :param doc_id: ไอดีเอกสาร
    :param page_number: เลขหน้า (เริ่ม 1)
    :param start_index: index เริ่มต้นสำหรับ running id
    :return: list[TextBlock]
    """
    page_dict = pdf_page.get_text("dict")
    blocks = page_dict.get("blocks", [])

    text_blocks: List[TextBlock] = []
    current_index = start_index

    for block in blocks:
        # บาง block อาจไม่มี "lines" (เช่น รูปภาพ) ข้ามไป
        if "lines" not in block:
            continue

        x0, y0, x1, y1 = block["bbox"]

        lines = block.get("lines", [])
        spans_text = []
        font_sizes = []

        for line in lines:
            for span in line.get("spans", []):
                text = span.get("text", "")
                if text and text.strip():
                    spans_text.append(text)
                    size = span.get("size", 0.0)
                    font_sizes.append(float(size))

        # ถ้า block นี้ไม่มีข้อความที่มีเนื้อ ก็ข้ามไป
        if not spans_text:
            continue

        # รวม text ทั้ง block เป็นข้อความเดียว (เว้นวรรค)
        content = " ".join(spans_text).strip()
        if not content:
            continue

        avg_font_size: Optional[float] = None
        if font_sizes:
            avg_font_size = sum(font_sizes) / len(font_sizes)

        current_index += 1
        block_id = f"txt_{current_index:04d}"

        text_block = TextBlock(
            id=block_id,
            doc_id=doc_id,
            page=page_number,
            content=content,
            section=None,        # ยังไม่รู้ section (ให้ segmenter ทำต่อในเฟสหน้า)
            category=None,       # ยังไม่จัด category (ให้ categorizer ทำต่อ)
            bbox=(float(x0), float(y0), float(x1), float(y1)),
            extra={
                "avg_font_size": avg_font_size,
            },
        )
        text_blocks.append(text_block)

    return text_blocks


def parse_pdf(
    file_path: str | Path,
    doc_type: str = "generic",
    doc_id: Optional[str] = None,
    source: str = "uploaded",
) -> IngestedDocument:
    """
    ฟังก์ชันหลัก: แปลง PDF 1 ไฟล์ -> IngestedDocument (metadata + text blocks)

    - ยังไม่ดึงตาราง (ให้ table_extractor จัดการในเฟสถัดไป)
    - ยังไม่ดึงรูป (ให้ image_extractor จัดการในเฟสถัดไป)

    :param file_path: path ไปยัง PDF
    :param doc_type: ประเภทเอกสาร เช่น "bank_statement", "receipt", "invoice"
    :param doc_id: ถ้าไม่ระบุ จะสร้างจากชื่อไฟล์
    :param source: แหล่งที่มา เช่น "uploaded"
    :return: IngestedDocument
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    # เปิดเอกสารด้วย PyMuPDF
    pdf_doc = fitz.open(path)

    try:
        if doc_id is None:
            doc_id = _generate_doc_id(path)

        # สร้าง metadata
        metadata = DocumentMetadata(
            doc_id=doc_id,
            file_name=path.name,
            doc_type=doc_type,
            page_count=pdf_doc.page_count,
            ingested_at=datetime.utcnow().isoformat(),
            source=source,
        )

        all_text_blocks: List[TextBlock] = []
        current_index = 0

        # loop ทุกหน้า
        for page_index in range(pdf_doc.page_count):
            page = pdf_doc[page_index]
            page_number = page_index + 1
            page_text_blocks = _extract_text_blocks_from_page(
                pdf_page=page,
                doc_id=doc_id,
                page_number=page_number,
                start_index=current_index,
            )
            all_text_blocks.extend(page_text_blocks)
            current_index += len(page_text_blocks)

        # คืนค่า document ที่มี metadata + text ทั้งหมด
        ingested = IngestedDocument(
            metadata=metadata,
            texts=all_text_blocks,
            tables=[],
            images=[],
        )
        return ingested

    finally:
        pdf_doc.close()


# เผื่ออยากรันทดสอบจาก command line โดยตรง
if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(description="Parse PDF into structured text blocks.")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument(
        "--doc-type",
        default="generic",
        help="Document type (e.g., bank_statement, receipt, invoice)",
    )
    args = parser.parse_args()

    doc = parse_pdf(args.pdf_path, doc_type=args.doc_type)
    print(json.dumps(doc.to_dict(), ensure_ascii=False, indent=2))
