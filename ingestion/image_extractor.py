from __future__ import annotations

"""
image_extractor.py

หน้าที่:
- เปิดไฟล์ PDF
- ดึงรูปภาพทุกภาพในเอกสาร
- บันทึกลงโฟลเดอร์ (เช่น ingested/{doc_id}/images/img_001_001.png)
- แปลงผลลัพธ์เป็น list[ImageBlock] ตาม schema
"""

from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF

from .schema import ImageBlock


def extract_images(
    file_path: str | Path,
    doc_id: str,
    output_root: str | Path = "ingested",
) -> List[ImageBlock]:
    """
    ดึงรูปภาพทั้งหมดจาก PDF

    :param file_path: path ไปยังไฟล์ PDF
    :param doc_id: ใช้ผูกกับ DocumentMetadata.doc_id และโฟลเดอร์ output
    :param output_root: โฟลเดอร์รากสำหรับเก็บผลลัพธ์รูป เช่น "ingested"
    :return: list[ImageBlock]
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    output_root = Path(output_root)
    image_dir = output_root / doc_id / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    pdf_doc = fitz.open(path)
    image_blocks: List[ImageBlock] = []

    try:
        image_counter = 0

        # loop ทุกหน้า
        for page_index in range(pdf_doc.page_count):
            page = pdf_doc[page_index]
            page_number = page_index + 1

            # get_images(full=True) คืน list ของ image objects ในหน้านั้น
            images = page.get_images(full=True)

            for img_index, img in enumerate(images, start=1):
                xref = img[0]  # image reference id ใน PDF
                base_image = pdf_doc.extract_image(xref)

                img_bytes: bytes = base_image["image"]
                img_ext: str = base_image.get("ext", "png")
                width: int = base_image.get("width", 0)
                height: int = base_image.get("height", 0)

                image_counter += 1
                img_id = f"img_{image_counter:04d}"

                # ตั้งชื่อไฟล์ เช่น img_p001_001.png
                filename = f"img_p{page_number:03d}_{img_index:03d}.{img_ext}"
                file_path_on_disk = image_dir / filename

                # เซฟรูปลงดิสก์
                with open(file_path_on_disk, "wb") as f:
                    f.write(img_bytes)

                # ในเฟสนี้เรายังไม่รู้ bbox ที่แน่นอนของรูปในหน้า → ให้ bbox=None ไปก่อน
                image_block = ImageBlock(
                    id=img_id,
                    doc_id=doc_id,
                    page=page_number,
                    file_path=str(file_path_on_disk),
                    caption=None,             # ภายหลังสามารถใช้ Gemini ช่วยเดา caption ได้
                    section=None,
                    category=None,            # เช่น "logo", "chart", "signature" (ภายหลังค่อยใส่)
                    bbox=None,
                    extra={
                        "width": width,
                        "height": height,
                        "xref": xref,
                    },
                )
                image_blocks.append(image_block)

        return image_blocks

    finally:
        pdf_doc.close()


if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(description="Extract images from PDF into ImageBlock list.")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--doc-id", help="Document ID (default: stem of file name)")
    parser.add_argument(
        "--output-root",
        default="ingested",
        help="Root folder for saving images (default: 'ingested')",
    )
    args = parser.parse_args()

    pdf_path = args.pdf_path
    doc_id = args.doc_id or Path(pdf_path).stem

    images = extract_images(
        file_path=pdf_path,
        doc_id=doc_id,
        output_root=args.output_root,
    )

    print(f"Extracted {len(images)} images.")
    data = [im.to_dict() for im in images]
    print(json.dumps(data, ensure_ascii=False, indent=2))
