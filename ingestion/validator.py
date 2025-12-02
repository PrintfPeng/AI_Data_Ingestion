from __future__ import annotations

"""
validator.py

ชุดฟังก์ชันตรวจความถูกต้องของ IngestedDocument:

- validate_document_structure: ตรวจ metadata + text baseline
- validate_tables: ตรวจตาราง
- validate_images: ตรวจรูป
- validate_all: รวมทุกอย่างแล้วคืน issues เป็น list[dict]

ใช้สำหรับ:
- เช็คคุณภาพ ingestion
- log ปัญหาไว้ใน validation.json
"""

from typing import List, Dict, Any
from .schema import IngestedDocument, TableBlock, ImageBlock, TextBlock


def _issue(
    level: str,
    code: str,
    message: str,
    context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "level": level,   # "info" | "warning" | "error"
        "code": code,
        "message": message,
        "context": context or {},
    }


def validate_document_structure(doc: IngestedDocument) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []

    if not doc.metadata.doc_id:
        issues.append(
            _issue(
                "error",
                "MISSING_DOC_ID",
                "Document metadata.doc_id is empty.",
            )
        )

    if not doc.metadata.file_name:
        issues.append(
            _issue(
                "warning",
                "MISSING_FILE_NAME",
                "Document metadata.file_name is empty.",
            )
        )

    if not doc.texts:
        issues.append(
            _issue(
                "error",
                "NO_TEXT_BLOCKS",
                "Document has no TextBlock entries.",
            )
        )

    return issues


def validate_tables(doc: IngestedDocument) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []

    for idx, tb in enumerate(doc.tables):
        header = getattr(tb, "header", [])
        rows = getattr(tb, "rows", [])

        if not header and rows:
            issues.append(
                _issue(
                    "warning",
                    "TABLE_NO_HEADER",
                    f"Table index={idx} has rows but empty header.",
                    {"table_index": idx},
                )
            )

        if header and not rows:
            issues.append(
                _issue(
                    "warning",
                    "TABLE_NO_ROWS",
                    f"Table index={idx} has header but no rows.",
                    {"table_index": idx},
                )
            )

        # เช็คความยาว row vs header (อย่างหยาบ ๆ)
        for r_idx, row in enumerate(rows):
            if len(row) != len(header):
                issues.append(
                    _issue(
                        "warning",
                        "ROW_LEN_MISMATCH",
                        f"Table index={idx} row={r_idx} len(row)={len(row)} != len(header)={len(header)}",
                        {"table_index": idx, "row_index": r_idx},
                    )
                )

    return issues


def validate_images(doc: IngestedDocument) -> List[Dict[str, Any]]:
    # ขึ้นกับว่า schema.ImageBlock มีอะไรบ้าง
    # สมมติว่าอย่างน้อยมี path หรือ ref
    issues: List[Dict[str, Any]] = []
    for idx, im in enumerate(doc.images):
        if not getattr(im, "image_path", None) and not getattr(im, "ref", None):
            issues.append(
                _issue(
                    "warning",
                    "IMAGE_NO_PATH",
                    f"Image index={idx} has no image_path/ref.",
                    {"image_index": idx},
                )
            )
    return issues


def validate_all(doc: IngestedDocument) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    issues.extend(validate_document_structure(doc))
    issues.extend(validate_tables(doc))
    issues.extend(validate_images(doc))
    return issues
