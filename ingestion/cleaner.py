from __future__ import annotations

"""
cleaner.py

Data Cleaning Engine ขั้นพื้นฐานสำหรับ:
- TextBlock: ล้าง whitespace, ตัด block ว่าง, ติด metadata เพิ่ม
- TableBlock: strip ช่องว่าง, ลบคอลัมน์/แถวที่ว่างเปล่า, normalize โครงสร้าง

ไฟล์นี้เน้น:
- ทำความสะอาดแบบ "ไม่ทำลายข้อมูล"
- เก็บ info เดิมไว้ใน extra.cleaning_metadata เผื่อ debug ทีหลัง
"""

from typing import List, Dict, Any
import re

from .schema import TextBlock, TableBlock


WHITESPACE_RE = re.compile(r"\s+")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")  # เว้น \t, \n, \r


def _normalize_text(s: str) -> str:
    """ล้าง control char + ยุบ whitespace ซ้ำ + strip"""
    if not s:
        return ""
    s = CONTROL_CHAR_RE.sub("", s)
    s = WHITESPACE_RE.sub(" ", s)
    return s.strip()


def clean_text_blocks(blocks: List[TextBlock]) -> List[TextBlock]:
    """
    ทำความสะอาด TextBlock:
    - ลบ control chars
    - ยุบ whitespace
    - ตัด block ที่ว่างหลังทำความสะอาด
    - บันทึกข้อมูลก่อน/หลังใน extra.cleaning
    """
    cleaned: List[TextBlock] = []

    for b in blocks:
        original = b.content or ""
        normalized = _normalize_text(original)

        if not normalized:
            # ถ้าไม่มีอะไรเหลือ → ทิ้ง block นี้ไป
            continue

        b.content = normalized

        extra = dict(b.extra or {})
        cleaning_meta: Dict[str, Any] = extra.get("cleaning", {})
        cleaning_meta.update(
            {
                "original_length": len(original),
                "cleaned_length": len(normalized),
                "removed_chars": len(original) - len(normalized),
            }
        )
        extra["cleaning"] = cleaning_meta
        b.extra = extra

        cleaned.append(b)

    return cleaned


def _clean_table_cell(cell: str) -> str:
    """ทำความสะอาดข้อความใน cell ตาราง"""
    return _normalize_text(cell)


def clean_table_blocks(tables: List[TableBlock]) -> List[TableBlock]:
    """
    ทำความสะอาด TableBlock:
    - strip / normalize whitespace ใน header + rows
    - ลบคอลัมน์ที่ว่างทุก cell
    - ลบแถวที่ว่างทุก cell
    """
    cleaned_tables: List[TableBlock] = []

    for tb in tables:
        header = list(getattr(tb, "header", []))
        rows = list(getattr(tb, "rows", []))

        header_clean = [_clean_table_cell(h) for h in header]
        rows_clean = [[_clean_table_cell(str(c)) for c in row] for row in rows]

        # ถ้ามีข้อมูล -> จัดคอลัมน์ใหม่
        if header_clean and rows_clean:
            col_count = max(len(header_clean), max(len(r) for r in rows_clean))
            header_padded = header_clean + [""] * (col_count - len(header_clean))
            rows_padded = [r + [""] * (col_count - len(r)) for r in rows_clean]

            keep_col_idx = []
            for idx in range(col_count):
                col_vals = [header_padded[idx]] + [r[idx] for r in rows_padded]
                if any(v.strip() for v in col_vals):
                    keep_col_idx.append(idx)

            header_final = [header_padded[i] for i in keep_col_idx]
            rows_final = [[row[i] for i in keep_col_idx] for row in rows_padded]
        else:
            header_final = header_clean
            rows_final = rows_clean

        # ลบแถวว่าง
        rows_final = [r for r in rows_final if any(c.strip() for c in r)]

        tb.header = header_final
        tb.rows = rows_final

        extra = dict(tb.extra or {})
        cleaning_meta: Dict[str, Any] = extra.get("cleaning", {})
        cleaning_meta.update(
            {
                "original_row_count": len(rows),
                "cleaned_row_count": len(rows_final),
                "original_header_len": len(header),
                "cleaned_header_len": len(header_final),
            }
        )
        extra["cleaning"] = cleaning_meta
        tb.extra = extra

        cleaned_tables.append(tb)

    return cleaned_tables
