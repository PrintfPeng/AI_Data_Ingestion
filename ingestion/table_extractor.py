from __future__ import annotations

"""
table_extractor.py

หน้าที่:
- ใช้ Camelot อ่านตารางจาก PDF
- แปลงตารางแต่ละตัวเป็น TableBlock ตาม schema
- คืนค่า list[TableBlock] เพื่อไปใส่ใน IngestedDocument.tables ภายหลัง
"""

from pathlib import Path
from typing import List, Optional, Any

import camelot
import pandas as pd

from .schema import TableBlock, BBox


def _guess_table_category(df: pd.DataFrame) -> str:
    """
    เดา category ของตารางแบบง่าย ๆ จากชื่อ column
    ภายหลังสามารถใช้ Gemini / LLM ช่วย classify ให้เนียนขึ้นได้

    ตัวอย่าง category:
    - "transaction_table"
    - "item_list"
    - "generic_table"
    """
    header_row = df.iloc[0].astype(str).str.lower().tolist()

    joined = " ".join(header_row)
    if any(k in joined for k in ["date", "วันที่"]) and any(
        k in joined for k in ["amount", "ยอด", "เงิน"]
    ):
        return "transaction_table"

    if any(k in joined for k in ["item", "description", "รายการ"]):
        return "item_list"

    return "generic_table"


def _dataframe_to_columns_rows(df: pd.DataFrame) -> tuple[list[str], list[list[Any]]]:
    """
    แปลง DataFrame ที่ Camelot คืนมาให้เป็น (columns, rows)
    โดยสมมติว่า row แรกคือ header
    """
    # แปลงทุก cell เป็น string ก่อน เพื่อความสม่ำเสมอ
    df_str = df.astype(str)

    # สมมติว่า row แรกคือ header
    header = df_str.iloc[0].tolist()
    data_rows = df_str.iloc[1:].values.tolist()

    # ลบ header ที่ว่างเปล่าบางส่วน
    header = [h.strip() for h in header]

    return header, data_rows


def extract_tables(
    file_path: str | Path,
    doc_id: str,
    doc_type: str = "generic",
    pages: str = "all",
    flavor_priority: Optional[list[str]] = None,
) -> List[TableBlock]:
    """
    ดึงตารางจาก PDF 1 ไฟล์ทั้งหมด

    :param file_path: path ไปยัง PDF
    :param doc_id: ไอดีเอกสาร (ใช้เชื่อมกับ DocumentMetadata)
    :param doc_type: ประเภทเอกสาร (เผื่อใช้ logic เพิ่มเติมในอนาคต)
    :param pages: หน้า เช่น "1", "1,2,3", "1-3", "all"
    :param flavor_priority: ลำดับการลอง flavor ของ Camelot เช่น ["lattice", "stream"]
    :return: list[TableBlock]
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    # ค่า default: ลอง lattice ก่อน ถ้าไม่ได้ค่อยลอง stream
    if flavor_priority is None:
        flavor_priority = ["lattice", "stream"]

    all_tables: List[TableBlock] = []
    table_index = 0

    # ลองแต่ละ flavor ตามลำดับจนกว่าจะเจอตาราง
    for flavor in flavor_priority:
        try:
            tables = camelot.read_pdf(str(path), pages=pages, flavor=flavor)
        except Exception as e:
            # ถ้า flavor นี้ใช้ไม่ได้ (เช่น PDF ไม่มีเส้นตารางสำหรับ lattice) ก็ข้าม
            print(f"[table_extractor] Error using flavor='{flavor}': {e}")
            continue

        if tables.n == 0:
            # ไม่มีตารางใน flavor นี้
            continue

        # ถ้า flavor นี้เจอตารางแล้ว เราจะใช้ผลของ flavor นี้เลย
        for t in tables:
            df: pd.DataFrame = t.df

            # แปลง DataFrame -> columns, rows
            columns, rows = _dataframe_to_columns_rows(df)

            # เดา category แบบง่าย ๆ
            category = _guess_table_category(df)

            # พยายามดึง bbox ถ้ามี (บาง version ของ Camelot มี attribute _bbox)
            bbox: Optional[BBox] = None
            if hasattr(t, "_bbox") and t._bbox is not None:
                x1, y1, x2, y2 = t._bbox
                bbox = (float(x1), float(y1), float(x2), float(y2))

            table_index += 1
            table_id = f"tbl_{table_index:04d}"

            table_block = TableBlock(
                id=table_id,
                doc_id=doc_id,
                page=t.page,              # page index ที่ Camelot แยกได้
                name=f"table_{table_index}",
                section=None,             # ภายหลังให้ segmenter ใส่
                category=category,        # เดาจาก header / columns
                columns=columns,
                rows=rows,
                bbox=bbox,
                extra={
                    "camelot_flavor": flavor,
                    "parsing_report": t.parsing_report,  # ใช้ debug ได้
                    "doc_type": doc_type,
                },
            )
            all_tables.append(table_block)

        # ถ้า flavor นี้เจอตารางแล้ว เราไม่จำเป็นต้องลอง flavor อื่นต่อ
        if all_tables:
            break

    return all_tables


if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(description="Extract tables from PDF into TableBlock list.")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--doc-id", help="Document ID (default: stem of file name)")
    parser.add_argument("--doc-type", default="generic", help="Document type")
    parser.add_argument("--pages", default="all", help="Pages to parse (e.g., '1', '1-3', 'all')")
    args = parser.parse_args()

    pdf_path = args.pdf_path
    doc_id = args.doc_id or Path(pdf_path).stem

    tables = extract_tables(
        file_path=pdf_path,
        doc_id=doc_id,
        doc_type=args.doc_type,
        pages=args.pages,
    )

    print(f"Extracted {len(tables)} tables.")
    data = [t.to_dict() for t in tables]
    print(json.dumps(data, ensure_ascii=False, indent=2))
