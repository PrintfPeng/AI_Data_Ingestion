from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class TextItem(BaseModel):
    """
    แทน 1 block ใน text.json

    {
      "id": "txt_001",
      "doc_type": "bank_statement",
      "doc_id": "doc_001",
      "page": 1,
      "section": "summary",
      "content": "ยอดคงเหลือรวมสิ้นงวด...",
      "bbox": [x1, y1, x2, y2]
    }
    """

    id: str
    doc_type: str
    doc_id: str
    page: int
    section: Optional[str] = None
    content: str
    bbox: List[float]


class TableItem(BaseModel):
    """
    แทน 1 table ใน table.json

    {
      "id": "tbl_001",
      "doc_type": "bank_statement",
      "doc_id": "doc_001",
      "page": 2,
      "name": "transaction_table",
      "columns": ["date", "description", "amount", "balance"],
      "rows": [
        ["2025-11-01", "โอนออก xxxxxx", "-1000.00", "5000.00"]
      ],
      "bbox": [x1, y1, x2, y2]
    }
    """

    id: str
    doc_type: str
    doc_id: str
    page: int
    name: str
    columns: List[str]
    rows: List[List[str]]
    bbox: List[float]


class ImageItem(BaseModel):
    """
    แทน 1 image ใน image.json

    {
      "id": "img_001",
      "doc_type": "invoice",
      "doc_id": "doc_010",
      "page": 1,
      "file_path": "images/doc_010/img_001.png",
      "caption": "Company logo",
      "bbox": [x1, y1, x2, y2]
    }
    """

    id: str
    doc_type: str
    doc_id: str
    page: int
    file_path: str
    caption: Optional[str] = None
    bbox: List[float]


class Metadata(BaseModel):
    """
    metadata.json

    {
      "doc_id": "doc_001",
      "file_name": "statement_nov_2025.pdf",
      "doc_type": "bank_statement",
      "page_count": 8,
      "ingested_at": "2025-12-01T10:00:00",
      "source": "uploaded_by_user"
    }
    """

    doc_id: str
    file_name: str
    doc_type: str
    page_count: int
    ingested_at: datetime
    source: str


class DocumentBundle(BaseModel):
    """
    ของฝั่งคุณเอง: รวมข้อมูลทุกอย่างของ doc_id เดียวกัน
    จะใช้เป็น input หลักของ pipeline ฝั่ง RAG/DB
    """

    metadata: Metadata
    texts: List[TextItem] = []
    tables: List[TableItem] = []
    images: List[ImageItem] = []
