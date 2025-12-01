from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple


# Bounding box: (x1, y1, x2, y2) ในหน่วยพิกัดของหน้า PDF
BBox = Tuple[float, float, float, float]


@dataclass
class DocumentMetadata:
    """ข้อมูลเมตาของเอกสารต้นฉบับ 1 ไฟล์"""
    doc_id: str                  # ไอดีภายในระบบ เช่น "doc_001"
    file_name: str               # ชื่อไฟล์จริง เช่น "statement_nov_2025.pdf"
    doc_type: str                # ประเภทเอกสาร เช่น "bank_statement", "receipt", "invoice"
    page_count: int              # จำนวนหน้า
    ingested_at: str             # เวลา ingest (ISO string) เช่น "2025-12-01T10:00:00"
    source: str = "uploaded"     # แหล่งที่มา เช่น "uploaded", "api", "scanner"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TextBlock:
    """บล็อกข้อความ 1 ก้อน ในเอกสาร"""
    id: str                      # ไอดีของ block เช่น "txt_001"
    doc_id: str                  # อ้างอิงไปที่ DocumentMetadata.doc_id
    page: int                    # หน้า (เริ่มจาก 1)
    content: str                 # เนื้อความจริง ๆ
    section: Optional[str] = None    # เช่น "summary", "header", "transaction_detail"
    category: Optional[str] = None   # label ที่ใช้กับ RAG เช่น "narrative", "note"
    bbox: Optional[BBox] = None      # พิกัดบนหน้า PDF
    extra: Dict[str, Any] = field(default_factory=dict)  # ช่องไว้เก็บอะไรเพิ่มในอนาคต

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TableBlock:
    """โครงสร้างตาราง 1 ตาราง"""
    id: str
    doc_id: str
    page: int
    name: Optional[str] = None       # ชื่อสั้น ๆ ของตาราง เช่น "transaction_table"
    section: Optional[str] = None    # เช่น "transaction", "summary"
    category: Optional[str] = None   # เช่น "transaction_table", "item_list"
    columns: List[str] = field(default_factory=list)
    rows: List[List[Any]] = field(default_factory=list)
    bbox: Optional[BBox] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ImageBlock:
    """ข้อมูลรูปภาพ 1 รูป ในเอกสาร"""
    id: str
    doc_id: str
    page: int
    file_path: str                   # path ไฟล์รูปที่ export ออกมา เช่น "images/doc_001/img_001.png"
    caption: Optional[str] = None    # caption หรือข้อความรอบ ๆ รูป
    section: Optional[str] = None
    category: Optional[str] = None   # เช่น "logo", "chart", "signature"
    bbox: Optional[BBox] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IngestedDocument:
    """
    ตัวแทนผลลัพธ์การ ingest เอกสาร 1 ไฟล์
    รวม metadata + text blocks + tables + images
    """
    metadata: DocumentMetadata
    texts: List[TextBlock] = field(default_factory=list)
    tables: List[TableBlock] = field(default_factory=list)
    images: List[ImageBlock] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """แปลงทั้งเอกสารเป็น dict พร้อมสำหรับ serialize เป็น JSON"""
        return {
            "metadata": self.metadata.to_dict(),
            "texts": [t.to_dict() for t in self.texts],
            "tables": [tb.to_dict() for tb in self.tables],
            "images": [im.to_dict() for im in self.images],
        }
