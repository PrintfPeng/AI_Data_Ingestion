from __future__ import annotations

import json
from pathlib import Path
from typing import List

from ..models import (
    DocumentBundle,
    ImageItem,
    Metadata,
    TableItem,
    TextItem,
)


def _load_json(path: Path):
    """
    helper เล็ก ๆ โหลด JSON จากไฟล์
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def load_document_bundle(base_dir: str, doc_id: str) -> DocumentBundle:
    """
    โหลดข้อมูลของเอกสาร 1 ชุด (doc_id เดียว) จากโฟลเดอร์ที่มี
    - text.json
    - table.json
    - image.json
    - metadata.json

    แล้วรวมออกมาเป็น DocumentBundle
    """

    base_path = Path(base_dir)

    # 1) metadata.json – เป็น object เดียว
    metadata_raw = _load_json(base_path / "metadata.json")
    # ถ้าต่อไปมีหลาย doc share metadata ค่อยเพิ่ม logic ตรงนี้อีกที
    if metadata_raw.get("doc_id") != doc_id:
        # ตอนนี้ assume ว่า 1 โฟลเดอร์ = 1 doc
        # ถ้าไม่ตรงก็เตือนให้รู้ก่อน
        raise ValueError(
            f"metadata.doc_id ({metadata_raw.get('doc_id')}) != requested doc_id ({doc_id})"
        )

    metadata = Metadata(**metadata_raw)

    # 2) text.json – list ของ block
    text_list_raw = _load_json(base_path / "text.json")
    texts: List[TextItem] = [
        TextItem(**item)
        for item in text_list_raw
        if item.get("doc_id") == doc_id
    ]

    # 3) table.json – list ของ table
    table_list_raw = _load_json(base_path / "table.json")
    tables: List[TableItem] = [
        TableItem(**item)
        for item in table_list_raw
        if item.get("doc_id") == doc_id
    ]

    # 4) image.json – list ของ image
    image_list_raw = _load_json(base_path / "image.json")
    images: List[ImageItem] = [
        ImageItem(**item)
        for item in image_list_raw
        if item.get("doc_id") == doc_id
    ]

    bundle = DocumentBundle(
        metadata=metadata,
        texts=texts,
        tables=tables,
        images=images,
    )
    return bundle
