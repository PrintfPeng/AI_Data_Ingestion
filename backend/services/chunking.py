from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

from ..models import DocumentBundle, TableItem


class Chunk(BaseModel):
    """
    หนึ่งชิ้นข้อมูลที่เราจะส่งเข้า Vector DB
    """

    id: str          # ต้อง unique ทั่วทุก doc
    doc_id: str
    doc_type: str
    source: Literal["text", "table", "image"]
    page: Optional[int] = None
    content: str
    metadata: Dict[str, Any] = {}


def _table_to_text(table: TableItem) -> str:
    header = f"Table {table.name} (page {table.page})"
    col_line = " | ".join(table.columns)
    row_lines = []

    max_rows = min(len(table.rows), 10)
    for row in table.rows[:max_rows]:
        row_lines.append(" | ".join(row))

    body = "\n".join(row_lines)
    text = f"{header}\nColumns: {col_line}\nRows:\n{body}"
    return text


def text_items_to_chunks(bundle: DocumentBundle) -> List[Chunk]:
    chunks: List[Chunk] = []

    for item in bundle.texts:
        if not item.content:
            continue

        chunk = Chunk(
            id=f"{item.doc_id}::text::{item.id}",   # <<< ใส่ doc_id เข้าไป
            doc_id=item.doc_id,
            doc_type=item.doc_type,
            source="text",
            page=item.page,
            content=item.content,
            metadata={
                "block_id": item.id,
                "section": item.section,
                "bbox": item.bbox,
                "page": item.page,
                "doc_type": item.doc_type,
            },
        )
        chunks.append(chunk)

    return chunks


def table_items_to_chunks(bundle: DocumentBundle) -> List[Chunk]:
    chunks: List[Chunk] = []

    for item in bundle.tables:
        text_representation = _table_to_text(item)

        chunk = Chunk(
            id=f"{item.doc_id}::table::{item.id}",  # <<< ใส่ doc_id เข้าไป
            doc_id=item.doc_id,
            doc_type=item.doc_type,
            source="table",
            page=item.page,
            content=text_representation,
            metadata={
                "table_id": item.id,
                "name": item.name,
                "columns": item.columns,
                "bbox": item.bbox,
                "page": item.page,
                "doc_type": item.doc_type,
            },
        )
        chunks.append(chunk)

    return chunks


def image_items_to_chunks(bundle: DocumentBundle) -> List[Chunk]:
    chunks: List[Chunk] = []

    for item in bundle.images:
        if not item.caption:
            continue

        chunk = Chunk(
            id=f"{item.doc_id}::image::{item.id}",  # <<< ใส่ doc_id เข้าไป
            doc_id=item.doc_id,
            doc_type=item.doc_type,
            source="image",
            page=item.page,
            content=item.caption,
            metadata={
                "image_id": item.id,
                "file_path": item.file_path,
                "bbox": item.bbox,
                "page": item.page,
                "doc_type": item.doc_type,
            },
        )
        chunks.append(chunk)

    return chunks
