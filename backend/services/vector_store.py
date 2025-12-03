from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.vectorstores import Chroma

from .chunking import Chunk
from .embeddings import get_embedding_client

from fastapi import HTTPException
from langchain_google_genai._common import GoogleGenerativeAIError

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"


def search_similar(query: str, top_k: int = 5):
    try:
        docs = vectordb.similarity_search(query, k=top_k)
        return docs
    except GoogleGenerativeAIError as e:
        print("Embedding error:", e)  # log หลังบ้าน
        raise HTTPException(
            status_code=500,
            detail="Embedding error: โปรดตรวจสอบ GOOGLE_API_KEY ใน .env (อาจใช้ไม่ได้ / หมดอายุ / ถูกปิด)"
        )


def get_vector_store(
    persist_directory: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """
    คืน Chroma vector store ที่ผูกกับ Gemini embeddings
    """
    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)

    embeddings = get_embedding_client()

    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_path),
    )
    return vectordb


def _normalize_metadata(md: dict) -> dict:
    """
    Chroma รับได้เฉพาะค่าแบบ str/int/float/bool/None
    อันนี้เลยแปลงพวก list/dict/object ให้กลายเป็น string
    """
    simple: dict = {}
    for k, v in md.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            simple[k] = v
        else:
            # แปลงของซับซ้อนเป็น string เช่น bbox, columns, ฯลฯ
            simple[k] = str(v)
    return simple


def index_chunks(
    chunks: List[Chunk],
    persist_directory: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """
    เอา chunks ทั้งหมดไปเก็บใน Chroma
    """
    if not chunks:
        return

    vectordb = get_vector_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    texts = [c.content for c in chunks]

    raw_metadatas = [
        c.metadata
        | {
            "doc_id": c.doc_id,
            "doc_type": c.doc_type,
            "source": c.source,
            "page": c.page,
            "chunk_id": c.id,
        }
        for c in chunks
    ]

    metadatas = [_normalize_metadata(md) for md in raw_metadatas]

    ids = [c.id for c in chunks]

    vectordb.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
    )

    vectordb.persist()

def search_similar(
    query: str,
    k: int = 5,
    persist_directory: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    doc_ids: list[str] | None = None,
    sources: list[str] | None = None,
):
    """
    search จาก Chroma พร้อม filter ตาม doc_ids / source ได้
    ใช้ syntax filter ใหม่ของ Chroma:

    - ถ้ามีเงื่อนไขเดียว → {"doc_id": {"$in": [...]}}
    - ถ้ามีหลายเงื่อนไข → {"$and": [ {...}, {...} ]}
    """

    vectordb = get_vector_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    conditions: list[dict] = []

    if doc_ids:
        conditions.append({"doc_id": {"$in": doc_ids}})

    if sources:
        conditions.append({"source": {"$in": sources}})

    if len(conditions) == 0:
        filter_dict = None
    elif len(conditions) == 1:
        filter_dict = conditions[0]
    else:
        filter_dict = {"$and": conditions}

    if filter_dict:
        docs = vectordb.similarity_search(
            query,
            k=k,
            filter=filter_dict,
        )
    else:
        docs = vectordb.similarity_search(query, k=k)

    return docs
