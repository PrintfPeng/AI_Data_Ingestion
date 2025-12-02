from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings


_EMBEDDING_MODEL_NAME = "models/text-embedding-004"


def get_embedding_client() -> GoogleGenerativeAIEmbeddings:
    """
    เตรียม client สำหรับสร้าง embedding ด้วย Gemini
    ต้องมี GOOGLE_API_KEY อยู่ใน .env หรือ environment
    """
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        # ให้ error ชัด ๆ เผื่อลืมตั้งค่า
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Please add it to your environment or .env file."
        )

    # langchain-google-genai จะอ่าน key จาก env อยู่แล้ว ไม่ต้องส่งซ้ำ
    embeddings = GoogleGenerativeAIEmbeddings(
        model=_EMBEDDING_MODEL_NAME,
    )
    return embeddings


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    helper embed เป็น batch
    """
    embeddings = get_embedding_client()
    vectors = embeddings.embed_documents(texts)
    return vectors
