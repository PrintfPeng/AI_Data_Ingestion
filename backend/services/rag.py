from __future__ import annotations

from typing import Dict, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from .vector_store import search_similar


def _get_llm() -> ChatGoogleGenerativeAI:
    """
    เตรียม LLM (Gemini) สำหรับตอบคำถาม
    ใช้ GOOGLE_API_KEY จาก .env เหมือน embeddings
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
    )
    return llm

async def classify_query_intent(query: str) -> str:
    """
    ใช้ LLM ช่วยบอกว่า query ต้องไปดู
    - "text"
    - "table"
    - "both"

    return เป็น string หนึ่งในชุดนั้น
    """
    llm = _get_llm()

    system_prompt = (
        "คุณเป็นตัวจัดประเภทคำถามเกี่ยวกับเอกสารการเงิน/ธุรกรรม.\n"
        "ให้ตอบสั้น ๆ เป็นคำเดียวเท่านั้น หนึ่งใน: text, table, both.\n\n"
        "- ถ้าคำถามถามถึงยอดคงเหลือ, คำอธิบาย, เนื้อหาบรรยาย → text\n"
        "- ถ้าคำถามถามถึงข้อมูลในตาราง, แถว/คอลัมน์, ยอดรวมจากตาราง → table\n"
        "- ถ้าดูเหมือนต้องใช้ทั้งเนื้อหาและตาราง → both\n"
    )

    user_prompt = f"คำถาม: {query}\n\nตอบแค่หนึ่งคำ: text, table หรือ both"

    resp = await llm.ainvoke(
        [("system", system_prompt), ("user", user_prompt)]
    )
    raw = resp.content.strip().lower()

    if "table" in raw:
        return "table"
    if "both" in raw:
        return "both"
    return "text"



def _build_context_text(docs) -> str:
    """
    รวม context จากเอกสารที่ค้นมาให้ LLM
    """
    parts: List[str] = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        doc_id = meta.get("doc_id", "unknown")
        page = meta.get("page", "?")
        source = meta.get("source", "text")
        header = f"[{i}] (doc_id={doc_id}, page={page}, source={source})"
        parts.append(f"{header}\n{d.page_content}")
    return "\n\n".join(parts)


async def answer_question(
    query: str,
    doc_ids: Optional[List[str]] = None,
    top_k: int = 5,
    mode: str = "auto",   # "auto" | "text" | "table" | "both"
) -> Dict:

    """
    RAG flow:
    1) classify intent → text / table / both
    2) search จาก vector DB ด้วย filter ที่เหมาะสม
    3) รวม context เป็น prompt
    4) ให้ LLM ตอบ
    """

    # 1) classify intent
    # 1) ตัดสินใจ intent / source_filter ตาม mode
    if mode == "auto":
        # ให้ LLM ช่วย classify
        intent = await classify_query_intent(query)
    elif mode in ("text", "table", "both"):
        # ใช้ค่าที่ user เลือกบังคับเลย
        intent = mode
    else:
        intent = "auto"  # กันพลาด ใส่ค่าแปลกๆ มาก็ถือว่า auto
        intent = await classify_query_intent(query)

    if intent == "text":
        source_filter = ["text"]
    elif intent == "table":
        source_filter = ["table"]
    else:
        # both หรือ auto แบบไม่จำกัด
        source_filter = None


    # 2) search similar docs (ให้ vector_store ใช้ filter ตาม doc_ids + source)
    docs = search_similar(
        query,
        k=top_k,
        doc_ids=doc_ids,
        sources=source_filter,
    )

    if not docs:
        return {
            "answer": "ไม่พบข้อมูลที่เกี่ยวข้องเพียงพอในฐานข้อมูลเอกสาร",
            "sources": [],
        }

    context_text = _build_context_text(docs)

    system_prompt = (
        "คุณเป็นผู้ช่วยวิเคราะห์เอกสารการเงิน/ธุรกรรมจาก PDF.\n"
        "ให้ตอบคำถามโดยอ้างอิงเฉพาะจาก CONTEXT ด้านล่างนี้เท่านั้น.\n"
        "ถ้าข้อมูลไม่พอ ให้ตอบว่า 'ไม่ทราบจากข้อมูลที่มีอยู่'.\n\n"
        f"(query intent: {intent})\n\n"
        "=== CONTEXT START ===\n"
        f"{context_text}\n"
        "=== CONTEXT END ===\n\n"
        "ตอนนี้ให้ตอบคำถามของผู้ใช้ด้านล่างให้กระชับและชัดเจน."
    )

    user_prompt = query

    llm = _get_llm()
    resp = await llm.ainvoke(
        [
            ("system", system_prompt),
            ("user", user_prompt),
        ]
    )
    answer_text = resp.content if hasattr(resp, "content") else str(resp)

    sources = []
    for d in docs:
        meta = d.metadata or {}
        sources.append(
            {
                "doc_id": meta.get("doc_id"),
                "page": meta.get("page"),
                "source": meta.get("source"),
                "chunk_id": meta.get("chunk_id"),
            }
        )

    return {
        "answer": answer_text,
        "sources": sources,
        "intent": intent,
    }
