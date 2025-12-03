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


# -------------------------------------------------------------------
# 1) Rule-based intent (ถูก ๆ เร็ว ๆ ก่อน)
# -------------------------------------------------------------------
def _rule_based_intent(query: str) -> Optional[str]:
    """
    เดา intent แบบ rule-based ง่าย ๆ จาก keyword
    คืนค่า: "text" | "table" | "both" | None

    ใช้ได้กับเอกสารหลายประเภท ไม่ได้จำกัดแค่การเงิน
    """
    q = query.lower().strip()

    # keyword ที่มักเกี่ยวข้องกับ "ตารางข้อมูล" ในหลาย ๆ โดเมน
    table_keywords = [
        "ตาราง",
        "table",
        "รายการ",
        "รายชื่อ",
        "สรุปข้อมูล",
        "สรุปผล",
        "สถิติ",
        "สรุปคะแนน",
        "แถวที่",
        "คอลัมน์",
        "column",
        "row",
        "ชีท",
        "sheet",
    ]

    # keyword ที่มักเกี่ยวข้องกับรูป / กราฟ
    image_keywords = [
        "รูป",
        "รูปภาพ",
        "image",
        "logo",
        "โลโก้",
        "กราฟ",
        "graph",
        "chart",
        "แผนภาพ",
        "diagram",
        "แผนภูมิ",
    ]

    is_table = any(kw in q for kw in table_keywords)
    is_image = any(kw in q for kw in image_keywords)

    if is_table and not is_image:
        return "table"
    if is_image and not is_table:
        # ตอนนี้เรายังไม่มี RAG ฝั่ง image แยกชัด → ถือเป็น both ไปก่อน
        return "both"
    if is_table and is_image:
        return "both"

    # ถ้าไม่มี keyword พิเศษ → น่าจะถามเชิงเนื้อหาบรรยาย
    if q:
        return "text"

    return None


# -------------------------------------------------------------------
# 2) LLM-based intent (ละเอียดแต่แพงกว่า)
# -------------------------------------------------------------------
async def classify_query_intent(query: str) -> str:
    """
    ใช้ LLM ช่วยบอกว่า query ต้องไปดู
    - "text"
    - "table"
    - "both"

    ใช้กับเอกสารทั่วไป: รายงาน, คู่มือ, สัญญา, เอกสารการเงิน ฯลฯ
    """
    llm = _get_llm()

    system_prompt = (
        "คุณเป็นตัวจัดประเภทคำถามเกี่ยวกับเอกสาร PDF หลายประเภท "
        "เช่น รายงานบริษัท รายงานวิชาการ คู่มือ สัญญา เอกสารการเงิน ฯลฯ\n"
        "เป้าหมายคือบอกว่าเมื่อจะตอบคำถามนี้ เราควรโฟกัสข้อมูลจากไหนเป็นหลัก:\n"
        "- text  = เนื้อหาบรรยาย / ย่อหน้า / ข้อความยาว ๆ\n"
        "- table = ข้อมูลในตาราง เช่น แถว-คอลัมน์ รายการ สรุปตัวเลข\n"
        "- both  = ต้องใช้ทั้งข้อความและข้อมูลตารางร่วมกัน\n\n"
        "ให้ตอบสั้น ๆ เป็นคำเดียวเท่านั้น หนึ่งใน: text, table, both.\n"
    )

    user_prompt = f"คำถาม: {query}\n\nตอบแค่หนึ่งคำ: text, table หรือ both"

    resp = await llm.ainvoke(
        [("system", system_prompt), ("user", user_prompt)]
    )
    raw = (resp.content or "").strip().lower()

    if "both" in raw:
        return "both"
    if "table" in raw:
        return "table"
    return "text"


# -------------------------------------------------------------------
# 3) รวม context จากเอกสาร
# -------------------------------------------------------------------
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
        doc_type = meta.get("doc_type") or "unknown"
        header = (
            f"[{i}] (doc_id={doc_id}, page={page}, "
            f"source={source}, doc_type={doc_type})"
        )
        parts.append(f"{header}\n{d.page_content}")
    return "\n\n".join(parts)


# -------------------------------------------------------------------
# 4) main RAG function
# -------------------------------------------------------------------
async def answer_question(
    query: str,
    doc_ids: Optional[List[str]] = None,
    top_k: int = 5,
    mode: str = "auto",  # "auto" | "text" | "table" | "both"
) -> Dict:
    """
    RAG flow:
    1) ตัดสินใจ intent → text / table / both (rule-based + LLM)
    2) search จาก vector DB ด้วย filter ที่เหมาะสม (doc_ids + source)
    3) รวม context เป็น prompt
    4) ให้ LLM ตอบ

    รองรับเอกสารหลายประเภท ไม่จำกัดแค่ statement การเงิน
    """

    # ----------------------------------------
    # 1) ตัดสินใจ intent ตาม mode
    # ----------------------------------------
    if mode == "auto":
        # ลองใช้ rule-based ก่อน
        intent = _rule_based_intent(query)
        if intent is None:
            # ถ้าดูไม่ออก → ให้ LLM ช่วย classify
            intent = await classify_query_intent(query)
    elif mode in ("text", "table", "both"):
        # ใช้ค่าที่ user เลือกบังคับเลย
        intent = mode
    else:
        # mode แปลก → ถือว่า auto
        intent = _rule_based_intent(query) or await classify_query_intent(query)

    # map intent -> source_filter
    if intent == "text":
        source_filter = ["text"]
    elif intent == "table":
        source_filter = ["table"]
    elif intent == "both":
        # กรณี both: เราสนทั้ง text + table เป็นหลัก
        source_filter = ["text", "table"]
    else:
        # กันพลาด
        source_filter = None

    # ----------------------------------------
    # 2) search similar docs จาก vector DB
    # ----------------------------------------
    docs = search_similar(
        query=query,
        k=top_k,
        doc_ids=doc_ids,
        sources=source_filter,
    )

    if not docs:
        return {
            "answer": "ไม่พบข้อมูลที่เกี่ยวข้องเพียงพอในฐานข้อมูลเอกสาร",
            "sources": [],
            "intent": intent,
            "mode": mode,
        }

    # ----------------------------------------
    # 3) เตรียม context ให้ LLM
    # ----------------------------------------
    context_text = _build_context_text(docs)

    system_prompt = (
        "คุณเป็นผู้ช่วยอ่านและวิเคราะห์เอกสาร PDF หลายประเภท "
        "(เช่น รายงานบริษัท รายงานวิชาการ คู่มือ สัญญา เอกสารการเงิน ฯลฯ).\n"
        "ให้ตอบคำถามโดยอ้างอิงเฉพาะจาก CONTEXT ด้านล่างนี้เท่านั้น "
        "ห้ามเดาเกินข้อมูลในเอกสาร.\n"
        "ถ้าข้อมูลไม่พอ ให้ตอบว่า 'ไม่ทราบจากข้อมูลที่มีอยู่'.\n\n"
        f"(query intent: {intent}, mode: {mode})\n\n"
        "=== CONTEXT START ===\n"
        f"{context_text}\n"
        "=== CONTEXT END ===\n\n"
        "ตอนนี้ให้ตอบคำถามของผู้ใช้ด้านล่างให้กระชับ ชัดเจน "
        "และอ้างอิงจากเนื้อหาใน CONTEXT เท่านั้น."
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

    # ----------------------------------------
    # 4) เตรียม sources สำหรับ frontend + history
    # ----------------------------------------
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
        "mode": mode,
    }
