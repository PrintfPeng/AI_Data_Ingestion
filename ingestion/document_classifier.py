from __future__ import annotations

"""
document_classifier.py

หน้าที่:
- จำแนกประเภทเอกสารจากข้อความ text blocks และชื่อไฟล์
- รองรับ 2 โหมด:
    1) Rule-based (ไม่ใช้โมเดล)
    2) Gemini LLM-based (ใช้โมเดล gemini-2.5 / 2.0)

ขั้นตอน:
- อ่าน TextBlock
- รวมข้อความบางส่วน (sample_text)
- Rule-based → ถ้าดูไม่ออก
- ถ้า use_gemini=True → ใช้ LLM ช่วย classify
"""

from typing import List, Optional
from ingestion.schema import IngestedDocument, TextBlock, DocumentMetadata


# -------------------------
# Document Label Set
# -------------------------
CANDIDATE_TYPES = [
    "bank_statement",
    "invoice",
    "receipt",
    "purchase_order",
    "delivery_note",
    "tax_form",
    "generic",
]

# -------------------------
# Gemini Model Candidates
# -------------------------
MODEL_CANDIDATES = [
    "models/gemini-2.5-flash",        # ตัวเร็ว ใช้แทนได้
]

GEMINI_MODEL_NAME = "models/gemini-2.5-pro"


# -------------------------
# HELPER FUNCTION
# -------------------------
def _collect_sample_text(texts: List[TextBlock], max_chars: int = 6000) -> str:
    """รวม text block แรก ๆ เอามาเป็น sample text สำหรับ rule/LLM"""
    chunks = []
    total = 0
    for t in texts:
        if not t.content:
            continue
        if total + len(t.content) > max_chars:
            break
        chunks.append(t.content)
        total += len(t.content)
    return "\n".join(chunks)


# ============================================================
# 1) RULE-BASED CLASSIFIER (พื้นฐาน)
# ============================================================
def classify_document_rule_based(doc: IngestedDocument) -> str:
    """จำแนกเอกสารแบบง่าย ๆ ไม่ใช้ AI"""
    file_name = doc.metadata.file_name.lower()
    sample = _collect_sample_text(doc.texts).lower()

    # rule from file name
    if "statement" in file_name and "bank" in file_name:
        return "bank_statement"
    if "invoice" in file_name:
        return "invoice"
    if "receipt" in file_name:
        return "receipt"
    if "po_" in file_name or "purchase_order" in file_name:
        return "purchase_order"

    # rule from content
    if "account summary" in sample and "statement period" in sample:
        return "bank_statement"
    if "invoice no" in sample or "tax invoice" in sample:
        return "invoice"
    if "receipt no" in sample or "thank you for your payment" in sample:
        return "receipt"
    if "purchase order" in sample:
        return "purchase_order"

    return "generic"


# ============================================================
# 2) GEMINI-BASED CLASSIFIER
# ============================================================
def classify_document_with_gemini(
    doc: IngestedDocument,
    model_name: Optional[str] = None,
) -> str:
    """
    ใช้ Gemini จำแนกประเภทเอกสาร
    - ใช้โมเดล fix ตัวเดียว (GEMINI_MODEL_NAME)
    - ไม่เรียก list_models() แล้ว เพื่อลดโอกาสเจอ error แปลก ๆ จาก API
    """
    try:
        import google.generativeai as genai
        import os

        # โหลด API KEY
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[document_classifier] GEMINI_API_KEY not set → fallback")
            return classify_document_rule_based(doc)

        genai.configure(api_key=api_key)

        # ถ้าไม่ส่งชื่อโมเดลมา ใช้ค่าคงที่ของเราเอง
        if model_name is None:
            model_name = GEMINI_MODEL_NAME

        print(f"[document_classifier] Using Gemini model: {model_name}")
        model = genai.GenerativeModel(model_name)

        # เตรียมข้อความ
        sample_text = _collect_sample_text(doc.texts, max_chars=6000)

        prompt = f"""
You are a professional document classifier.

Classify the following PDF text into ONE label:

{CANDIDATE_TYPES}

File name: {doc.metadata.file_name}

Text sample:
\"\"\"{sample_text}\"\"\"
"""

        resp = model.generate_content(prompt)
        answer = (resp.text or "").strip().lower()
        print("[document_classifier] Gemini raw answer:", answer)

        # normalize
        answer = answer.replace("label:", "").strip()

        # fuzzy match แบบง่าย ๆ
        if "bank" in answer and "statement" in answer:
            return "bank_statement"
        if "invoice" in answer:
            return "invoice"
        if "receipt" in answer:
            return "receipt"
        if "purchase" in answer:
            return "purchase_order"

        return "generic"

    except Exception as e:
        print(f"[document_classifier] Gemini classify failed: {e}")
        print("[document_classifier] Fallback to rule-based")
        return classify_document_rule_based(doc)


# ============================================================
# PUBLIC ENTRYPOINT
# ============================================================
def classify_document(doc: IngestedDocument, use_gemini: bool = True) -> str:
    """
    เลือกว่าจะใช้ rule-based หรือ Gemini
    """
    if not use_gemini:
        return classify_document_rule_based(doc)

    # พยายามใช้ Gemini ก่อน
    return classify_document_with_gemini(doc)


# ============================================================
# CLI TEST
# ============================================================
if __name__ == "__main__":
    import json
    from pathlib import Path

    # ทดสอบโหลดจาก ingested/sample
    root = Path("ingested") / "sample"
    meta_path = root / "metadata.json"
    text_path = root / "text.json"

    if not meta_path.exists() or not text_path.exists():
        print("Please run ingestion first.")
    else:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        texts = json.loads(text_path.read_text(encoding="utf-8"))

        doc = IngestedDocument(
            metadata=DocumentMetadata(**meta),
            texts=[TextBlock(**t) for t in texts],
            tables=[],
            images=[],
        )

        print("Rule-based:", classify_document(doc, use_gemini=False))
        print("Gemini:", classify_document(doc, use_gemini=True))
