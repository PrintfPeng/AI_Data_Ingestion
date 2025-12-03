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
    if "statement" in file_name or "bank" in file_name:
        return "bank_statement"
    if "invoice" in file_name:
        return "invoice"
    if "receipt" in file_name or "slip" in file_name:
        return "receipt"
    if "po" in file_name or "purchase" in file_name:
        return "purchase_order"

    # rule จาก keyword
    if "ใบแจ้งยอด" in sample or "ยอดคงเหลือ" in sample or "account" in sample:
        return "bank_statement"

    if "tax invoice" in sample or "ใบกำกับภาษี" in sample:
        return "invoice"

    if "received with thanks" in sample or "ใบเสร็จ" in sample:
        return "receipt"

    # ตรวจ pattern ตัวเลขยอดเงิน
    import re
    if re.search(r"\d{1,3}(,\d{3})+\.\d{2}", sample):
        return "bank_statement"

    # ธนาคารรัฐไทย → GFMIS
    if "gfmis" in sample or "gfmis thai" in sample:
        return "bank_statement"

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
    - ถ้า model_name ไม่มี → auto select จาก MODEL_CANDIDATES
    """

    try:
        import google.generativeai as genai
        import os

        # โหลด API KEY
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("[document_classifier] GOOGLE_API_KEY not set → fallback")
            return classify_document_rule_based(doc)

        genai.configure(api_key=api_key)

        # เลือกโมเดลอัตโนมัติ
        if model_name is None:
            available = [m.name for m in genai.list_models()]
            chosen = None
            for cand in MODEL_CANDIDATES:
                if cand in available:
                    chosen = cand
                    break

            if chosen is None:
                print("[document_classifier] No valid Gemini model → fallback")
                return classify_document_rule_based(doc)

            model_name = chosen

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

Respond ONLY with a label from the list above.
"""

        resp = model.generate_content(prompt)
        answer = (resp.text or "").strip().lower()

        # ทำให้ label เข้ากับชุดที่เรามี
        for t in CANDIDATE_TYPES:
            if answer == t:
                return t

        # fuzzy match
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
# MAIN ENTRY
# ============================================================
def classify_document(
    doc: IngestedDocument,
    use_gemini: bool = False,
) -> str:
    """เลือกว่าจะใช้ AI หรือ rule-based"""
    if use_gemini:
        return classify_document_with_gemini(doc)
    return classify_document_rule_based(doc)


# TEST MODE
if __name__ == "__main__":
    import json
    from pathlib import Path

    meta_path = Path("ingested/sample/metadata.json")
    text_path = Path("ingested/sample/text.json")

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
