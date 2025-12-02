from __future__ import annotations

"""
semantic_enricher.py

รวมฟังก์ชัน:
1) Section Segmentation / Categorization สำหรับ TextBlock
2) Text Role Categorization (title, account_info, transaction_row ฯลฯ)
3) Table Normalization (header → canonical names)
4) Table Role Categorization (transaction_table / summary_table / other_table)
5) Mapping Prepare: ดึงรายการ transaction ออกมาในรูปแบบโครงสร้าง

ทำงานได้ทั้งแบบ:
- rule-based อย่างเดียว (ถ้าไม่มี GEMINI_API_KEY)
- ใช้ Gemini ช่วย (ถ้ามี GEMINI_API_KEY)
"""

from typing import List, Dict, Any, Optional
import os

from .schema import IngestedDocument, TextBlock, TableBlock

# ---------------------------
# Helper: Gemini model
# ---------------------------

GEMINI_MODEL = "models/gemini-2.5-pro"


def _get_gemini_model():
    """คืนโมเดล Gemini ถ้ามี API KEY; ถ้าไม่มีให้คืน None"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        return genai.GenerativeModel(GEMINI_MODEL)
    except Exception as e:
        print("[semantic_enricher] Cannot init Gemini:", e)
        return None


# ===========================
# 1) SECTION TAGGING
# ===========================

SECTION_LABELS = ["header", "summary", "transactions", "footer", "other"]


def _guess_section_rule(block: TextBlock) -> str:
    """
    rule-based แบบง่าย ๆ พอให้มี section ใช้งาน
    (ไม่ใช้ page_index/bbox เพราะ schema ไม่ได้การันตีว่ามี field นี้)
    """
    txt = (block.content or "").lower()

    if any(k in txt for k in ["summary", "สรุป", "overview"]):
        return "summary"

    if any(k in txt for k in ["รายการเดินบัญชี", "transaction", "movement"]):
        return "transactions"

    if any(k in txt for k in ["รายการ", "รายละเอียดบัญชี", "statement"]):
        return "transactions"

    if any(k in txt for k in ["ลงชื่อ", "ผู้มีอำนาจลงนาม", "ขอแสดงความนับถือ", "signature"]):
        return "footer"

    return "other"


def tag_sections(
    doc: IngestedDocument,
    use_gemini: bool = False,
) -> IngestedDocument:
    """
    ใส่ section label ลงใน TextBlock.extra["section"]
    ถ้า use_gemini=True + มี GEMINI_API_KEY → ใช้ LLM ช่วย
    ถ้า error หรือไม่มี KEY → fallback เป็น rule-based (_guess_section_rule)
    """

    model = _get_gemini_model() if use_gemini else None

    if model:
        # ทำทีละก้อนใหญ่ ให้โมเดลช่วย tag section เฉพาะบาง block แรก
        joined = []
        for i, b in enumerate(doc.texts):
            joined.append(f"[{i}] {b.content}")
        prompt_text = "\n".join(joined[:200])  # limit 200 blocks แรก

        prompt = f"""
You are a document segmenter.

For each numbered text block below, assign ONE section label from:
{SECTION_LABELS}

Format: one line per block, in the form:
index: label

Text blocks:
{prompt_text}
"""

        try:
            resp = model.generate_content(prompt)
            mapping: Dict[int, str] = {}
            for line in (resp.text or "").splitlines():
                line = line.strip()
                if not line or ":" not in line:
                    continue
                idx_str, label = line.split(":", 1)
                idx_str = idx_str.strip().strip("[]")
                label = label.strip().lower()
                try:
                    idx = int(idx_str)
                except ValueError:
                    continue
                if label not in SECTION_LABELS:
                    label = "other"
                mapping[idx] = label

            # apply mapping + fallback rule-based ที่ไม่มีใน mapping
            for i, b in enumerate(doc.texts):
                extra = dict(b.extra or {})
                extra["section"] = mapping.get(i, _guess_section_rule(b))
                b.extra = extra

            return doc

        except Exception as e:
            print("[semantic_enricher] Gemini section tagging failed:", e)
            print("[semantic_enricher] Fallback to rule-based tagging")

    # fallback: rule-based ทั้งหมด
    for b in doc.texts:
        extra = dict(b.extra or {})
        extra["section"] = _guess_section_rule(b)
        b.extra = extra

    return doc


# ===========================
# 2) TEXT ROLE CATEGORIZATION
# ===========================

TEXT_ROLE_LABELS = [
    "title",                # ชื่อรายงาน / header ใหญ่
    "account_info",         # ชื่อบัญชี / เลขบัญชี / ธนาคาร
    "transaction_header",   # header ส่วนหัวของตารางรายการเดินบัญชี
    "transaction_row",      # ข้อความบรรยายรายการ (เช่น “โอนจาก XXX”)
    "note",                 # หมายเหตุ / ข้อมูลเพิ่มเติม
    "footer_text",          # ข้อความปิดท้าย
    "other",
]


def _guess_text_role_rule(block: TextBlock) -> str:
    txt = (block.content or "").strip()
    lower = txt.lower()

    # title: ตัวใหญ่, มีคำพวก statement, report
    if len(txt) < 80 and any(k in lower for k in ["statement", "รายงาน", "account statement"]):
        return "title"

    if any(k in lower for k in ["เลขที่บัญชี", "account no", "account number", "branch", "ธนาคาร"]):
        return "account_info"

    if any(k in lower for k in ["วันที่", "วันเดือนปี", "transaction", "ยอดคงเหลือ", "จำนวนเงิน"]):
        return "transaction_header"

    if any(k in lower for k in ["หมายเหตุ", "note:", "หมาย เหตุ"]):
        return "note"

    if any(k in lower for k in ["ลงชื่อ", "ผู้มีอำนาจลงนาม", "ขอแสดงความนับถือ"]):
        return "footer_text"

    # heuristic: block อยู่ใน section=transactions และความยาวปานกลาง
    section = (block.extra or {}).get("section")
    if section == "transactions" and 10 <= len(txt) <= 200:
        return "transaction_row"

    return "other"


def categorize_text_blocks(
    doc: IngestedDocument,
    use_gemini: bool = False,
) -> IngestedDocument:
    """
    ใส่ role ให้ TextBlock.extra["role"] เช่น:
    - title
    - account_info
    - transaction_header
    - transaction_row
    - note
    - footer_text
    - other
    """

    model = _get_gemini_model() if use_gemini else None

    if model:
        # ส่งเฉพาะ subset ไปให้โมเดลช่วย classify
        joined = []
        for i, b in enumerate(doc.texts[:200]):
            section = (b.extra or {}).get("section", "unknown")
            joined.append(f"[{i}] (section={section}) {b.content}")
        prompt_text = "\n".join(joined)

        prompt = f"""
You are a document text role classifier for bank/financial PDFs.

For each text block, assign ONE role from:
{TEXT_ROLE_LABELS}

Format: one line per block:
index: role

Text blocks:
{prompt_text}
"""

        try:
            resp = model.generate_content(prompt)
            mapping: Dict[int, str] = {}
            for line in (resp.text or "").splitlines():
                line = line.strip()
                if not line or ":" not in line:
                    continue
                idx_str, label = line.split(":", 1)
                idx_str = idx_str.strip().strip("[]")
                label = label.strip().lower()
                try:
                    idx = int(idx_str)
                except ValueError:
                    continue
                if label not in TEXT_ROLE_LABELS:
                    label = "other"
                mapping[idx] = label

            for i, b in enumerate(doc.texts):
                extra = dict(b.extra or {})
                extra["role"] = mapping.get(i, _guess_text_role_rule(b))
                b.extra = extra

            return doc

        except Exception as e:
            print("[semantic_enricher] Gemini text role tagging failed:", e)
            print("[semantic_enricher] Fallback to rule-based text role")

    # fallback rule-based
    for b in doc.texts:
        extra = dict(b.extra or {})
        extra["role"] = _guess_text_role_rule(b)
        b.extra = extra

    return doc


# ===========================
# 4) TABLE NORMALIZER + ROLE
# ===========================

HEADER_NORMALIZATION_MAP = {
    # date
    "date": "date",
    "วันที่": "date",
    "วันเดือนปี": "date",
    # description
    "description": "description",
    "details": "description",
    "รายละเอียด": "description",
    "รายการ": "description",
    # debit / credit
    "debit": "amount_out",
    "withdrawal": "amount_out",
    "ถอน": "amount_out",
    "จ่าย": "amount_out",
    "credit": "amount_in",
    "deposit": "amount_in",
    "ฝาก": "amount_in",
    "รับ": "amount_in",
    # balance
    "balance": "balance",
    "ยอดคงเหลือ": "balance",
    "คงเหลือ": "balance",
    # amount generic
    "amount": "amount",
    "ยอดเงิน": "amount",
    "จำนวนเงิน": "amount",
}


def _normalize_header_name(h: str) -> str:
    """normalize header ชื่อ → canonical name ถ้าเจอ"""
    h_clean = (h or "").strip().lower()
    if not h_clean:
        return ""
    for key, canonical in HEADER_NORMALIZATION_MAP.items():
        if key in h_clean:
            return canonical
    return h_clean


TABLE_ROLE_LABELS = ["transaction_table", "summary_table", "other_table"]


def _guess_table_role(tb: TableBlock) -> str:
    header = getattr(tb, "header", []) or []
    header_lower = [str(h).lower() for h in header]

    if any("date" in h for h in header_lower) and any(
        x in h for h in header_lower for x in ["amount", "ยอดเงิน", "debit", "credit", "ยอดคงเหลือ", "balance"]
    ):
        return "transaction_table"

    if any(k in " ".join(header_lower) for k in ["summary", "สรุป", "total", "รวม"]):
        return "summary_table"

    return "other_table"


def normalize_tables(tables: List[TableBlock]) -> List[TableBlock]:
    """
    ปรับ header ของตารางให้เป็นชื่อมาตรฐาน เช่น
    - date
    - description
    - amount_in / amount_out
    - balance

    และใส่ role ลงใน TableBlock.extra["role"]
    """
    for tb in tables:
        header = list(getattr(tb, "header", []))
        normalized_header = [_normalize_header_name(h) for h in header]

        tb.header = normalized_header

        extra = dict(tb.extra or {})
        extra_norm = extra.get("header_normalization", {})
        extra_norm.update(
            {
                "original_header": header,
                "normalized_header": normalized_header,
            }
        )
        extra["header_normalization"] = extra_norm

        # ใส่ role ให้ table ด้วย
        extra["role"] = _guess_table_role(tb)
        tb.extra = extra

    return tables


# ===========================
# 5) MAPPING PREPARE
# ===========================

from typing import Optional


def _parse_float_safe(val: Optional[str]) -> Optional[float]:
    if val is None:
        return None
    s = str(val).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def extract_transactions_from_table(tb: TableBlock) -> List[Dict[str, Any]]:
    """
    พยายาม map ตารางให้กลายเป็น transaction records:
    - หา column index ของ date / description / amount / amount_in / amount_out / balance
    - คืน list ของ dict ที่มี key เหล่านี้
    """
    header = getattr(tb, "header", [])
    rows = getattr(tb, "rows", [])

    # map header → index
    name_to_idx: Dict[str, int] = {}
    for i, h in enumerate(header):
        if not h:
            continue
        name_to_idx[h] = i

    records: List[Dict[str, Any]] = []

    for row in rows:
        def col(name: str) -> Optional[str]:
            idx = name_to_idx.get(name)
            if idx is None or idx >= len(row):
                return None
            return str(row[idx]).strip()

        date = col("date")
        desc = col("description")

        amount_in = col("amount_in")
        amount_out = col("amount_out")
        amount = col("amount")
        balance = col("balance")

        if not any([date, desc, amount_in, amount_out, amount, balance]):
            continue

        record: Dict[str, Any] = {
            "date_raw": date,
            "description": desc,
            "amount_in_raw": amount_in,
            "amount_out_raw": amount_out,
            "amount_raw": amount,
            "balance_raw": balance,
            "amount_in": _parse_float_safe(amount_in) if amount_in else None,
            "amount_out": _parse_float_safe(amount_out) if amount_out else None,
            "amount": _parse_float_safe(amount) if amount else None,
            "balance": _parse_float_safe(balance) if balance else None,
        }

        records.append(record)

    return records


def prepare_mapping_payload(doc: IngestedDocument) -> Dict[str, Any]:
    """
    ดึงข้อมูลที่จำเป็นสำหรับทำ mapping ข้ามเอกสาร:
    - doc metadata
    - transaction records (จากตารางที่ normalize แล้ว)
    """
    all_transactions: List[Dict[str, Any]] = []

    for tb in doc.tables:
        txs = extract_transactions_from_table(tb)
        if not txs:
            continue
        all_transactions.extend(txs)

    payload: Dict[str, Any] = {
        "doc_id": doc.metadata.doc_id,
        "doc_type": doc.metadata.doc_type,
        "file_name": doc.metadata.file_name,
        "transactions": all_transactions,
    }
    return payload
