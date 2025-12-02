import io
import fitz  # PyMuPDF
from PIL import Image
from google import genai
from ingestion.config import GEMINI_API_KEY

def _get_gemini_model():
    if not GEMINI_API_KEY:
        raise ValueError("❌ Missing GEMINI_API_KEY in config.py")

    client = genai.Client(api_key=GEMINI_API_KEY)
    model = client.models.get("gemini-2.0-flash")
    return client, model

def pdf_page_to_image_bytes(page):
    pix = page.get_pixmap(dpi=200)
    return pix.tobytes("png")

def ocr_page(client, model, image_bytes):
    response = client.models.generate_content(
        model=model.name,
        contents=[
            {
                "mime_type": "image/png",
                "data": image_bytes
            }
        ]
    )
    return response.text

class OCRDocument:
    def __init__(self):
        self.texts = []  # list of {"page": int, "content": str}

def ocr_extract_document(pdf_path: str) -> OCRDocument:
    client, model = _get_gemini_model()

    doc = fitz.open(pdf_path)
    result = OCRDocument()

    print(f"[OCR] Total pages: {len(doc)}")

    for idx, page in enumerate(doc):
        print(f"[OCR] Processing page {idx + 1}/{len(doc)}")

        image_bytes = pdf_page_to_image_bytes(page)
        text = ocr_page(client, model, image_bytes)

        result.texts.append({
            "page": idx + 1,
            "content": text
        })

    print("[OCR] Completed")
    return result
