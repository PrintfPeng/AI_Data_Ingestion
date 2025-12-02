from __future__ import annotations

from typing import List, Optional, Literal

from fastapi import FastAPI
from pydantic import BaseModel

from fastapi.staticfiles import StaticFiles
from pathlib import Path

from backend.services.logger import append_log
from backend.services.logger import read_logs

from .services.rag import answer_question

from fastapi.responses import RedirectResponse


app = FastAPI(
    title="AI Data Ingestion Backend",
    description="Backend for DB, Embeddings, RAG, API, and Evaluation",
    version="0.1.0",
)

frontend_path = Path(__file__).resolve().parents[1] / "frontend"
app.mount(
    "/app",
    StaticFiles(directory=str(frontend_path), html=True),
    name="frontend",
)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "backend",
        "version": "0.1.0",
    }


class AskRequest(BaseModel):
    query: str
    doc_ids: Optional[List[str]] = None
    top_k: int = 5
    mode: Literal["auto", "text", "table", "both"] = "auto"


class AskResponse(BaseModel):
    answer: str
    sources: List[dict]
    intent: str

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    result = await answer_question(
        query=req.query,
        doc_ids=req.doc_ids,
        top_k=req.top_k,
        mode=req.mode,
    )

    # เขียน log ลงไฟล์
    try:
        append_log(
            {
                "query": req.query,
                "doc_ids": req.doc_ids,
                "top_k": req.top_k,
                "mode": req.mode,
                "answer": result.get("answer"),
                "intent": result.get("intent"),
                "sources": result.get("sources"),
            }
        )
    except Exception as e:
        # กันไว้เฉย ๆ อย่าให้ logging ทำให้ API พัง
        print(f"[LOG_ERROR] {e!r}")

    return AskResponse(**result)

class HistoryItem(BaseModel):
    ts: str
    query: str
    answer: str
    doc_ids: Optional[List[str]] = None
    intent: Optional[str] = None
    mode: Optional[str] = None


@app.get("/history", response_model=List[HistoryItem])
def get_history(limit: int = 50):
    """
    ดึง history Q&A ย้อนหลังใหม่สุดไม่เกิน limit รายการ
    """
    logs = read_logs(limit=limit)
    items: List[HistoryItem] = []

    for e in logs:
        items.append(
            HistoryItem(
                ts=e.get("ts", ""),
                query=e.get("query", ""),
                answer=e.get("answer", ""),
                doc_ids=e.get("doc_ids"),
                intent=e.get("intent"),
                mode=e.get("mode"),
            )
        )

    return items

@app.get("/")
def root():
    # redirect ไปหน้า frontend หลัก
    return RedirectResponse(url="/app")
