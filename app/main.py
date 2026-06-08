
#논문 분석 서비스의 백엔드 API

#주요 기능
#1. /analyze : 논문 PDF를 업로드해서 분석
#2. /qa : 업로드한 논문에 대해 질문하기
#3. /related : 관련 논문 추천 받기


# 타입 힌트를 더 유연하게 사용하기 위한 설정
# 보통 파이썬에서 타입 힌트를 쓸 때 정의되지 않은 클래스 이름 때문에 문제가 발생할 수 있음
# 타입을 나중에 문자열처럼 처리해서 더 편하게 사용할 수 있음
# query: str | None = None

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .arxiv_mcp import search_related_arxiv
from .graph import build_analysis_graph
from .models import AnalysisResult
from .pdf_utils import extract_paper_artifacts
from .rag import answer_with_rag, build_or_replace_doc_index
from .vision_utils import enrich_figures_with_ocr_and_vision


BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# http://127.0.0.1:8000/apple.png

app = FastAPI(title="Multimodal Paper Analyzer")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

analysis_graph = build_analysis_graph()
analysis_store: Dict[str, AnalysisResult] = {}
related_store: Dict[str, list] = {}


class QARequest(BaseModel):
    doc_id: str
    question: str


class RelatedRequest(BaseModel):
    doc_id: str
    query: str | None = None

# http://127.0.0.1:8000/
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request,name="index.html")

# http://127.0.0.1:8000/analyze
@app.post("/analyze")
async def analyze_paper(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")

    doc_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{doc_id}.pdf"
    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    artifacts = extract_paper_artifacts(str(save_path), file.filename)
    artifacts.figures = enrich_figures_with_ocr_and_vision(str(save_path), artifacts.figures)
    state = {"artifacts": artifacts, "result": AnalysisResult()}
    out = analysis_graph.invoke(state)
    result: AnalysisResult = out["result"]
    analysis_store[doc_id] = result
    build_or_replace_doc_index(doc_id, artifacts)
    # Use only short filename keywords for stable arXiv auto-search.
    seed_query = artifacts.filename.replace(".pdf", "")
    related_papers, related_warning = search_related_arxiv(seed_query, max_results=5)
    related_store[doc_id] = related_papers

    tables_preview = []
    for table in artifacts.tables[:3]:
        tables_preview.append(
            {
                "page": table.page,
                "rows": table.rows[:5],
            }
        )

    figures_preview = [
        {
            "page": fig.page,
            "caption": fig.caption,
            "ocr_text": fig.ocr_text[:220],
            "vision_summary": fig.vision_summary,
        }
        for fig in artifacts.figures[:10]
    ]

    return {
        "doc_id": doc_id,
        "filename": artifacts.filename,
        "summary": result.summary,
        "key_contributions": result.key_contributions,
        "limitations": result.limitations,
        "reproducibility": result.reproducibility,
        "comparative_analysis": result.comparative_analysis,
        "references": artifacts.references[:20],
        "tables": tables_preview,
        "figures": figures_preview,
        "related_papers": related_papers,
        "search_warning": related_warning,
    }

# http://127.0.0.1:8000/qa
@app.post("/qa")
async def question_answer(payload: QARequest):
    result = analysis_store.get(payload.doc_id)
    if not result:
        raise HTTPException(status_code=404, detail="분석 결과를 찾을 수 없습니다. 먼저 PDF를 분석하세요.")

    try:
        answer = answer_with_rag(payload.doc_id, payload.question)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"answer": answer}

# http://127.0.0.1:8000/related
@app.post("/related")
async def related_papers(payload: RelatedRequest):
    if payload.doc_id not in analysis_store:
        raise HTTPException(status_code=404, detail="분석 결과를 찾을 수 없습니다. 먼저 PDF를 분석하세요.")
    if payload.query and payload.query.strip():
        papers, warn = search_related_arxiv(payload.query, max_results=5)
        related_store[payload.doc_id] = papers
        return {"related_papers": papers, "search_warning": warn}
    return {
        "related_papers": related_store.get(payload.doc_id, []),
        "search_warning": None,
    }