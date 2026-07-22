"""FastAPI service exposing the RAG pipeline. Run: uvicorn app:app --reload"""
from __future__ import annotations
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from generation import GeminiBackend
from rag import RAGPipeline

load_dotenv()

app = FastAPI(title="DocuQuery AI", version="1.0")

_pipeline: RAGPipeline | None = None


class Query(BaseModel):
    question: str


class AnswerOut(BaseModel):
    answer: str
    sources: list[str]


@app.on_event("startup")
def _startup():
    global _pipeline
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    _pipeline = RAGPipeline(GeminiBackend(key))


@app.get("/health")
def health():
    return {"status": "ok", "index_loaded": _pipeline is not None}


@app.post("/query", response_model=AnswerOut)
def query(q: Query):
    if not q.question.strip():
        raise HTTPException(400, "question must not be empty")
    result = _pipeline.answer(q.question)
    return AnswerOut(answer=result.text, sources=result.sources)
