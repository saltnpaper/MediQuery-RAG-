from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .rag_service import RAGService


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT_DIR.parent.parent / "all_chunks.json"
DATA_PATH = Path(os.getenv("MEDIQUERY_CHUNKS_PATH", DEFAULT_DATA_PATH))

app = FastAPI(title="MediQuery VitaChat API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_service = RAGService(DATA_PATH)


class ChatRequest(BaseModel):
    message: str


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "data_path": str(DATA_PATH),
        "retrieval_backend": rag_service.backend_mode,
        "artifact_dir": str(rag_service.artifact_dir),
    }


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    return rag_service.answer(request.message)
