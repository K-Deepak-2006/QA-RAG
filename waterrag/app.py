from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .generator import generate
from .retriever import MODES, HybridRetriever

STATIC = Path(__file__).resolve().parent.parent / "static"
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["retriever"] = HybridRetriever()
    yield
    state.clear()


app = FastAPI(title="WaterRAG", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: str = "hybrid_rerank"
    k: int = Field(default=5, ge=1, le=10)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(state["retriever"].chunks)}


@app.post("/ask")
def ask(req: AskRequest):
    if req.mode not in MODES:
        raise HTTPException(400, f"mode must be one of {list(MODES)}")
    passages = state["retriever"].retrieve(req.question, k=req.k, mode=req.mode)
    return {
        "answer": generate(req.question, passages),
        "sources": [{"n": i, "source": p["source"], "page": p["page"], "text": p["text"]}
                    for i, p in enumerate(passages, 1)],
    }
