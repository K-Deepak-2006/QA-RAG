import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data" / "docs"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "index"))
COLLECTION = "waterrag"

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", 120))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 30))

CANDIDATES = 20   # per retriever before fusion
FINAL_K = 5       # passages sent to the LLM
RRF_K = 60        # reciprocal rank fusion constant

LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
