"""Hybrid retriever: dense (Qdrant) + BM25, fused with RRF, optionally cross-encoder reranked."""
import json

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from . import config
from .fusion import rrf_fuse, tokenize

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


class HybridRetriever:
    def __init__(self):
        self.chunks = json.loads((config.INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.embedder = SentenceTransformer(config.EMBED_MODEL)
        self.reranker = CrossEncoder(config.RERANK_MODEL)
        self.client = QdrantClient(path=str(config.INDEX_DIR / "qdrant"))

    def _dense(self, query: str, k: int) -> list[int]:
        vec = self.embedder.encode(config.QUERY_PREFIX + query, normalize_embeddings=True)
        hits = self.client.query_points(config.COLLECTION, query=vec.tolist(), limit=k).points
        return [h.id for h in hits]

    def _sparse(self, query: str, k: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:k]

    def _rerank(self, query: str, ids: list[int]) -> list[int]:
        pairs = [(query, self.chunks[i]["text"]) for i in ids]
        scores = self.reranker.predict(pairs)
        return [i for i, _ in sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)]

    def ranked_ids(self, query: str, mode: str = "hybrid_rerank") -> list[int]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        n = config.CANDIDATES
        if mode == "dense":
            return self._dense(query, n)
        if mode == "bm25":
            return self._sparse(query, n)
        fused = rrf_fuse([self._dense(query, n), self._sparse(query, n)])[:n]
        return self._rerank(query, fused) if mode == "hybrid_rerank" else fused

    def retrieve(self, query: str, k: int = config.FINAL_K, mode: str = "hybrid_rerank") -> list[dict]:
        return [self.chunks[i] for i in self.ranked_ids(query, mode)[:k]]
