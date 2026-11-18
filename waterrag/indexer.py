"""Build the index: chunk documents, embed them, store vectors in Qdrant (local mode)."""
import json

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from . import config
from .chunking import chunk_text
from .loader import load_documents


def build_chunks() -> list[dict]:
    chunks = []
    for doc in load_documents(config.DATA_DIR):
        for piece in chunk_text(doc["text"], config.CHUNK_WORDS, config.CHUNK_OVERLAP):
            chunks.append({"id": len(chunks), "source": doc["source"],
                           "page": doc["page"], "text": piece})
    return chunks


def build_index() -> int:
    chunks = build_chunks()
    if not chunks:
        raise SystemExit(f"No documents found in {config.DATA_DIR}")

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    (config.INDEX_DIR / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")

    model = SentenceTransformer(config.EMBED_MODEL)
    vectors = model.encode([c["text"] for c in chunks], normalize_embeddings=True,
                           show_progress_bar=True)

    client = QdrantClient(path=str(config.INDEX_DIR / "qdrant"))
    if client.collection_exists(config.COLLECTION):
        client.delete_collection(config.COLLECTION)
    client.create_collection(
        config.COLLECTION,
        vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
    )
    client.upsert(
        config.COLLECTION,
        points=[PointStruct(id=c["id"], vector=v.tolist(),
                            payload={"source": c["source"], "page": c["page"]})
                for c, v in zip(chunks, vectors)],
    )
    client.close()
    return len(chunks)


if __name__ == "__main__":
    n = build_index()
    print(f"Indexed {n} chunks into {config.INDEX_DIR}")
