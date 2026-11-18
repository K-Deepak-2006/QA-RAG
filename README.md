# WaterRAG

Retrieval-Augmented Generation over water-harvesting, irrigation and soil-sensing documents.
Hybrid retrieval (dense + BM25 fused with Reciprocal Rank Fusion), cross-encoder reranking,
cited answers, an evaluation harness, a FastAPI service, a small web UI and Docker packaging.

## Architecture

```
docs (.pdf/.txt/.md) -> chunk (120 words, 30 overlap) -> embed (bge-small) -> Qdrant (local)
                                                     \-> BM25 index
query -> dense top-20 + BM25 top-20 -> RRF fusion -> cross-encoder rerank -> top-5 -> LLM -> cited answer
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # optional: add ANTHROPIC_API_KEY for generated answers
python -m waterrag.indexer      # builds the index from data/docs
uvicorn waterrag.app:app --reload
# open http://localhost:8000
```

Without an API key the app runs in extractive mode (returns the top passage). Models are
downloaded from Hugging Face on first run.

## Use your own documents

Delete the sample files in `data/docs/`, drop in your own PDFs/text (research papers, your
project reports, university documents), then re-run `python -m waterrag.indexer`.
Edit `eval/questions.jsonl` with 30+ questions of your own:
`{"question": ..., "source": "file.pdf", "keyword": "phrase that appears in the answer chunk"}`.

## Evaluate (this produces your resume numbers)

```bash
python -m eval.evaluate         # dense vs BM25 vs hybrid vs hybrid+rerank: hit@k, MRR, latency
python -m eval.faithfulness     # LLM-judged answer faithfulness (needs ANTHROPIC_API_KEY)
python -m pytest tests          # unit tests
```

The sample corpus is tiny (15 chunks), so its scores are only a smoke test. Report numbers
from your own corpus and question set.

## Docker

```bash
docker build -t waterrag . && docker run -p 8000:8000 -e ANTHROPIC_API_KEY=... waterrag
```

## API

`POST /ask` with `{"question": "...", "mode": "hybrid_rerank", "k": 5}` returns
`{"answer": "...", "sources": [{"n", "source", "page", "text"}]}`. `GET /health` reports chunk count.

## Design choices (interview notes)

- **Hybrid retrieval:** dense search handles paraphrases, BM25 handles exact terms (units, acronyms, names).
- **RRF:** fuses rankings without having to normalise incompatible score scales.
- **Cross-encoder rerank:** slower but more precise; applied only to the ~20 fused candidates.
- **Grounded prompt:** answers must cite numbered passages and say "I don't know" when context is missing.
- **Ablation harness:** every retrieval choice is measured, not assumed.
