"""Retrieval ablation: dense vs BM25 vs hybrid vs hybrid+rerank.

A retrieved chunk counts as a hit if it comes from the expected source file AND
contains the expected keyword (so the answer is really in the chunk).

Usage:  python -m eval.evaluate
"""
import json
import time
from pathlib import Path

from waterrag.retriever import MODES, HybridRetriever

QUESTIONS = Path(__file__).parent / "questions.jsonl"
OUT = Path(__file__).parent / "results.json"


def is_hit(chunk: dict, q: dict) -> bool:
    return chunk["source"] == q["source"] and q["keyword"].lower() in chunk["text"].lower()


def first_hit_rank(retriever: HybridRetriever, q: dict, mode: str, depth: int = 10):
    ids = retriever.ranked_ids(q["question"], mode)[:depth]
    for rank, cid in enumerate(ids, start=1):
        if is_hit(retriever.chunks[cid], q):
            return rank
    return None


def main():
    questions = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    retriever = HybridRetriever()
    results = {}
    for mode in MODES:
        ranks, start = [], time.perf_counter()
        for q in questions:
            ranks.append(first_hit_rank(retriever, q, mode))
        ms = (time.perf_counter() - start) / len(questions) * 1000
        n = len(questions)
        results[mode] = {
            "hit@1": sum(r == 1 for r in ranks) / n,
            "hit@3": sum(r is not None and r <= 3 for r in ranks) / n,
            "hit@5": sum(r is not None and r <= 5 for r in ranks) / n,
            "mrr@10": sum(1 / r for r in ranks if r) / n,
            "avg_latency_ms": round(ms, 1),
        }
    OUT.write_text(json.dumps(results, indent=2))

    print(f"\n{len(questions)} questions\n")
    print(f"{'mode':<15}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'mrr@10':>9}{'ms/query':>10}")
    for mode, m in results.items():
        print(f"{mode:<15}{m['hit@1']:>8.2f}{m['hit@3']:>8.2f}{m['hit@5']:>8.2f}"
              f"{m['mrr@10']:>9.2f}{m['avg_latency_ms']:>10.1f}")


if __name__ == "__main__":
    main()
