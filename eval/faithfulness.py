"""LLM-as-judge faithfulness: what fraction of each answer is supported by its retrieved context?

Needs ANTHROPIC_API_KEY.   Usage:  python -m eval.faithfulness
"""
import json
import os
import re
from pathlib import Path

import anthropic

from waterrag import config
from waterrag.generator import build_prompt, generate
from waterrag.retriever import HybridRetriever

QUESTIONS = Path(__file__).parent / "questions.jsonl"

JUDGE = (
    "You are a strict grader. Given CONTEXT passages and an ANSWER, list the factual claims "
    "in the answer and decide how many are directly supported by the context. "
    'Reply with JSON only: {"claims": <int>, "supported": <int>}.'
)


def judge(client, context_prompt: str, answer: str) -> float:
    msg = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=100,
        system=JUDGE,
        messages=[{"role": "user", "content": f"{context_prompt}\n\nANSWER:\n{answer}"}],
    )
    match = re.search(r"\{.*\}", msg.content[0].text, re.S)
    data = json.loads(match.group(0))
    return data["supported"] / data["claims"] if data["claims"] else 1.0


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run the faithfulness evaluation.")
    client = anthropic.Anthropic()
    retriever = HybridRetriever()
    questions = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    scores = []
    for q in questions:
        passages = retriever.retrieve(q["question"])
        answer = generate(q["question"], passages)
        scores.append(judge(client, build_prompt(q["question"], passages), answer))
        print(f"{scores[-1]:.2f}  {q['question']}")
    print(f"\nMean faithfulness over {len(scores)} questions: {sum(scores) / len(scores):.2f}")


if __name__ == "__main__":
    main()
