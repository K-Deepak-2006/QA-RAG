"""Answer generation grounded in retrieved passages, with citations."""
import os

from . import config

SYSTEM = (
    "You answer questions using ONLY the numbered context passages provided. "
    "Cite passages inline like [1] or [2]. If the context does not contain the answer, "
    "say you don't know. Be concise."
)


def build_prompt(question: str, passages: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i}] ({p['source']}, p.{p['page']})\n{p['text']}" for i, p in enumerate(passages, 1)
    )
    return f"Context:\n{context}\n\nQuestion: {question}"


def generate(question: str, passages: list[dict]) -> str:
    if not os.getenv("ANTHROPIC_API_KEY"):
        top = passages[0]["text"] if passages else "No relevant passages found."
        return f"(extractive mode, set ANTHROPIC_API_KEY for generated answers)\n\n{top} [1]"

    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=600,
        system=SYSTEM,
        messages=[{"role": "user", "content": build_prompt(question, passages)}],
    )
    return msg.content[0].text
