"""
Answer generation behind a tiny interface, so the LLM backend is swappable.
Default backend is Gemini (free tier). Add another by implementing generate().
"""
from __future__ import annotations
from typing import List, Protocol

from config import CONFIG
from retrieval import Chunk


class LLMBackend(Protocol):
    def generate(self, prompt: str) -> str: ...


class GeminiBackend:
    def __init__(self, api_key: str, model: str = CONFIG.gemini_model):
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(self, prompt: str) -> str:
        try:
            resp = self._client.models.generate_content(model=self._model, contents=prompt)
            return (resp.text or "No answer generated.").strip()
        except Exception as e:  # network, rate limit, safety block
            return f"[generation error] {e}"


def build_prompt(question: str, chunks: List[Chunk]) -> str:
    context_parts, running = [], 0
    for c in chunks:
        piece = f"[{c.cite()}]\n{c.text}"
        if running + len(piece) > CONFIG.max_context_chars:
            break
        context_parts.append(piece)
        running += len(piece)
    context = "\n\n".join(context_parts)
    return (
        f"{CONFIG.system_instruction}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
