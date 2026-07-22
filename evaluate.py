"""
Evaluation harness. Measures two things over a hand-written question set:

  1. Retrieval hit rate @k: did a chunk from the expected source page make it
     into the retrieved top-k? (Pure retrieval quality, no LLM involved.)
  2. Answer quality: LLM-as-judge scores the generated answer against a
     reference answer on a 1-5 scale.

Lets you compare configs (hybrid vs vector-only, k=3 vs k=5, chunk sizes) with
numbers instead of vibes. Ground-truth lives in eval_set.json.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()

from config import CONFIG
from generation import LLMBackend, GeminiBackend
from rag import RAGPipeline


@dataclass
class EvalItem:
    question: str
    reference: str
    expected_source: str   # "filename.pdf"
    expected_page: int


def load_eval_set(path: Path) -> List[EvalItem]:
    return [EvalItem(**r) for r in json.loads(path.read_text())]


def retrieval_hit(pipeline: RAGPipeline, item: EvalItem) -> bool:
    chunks = pipeline._retriever.retrieve(item.question, CONFIG.top_k)
    got_pages = [(c.source, c.page) for c in chunks]
    hit = any(
        c.source == item.expected_source and c.page == item.expected_page
        for c in chunks
    )
    if not hit:
        print(f"    [debug] expected {item.expected_source} p.{item.expected_page}, got pages: {[p for _, p in got_pages]}")
    return hit


def judge_answer(backend: LLMBackend, item: EvalItem, got: str) -> int:
    prompt = (
        "Score how well the CANDIDATE answer matches the REFERENCE answer for the "
        "question, on a scale of 1 (wrong/irrelevant) to 5 (fully correct). "
        "Reply with ONLY the integer.\n\n"
        f"Question: {item.question}\nReference: {item.reference}\nCandidate: {got}\n\nScore:"
    )
    raw = backend.generate(prompt)
    for tok in raw.split():
        if tok.strip().isdigit():
            return max(1, min(5, int(tok.strip())))
    return 1


def run(eval_path: str = "eval_set.json") -> None:
    import os
    backend = GeminiBackend(os.environ["GEMINI_API_KEY"])
    pipeline = RAGPipeline(backend)
    items = load_eval_set(Path(eval_path))

    hits, scores = 0, []
    for it in items:
        hit = retrieval_hit(pipeline, it)
        ans = pipeline.answer(it.question)
        score = judge_answer(backend, it, ans.text)
        hits += int(hit)
        scores.append(score)
        print(f"  hit={hit!s:5} score={score}  {it.question[:50]}")

    n = len(items)
    print(f"\nConfig: hybrid={CONFIG.use_hybrid} top_k={CONFIG.top_k} "
          f"chunk={CONFIG.chunk_size}/{CONFIG.chunk_overlap}")
    print(f"Retrieval hit rate @{CONFIG.top_k}: {hits}/{n} = {hits/n:.0%}")
    print(f"Mean answer score: {sum(scores)/n:.2f} / 5")


if __name__ == "__main__":
    run()
