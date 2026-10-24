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

    @property
    def answerable(self) -> bool:
        return self.expected_page > 0


def load_eval_set(path: Path) -> List[EvalItem]:
    return [EvalItem(**r) for r in json.loads(path.read_text())]


def retrieval_hit(pipeline: RAGPipeline, item: EvalItem) -> bool:
    if not item.answerable:
        raise ValueError("Retrieval hit rate applies only to answerable questions")
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

    answerable = [it for it in items if it.answerable]
    out_of_scope = [it for it in items if not it.answerable]
    hits, scores = 0, []
    for it in answerable:
        hit = retrieval_hit(pipeline, it)
        ans = pipeline.answer(it.question)
        score = judge_answer(backend, it, ans.text)
        hits += int(hit)
        scores.append(score)
        print(f"  hit={hit!s:5} score={score}  {it.question[:50]}")

    print(f"\\nConfig: hybrid={CONFIG.use_hybrid} top_k={CONFIG.top_k} "
          f"chunk={CONFIG.chunk_size}/{CONFIG.chunk_overlap}")
    if answerable:
        print(f"Answerable retrieval hit rate @{CONFIG.top_k}: "
              f"{hits}/{len(answerable)} = {hits/len(answerable):.0%}")
        print(f"Mean answer score: {sum(scores)/len(scores):.2f} / 5")
    else:
        print("No answerable retrieval questions.")

    # Negative questions are intentionally excluded from retrieval hit-rate:
    # no valid expected source page exists to retrieve.
    print(f"Out-of-scope questions: {len(out_of_scope)} (excluded from retrieval hit rate)")
    for item in out_of_scope:
        answer = pipeline.answer(item.question)
        print(f"  [manual abstention review] {item.question} -> {answer.text}")
    print("Out-of-scope responses require manual review; no automatic abstention "
          "accuracy is claimed.")


if __name__ == "__main__":
    run()
