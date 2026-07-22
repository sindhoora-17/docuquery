"""
Core RAG orchestration: load the persisted index, wire up hybrid retrieval,
answer questions WITH their sources.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from config import CONFIG
from retrieval import HybridRetriever, Chunk
from generation import LLMBackend, build_prompt


@dataclass
class Answer:
    text: str
    sources: List[str]   # e.g. ["report.pdf (p.3)", "notes.pdf (p.1)"]


class RAGPipeline:
    def __init__(self, backend: LLMBackend, index_dir: Path = CONFIG.index_dir):
        self.backend = backend
        self._load(index_dir)

    def _load(self, index_dir: Path):
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS

        embeddings = HuggingFaceEmbeddings(model_name=CONFIG.embedding_model)
        self._store = FAISS.load_local(
            str(index_dir), embeddings, allow_dangerous_deserialization=True
        )
        manifest = json.loads((index_dir / "chunks.json").read_text())
        self._chunks = [Chunk(**m) for m in manifest]

        # Map FAISS similarity results back to manifest indices by chunk text.
        self._text_to_idx = {c.text: c.chunk_id for c in self._chunks}
        self._retriever = HybridRetriever(
            self._chunks, self._vector_search,
            candidate_k=CONFIG.candidate_k, rrf_k=CONFIG.rrf_k,
            use_hybrid=CONFIG.use_hybrid,
        )

    def _vector_search(self, query: str, k: int) -> List[int]:
        docs = self._store.similarity_search(query, k=k)
        idxs = []
        for d in docs:
            idx = self._text_to_idx.get(d.page_content)
            if idx is not None:
                idxs.append(idx)
        return idxs

    def answer(self, question: str) -> Answer:
        chunks = self._retriever.retrieve(question, CONFIG.top_k)
        if not chunks:
            return Answer("I could not find anything relevant in the documents.", [])
        prompt = build_prompt(question, chunks)
        text = self.backend.generate(prompt)
        # De-dup sources, preserve order.
        seen, sources = set(), []
        for c in chunks:
            cite = c.cite()
            if cite not in seen:
                seen.add(cite)
                sources.append(cite)
        return Answer(text=text, sources=sources)
