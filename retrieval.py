"""
Hybrid retrieval: BM25 (keyword) + FAISS vector search, merged with
Reciprocal Rank Fusion (RRF).

Why hybrid: vector search finds semantically similar passages but misses exact
tokens (names, IDs, rare terms); BM25 nails exact tokens but misses paraphrase.
RRF merges the two ranked lists without needing to normalize their incompatible
score scales — it fuses on rank position, not raw score.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Sequence

from rank_bm25 import BM25Okapi


@dataclass
class Chunk:
    """A retrievable unit of text plus provenance for citation."""
    text: str
    source: str      # filename
    page: int        # 1-indexed page number
    chunk_id: int

    def cite(self) -> str:
        return f"{self.source} (p.{self.page})"


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    """
    Combines a vector retriever with BM25 over the same chunk set.

    The vector side is injected as a callable so this class stays independent of
    the embedding backend (real HuggingFace model in prod, fake in tests).
    `vector_search(query, k) -> List[int]` returns chunk indices, best first.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        vector_search,
        candidate_k: int = 10,
        rrf_k: int = 60,
        use_hybrid: bool = True,
    ):
        self.chunks = list(chunks)
        self.vector_search = vector_search
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.use_hybrid = use_hybrid
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in self.chunks]) if self.chunks else None

    def _bm25_rank(self, query: str, k: int) -> List[int]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[:k]

    @staticmethod
    def _rrf(ranked_lists: List[List[int]], rrf_k: int) -> List[int]:
        """Reciprocal Rank Fusion: score(d) = sum 1/(rrf_k + rank_in_list)."""
        scores: dict[int, float] = {}
        for lst in ranked_lists:
            for rank, doc_idx in enumerate(lst):
                scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (rrf_k + rank)
        return sorted(scores, key=lambda d: scores[d], reverse=True)

    def retrieve(self, query: str, top_k: int) -> List[Chunk]:
        vector_ranked = self.vector_search(query, self.candidate_k)
        if not self.use_hybrid:
            return [self.chunks[i] for i in vector_ranked[:top_k]]

        bm25_ranked = self._bm25_rank(query, self.candidate_k)
        fused = self._rrf([vector_ranked, bm25_ranked], self.rrf_k)
        return [self.chunks[i] for i in fused[:top_k]]
