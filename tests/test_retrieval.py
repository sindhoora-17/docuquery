"""Unit tests for hybrid retrieval and RRF fusion."""
from retrieval import HybridRetriever, Chunk


def _chunks(texts):
    return [Chunk(t, f"doc{i}.pdf", 1, i) for i, t in enumerate(texts)]


def test_rrf_rewards_agreement():
    # Docs appearing in both lists should outrank docs in only one.
    fused = HybridRetriever._rrf([[2, 0, 1], [0, 2, 3]], 60)
    assert set(fused[:2]) == {0, 2}


def test_rrf_agreement_beats_single_list_top():
    # doc 5 is #1 in list A only; doc 1 is #2 in BOTH lists.
    # RRF should rank doc 1 above doc 5 because agreement compounds.
    fused = HybridRetriever._rrf([[5, 1, 9], [7, 1, 8]], 60)
    assert fused[0] == 1


def test_bm25_lifts_keyword_doc_via_fusion():
    # Vector search ranks the keyword doc LAST; BM25 ranks it first.
    # After fusion it should climb into the top result.
    chunks = _chunks([
        "general text about databases and storage systems",
        "unrelated content about cooking recipes",
        "FAISS enables fast similarity search over dense vectors",
    ])
    # vector puts the FAISS doc (idx 2) dead last
    r = HybridRetriever(chunks, lambda q, k: [0, 1, 2][:k], candidate_k=3, use_hybrid=True)
    top_ids = [c.chunk_id for c in r.retrieve("FAISS", top_k=3)]
    # BM25 pulls idx 2 up; it should no longer be last after fusion
    assert top_ids[0] == 2 or top_ids.index(2) < 2


def test_hybrid_off_is_vector_only():
    chunks = _chunks(["a", "b", "c"])
    r = HybridRetriever(chunks, lambda q, k: [2, 1, 0][:k], use_hybrid=False)
    assert r.retrieve("anything", top_k=1)[0].chunk_id == 2


def test_cite_format():
    assert Chunk("x", "report.pdf", 4, 0).cite() == "report.pdf (p.4)"
