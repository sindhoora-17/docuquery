"""Central configuration. Single source of truth so ingest and query never drift."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # Paths
    data_dir: Path = Path("data")
    index_dir: Path = Path("faiss_index")

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 100

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Retrieval
    top_k: int = 5                # final chunks passed to the LLM
    candidate_k: int = 10         # per-retriever candidates before fusion
    rrf_k: int = 60               # RRF smoothing constant (standard default)
    use_hybrid: bool = False       # BM25 + vector; False = vector only

    # Generation
    gemini_model: str = "gemini-flash-latest"
    max_context_chars: int = 6000

    # Prompt
    system_instruction: str = (
        "You are a document question-answering assistant. Answer using only the "
        "provided context. If the answer is not in the context, say you could not "
        "find it in the documents. Be concise and cite nothing outside the context."
    )


CONFIG = Config()
