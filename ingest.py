"""
Ingest PDFs -> chunks -> FAISS index (persisted) + a chunks manifest for BM25.

Fixes over the original:
  - Counts DOCUMENTS (files) and PAGES separately. PyPDFLoader yields one
    Document per page, so the old `len(documents)` was really a page count.
  - Single embeddings import (langchain_huggingface), matching query time.
  - Persists chunk text + metadata so BM25 can be rebuilt without re-embedding.
"""
from __future__ import annotations
import json
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import CONFIG

load_dotenv()


def load_pages(data_dir: Path):
    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs in {data_dir.resolve()}")
    pages = []
    for pdf in pdfs:
        loaded = PyPDFLoader(str(pdf)).load()   # one Document per page
        for d in loaded:
            d.metadata["source"] = pdf.name
        pages.extend(loaded)
    print(f"Loaded {len(pdfs)} documents ({len(pages)} pages)")
    return pages


def build_index() -> None:
    pages = load_pages(CONFIG.data_dir)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG.chunk_size,
        chunk_overlap=CONFIG.chunk_overlap,
    )
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name=CONFIG.embedding_model)
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(CONFIG.index_dir))

    # Persist a chunk manifest (text + provenance) for BM25 at query time.
    manifest = [
        {
            "text": c.page_content,
            "source": c.metadata.get("source", "unknown"),
            "page": int(c.metadata.get("page", 0)) + 1,  # PyPDF pages are 0-indexed
            "chunk_id": i,
        }
        for i, c in enumerate(chunks)
    ]
    (CONFIG.index_dir / "chunks.json").write_text(json.dumps(manifest))
    print(f"Saved FAISS index + {len(manifest)} chunk manifest to {CONFIG.index_dir}/")


if __name__ == "__main__":
    build_index()
