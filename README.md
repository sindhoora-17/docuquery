# DocuQuery AI

A document Q&A system over your own PDFs. It retrieves relevant passages using
**hybrid search** (keyword + semantic), generates an answer grounded only in
those passages, and **cites the source page** for every answer. Ships with an
evaluation harness so retrieval and answer quality are measured, not guessed.

## Why this design

- **Hybrid retrieval (BM25 + vector, fused with RRF).** Vector search finds
  semantically similar text but misses exact tokens (names, IDs, rare terms like
  "FAISS"); BM25 nails exact tokens but misses paraphrase. Reciprocal Rank Fusion
  merges the two ranked lists on rank position, so their incompatible score
  scales don't need normalizing.
- **Source citations.** Every answer lists the `file (p.N)` passages it used —
  the difference between a trustworthy tool and a black box.
- **Evaluated.** `evaluate.py` measures retrieval hit rate @k and answer quality
  (LLM-as-judge) over a hand-written question set, so config changes (hybrid vs
  vector-only, k, chunk size) can be compared with numbers.
- **Swappable LLM backend** behind a small interface; Gemini 1.5 Flash by default.

## Architecture

```
 PDFs ──► ingest.py ──► FAISS index + chunks.json (text + provenance)
                              │
 question ──► RAGPipeline ────┤
                              ├─ vector search (FAISS) ─┐
                              ├─ BM25 (rank_bm25) ──────┤─► RRF fusion ─► top-k
                              │                          │
                              └─ build prompt (top-k + citations) ─► Gemini
                                          │
                                          ▼
                              Answer{ text, sources[] }
                                          │
                        ┌─────────────────┼─────────────────┐
                     app.py            streamlit_app.py   evaluate.py
                   (REST /query)          (UI)            (metrics)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env    # add your GEMINI_API_KEY
```

Get a free Gemini API key at https://aistudio.google.com/apikey

## Usage

```bash
# 1. Put PDFs in data/, then build the index
python ingest.py

# 2a. Ask via REST API
uvicorn app:app --reload
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "what is the refund policy?"}'

# 2b. Or the UI
streamlit run streamlit_app.py
```

## Evaluation

Fill in `eval_set.json` with question / reference / expected-source triples, then:

```bash
python evaluate.py
```

Reports retrieval hit rate @k and mean answer score. To compare configs, edit
`config.py` (e.g. `use_hybrid=False`, or `top_k=3`) and re-run — the printout
labels each run with its config.

## Tests

```bash
pytest
```

Covers RRF fusion correctness and hybrid vs vector-only behavior.

## Config

All knobs live in `config.py`: chunk size/overlap, embedding model, `top_k`,
`candidate_k`, RRF constant, and the hybrid on/off switch.
