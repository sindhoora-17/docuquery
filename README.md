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
- **Refuses rather than hallucinates.** The prompt is constrained to the retrieved
  context, so when the answer genuinely isn't in the documents the system says so
  instead of inventing one from the model's general knowledge.
- **Evaluated.** `evaluate.py` measures retrieval hit rate @k and answer quality
  (LLM-as-judge) over a hand-written question set, so config changes (hybrid vs
  vector-only, k, chunk size) can be compared with numbers.
- **Swappable LLM backend** behind a small interface; defaults to Google's
  `gemini-flash-latest` alias so the app survives provider version changes.

## Results

Evaluated on a hand-built question set over a technical documentation corpus,
measuring whether the correct source page appears in the retrieved top-k:

| Retrieval mode        | Hit rate @5 | Mean answer score |
|-----------------------|-------------|-------------------|
| Hybrid (BM25 + vector)| **97%**     | 5.0 / 5           |
| Vector only           | 90%         | 5.0 / 5           |

Hybrid retrieval recovered a keyword-heavy question that vector search alone
missed entirely — semantic search returned neighbouring pages, while BM25 matched
the exact term and pulled the right page into the results through fusion.

> Small benchmark (13 questions, one corpus), so this is a demonstration of
> the method rather than a robust benchmark.

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

## Project structure

```
config.py           all tuning knobs in one place (chunk size, top_k, RRF constant, hybrid on/off)
ingest.py           PDF loading → chunking → embeddings → FAISS index + chunk manifest
retrieval.py        hybrid retriever: BM25 + vector search fused with RRF
generation.py       prompt construction + swappable LLM backend
rag.py              orchestration: retrieve → build prompt → generate → return answer + sources
app.py              FastAPI /query endpoint
streamlit_app.py    Streamlit UI (shows answers with their source pages)
evaluate.py         evaluation harness (retrieval hit rate + LLM-as-judge scoring)
tests/              unit tests for RRF fusion and hybrid vs vector-only behaviour
```

## Setup

Use a dedicated virtual environment — the pinned versions in
`requirements.txt` are chosen to work together, and installing them into a
shared/base environment can conflict with other projects' dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env             # then add your GEMINI_API_KEY
```

Get a free Gemini API key at https://aistudio.google.com/apikey

## Usage

```bash
# 1. Put PDFs in data/, then build the index
python ingest.py

# 2a. Ask via the REST API
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
labels each run with its config, so runs are directly comparable.

## Tests

```bash
pytest
```

Covers RRF fusion correctness and hybrid vs vector-only retrieval behaviour.

## Config

All knobs live in `config.py`: chunk size/overlap, embedding model, `top_k`,
`candidate_k`, RRF constant, and the hybrid on/off switch.