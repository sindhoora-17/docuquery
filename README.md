# DocuQuery AI

A document Q&A system over your own PDFs. It retrieves relevant passages using **hybrid search** (BM25 + semantic vector search), generates an answer grounded only in those passages, and cites the source page for every answer. The project also includes an evaluation harness so retrieval quality can be measured instead of guessed.

## Why this design

- **Hybrid retrieval (BM25 + vector, fused with RRF).** Vector search is strong at semantic similarity but can miss exact tokens such as names, IDs, or uncommon technical terms. BM25 complements it with lexical matching, while Reciprocal Rank Fusion combines both ranked lists without requiring score normalization.
- **Source citations.** Every answer carries `file (p.N)` provenance from the retrieved chunks.
- **Grounded generation.** The prompt constrains the model to retrieved context and instructs it to say when the answer is not present instead of filling gaps from general model knowledge.
- **Evaluation harness.** `evaluate.py` measures retrieval hit rate @k and uses an LLM-as-judge score for answer quality over a hand-written evaluation set.
- **Swappable generation backend.** LLM generation is isolated behind a small interface; the default backend uses Google's Gemini API.

## Results

The checked-in evaluation set contains 13 questions over a React documentation corpus. Retrieval hit rate measures whether the expected source page appears in the retrieved top-5 results.

| Retrieval mode | Correct source page in top 5 | Hit rate @5 |
|---|---:|---:|
| Hybrid (BM25 + vector) | 12 / 13 | **92%** |
| Vector only | 11 / 13 | 85% |

The improvement came from a keyword-heavy query where semantic retrieval returned nearby pages but BM25 matched the exact term and brought the expected page into the fused result set.

> This is intentionally a small project benchmark, not a general retrieval benchmark. The included `eval_set.json` and `evaluate.py` make the experiment reproducible and easy to extend with additional documents and questions.

## Architecture

```text
 PDFs ──► ingest.py ──► FAISS index + chunks.json (text + provenance)
                              │
 question ──► RAGPipeline ────┤
                              ├─ vector search (FAISS) ─┐
                              ├─ BM25 (rank_bm25) ──────┤─► RRF fusion ─► top-k
                              │                          │
                              └─ build grounded prompt ──► Gemini
                                          │
                                          ▼
                              Answer{ text, sources[] }
                                          │
                        ┌─────────────────┼─────────────────┐
                     app.py            streamlit_app.py   evaluate.py
                   (REST /query)          (UI)            (metrics)
```

## Project structure

```text
config.py           retrieval and generation configuration
ingest.py           PDF loading, chunking, embeddings, FAISS index, chunk manifest
retrieval.py        BM25 + vector retrieval with Reciprocal Rank Fusion
generation.py       prompt construction and LLM backend abstraction
rag.py              orchestration: retrieve → prompt → generate → answer + sources
app.py              FastAPI /query endpoint
streamlit_app.py    Streamlit UI
evaluate.py         retrieval hit-rate and answer-quality evaluation harness
eval_set.json       hand-written evaluation questions and expected source pages
tests/              unit tests for RRF and hybrid retrieval behavior
```

## Setup

Use a dedicated virtual environment so the project's Python dependencies stay isolated from other environments.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
```

Then add your Gemini API key to `.env`:

```text
GEMINI_API_KEY=your_api_key_here
```

## Usage

First, place PDFs in `data/` and build the index:

```bash
python ingest.py
```

Run the REST API:

```bash
uvicorn app:app --reload
```

Example request:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"How do you add state to a React component?"}'
```

Or launch the Streamlit UI:

```bash
streamlit run streamlit_app.py
```

## Evaluation

The evaluation set stores each question, reference answer, expected source document, and expected page. Run:

```bash
python evaluate.py
```

The harness reports retrieval hit rate @k and a mean answer-quality score. To compare configurations, change retrieval settings in `config.py` — for example `use_hybrid=False` for vector-only retrieval — and rerun the evaluation.

## Tests

```bash
pytest
```

The unit tests cover:

- Reciprocal Rank Fusion behavior
- agreement between lexical and vector rankings
- BM25 lifting exact-keyword matches
- vector-only fallback behavior
- citation formatting

GitHub Actions runs the test suite on every push to `main` and on pull requests.

## Configuration

The main tuning parameters live in `config.py`:

- chunk size and overlap
- embedding model
- final `top_k`
- per-retriever candidate count
- RRF smoothing constant
- hybrid/vector-only switch
- Gemini model
- maximum prompt context size
