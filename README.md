# Local RAG Web

A local retrieval-augmented generation application for searching a Turkish
document corpus and producing grounded answers with Foundry Local.

The project combines a FastAPI backend, a React/Vite frontend, SQLite vector
storage, multilingual sentence embeddings, hybrid retrieval, and streamed LLM
responses.

## Features

- Local document search for TXT, PDF, and DOCX files
- Dense embedding search combined with BM25 through Reciprocal Rank Fusion
- Turkish-aware normalization, suffix matching, and stopword filtering
- Groundedness checks and extractive fallback answers
- Server-Sent Events for retrieval, model, generation, and token updates
- Source filtering, evidence display, session history, and export
- Document upload, removal, reindexing, health checks, and benchmarking
- Local inference through Foundry Local and `phi-4-mini`

## Requirements

- Python 3.11 or newer
- Node.js and npm
- Foundry Local with the configured chat model downloaded

The default chat model is `phi-4-mini`. Override it with
`LOCAL_RAG_MODEL` when needed.

## Installation

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

### Frontend

```bash
cd frontend
npm install
```

## Development

Start the backend:

```bash
cd backend
.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Start the frontend in another terminal:

```bash
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). Vite proxies `/api`
requests to the FastAPI service on port `8000`.

## Production Build

Build the frontend:

```bash
cd frontend
npm run build
```

When `frontend/dist` exists, FastAPI serves the built frontend together with
the API:

```bash
cd backend
.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Project Structure

```text
backend/
  api/
    main.py          FastAPI application setup
    routes.py        HTTP and SSE endpoints
    state.py         In-memory session and filter state
  app/
    conversation.py  Follow-up resolution and session exports
    database.py      SQLite persistence
    documents.py     Document validation, parsing, chunking, and indexing
    evaluation.py    Retrieval metrics and model benchmarks
    health.py        Runtime health checks
    models.py        Embeddings, reranking, and Foundry Local access
    rag.py           Prompting, grounding, and answer orchestration
    retrieval.py     Dense, BM25, hybrid, and neighbor retrieval
    settings.py      Thresholds and project paths
    stopwords.py     NLTK Turkish stopwords and query-noise terms
  docs/              Indexed document corpus and data attribution
  tests/             Backend unit and integration tests
frontend/
  src/               React application
```

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/ask` | Stream a grounded answer through SSE |
| `GET` | `/api/health` | Return backend, index, and model health |
| `GET` | `/api/model` | Return the active model configuration |
| `GET` | `/api/config` | Return retrieval and generation settings |
| `GET` | `/api/stats` | Return index statistics |
| `GET` | `/api/sources` | List indexed sources |
| `GET` | `/api/chunks/{id}` | Return a single indexed chunk |
| `POST` | `/api/documents` | Upload a document |
| `DELETE` | `/api/documents/{name}` | Remove a managed document |
| `POST` | `/api/reindex` | Rebuild the document index |
| `GET` | `/api/history` | Return session history |
| `GET` | `/api/history/{id}/repeat` | Return a previous question for repetition |
| `GET` | `/api/filter` | Return the active source filter |
| `POST` | `/api/filter` | Set or clear the source filter |
| `POST` | `/api/export` | Export the session as Markdown or JSON |
| `POST` | `/api/benchmark` | Stream benchmark progress through SSE |

## Tests and Checks

Run backend tests:

```bash
cd backend
.venv/bin/python -m pytest -q
```

Run frontend checks:

```bash
cd frontend
npm run lint
npm run build
```

The current backend suite contains 217 passing tests.

## Configuration

Important values live in `backend/app/settings.py`. Calibration rationale is
recorded in `backend/docs/CALIBRATION.md`.

| Environment variable | Purpose |
|---|---|
| `LOCAL_RAG_MODEL` | Foundry Local chat model alias |
| `LOCAL_RAG_HOME` | Alternative backend project root |

Runtime data is stored under `backend/data` and is ignored by Git.

## Turkish Language Data

The base Turkish stopword set comes from the official NLTK `stopwords` corpus.
Application-specific query-noise terms are stored separately and combined at
import time. The list is vendored locally, so retrieval does not require an
internet connection at runtime.

Document and stopword sources are recorded in
`backend/docs/ATTRIBUTION.md`.

## License

This project is distributed under the MIT License. See `LICENSE`.
