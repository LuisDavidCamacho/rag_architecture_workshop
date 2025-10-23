# RAG Architecture Workshop

This repository scaffolds a hands-on workshop focused on building and comparing three Retrieval-Augmented Generation (RAG) patterns:

- Advanced RAG
- Graph RAG
- Reflective RAG

Each architecture will live on its own branch. The `main` branch contains the baseline project layout, development tooling, and shared frontend experience. Workshop participants will branch off from `main` to implement the specific architecture components.

## Repository Layout

- `backend/`
  - `app/main.py` — FastAPI entrypoint registering shared routers and health checks.
  - `app/api/api.py` — REST endpoints for the Advanced RAG scaffold:
    - `POST /api/query` starts a new chat, returning `{"chat_id": "<uuid>", "response": "<model output>"}`.
    - `POST /api/query/{chat_id}` continues an existing chat, reusing the provided identifier.
    - `POST /api/embed` triggers corpus chunking + embedding generation and returns a simple status payload.
  - `app/services/services.py` — Service-layer functions (`start_new_chat`, `continue_chat`, `embed_documents`) intentionally left unfinished. Participants will connect these to chunking, embedding, and FAISS utilities to complete the RAG loop.
  - `app/core/` — Helper abstractions used across architectures:
    - `database.py` wraps FAISS index creation, persistence, and nearest-neighbor search.
    - `chunking.py` implements a reusable document chunker with overlap control.
    - `embeddings.py` converts Polars DataFrames into embedding vectors via Ollama.
    - `llm.py` provides an `LLMChatAgent` wrapper that automatically replays log history from `ConversationStore` and records every turn for later analysis.
    - `history.py` offers a `ConversationStore` (with a `default()` helper pointing to `outputs/advanced_rag/conversations`) so transcripts are written to JSONL and ready for post-workshop analytics.
  - `app/models/schemas.py` — Pydantic request/response models backing the API definitions.
- `frontend/` — React + Material UI single-page application scaffolded with Vite (TypeScript). `src/App.tsx` wires the initial UI, while `src/components/WorkshopShell.tsx` renders the landing shell participants will extend in each branch.
- `infra/`
  - `docker/` contains Dockerfiles for backend (FastAPI + Poetry), frontend (Node build + Nginx serve), and the Ollama runtime with the preloaded `llama3.1:8b` model.
  - `docker-compose.yml` orchestrates all services for a local workshop environment.
- `data/` — Placeholder directory for corpus inputs that will be chunked and embedded.
- `outputs/` — Architecture-specific folders for storing model responses or evaluation artifacts.
- `scripts/` — Reserved for helper scripts (currently empty).

## REST Examples

Start a chat:

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the onboarding guide"}'
```

Continue the same chat:

```bash
curl -X POST http://localhost:8000/api/query/<chat_id> \
  -H "Content-Type: application/json" \
  -d '{"query": "Can you provide key action items?"}'
```

Request embeddings:

```bash
curl -X POST http://localhost:8000/api/embed \
  -H "Content-Type: application/json" \
  -d '{"documents": ["Doc 1 text", "Doc 2 text"], "chunk_size": 512, "overlap": 50}'
```

## Workshop Stages

1. Establish core project structure (current stage).
2. Add Docker assets for local orchestration.
3. Define Docker Compose topologies per architecture.
4. Manage Python dependencies via Poetry.
5. Expand the frontend to support RAG experimentation flows.
6. Create architecture-specific branches and fill in implementation challenges.

## Why the Services Matter

The `services.py` module is where each architecture’s secret sauce lives. By implementing `start_new_chat`, `continue_chat`, and `embed_documents`, participants will connect API payloads to the chunking, embedding, and FAISS utilities provided above. This layer is intentionally unfinished so you can experiment with prompt engineering, retrieval strategies, and evaluation techniques that differentiate Advanced RAG from the other branches.

Follow the instructor-led steps to progressively build out each architecture. The backend and frontend currently expose only the minimal scaffolding required to get started.
