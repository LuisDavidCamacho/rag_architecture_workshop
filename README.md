# RAG Architecture Workshop

This repository scaffolds a hands-on workshop focused on building and comparing three Retrieval-Augmented Generation (RAG) patterns:

- Advanced RAG
- Graph RAG
- Reflective RAG

Each architecture will live on its own branch. The `main` branch contains the baseline project layout, development tooling, and shared frontend experience. Workshop participants will branch off from `main` to implement the specific architecture components.

## Repository Layout

- `backend/`
  - `app/main.py` — FastAPI entrypoint that wires routers and middleware.
 - `app/api/` — REST endpoints shared across workshop branches (e.g., `/api/query`, `/api/embed`).
 - `app/models/` — Pydantic request/response schemas.
 - `app/services/` — Placeholder business logic where each architecture will plug in its pipeline.
  - `app/core/`
    - `chunking.py` — Utilities to split documents into manageable chunks; includes helpers tuned for the Enron email corpus (e.g., `chunk_email_records`).
    - `database.py` — Helpers for vector/graph persistence (FAISS, file stores, GraphRAG scaffolds) including `GraphFileStore` for reading pipeline artifacts.
    - `embeddings.py` — Embedding + graph artifact generation scaffolds, with methods to embed dataframe rows or raw text lists via LangChain and helper utilities for entity extraction/co-occurrence graphs.
    - `llm.py` — LangChain/Ollama wrapper that records conversations to disk.
    - `history.py` — JSONL conversation logging for post-workshop metrics.
- `frontend/` — React + Material UI interface with a WhatsApp-style chat and embedding trigger.
- `infra/` — Dockerfiles (backend, frontend, Ollama) and top-level `docker-compose.yml`.
- `data/` — Drop workshop corpora here before chunking/embedding.
- `outputs/` — Generated artifacts (embeddings, graph files, chat logs) per architecture.
- `scripts/` — Reserved for automation helpers (currently empty).

## Sample Corpus

- `data/corpus/emails.csv`
  - Columns:
    - `file` — Relative path/name for the original PST export message.
    - `message` — Full raw email content (headers + body).
  - Origin: Extracted from the Enron Email Dataset published on Kaggle after the Enron scandal. The workshop ships this curated CSV so everyone works from the same corpus without needing to download the full archive.
  - The embedding endpoint is intentionally opinionated: it always reads this CSV from the backend filesystem and prepares chunks/embeddings for the “Advanced RAG” exercises.

## Backend Directory Hints

- `app/api/api.py` exposes three core endpoints:
  - `POST /api/query` — creates a new chat session and returns `chat_id` + model reply.
  - `POST /api/query/{chat_id}` — continues an existing conversation.
  - `POST /api/embed` — ingests documents and triggers chunking/embedding.
- `app/services/services.py` contains unfinished functions (`start_new_chat`, `continue_chat`, `embed_documents`) that the workshop participants will complete.
- `app/core/embeddings.py` already provides convenience methods for creating embeddings, extracting simple entities, persisting graph artifacts under `outputs/graph_rag`, and generating vectors directly from raw text (via LangChain's Ollama integration).
- `app/services/services.py` now contains a concrete `embed_documents` implementation that reads `data/corpus/emails.csv`, calls the shared chunking/embedding utilities, and persists embeddings for downstream use. It also exposes Graph RAG-specific stubs (`build_graph_rag_index`, `graph_rag_query`) that workshop participants will implement.
- `app/core/database.py` exposes both `FaissVectorStore` and `GraphFileStore`, enabling you to read GraphRAG pipeline artifacts directly from the filesystem.
- `app/core/history.py` shows how chats are persisted so you can reuse transcripts after implementing retrieval.

## Advanced RAG Challenge Guidance

The goal is to transform the baseline endpoints into a full retrieval-augmented experience. Here are broad steps to guide your implementation without revealing the final solution:

1. **Document Preparation**
   - The reference implementation already demonstrates reading `data/corpus/emails.csv` (Enron email corpus) and chunking each message; use it as a baseline before experimenting with alternative datasets or chunking logic.
   - Consider recording metadata (e.g., document titles, chunk indices) so retrieval can point back to source context.
2. **Corpus Storage**
   - Persist embeddings via `FaissVectorStore` (or a store of your choosing). Decide how to map chunk IDs to original documents and where to serialize the index (`outputs/advanced_rag` is the suggested location).
   - Optionally leverage `EmbeddingGenerator.build_graph` for relationship insights, but keep the initial focus on vector retrieval.
3. **Retrieval Pipeline**
   - In `start_new_chat`, translate the user’s query into an embedding and query the vector store. Format the results as a compact context bundle (snippets, citations, etc.).
   - Weave retrieved snippets into a prompt template that instructs the LLM how to use evidence. The `LLMChatAgent.chat` helper already handles conversation history and logging.
   - Decide how many results to retrieve and how to rank them (similarity threshold, top-k, etc.). Justify your choice in comments or documentation.
4. **Chat Continuation**
   - `continue_chat` should reuse the stored `chat_id`, gather the previous context, and optionally perform follow-up retrieval (e.g., based on dialogue intent). Think about whether you want to store the user’s question before or after the model reply.
5. **Evaluation & Observability**
   - Use `ConversationStore.export` or the JSONL transcripts to review how well the model references retrieved evidence.
   - You might add lightweight instrumentation (timings, retrieved chunk IDs) to better understand retrieval quality during the workshop.

Treat the above as a checklist rather than a prescription. The workshop is designed to provoke experimentation—adapt chunking strategies, improve prompt wording, or bolt on re-ranking as you see fit. The only constraint: keep everything runnable locally via the provided Docker/Vite setup so the audience can iterate quickly.

## Graph RAG Challenge Guidance

When you switch to the Graph RAG branch, build upon the new scaffolding:

1. **Graph Construction**
   - Use `EmbeddingGenerator.extract_entities` and `EmbeddingGenerator.build_graph` to emit co-occurrence nodes/edges for the Enron corpus.
   - Extend or replace the heuristics with GraphRAG’s extraction pipelines (e.g., claim/relationship extraction) and persist results using the same directory structure.
2. **Graph Storage & Inspection**
   - Leverage `GraphFileStore.query_graph_file` to read GraphRAG-compatible artifacts (JSON/JSONL) from disk. This mirrors the default file-based pipeline storage.
3. **Service Layer Stubs**
   - Implement `build_graph_rag_index` to orchestrate extraction, graph persistence, and optional summarisation ahead of query time.
   - Implement `graph_rag_query` to combine graph traversal (communities, relationship hops, claim summaries) with language model prompting.
4. **Prompting & Evaluation**
   - Compare graph-enriched answers with the Advanced RAG baseline. Capture which nodes/edges were used and how they influence responses.

Document your choices so participants can appreciate the trade-offs between vector-only and graph-aware retrieval.

## REST Examples

- Start a new chat session:

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarise Enron's West Coast strategy."}'
```

- Continue an existing chat (replace `<chat_id>` with the identifier returned above):

```bash
curl -X POST http://localhost:8000/api/query/<chat_id> \
  -H "Content-Type: application/json" \
  -d '{"query": "Highlight any risk factors mentioned previously."}'
```

- Trigger embeddings for the bundled Enron email corpus:

```bash
curl -X POST http://localhost:8000/api/embed \
  -H "Content-Type: application/json" \
  -d '{"filename": "emails.csv"}'
```

The embed endpoint reads `data/corpus/emails.csv`, produces chunk-level embeddings, and stores artifacts in `outputs/advanced_rag/` for downstream retrieval experiments.
