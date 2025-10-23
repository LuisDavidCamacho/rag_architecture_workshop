# RAG Architecture Workshop

This repository scaffolds a hands-on workshop focused on building and comparing three Retrieval-Augmented Generation (RAG) patterns:

- Advanced RAG
- Graph RAG
- Reflective RAG

Each architecture will live on its own branch. The `main` branch contains the baseline project layout, development tooling, and shared frontend experience. Workshop participants will branch off from `main` to implement the specific architecture components.

## Repository Layout

- `backend/` — FastAPI service scaffold that will host shared workshop APIs.
- `frontend/` — React + Material UI single-page application for interacting with each RAG variant.
- `infra/` — Dockerfiles, compose definitions, and infrastructure helpers (to be filled in stage-by-stage).
- `scripts/` — Utility scripts that make it easy to bootstrap and manage workshop environments.

## Workshop Stages

1. Establish core project structure (current stage).
2. Add Docker assets for local orchestration.
3. Define Docker Compose topologies per architecture.
4. Manage Python dependencies via Poetry.
5. Expand the frontend to support RAG experimentation flows.
6. Create architecture-specific branches and fill in implementation challenges.

Follow the instructor-led steps to progressively build out each architecture. The backend and frontend currently expose only the minimal scaffolding required to get started.

