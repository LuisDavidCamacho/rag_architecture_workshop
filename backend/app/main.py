"""FastAPI application entrypoint for the RAG workshop backend."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.api import router as advanced_rag_router


def create_app() -> FastAPI:
    """Create the FastAPI app with minimal scaffolding."""
    app = FastAPI(
        title="RAG Architecture Workshop",
        description=(
            "Backend scaffold for experimenting with multiple Retrieval-Augmented "
            "Generation architectures."
        ),
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(advanced_rag_router)

    @app.get("/health", tags=["health"])
    def health_check() -> dict[str, str]:
        """Basic liveness probe to verify the scaffold is wired correctly."""
        return {"status": "ok", "message": "Backend scaffold ready"}

    return app


app = create_app()
