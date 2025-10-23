"""FastAPI application entrypoint for the RAG workshop backend."""

from fastapi import FastAPI


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

    @app.get("/healthz", tags=["health"])
    def health_check() -> dict[str, str]:
        """Basic liveness probe to verify the scaffold is wired correctly."""
        return {"status": "ok", "message": "Backend scaffold ready"}

    return app


app = create_app()
