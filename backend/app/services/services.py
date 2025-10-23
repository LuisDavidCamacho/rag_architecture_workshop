"""Service layer skeleton for Advanced RAG operations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple
from uuid import UUID

from ..core import EmbeddingGenerator
from ..core.chunking import chunk_email_records


def start_new_chat(user_query: str) -> Tuple[UUID, str]:
    """
    Initialize a new conversational session for the Advanced RAG workflow.

    Returns the chat session UUID and the model response to the initial query.
    """
    raise NotImplementedError("Advanced RAG start_new_chat service not implemented.")


def continue_chat(chat_id: UUID, user_query: str) -> str:
    """
    Continue an existing chat session by appending a new user query.

    Returns the model-generated response tied to the provided chat_id.
    """
    raise NotImplementedError("Advanced RAG continue_chat service not implemented.")


def embed_documents(filename: str, *, chunk_size: int, overlap: int) -> int:
    """
    Generate embeddings for the specified CSV corpus located under data/corpus.

    This implementation is tailored for the Enron email dataset (emails.csv) shipped
    with the workshop materials.
    """
    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "polars is required to process the email corpus. Add it via Poetry."
        ) from exc

    safe_name = Path(filename).name
    corpus_path = Path("data/corpus") / safe_name
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus file '{safe_name}' was not found in data/corpus (looked for {corpus_path})."
        )

    dataframe = pl.read_csv(corpus_path)
    expected_columns = {"file", "message"}
    if not expected_columns.issubset(dataframe.columns):
        raise ValueError(
            f"Corpus file '{filename}' must contain columns {expected_columns}. "
            f"Found {set(dataframe.columns)} instead."
        )

    records = (
        (row["file"], row["message"])
        for row in dataframe.iter_rows(named=True)
    )
    chunk_ids, chunk_texts, chunk_sources = chunk_email_records(
        records,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if not chunk_ids:
        return 0

    embedding_generator = EmbeddingGenerator(model_name="llama3.1:8b")
    embeddings = embedding_generator.generate_from_texts(chunk_ids, chunk_texts)

    # Persist embeddings for downstream retrieval challenges.
    output_dir = Path("outputs/advanced_rag/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "email_embeddings.jsonl"

    with embeddings_path.open("w", encoding="utf-8") as handle:
        for chunk_id, vector in embeddings:
            payload = {
                "chunk_id": chunk_id,
                "source_file": chunk_sources.get(chunk_id, chunk_id.split("::")[0]),
                "embedding": vector,
            }
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")

    return len(embeddings)


def build_graph_rag_index(filename: str, *, chunk_size: int, overlap: int) -> None:
    """Prepare graph-based artifacts for the specified corpus (Graph RAG)."""
    raise NotImplementedError("Graph RAG index construction not implemented yet.")


def graph_rag_query(chat_id: UUID, user_query: str) -> str:
    """Execute a Graph RAG retrieval + generation workflow."""
    raise NotImplementedError("Graph RAG query flow not implemented yet.")
