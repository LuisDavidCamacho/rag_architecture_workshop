"""Embedding generation utilities for Advanced RAG."""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl
    from fastembed import TextEmbedding


class EmbeddingGenerator:
    """
    Abstraction over the embedding model interface.

    The workshop will extend this class to call a concrete embedding model (e.g.,
    via FastEmbed or a Hugging Face pipeline) and convert results into vectors ready
    for persistence.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        id_column: str = "id",
        text_column: str = "text",
        embedder: Optional["TextEmbedding"] = None,
        batch_size: int = 128,
    ) -> None:
        self.model_name = model_name or os.getenv(
            "FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5"
        )
        self.id_column = id_column
        self.text_column = text_column
        env_batch_size = os.getenv("FASTEMBED_BATCH_SIZE")
        if env_batch_size:
            try:
                batch_size = int(env_batch_size)
            except ValueError:
                raise ValueError("FASTEMBED_BATCH_SIZE must be an integer.") from None

        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        self.batch_size = batch_size
        self._embedder: Optional["TextEmbedding"] = embedder

    def generate(self, dataframe: "pl.DataFrame") -> List[Tuple[str, List[float]]]:
        """
        Produce embeddings for the provided Polars DataFrame.

        The DataFrame is expected to contain at least an identifier column and a
        textual payload column. Returns (document_id, embedding_vector) tuples.
        """
        try:
            import polars as pl
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "polars is not installed. Add it to the Poetry dependencies to use "
                "the embedding generator."
            ) from exc

        if not isinstance(dataframe, pl.DataFrame):
            raise TypeError("Expected a Polars DataFrame as input.")

        if self.id_column not in dataframe.columns:
            raise ValueError(
                f"Missing identifier column '{self.id_column}' in DataFrame."
            )

        if self.text_column not in dataframe.columns:
            raise ValueError(
                f"Missing text column '{self.text_column}' in DataFrame."
            )

        id_series = dataframe[self.id_column].cast(pl.Utf8, strict=False)
        text_series = dataframe[self.text_column].cast(pl.Utf8, strict=False)

        identifiers = id_series.to_list()
        documents = text_series.to_list()

        if not documents:
            return []

        embeddings = self._embed_documents(documents)

        if len(embeddings) != len(identifiers):
            raise RuntimeError(
                "Embedding output size mismatch. Ensure the embedding model returns "
                "one vector per input document."
            )

        return [
            (identifier, vector)
            for identifier, vector in zip(identifiers, embeddings, strict=True)
        ]

    def generate_from_texts(
        self, identifiers: Iterable[str], documents: Iterable[str]
    ) -> List[Tuple[str, List[float]]]:
        """Generate embeddings directly from lists of ids and documents."""
        ids = list(identifiers)
        docs = list(documents)

        if not ids or not docs:
            return []

        if len(ids) != len(docs):
            raise ValueError("Identifiers and documents must have the same length.")

        print("*" * 20)
        print("generating embeddings in embedding generator")
        print("ids length:", len(docs))
        print("*" * 20)
        embeddings = self._embed_documents(docs)
        if len(embeddings) != len(ids):
            raise RuntimeError("Embedding model returned unexpected number of vectors.")
        print("*" * 20)
        print("embedding generation complete")
        print("returning embeddings")
        print("*" * 20)
        return list(zip(ids, embeddings, strict=True))

    def embed_batch(self, documents: List[str]) -> List[List[float]]:
        """Embed a batch of documents without extra bookkeeping."""
        if not documents:
            return []
        return self._embed_documents(documents)

    def _ensure_embedder(self) -> "TextEmbedding":
        if self._embedder is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "FastEmbed is not installed. Add the 'fastembed' package via Poetry."
                ) from exc

            self._embedder = TextEmbedding(model_name=self.model_name)
        return self._embedder

    def _embed_documents(self, docs: List[str]) -> List[List[float]]:
        embedder = self._ensure_embedder()

        embeddings: List[List[float]] = []
        for batch in _batched(docs, self.batch_size):
            if not batch:
                continue
            for vector in embedder.embed(batch, batch_size=self.batch_size):
                embeddings.append(vector.tolist())

        return embeddings


def _batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
