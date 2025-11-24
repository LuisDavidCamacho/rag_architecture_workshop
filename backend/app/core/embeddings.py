"""Embedding and entity extraction utilities for RAG workflows."""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Iterable, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl
    from fastembed import TextEmbedding


class EmbeddingGenerator:
    """
    Abstraction over the embedding model interface.

    The workshop will extend this class to call a concrete embedding model (e.g.,
    via Ollama or a Hugging Face pipeline) and convert results into vectors ready
    for persistence.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        id_column: str = "id",
        text_column: str = "text",
        embedder: Optional["TextEmbedding"] = None,
        batch_size: int = 256,
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
            except ValueError as exc:  # pragma: no cover
                raise ValueError("FASTEMBED_BATCH_SIZE must be an integer.") from exc

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
        return list(self.iter_embeddings(identifiers, documents))

    def iter_embeddings(
        self, identifiers: Iterable[str], documents: Iterable[str]
    ) -> Iterable[Tuple[str, List[float]]]:
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
        for id_batch, doc_batch in _batched_pairs(ids, docs, self.batch_size):
            vectors = self._embed_documents(doc_batch)
            if len(vectors) != len(id_batch):
                raise RuntimeError(
                    "Embedding model returned unexpected number of vectors."
                )
            for chunk_id, vector in zip(id_batch, vectors, strict=True):
                yield chunk_id, vector
        print("*" * 20)
        print("embedding generation complete")
        print("returning embeddings")
        print("*" * 20)

    def embed_batch(self, documents: List[str]) -> List[List[float]]:
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
        start_time = perf_counter()
        for batch in _batched(docs, self.batch_size):
            if not batch:
                continue
            for vector in embedder.embed(batch, batch_size=self.batch_size):
                embeddings.append(vector.tolist())
        elapsed = perf_counter() - start_time
        if elapsed:
            print(
                f"FastEmbed generated {len(embeddings)} vectors in "
                f"{elapsed:.2f}s (~{len(embeddings) / max(elapsed, 1e-6):.1f} chunks/s)."
            )

        return embeddings

    def extract_entities(self, text: str) -> List[str]:
        """Extract entities (people, orgs, emails) using simple heuristics."""
        email_pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
        capitalised_pattern = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")

        candidates = email_pattern.findall(text)
        candidates.extend(capitalised_pattern.findall(text))

        seen: set[str] = set()
        unique_entities: List[str] = []
        for candidate in candidates:
            key = candidate.lower()
            if key not in seen:
                seen.add(key)
                unique_entities.append(candidate)
        return unique_entities

def build_graph(
        self,
        dataframe: "pl.DataFrame",
        *,
        output_dir: str | Path = "outputs/graph_rag",
    ) -> dict[str, int]:
        """
        Build a lightweight co-occurrence graph from a dataframe of emails.

        Returns a summary dictionary containing the number of nodes and edges.
        """
        try:
            import polars as pl
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("polars is required to build graph artifacts.") from exc

        if not isinstance(dataframe, pl.DataFrame):
            raise TypeError("Expected a polars.DataFrame when building the graph.")

        if not {"file", "message"}.issubset(dataframe.columns):
            raise ValueError("DataFrame must contain 'file' and 'message' columns.")

        node_frequency: Counter[str] = Counter()
        edges: defaultdict[tuple[str, str], int] = defaultdict(int)

        for row in dataframe.iter_rows(named=True):
            entities = self.extract_entities(row["message"] or "")
            if not entities:
                continue
            node_frequency.update(entities)
            for i, source in enumerate(entities):
                for target in entities[i + 1 :]:
                    edge = tuple(sorted((source, target)))
                    edges[edge] += 1

        graph_path = Path(output_dir)
        graph_path.mkdir(parents=True, exist_ok=True)

        nodes_file = graph_path / "nodes.jsonl"
        edges_file = graph_path / "edges.jsonl"

        with nodes_file.open("w", encoding="utf-8") as nf:
            for entity, freq in node_frequency.items():
                nf.write(
                    json.dumps({"id": entity, "label": entity, "frequency": freq})
                )
                nf.write("\n")

        with edges_file.open("w", encoding="utf-8") as ef:
            for (source, target), weight in edges.items():
                ef.write(
                    json.dumps(
                        {
                            "source": source,
                            "target": target,
                            "weight": weight,
                        }
                    )
                )
                ef.write("\n")

        return {"nodes": len(node_frequency), "edges": len(edges)}


def _batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _batched_pairs(
    ids: List[str], docs: List[str], batch_size: int
) -> Iterable[Tuple[List[str], List[str]]]:
    for start in range(0, len(ids), batch_size):
        yield ids[start : start + batch_size], docs[start : start + batch_size]
