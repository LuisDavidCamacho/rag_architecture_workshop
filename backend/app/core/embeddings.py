"""Embedding and entity extraction utilities for RAG workflows."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, TYPE_CHECKING

import os
from langchain_core.embeddings import Embeddings

if TYPE_CHECKING:
    import polars as pl


class EmbeddingGenerator:
    """
    Abstraction over the embedding model interface.

    The workshop will extend this class to call a concrete embedding model (e.g.,
    via Ollama or a Hugging Face pipeline) and convert results into vectors ready
    for persistence.
    """

    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        *,
        id_column: str = "id",
        text_column: str = "text",
        embedder: Optional[Embeddings] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.id_column = id_column
        self.text_column = text_column
        self._embedder: Optional[Embeddings] = embedder
        self._base_url = base_url or os.getenv("OLLAMA_BASE_URL")

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

        if self._embedder is None:
            try:
                from langchain_ollama import OllamaEmbeddings
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "langchain-ollama is not installed. Add it via Poetry to generate "
                    "embeddings using Ollama."
                ) from exc

            self._embedder = OllamaEmbeddings(
                model=self.model_name,
                base_url=self._base_url,
            )

        embeddings = self._embedder.embed_documents(documents)

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

        if self._embedder is None:
            try:
                from langchain_ollama import OllamaEmbeddings
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "langchain-ollama is not installed. Add it via Poetry to generate embeddings."
                ) from exc

            self._embedder = OllamaEmbeddings(
                model=self.model_name,
                base_url=self._base_url,
            )

        embeddings = self._embedder.embed_documents(docs)
        if len(embeddings) != len(ids):
            raise RuntimeError("Embedding model returned unexpected number of vectors.")

        return list(zip(ids, embeddings))

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
