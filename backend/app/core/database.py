"""Abstractions for working with FAISS-backed vector storage."""

from __future__ import annotations

import json
import os
import asyncio
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from graphrag.storage.file_pipeline_storage import FilePipelineStorage


class FaissVectorStore:
    """
    Wrapper responsible for initializing and interacting with a FAISS index.

    This class delegates the concrete FAISS operations to be filled in during the
    workshop exercises.
    """

    def __init__(self, dimension: int, index_path: Optional[str] = None) -> None:
        self.dimension = dimension
        self.index_path = index_path
        self._ids_path = f"{index_path}.ids.json" if index_path else None
        self._index = None  # lazily populated faiss.IndexFlatL2 instance.
        self._document_ids: List[str] = []

    def initialize(self) -> None:
        """
        Instantiate the FAISS index structure for the configured dimensionality.
        """
        try:
            import faiss  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "faiss-cpu package not installed. Add it via Poetry before "
                "initializing the vector store."
            ) from exc

        if self._index is None:
            self._index = faiss.IndexFlatL2(self.dimension)

            if self.index_path:
                try:
                    self._index = faiss.read_index(self.index_path)
                except Exception:  # noqa: BLE001 - best-effort restore
                    self._index = faiss.IndexFlatL2(self.dimension)
                else:
                    self._restore_document_ids()

    def add_embeddings(
        self,
        embeddings: Iterable[Tuple[str, List[float]]],
    ) -> None:
        """
        Persist the provided embeddings into the FAISS index.

        The iterable should yield pairs of (document_id, vector).
        """
        if self._index is None:
            raise RuntimeError("FAISS index not initialized. Call initialize() first.")

        try:
            import faiss  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "faiss-cpu package not installed. Add it via Poetry before "
                "adding embeddings."
            ) from exc

        new_ids: List[str] = []
        vectors: List[List[float]] = []

        for doc_id, vector in embeddings:
            if len(vector) != self.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch. Expected {self.dimension}, "
                    f"got {len(vector)} for document {doc_id}."
                )
            new_ids.append(doc_id)
            vectors.append(vector)

        if not vectors:
            return

        vector_array = np.asarray(vectors, dtype="float32")
        self._index.add(vector_array)
        self._document_ids.extend(new_ids)

        if self.index_path:
            faiss.write_index(self._index, self.index_path)
        self._persist_document_ids()

    def query(self, vector: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Perform a similarity search against the FAISS index.

        Returns a list of (document_id, score) pairs representing nearest neighbors.
        """
        if self._index is None:
            raise RuntimeError("FAISS index not initialized. Call initialize() first.")

        if len(vector) != self.dimension:
            raise ValueError(
                f"Query dimension mismatch. Expected {self.dimension}, "
                f"got {len(vector)}."
            )

        query_array = np.asarray([vector], dtype="float32")
        distances, indices = self._index.search(query_array, top_k)

        results: List[Tuple[str, float]] = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            document_id = (
                self._document_ids[idx] if idx < len(self._document_ids) else str(idx)
            )
            results.append((document_id, float(distance)))

        return results

    def _restore_document_ids(self) -> None:
        if not self._ids_path or not os.path.exists(self._ids_path):
            self._document_ids = []
            return

        try:
            with open(self._ids_path, "r", encoding="utf-8") as file:
                self._document_ids = json.load(file)
        except Exception:  # noqa: BLE001 - best-effort restore
            self._document_ids = []

    def _persist_document_ids(self) -> None:
        if not self._ids_path:
            return

        try:
            with open(self._ids_path, "w", encoding="utf-8") as file:
                json.dump(self._document_ids, file)
        except Exception:  # noqa: BLE001 - persistence failures should not crash callers
            pass
class GraphFileStore:
    """File-backed graph store compatible with GraphRAG defaults."""

    def __init__(self, base_dir: str | Path = "outputs/graph_rag") -> None:
        self.base_dir = Path(base_dir)
        self.storage = FilePipelineStorage(base_dir=str(self.base_dir), encoding="utf-8")

    async def _get_async(
        self,
        key: str,
        *,
        as_bytes: bool = False,
        encoding: Optional[str] = None,
    ) -> str | bytes:
        data = await self.storage.get(key, as_bytes=as_bytes, encoding=encoding)
        if data is None:
            raise FileNotFoundError(
                f"Graph artifact '{key}' not found under {self.base_dir}."
            )
        return data

    def query_graph_file(
        self,
        key: str,
        *,
        parse_json: bool = True,
    ) -> list | dict | str | bytes:
        async def loader() -> str | bytes:
            return await self._get_async(key)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            payload = asyncio.run(loader())
        else:
            payload = loop.run_until_complete(loader())

        if not parse_json:
            return payload

        if isinstance(payload, bytes):
            payload_text = payload.decode("utf-8")
        else:
            payload_text = payload

        payload_text = payload_text.strip()
        if not payload_text:
            return []

        try:
            return json.loads(payload_text)
        except json.JSONDecodeError:
            records = []
            for line in payload_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
            return records
