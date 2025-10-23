"""Utilities for chunking raw documents prior to embedding."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    """
    Strategy object responsible for splitting documents into manageable chunks.

    Concrete chunking logic is intentionally left unimplemented for the workshop.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, documents: Iterable[str]) -> List[str]:
        """
        Split the incoming documents into chunks based on configured parameters.

        Returns a list of textual chunks ready for embedding.
        """
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")

        if self.overlap < 0:
            raise ValueError("overlap cannot be negative.")

        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size to avoid loops.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
        )

        chunks: List[str] = []

        for document in documents:
            if not document:
                continue
            split_chunks = splitter.split_text(document)
            chunks.extend(segment.strip() for segment in split_chunks if segment.strip())

        return chunks


def chunk_email_corpus(message: str, *, chunk_size: int, overlap: int) -> List[str]:
    """Chunk a single email using LangChain's recursive splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    return [segment.strip() for segment in splitter.split_text(message) if segment.strip()]


def chunk_email_records(
    records: Iterable[tuple[str, str]], *, chunk_size: int, overlap: int
) -> Tuple[List[str], List[str], Dict[str, str]]:
    """Chunk (file, message) pairs into overlapping snippets."""
    chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap)
    chunk_ids: List[str] = []
    chunk_texts: List[str] = []
    chunk_sources: Dict[str, str] = {}

    for source_file, message in records:
        text = (message or "").strip()
        if not text:
            continue
        chunks = chunker.chunk([text])
        for index, chunk in enumerate(chunks):
            chunk_id = f"{source_file}::chunk-{index}"
            chunk_ids.append(chunk_id)
            chunk_texts.append(chunk)
            chunk_sources[chunk_id] = source_file

    return chunk_ids, chunk_texts, chunk_sources
