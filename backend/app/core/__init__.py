"""Core application utilities and abstractions."""

from .chunking import DocumentChunker, chunk_email_records
from .database import FaissVectorStore
from .embeddings import EmbeddingGenerator
from .history import ConversationStore, StoredMessage
from .llm import LLMChatAgent

__all__ = [
    "FaissVectorStore",
    "DocumentChunker",
    "EmbeddingGenerator",
    "LLMChatAgent",
    "ConversationStore",
    "StoredMessage",
    "chunk_email_records",
]
